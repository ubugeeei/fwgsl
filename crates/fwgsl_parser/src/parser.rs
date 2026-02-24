//! Recursive descent parser for fwgsl.
//!
//! Produces a simple AST. Expression parsing uses Pratt (precedence climbing).

use fwgsl_diagnostics::{Diagnostic, DiagnosticSink, Label};
use fwgsl_span::Span;
use fwgsl_syntax::SyntaxKind;

use crate::layout::resolve_layout;
use crate::lexer::{lex, Token};

// ═══════════════════════════════════════════════════════════════════════════
// AST types
// ═══════════════════════════════════════════════════════════════════════════

/// A parsed fwgsl program.
#[derive(Debug, Clone)]
pub struct Program {
    pub decls: Vec<Decl>,
}

#[derive(Debug, Clone)]
pub enum Decl {
    TypeSig {
        name: String,
        ty: Type,
        span: Span,
    },
    FunDecl {
        name: String,
        params: Vec<Pat>,
        body: Expr,
        where_binds: Vec<(String, Expr)>,
        span: Span,
    },
    DataDecl {
        name: String,
        type_params: Vec<String>,
        constructors: Vec<ConDecl>,
        span: Span,
    },
    EntryPoint {
        attributes: Vec<Attribute>,
        name: String,
        params: Vec<Pat>,
        body: Expr,
        span: Span,
    },
    TypeAlias {
        name: String,
        params: Vec<String>,
        ty: Type,
        span: Span,
    },
}

#[derive(Debug, Clone)]
pub struct ConDecl {
    pub name: String,
    pub fields: ConFields,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum ConFields {
    Positional(Vec<Type>),
    Record(Vec<(String, Type)>),
    Empty,
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<String>,
    pub span: Span,
}

#[derive(Debug, Clone)]
pub enum Expr {
    Lit(Lit, Span),
    Var(String, Span),
    Con(String, Span),
    App(Box<Expr>, Box<Expr>, Span),
    Infix(Box<Expr>, String, Box<Expr>, Span),
    Lambda(Vec<Pat>, Box<Expr>, Span),
    Let(Vec<(String, Expr)>, Box<Expr>, Span),
    Case(Box<Expr>, Vec<(Pat, Expr)>, Span),
    If(Box<Expr>, Box<Expr>, Box<Expr>, Span),
    Paren(Box<Expr>, Span),
    Tuple(Vec<Expr>, Span),
    Record(Vec<(String, Expr)>, Span),
    FieldAccess(Box<Expr>, String, Span),
    OpSection(String, Span),
    Neg(Box<Expr>, Span),
    Do(Vec<DoStmt>, Span),
    /// Vec literal: `[a, b, c]` — desugared to vecN constructor call.
    VecLit(Vec<Expr>, Span),
}

#[derive(Debug, Clone)]
pub enum DoStmt {
    Bind(String, Expr, Span),
    Expr(Expr, Span),
    Let(String, Expr, Span),
}

#[derive(Debug, Clone)]
pub enum Lit {
    Int(i64),
    Float(f64),
    String(String),
    Char(char),
}

#[derive(Debug, Clone)]
pub enum Pat {
    Wild(Span),
    Var(String, Span),
    Con(String, Vec<Pat>, Span),
    Lit(Lit, Span),
    Paren(Box<Pat>, Span),
    Tuple(Vec<Pat>, Span),
    Record(String, Vec<(String, Option<Pat>)>, Span),
    As(String, Box<Pat>, Span),
}

#[derive(Debug, Clone)]
pub enum Type {
    Con(String, Span),
    Var(String, Span),
    App(Box<Type>, Box<Type>, Span),
    Arrow(Box<Type>, Box<Type>, Span),
    Paren(Box<Type>, Span),
    Tuple(Vec<Type>, Span),
    Unit(Span),
}

// ═══════════════════════════════════════════════════════════════════════════
// Span accessors for AST nodes
// ═══════════════════════════════════════════════════════════════════════════

impl Expr {
    pub fn span(&self) -> Span {
        match self {
            Expr::Lit(_, s)
            | Expr::Var(_, s)
            | Expr::Con(_, s)
            | Expr::App(_, _, s)
            | Expr::Infix(_, _, _, s)
            | Expr::Lambda(_, _, s)
            | Expr::Let(_, _, s)
            | Expr::Case(_, _, s)
            | Expr::If(_, _, _, s)
            | Expr::Paren(_, s)
            | Expr::Tuple(_, s)
            | Expr::Record(_, s)
            | Expr::FieldAccess(_, _, s)
            | Expr::OpSection(_, s)
            | Expr::Neg(_, s)
            | Expr::Do(_, s)
            | Expr::VecLit(_, s) => *s,
        }
    }
}

impl Type {
    pub fn span(&self) -> Span {
        match self {
            Type::Con(_, s)
            | Type::Var(_, s)
            | Type::App(_, _, s)
            | Type::Arrow(_, _, s)
            | Type::Paren(_, s)
            | Type::Tuple(_, s)
            | Type::Unit(s) => *s,
        }
    }
}

impl Pat {
    pub fn span(&self) -> Span {
        match self {
            Pat::Wild(s)
            | Pat::Var(_, s)
            | Pat::Con(_, _, s)
            | Pat::Lit(_, s)
            | Pat::Paren(_, s)
            | Pat::Tuple(_, s)
            | Pat::Record(_, _, s)
            | Pat::As(_, _, s) => *s,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Parser
// ═══════════════════════════════════════════════════════════════════════════

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    fuel: u32,
    diagnostics: DiagnosticSink,
    source: String,
}

const MAX_FUEL: u32 = 10_000;

impl Parser {
    /// Create a new parser from source text. Lexes and resolves layout.
    pub fn new(source: &str) -> Self {
        let raw_tokens = lex(source);
        let tokens = resolve_layout(raw_tokens, source);
        Self {
            tokens,
            pos: 0,
            fuel: MAX_FUEL,
            diagnostics: DiagnosticSink::new(),
            source: source.to_owned(),
        }
    }

    /// Return the diagnostics accumulated during parsing.
    pub fn diagnostics(&self) -> &DiagnosticSink {
        &self.diagnostics
    }

    // -- Navigation helpers ------------------------------------------------

    fn current_token(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn peek(&self) -> SyntaxKind {
        self.current_token().kind
    }

    fn at(&self, kind: SyntaxKind) -> bool {
        self.peek() == kind
    }

    fn at_end(&self) -> bool {
        self.peek() == SyntaxKind::Eof
    }

    fn bump(&mut self) -> Token {
        let tok = self.current_token().clone();
        if !self.at_end() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, kind: SyntaxKind) -> Token {
        if self.at(kind) {
            self.bump()
        } else {
            let tok = self.current_token().clone();
            self.diagnostics.push(
                Diagnostic::error(format!("expected {}, found {}", kind, tok.kind))
                    .with_label(Label::primary(tok.span, format!("expected {}", kind))),
            );
            tok
        }
    }

    fn eat(&mut self, kind: SyntaxKind) -> bool {
        if self.at(kind) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn skip_trivia(&mut self) {
        while self.peek().is_trivia() {
            self.bump();
        }
    }

    /// Peek at the next non-trivia token kind without consuming.
    fn peek_non_trivia(&self) -> SyntaxKind {
        let mut i = self.pos;
        loop {
            if i >= self.tokens.len() {
                return SyntaxKind::Eof;
            }
            let kind = self.tokens[i].kind;
            if !kind.is_trivia() {
                return kind;
            }
            i += 1;
        }
    }

    fn span_from(&self, start: u32) -> Span {
        let end = if self.pos > 0 {
            self.tokens[self.pos - 1].span.end
        } else {
            start
        };
        Span::new(start, end)
    }

    fn current_span(&self) -> Span {
        self.current_token().span
    }

    fn text_of(&self, tok: &Token) -> &str {
        tok.span.source_text(&self.source)
    }

    fn consume_fuel(&mut self) -> bool {
        if self.fuel == 0 {
            return false;
        }
        self.fuel -= 1;
        true
    }

    // -- Layout helpers ----------------------------------------------------

    fn eat_layout_semi(&mut self) -> bool {
        self.skip_trivia();
        self.eat(SyntaxKind::LayoutSemicolon)
    }

    fn at_layout_end(&self) -> bool {
        let k = self.peek_non_trivia();
        matches!(k, SyntaxKind::LayoutBraceClose | SyntaxKind::Eof)
    }

    fn eat_layout_close(&mut self) -> bool {
        self.skip_trivia();
        self.eat(SyntaxKind::LayoutBraceClose)
    }

    // ═════════════════════════════════════════════════════════════════════
    // Top-level: Program
    // ═════════════════════════════════════════════════════════════════════

    pub fn parse_program(&mut self) -> Program {
        let mut decls = Vec::new();
        loop {
            self.skip_trivia();
            // Also consume layout tokens between decls at top level
            while self.eat(SyntaxKind::LayoutSemicolon)
                || self.eat(SyntaxKind::LayoutBraceClose)
                || self.eat(SyntaxKind::LayoutBraceOpen)
            {
                self.skip_trivia();
            }

            if self.at_end() {
                break;
            }
            if !self.consume_fuel() {
                break;
            }
            if let Some(decl) = self.parse_decl() {
                decls.push(decl);
            } else {
                // Error recovery: skip one token
                self.bump();
            }
        }
        Program { decls }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Declarations
    // ═════════════════════════════════════════════════════════════════════

    fn parse_decl(&mut self) -> Option<Decl> {
        self.skip_trivia();
        match self.peek_non_trivia() {
            SyntaxKind::At => {
                // Parse attributes, then the annotated declaration.
                // Attributes may be followed by a type signature and/or
                // a function definition on separate layout lines.
                let start = self.current_span().start;
                let mut attributes = Vec::new();
                while self.peek_non_trivia() == SyntaxKind::At {
                    self.skip_trivia();
                    attributes.push(self.parse_attribute());
                    self.skip_trivia();
                }

                // Consume LayoutSemicolon between attributes and the declaration
                self.eat_layout_semi();
                self.skip_trivia();

                // Skip optional type signature (e.g. `main : ComputeInput -> ()`)
                if self.at(SyntaxKind::Ident) {
                    let saved = self.pos;
                    let name_tok = self.bump();
                    self.skip_trivia();
                    if self.at(SyntaxKind::Colon) {
                        // It's a type signature – parse and discard it, then
                        // consume the LayoutSemicolon and continue to the fun decl.
                        let name = self.text_of(&name_tok).to_owned();
                        let _ty_sig = self.parse_type_sig(name, name_tok.span.start);
                        self.eat_layout_semi();
                        self.skip_trivia();
                    } else {
                        // Not a type sig – backtrack
                        self.pos = saved;
                    }
                }

                // Now parse the actual function declaration
                self.skip_trivia();
                let name_tok = self.expect(SyntaxKind::Ident);
                let name = self.text_of(&name_tok).to_owned();
                self.skip_trivia();

                let mut params = Vec::new();
                while !self.at(SyntaxKind::Equals) && !self.at_end() && self.consume_fuel() {
                    if matches!(
                        self.peek_non_trivia(),
                        SyntaxKind::LayoutSemicolon | SyntaxKind::LayoutBraceClose
                    ) {
                        break;
                    }
                    let p = self.parse_pat_atom();
                    params.push(p);
                    self.skip_trivia();
                }

                self.expect(SyntaxKind::Equals);
                self.skip_trivia();
                let body = self.parse_expr();

                let span = self.span_from(start);
                Some(Decl::EntryPoint {
                    attributes,
                    name,
                    params,
                    body,
                    span,
                })
            }
            SyntaxKind::KwData => Some(self.parse_data_decl()),
            SyntaxKind::KwType => Some(self.parse_type_alias()),
            SyntaxKind::Ident => {
                // Could be a type signature or function declaration.
                // Look ahead: name then `:` means type sig; otherwise fun decl.
                self.skip_trivia();
                let name_tok = self.bump();
                let name = self.text_of(&name_tok).to_owned();
                let start = name_tok.span.start;

                self.skip_trivia();
                if self.at(SyntaxKind::Colon) {
                    Some(self.parse_type_sig(name, start))
                } else {
                    Some(self.parse_fun_decl(name, start))
                }
            }
            _ => None,
        }
    }

    fn parse_type_sig(&mut self, name: String, start: u32) -> Decl {
        self.expect(SyntaxKind::Colon); // consume `:`
        self.skip_trivia();
        let ty = self.parse_type();
        let span = self.span_from(start);
        Decl::TypeSig { name, ty, span }
    }

    fn parse_fun_decl(&mut self, name: String, start: u32) -> Decl {
        // Parse patterns before `=` or `|` (guard)
        let mut params = Vec::new();
        self.skip_trivia();
        while !self.at(SyntaxKind::Equals)
            && !self.at(SyntaxKind::Pipe)
            && !self.at_end()
            && self.consume_fuel()
        {
            if matches!(
                self.peek_non_trivia(),
                SyntaxKind::LayoutSemicolon | SyntaxKind::LayoutBraceClose
            ) {
                break;
            }
            let p = self.parse_pat_atom();
            params.push(p);
            self.skip_trivia();
        }

        // Guard clauses: f x | cond1 = body1 | cond2 = body2 | otherwise = bodyN
        if self.at(SyntaxKind::Pipe) {
            let body = self.parse_guard_clauses(start);

            // Optional `where` clause
            let mut where_binds = Vec::new();
            self.skip_trivia();
            if self.eat(SyntaxKind::KwWhere) {
                where_binds = self.parse_where_binds();
            }

            let span = self.span_from(start);
            return Decl::FunDecl {
                name,
                params,
                body,
                where_binds,
                span,
            };
        }

        self.expect(SyntaxKind::Equals);
        self.skip_trivia();
        let body = self.parse_expr();

        // Optional `where` clause
        let mut where_binds = Vec::new();
        self.skip_trivia();
        if self.eat(SyntaxKind::KwWhere) {
            where_binds = self.parse_where_binds();
        }

        let span = self.span_from(start);
        Decl::FunDecl {
            name,
            params,
            body,
            where_binds,
            span,
        }
    }

    /// Parse guard clauses and desugar into nested if-then-else.
    /// `| cond1 = body1 | cond2 = body2 | otherwise = bodyN`
    fn parse_guard_clauses(&mut self, start: u32) -> Expr {
        let mut guards: Vec<(Expr, Expr)> = Vec::new();

        while self.at(SyntaxKind::Pipe) && self.consume_fuel() {
            self.bump(); // consume `|`
            self.skip_trivia();
            let cond = self.parse_expr();
            self.skip_trivia();
            self.expect(SyntaxKind::Equals);
            self.skip_trivia();
            let body = self.parse_expr();
            guards.push((cond, body));
            self.skip_trivia();
            // Consume layout semicolons between guards
            self.eat_layout_semi();
            self.skip_trivia();
        }

        // Desugar: fold guards right into if-then-else chain.
        // The last guard is treated as the else branch if its condition is
        // `otherwise` (i.e. Var("otherwise")) or `True`.
        if guards.is_empty() {
            let span = self.span_from(start);
            return Expr::Var("<error>".to_owned(), span);
        }

        let mut result = None;
        for (cond, body) in guards.into_iter().rev() {
            let span = cond.span().merge(body.span());
            if let Expr::Var(ref name, _) = cond {
                if name == "otherwise" {
                    // `otherwise` guard: becomes the else branch
                    result = Some(body);
                    continue;
                }
            }
            let else_branch = result.unwrap_or_else(|| {
                // Fallback: return a zero literal if no otherwise clause
                Expr::Lit(Lit::Int(0), span)
            });
            result = Some(Expr::If(
                Box::new(cond),
                Box::new(body),
                Box::new(else_branch),
                span,
            ));
        }

        result.unwrap_or_else(|| {
            let span = self.span_from(start);
            Expr::Var("<error>".to_owned(), span)
        })
    }

    fn parse_where_binds(&mut self) -> Vec<(String, Expr)> {
        let mut binds = Vec::new();
        // Layout should have inserted LayoutBraceOpen
        self.skip_trivia();
        // Consume optional layout brace open
        self.eat(SyntaxKind::LayoutBraceOpen);

        loop {
            self.skip_trivia();
            if self.at_layout_end() || self.at_end() {
                break;
            }
            if !self.consume_fuel() {
                break;
            }

            if self.at(SyntaxKind::Ident) {
                let name_tok = self.bump();
                let name = self.text_of(&name_tok).to_owned();
                self.skip_trivia();
                self.expect(SyntaxKind::Equals);
                self.skip_trivia();
                let expr = self.parse_expr();
                binds.push((name, expr));
                self.eat_layout_semi();
            } else {
                break;
            }
        }

        self.eat_layout_close();
        binds
    }

    fn parse_data_decl(&mut self) -> Decl {
        let start = self.current_span().start;
        self.expect(SyntaxKind::KwData);
        self.skip_trivia();

        let name_tok = self.expect(SyntaxKind::UpperIdent);
        let name = self.text_of(&name_tok).to_owned();

        // Type parameters
        let mut type_params = Vec::new();
        self.skip_trivia();
        while self.at(SyntaxKind::Ident) {
            let p = self.bump();
            type_params.push(self.text_of(&p).to_owned());
            self.skip_trivia();
        }

        self.expect(SyntaxKind::Equals);
        self.skip_trivia();

        // Parse constructors separated by `|`
        let mut constructors = Vec::new();
        constructors.push(self.parse_con_decl());
        loop {
            self.skip_trivia();
            if self.eat(SyntaxKind::Pipe) {
                self.skip_trivia();
                constructors.push(self.parse_con_decl());
            } else {
                break;
            }
        }

        let span = self.span_from(start);
        Decl::DataDecl {
            name,
            type_params,
            constructors,
            span,
        }
    }

    fn parse_con_decl(&mut self) -> ConDecl {
        let start = self.current_span().start;
        let name_tok = self.expect(SyntaxKind::UpperIdent);
        let name = self.text_of(&name_tok).to_owned();
        self.skip_trivia();

        // Check for record syntax `{`
        let fields = if self.at(SyntaxKind::LBrace) {
            self.bump();
            let mut flds = Vec::new();
            loop {
                self.skip_trivia();
                if self.at(SyntaxKind::RBrace) || self.at_end() {
                    break;
                }
                let field_name_tok = self.expect(SyntaxKind::Ident);
                let field_name = self.text_of(&field_name_tok).to_owned();
                self.skip_trivia();
                self.expect(SyntaxKind::Colon);
                self.skip_trivia();
                let ty = self.parse_type_atom();
                flds.push((field_name, ty));
                self.skip_trivia();
                if !self.eat(SyntaxKind::Comma) {
                    break;
                }
            }
            self.expect(SyntaxKind::RBrace);
            ConFields::Record(flds)
        } else {
            // Positional fields: type atoms until we see `|`, newline-level token, or EOF
            let mut tys = Vec::new();
            while !self.at_end() && self.consume_fuel() {
                let k = self.peek_non_trivia();
                if matches!(
                    k,
                    SyntaxKind::Pipe
                        | SyntaxKind::Eof
                        | SyntaxKind::LayoutSemicolon
                        | SyntaxKind::LayoutBraceClose
                        | SyntaxKind::KwData
                        | SyntaxKind::KwWhere
                        | SyntaxKind::Newline
                ) {
                    break;
                }
                // Only consume type atoms (Con / Var / Paren)
                if !matches!(
                    k,
                    SyntaxKind::UpperIdent | SyntaxKind::Ident | SyntaxKind::LParen
                ) {
                    break;
                }
                self.skip_trivia();
                let ty = self.parse_type_atom();
                tys.push(ty);
            }
            if tys.is_empty() {
                ConFields::Empty
            } else {
                ConFields::Positional(tys)
            }
        };

        let span = self.span_from(start);
        ConDecl { name, fields, span }
    }

    fn parse_type_alias(&mut self) -> Decl {
        let start = self.current_span().start;
        self.expect(SyntaxKind::KwType);
        self.skip_trivia();

        let name_tok = self.expect(SyntaxKind::UpperIdent);
        let name = self.text_of(&name_tok).to_owned();

        // Type parameters
        let mut params = Vec::new();
        self.skip_trivia();
        while self.at(SyntaxKind::Ident) {
            let p = self.bump();
            params.push(self.text_of(&p).to_owned());
            self.skip_trivia();
        }

        self.expect(SyntaxKind::Equals);
        self.skip_trivia();
        let ty = self.parse_type();

        let span = self.span_from(start);
        Decl::TypeAlias {
            name,
            params,
            ty,
            span,
        }
    }

    fn parse_attribute(&mut self) -> Attribute {
        let start = self.current_span().start;
        self.expect(SyntaxKind::At);
        self.skip_trivia();

        // Attribute name: could be Ident or UpperIdent
        let name_tok = if self.at(SyntaxKind::Ident) {
            self.bump()
        } else {
            self.expect(SyntaxKind::UpperIdent)
        };
        let name = self.text_of(&name_tok).to_owned();

        // Optional arguments: either parenthesized `@name(a, b)` or
        // bare integer literals `@name 64 1 1`.
        let mut args = Vec::new();
        self.skip_trivia();
        if self.at(SyntaxKind::LParen) {
            self.bump();
            loop {
                self.skip_trivia();
                if self.at(SyntaxKind::RParen) || self.at_end() {
                    break;
                }
                let arg_tok = self.bump();
                args.push(self.text_of(&arg_tok).to_owned());
                self.skip_trivia();
                if !self.eat(SyntaxKind::Comma) {
                    break;
                }
            }
            self.expect(SyntaxKind::RParen);
        } else {
            // Bare integer/float literal arguments (e.g. `@workgroup_size 64 1 1`)
            while matches!(
                self.peek_non_trivia(),
                SyntaxKind::IntLiteral | SyntaxKind::FloatLiteral
            ) {
                self.skip_trivia();
                let arg_tok = self.bump();
                args.push(self.text_of(&arg_tok).to_owned());
            }
        }

        let span = self.span_from(start);
        Attribute { name, args, span }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Expressions (Pratt / precedence climbing)
    // ═════════════════════════════════════════════════════════════════════

    pub fn parse_expr(&mut self) -> Expr {
        self.parse_expr_bp(0)
    }

    fn parse_expr_bp(&mut self, min_bp: u8) -> Expr {
        self.skip_trivia();

        // -- Prefix / atoms ------------------------------------------------
        let mut lhs = match self.peek_non_trivia() {
            SyntaxKind::Minus => {
                // Negation prefix
                self.skip_trivia();
                let start = self.current_span().start;
                self.bump();
                self.skip_trivia();
                let rhs = self.parse_expr_bp(13); // high precedence for neg
                let span = self.span_from(start);
                Expr::Neg(Box::new(rhs), span)
            }
            SyntaxKind::Backslash => self.parse_lambda(),
            SyntaxKind::KwIf => self.parse_if(),
            SyntaxKind::KwMatch => self.parse_match(),
            SyntaxKind::KwLet => self.parse_let(),
            SyntaxKind::KwDo => self.parse_do(),
            _ => self.parse_atom(),
        };

        // -- Infix / postfix loop ------------------------------------------
        loop {
            if !self.consume_fuel() {
                break;
            }
            self.skip_trivia();
            let op_kind = self.peek_non_trivia();

            // Function application (juxtaposition): if the next token could
            // start an atom and we have enough binding power.
            if is_atom_start(op_kind) && min_bp <= 11 {
                self.skip_trivia();
                let arg = self.parse_atom();
                let span = lhs.span().merge(arg.span());
                lhs = Expr::App(Box::new(lhs), Box::new(arg), span);
                continue;
            }

            // Dot for field access
            if op_kind == SyntaxKind::Dot && min_bp <= 13 {
                self.skip_trivia();
                self.bump(); // consume `.`
                self.skip_trivia();
                if self.at(SyntaxKind::Ident) {
                    let field_tok = self.bump();
                    let field = self.text_of(&field_tok).to_owned();
                    let span = lhs.span().merge(field_tok.span);
                    lhs = Expr::FieldAccess(Box::new(lhs), field, span);
                    continue;
                } else {
                    // Error: expected field name after `.`
                    let tok = self.current_token().clone();
                    self.diagnostics.push(
                        Diagnostic::error("expected field name after '.'")
                            .with_label(Label::primary(tok.span, "expected identifier")),
                    );
                    break;
                }
            }

            // Binary operators
            if let Some((l_bp, r_bp)) = infix_binding_power(op_kind) {
                if l_bp < min_bp {
                    break;
                }
                self.skip_trivia();
                let op_tok = self.bump();
                let op_text = self.text_of(&op_tok).to_owned();
                self.skip_trivia();
                let rhs = self.parse_expr_bp(r_bp);
                let span = lhs.span().merge(rhs.span());
                lhs = Expr::Infix(Box::new(lhs), op_text, Box::new(rhs), span);
                continue;
            }

            // Backtick infix: expr `func` expr
            if op_kind == SyntaxKind::Backtick && min_bp <= 3 {
                self.skip_trivia();
                self.bump(); // consume opening backtick
                self.skip_trivia();
                let func_tok = self.bump();
                let func = self.text_of(&func_tok).to_owned();
                self.skip_trivia();
                self.expect(SyntaxKind::Backtick); // closing backtick
                self.skip_trivia();
                let rhs = self.parse_expr_bp(4);
                let span = lhs.span().merge(rhs.span());
                lhs = Expr::Infix(Box::new(lhs), func, Box::new(rhs), span);
                continue;
            }

            // Pipeline operator: x |> f  desugars to  f x
            if op_kind == SyntaxKind::PipeForward && min_bp <= 1 {
                self.skip_trivia();
                self.bump(); // consume `|>`
                self.skip_trivia();
                let func = self.parse_expr_bp(2);
                let span = lhs.span().merge(func.span());
                lhs = Expr::App(Box::new(func), Box::new(lhs), span);
                continue;
            }

            break;
        }

        lhs
    }

    fn parse_atom(&mut self) -> Expr {
        self.skip_trivia();

        match self.peek() {
            SyntaxKind::IntLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let val = parse_int_literal(text);
                Expr::Lit(Lit::Int(val), tok.span)
            }
            SyntaxKind::FloatLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let val: f64 = text.parse().unwrap_or(0.0);
                Expr::Lit(Lit::Float(val), tok.span)
            }
            SyntaxKind::StringLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                // Strip quotes
                let inner = &text[1..text.len() - 1];
                Expr::Lit(Lit::String(unescape_string(inner)), tok.span)
            }
            SyntaxKind::CharLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let inner = &text[1..text.len() - 1];
                let ch = unescape_char(inner);
                Expr::Lit(Lit::Char(ch), tok.span)
            }
            SyntaxKind::Ident => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Expr::Var(name, tok.span)
            }
            SyntaxKind::UpperIdent => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Expr::Con(name, tok.span)
            }
            SyntaxKind::LParen => self.parse_paren_expr(),
            SyntaxKind::LBracket => self.parse_vec_lit(),
            SyntaxKind::KwIf => self.parse_if(),
            SyntaxKind::KwMatch => self.parse_match(),
            SyntaxKind::KwLet => self.parse_let(),
            SyntaxKind::KwDo => self.parse_do(),
            SyntaxKind::Backslash => self.parse_lambda(),
            _ => {
                let tok = self.current_token().clone();
                self.diagnostics.push(
                    Diagnostic::error(format!("unexpected token: {}", tok.kind))
                        .with_label(Label::primary(tok.span, "unexpected")),
                );
                let span = tok.span;
                self.bump();
                // Return an error expression as a variable named `<error>`
                Expr::Var("<error>".to_owned(), span)
            }
        }
    }

    fn parse_paren_expr(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::LParen);
        self.skip_trivia();

        // Unit `()`
        if self.at(SyntaxKind::RParen) {
            self.bump();
            let span = self.span_from(start);
            return Expr::Tuple(Vec::new(), span);
        }

        // Operator section `(+)`, `(-)`, etc.
        if is_operator_token(self.peek()) {
            let op_tok = self.bump();
            self.skip_trivia();
            if self.at(SyntaxKind::RParen) {
                self.bump();
                let op = self.text_of(&op_tok).to_owned();
                let span = self.span_from(start);
                return Expr::OpSection(op, span);
            }
            // Not a simple section: this is something like `(+ 3)`.
            // For now, treat it as error and return the op as a section anyway.
            // A more complete parser would handle left/right sections.
            let op = self.text_of(&op_tok).to_owned();
            // Try to consume up to closing paren
            while !self.at(SyntaxKind::RParen) && !self.at_end() && self.consume_fuel() {
                self.bump();
            }
            self.eat(SyntaxKind::RParen);
            let span = self.span_from(start);
            return Expr::OpSection(op, span);
        }

        // Parse first expression
        let first = self.parse_expr();
        self.skip_trivia();

        // Tuple `(a, b, ...)`
        if self.at(SyntaxKind::Comma) {
            let mut elems = vec![first];
            while self.eat(SyntaxKind::Comma) {
                self.skip_trivia();
                elems.push(self.parse_expr());
                self.skip_trivia();
            }
            self.expect(SyntaxKind::RParen);
            let span = self.span_from(start);
            return Expr::Tuple(elems, span);
        }

        // Simple parenthesized expression
        self.expect(SyntaxKind::RParen);
        let span = self.span_from(start);
        Expr::Paren(Box::new(first), span)
    }

    fn parse_vec_lit(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::LBracket);
        self.skip_trivia();

        let mut elems = Vec::new();

        // Empty vec literal `[]` is not valid for WGSL, but parse it anyway
        if !self.at(SyntaxKind::RBracket) && !self.at_end() {
            elems.push(self.parse_expr());
            self.skip_trivia();

            while self.eat(SyntaxKind::Comma) {
                self.skip_trivia();
                if self.at(SyntaxKind::RBracket) {
                    break; // trailing comma
                }
                elems.push(self.parse_expr());
                self.skip_trivia();
            }
        }

        self.expect(SyntaxKind::RBracket);
        let span = self.span_from(start);
        Expr::VecLit(elems, span)
    }

    fn parse_if(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::KwIf);
        self.skip_trivia();
        let cond = self.parse_expr();
        self.skip_trivia();
        self.expect(SyntaxKind::KwThen);
        self.skip_trivia();
        let then_expr = self.parse_expr();
        self.skip_trivia();
        self.expect(SyntaxKind::KwElse);
        self.skip_trivia();
        let else_expr = self.parse_expr();
        let span = self.span_from(start);
        Expr::If(
            Box::new(cond),
            Box::new(then_expr),
            Box::new(else_expr),
            span,
        )
    }

    fn parse_match(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::KwMatch);
        self.skip_trivia();

        // Parse the scrutinee -- only parse atoms and applications,
        // stop before `|` (pipe) token. We use a limited expression parser.
        let scrutinee = self.parse_match_scrutinee();
        self.skip_trivia();

        // Parse arms: each starts with |
        let mut arms = Vec::new();
        loop {
            self.skip_trivia();
            if self.at_end() {
                break;
            }
            if !self.consume_fuel() {
                break;
            }
            if !self.at(SyntaxKind::Pipe) {
                break;
            }
            self.bump(); // consume |
            self.skip_trivia();

            let pat = self.parse_pat();
            self.skip_trivia();
            self.expect(SyntaxKind::Arrow);
            self.skip_trivia();
            let body = self.parse_expr();
            arms.push((pat, body));
        }

        let span = self.span_from(start);
        Expr::Case(Box::new(scrutinee), arms, span)
    }

    /// Parse the scrutinee of a match expression. Stops at `|` (Pipe) token or layout tokens.
    fn parse_match_scrutinee(&mut self) -> Expr {
        self.skip_trivia();
        let mut expr = self.parse_match_scrutinee_atom();

        // Allow function application in scrutinee
        loop {
            self.skip_trivia();
            let k = self.peek_non_trivia();
            if k == SyntaxKind::Pipe
                || k == SyntaxKind::Eof
                || k == SyntaxKind::LayoutBraceOpen
                || k == SyntaxKind::LayoutSemicolon
                || k == SyntaxKind::LayoutBraceClose
            {
                break;
            }
            if !is_atom_start(k) {
                break;
            }
            if !self.consume_fuel() {
                break;
            }
            self.skip_trivia();
            let arg = self.parse_match_scrutinee_atom();
            let span = expr.span().merge(arg.span());
            expr = Expr::App(Box::new(expr), Box::new(arg), span);
        }

        expr
    }

    fn parse_match_scrutinee_atom(&mut self) -> Expr {
        self.skip_trivia();
        match self.peek() {
            SyntaxKind::Ident => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Expr::Var(name, tok.span)
            }
            SyntaxKind::UpperIdent => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Expr::Con(name, tok.span)
            }
            SyntaxKind::IntLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let val = parse_int_literal(text);
                Expr::Lit(Lit::Int(val), tok.span)
            }
            SyntaxKind::LParen => self.parse_paren_expr(),
            _ => {
                let tok = self.current_token().clone();
                let span = tok.span;
                Expr::Var("<error>".to_owned(), span)
            }
        }
    }

    fn parse_let(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::KwLet);
        self.skip_trivia();

        // Layout brace open
        self.eat(SyntaxKind::LayoutBraceOpen);

        let mut binds = Vec::new();
        loop {
            self.skip_trivia();
            if self.at(SyntaxKind::KwIn) || self.at_layout_end() || self.at_end() {
                break;
            }
            if !self.consume_fuel() {
                break;
            }

            if self.at(SyntaxKind::Ident) {
                let name_tok = self.bump();
                let name = self.text_of(&name_tok).to_owned();
                self.skip_trivia();
                self.expect(SyntaxKind::Equals);
                self.skip_trivia();
                let expr = self.parse_expr();
                binds.push((name, expr));
                self.eat_layout_semi();
            } else {
                break;
            }
        }

        self.eat_layout_close();
        self.skip_trivia();
        self.expect(SyntaxKind::KwIn);
        self.skip_trivia();
        let body = self.parse_expr();
        let span = self.span_from(start);
        Expr::Let(binds, Box::new(body), span)
    }

    fn parse_do(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::KwDo);
        self.skip_trivia();

        // Layout brace open
        self.eat(SyntaxKind::LayoutBraceOpen);

        let mut stmts = Vec::new();
        loop {
            self.skip_trivia();
            if self.at_layout_end() || self.at_end() {
                break;
            }
            if !self.consume_fuel() {
                break;
            }

            let stmt_start = self.current_span().start;

            // `let x = expr`
            if self.at(SyntaxKind::KwLet) {
                self.bump();
                self.skip_trivia();
                // Consume possible LayoutBraceOpen from let in do
                self.eat(SyntaxKind::LayoutBraceOpen);
                self.skip_trivia();
                let name_tok = self.expect(SyntaxKind::Ident);
                let name = self.text_of(&name_tok).to_owned();
                self.skip_trivia();
                self.expect(SyntaxKind::Equals);
                self.skip_trivia();
                let expr = self.parse_expr();
                let span = self.span_from(stmt_start);
                stmts.push(DoStmt::Let(name, expr, span));
                self.eat_layout_semi();
                self.eat_layout_close();
                continue;
            }

            // Try to detect `x <- expr` pattern:
            // peek: Ident, then LeftArrow
            if self.at(SyntaxKind::Ident) {
                // Lookahead for `<-`
                let saved_pos = self.pos;
                let name_tok = self.bump();
                self.skip_trivia();
                if self.at(SyntaxKind::LeftArrow) {
                    self.bump(); // consume `<-`
                    self.skip_trivia();
                    let name = self.text_of(&name_tok).to_owned();
                    let expr = self.parse_expr();
                    let span = self.span_from(stmt_start);
                    stmts.push(DoStmt::Bind(name, expr, span));
                    self.eat_layout_semi();
                    continue;
                } else {
                    // Backtrack -- it's a plain expression
                    self.pos = saved_pos;
                }
            }

            // Plain expression statement
            let expr = self.parse_expr();
            let span = self.span_from(stmt_start);
            stmts.push(DoStmt::Expr(expr, span));
            self.eat_layout_semi();
        }

        self.eat_layout_close();

        let span = self.span_from(start);
        Expr::Do(stmts, span)
    }

    fn parse_lambda(&mut self) -> Expr {
        let start = self.current_span().start;
        self.expect(SyntaxKind::Backslash);
        self.skip_trivia();

        let mut params = Vec::new();
        while !self.at(SyntaxKind::Arrow) && !self.at_end() && self.consume_fuel() {
            let p = self.parse_pat_atom();
            params.push(p);
            self.skip_trivia();
        }

        self.expect(SyntaxKind::Arrow);
        self.skip_trivia();
        let body = self.parse_expr();
        let span = self.span_from(start);
        Expr::Lambda(params, Box::new(body), span)
    }

    // ═════════════════════════════════════════════════════════════════════
    // Patterns
    // ═════════════════════════════════════════════════════════════════════

    fn parse_pat(&mut self) -> Pat {
        self.skip_trivia();
        let start = self.current_span().start;

        match self.peek() {
            SyntaxKind::UpperIdent => {
                // Constructor pattern, possibly with sub-patterns
                let name_tok = self.bump();
                let name = self.text_of(&name_tok).to_owned();
                self.skip_trivia();

                // Check for record pattern Con { ... }
                if self.at(SyntaxKind::LBrace) {
                    return self.parse_record_pat(name, start);
                }

                // Collect sub-patterns (atoms only)
                let mut sub_pats = Vec::new();
                while is_pat_atom_start(self.peek_non_trivia()) && self.consume_fuel() {
                    self.skip_trivia();
                    sub_pats.push(self.parse_pat_atom());
                }
                let span = self.span_from(start);
                Pat::Con(name, sub_pats, span)
            }
            _ => self.parse_pat_atom(),
        }
    }

    fn parse_pat_atom(&mut self) -> Pat {
        self.skip_trivia();
        let start = self.current_span().start;

        match self.peek() {
            SyntaxKind::Underscore => {
                let tok = self.bump();
                Pat::Wild(tok.span)
            }
            SyntaxKind::Ident => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                // Check for as-pattern: name@pat
                if self.at(SyntaxKind::At) {
                    self.bump();
                    self.skip_trivia();
                    let inner = self.parse_pat_atom();
                    let span = self.span_from(start);
                    Pat::As(name, Box::new(inner), span)
                } else {
                    Pat::Var(name, tok.span)
                }
            }
            SyntaxKind::UpperIdent => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Pat::Con(name, Vec::new(), tok.span)
            }
            SyntaxKind::IntLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let val = parse_int_literal(text);
                Pat::Lit(Lit::Int(val), tok.span)
            }
            SyntaxKind::FloatLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let val: f64 = text.parse().unwrap_or(0.0);
                Pat::Lit(Lit::Float(val), tok.span)
            }
            SyntaxKind::StringLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let inner = &text[1..text.len() - 1];
                Pat::Lit(Lit::String(unescape_string(inner)), tok.span)
            }
            SyntaxKind::CharLiteral => {
                let tok = self.bump();
                let text = self.text_of(&tok);
                let inner = &text[1..text.len() - 1];
                Pat::Lit(Lit::Char(unescape_char(inner)), tok.span)
            }
            SyntaxKind::LParen => {
                self.bump();
                self.skip_trivia();
                // Unit `()`
                if self.at(SyntaxKind::RParen) {
                    self.bump();
                    let span = self.span_from(start);
                    return Pat::Tuple(Vec::new(), span);
                }
                let first = self.parse_pat();
                self.skip_trivia();
                if self.at(SyntaxKind::Comma) {
                    // Tuple pattern
                    let mut elems = vec![first];
                    while self.eat(SyntaxKind::Comma) {
                        self.skip_trivia();
                        elems.push(self.parse_pat());
                        self.skip_trivia();
                    }
                    self.expect(SyntaxKind::RParen);
                    let span = self.span_from(start);
                    Pat::Tuple(elems, span)
                } else {
                    self.expect(SyntaxKind::RParen);
                    let span = self.span_from(start);
                    Pat::Paren(Box::new(first), span)
                }
            }
            _ => {
                let tok = self.current_token().clone();
                self.diagnostics.push(
                    Diagnostic::error(format!("unexpected token in pattern: {}", tok.kind))
                        .with_label(Label::primary(tok.span, "unexpected")),
                );
                let span = tok.span;
                self.bump();
                Pat::Wild(span)
            }
        }
    }

    fn parse_record_pat(&mut self, con_name: String, start: u32) -> Pat {
        self.expect(SyntaxKind::LBrace);
        let mut fields = Vec::new();
        loop {
            self.skip_trivia();
            if self.at(SyntaxKind::RBrace) || self.at_end() {
                break;
            }

            let field_tok = self.expect(SyntaxKind::Ident);
            let field_name = self.text_of(&field_tok).to_owned();
            self.skip_trivia();

            if self.at(SyntaxKind::Equals) {
                self.bump();
                self.skip_trivia();
                let pat = self.parse_pat();
                fields.push((field_name, Some(pat)));
            } else {
                // Punned field: `field` means `field = field`
                fields.push((field_name, None));
            }

            self.skip_trivia();
            if !self.eat(SyntaxKind::Comma) {
                break;
            }
        }
        self.expect(SyntaxKind::RBrace);
        let span = self.span_from(start);
        Pat::Record(con_name, fields, span)
    }

    // ═════════════════════════════════════════════════════════════════════
    // Types
    // ═════════════════════════════════════════════════════════════════════

    pub fn parse_type(&mut self) -> Type {
        self.skip_trivia();
        let lhs = self.parse_type_app();
        self.skip_trivia();

        // Arrow types are right-associative
        if self.at(SyntaxKind::Arrow) {
            self.bump();
            self.skip_trivia();
            let rhs = self.parse_type(); // right-recursive for right-assoc
            let span = lhs.span().merge(rhs.span());
            Type::Arrow(Box::new(lhs), Box::new(rhs), span)
        } else {
            lhs
        }
    }

    fn parse_type_app(&mut self) -> Type {
        self.skip_trivia();
        let mut ty = self.parse_type_atom();

        loop {
            self.skip_trivia();
            let next = self.peek_non_trivia();
            if matches!(
                next,
                SyntaxKind::UpperIdent | SyntaxKind::Ident | SyntaxKind::LParen
            ) {
                let arg = self.parse_type_atom();
                let span = ty.span().merge(arg.span());
                ty = Type::App(Box::new(ty), Box::new(arg), span);
            } else {
                break;
            }
        }

        ty
    }

    fn parse_type_atom(&mut self) -> Type {
        self.skip_trivia();
        let start = self.current_span().start;

        match self.peek() {
            SyntaxKind::UpperIdent => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Type::Con(name, tok.span)
            }
            SyntaxKind::Ident => {
                let tok = self.bump();
                let name = self.text_of(&tok).to_owned();
                Type::Var(name, tok.span)
            }
            SyntaxKind::LParen => {
                self.bump();
                self.skip_trivia();

                // Unit type `()`
                if self.at(SyntaxKind::RParen) {
                    self.bump();
                    let span = self.span_from(start);
                    return Type::Unit(span);
                }

                let first = self.parse_type();
                self.skip_trivia();

                if self.at(SyntaxKind::Comma) {
                    // Tuple type
                    let mut elems = vec![first];
                    while self.eat(SyntaxKind::Comma) {
                        self.skip_trivia();
                        elems.push(self.parse_type());
                        self.skip_trivia();
                    }
                    self.expect(SyntaxKind::RParen);
                    let span = self.span_from(start);
                    Type::Tuple(elems, span)
                } else {
                    self.expect(SyntaxKind::RParen);
                    let span = self.span_from(start);
                    Type::Paren(Box::new(first), span)
                }
            }
            _ => {
                let tok = self.current_token().clone();
                self.diagnostics.push(
                    Diagnostic::error(format!("unexpected token in type: {}", tok.kind))
                        .with_label(Label::primary(tok.span, "unexpected")),
                );
                let span = tok.span;
                self.bump();
                Type::Con("<error>".to_owned(), span)
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Infix operator binding powers: returns `(left_bp, right_bp)`.
///
/// Left-associative:  `l_bp < r_bp`
/// Right-associative: `l_bp > r_bp`
fn infix_binding_power(kind: SyntaxKind) -> Option<(u8, u8)> {
    match kind {
        SyntaxKind::Dollar => Some((1, 0)), // right-assoc (l > r)
        SyntaxKind::OrOr => Some((1, 2)),
        SyntaxKind::AndAnd => Some((3, 4)),
        SyntaxKind::EqualEqual
        | SyntaxKind::NotEqual
        | SyntaxKind::Less
        | SyntaxKind::Greater
        | SyntaxKind::LessEqual
        | SyntaxKind::GreaterEqual => Some((5, 6)),
        SyntaxKind::Plus | SyntaxKind::Minus => Some((7, 8)),
        SyntaxKind::Star | SyntaxKind::Slash | SyntaxKind::Percent => Some((9, 10)),
        _ => None,
    }
}

/// Whether a token kind can begin an atom expression.
fn is_atom_start(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::IntLiteral
            | SyntaxKind::FloatLiteral
            | SyntaxKind::StringLiteral
            | SyntaxKind::CharLiteral
            | SyntaxKind::Ident
            | SyntaxKind::UpperIdent
            | SyntaxKind::LParen
            | SyntaxKind::LBracket
            | SyntaxKind::Backslash
    )
}

/// Whether a token kind can begin a pattern atom.
fn is_pat_atom_start(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::Underscore
            | SyntaxKind::Ident
            | SyntaxKind::UpperIdent
            | SyntaxKind::IntLiteral
            | SyntaxKind::FloatLiteral
            | SyntaxKind::StringLiteral
            | SyntaxKind::CharLiteral
            | SyntaxKind::LParen
    )
}

fn is_operator_token(kind: SyntaxKind) -> bool {
    matches!(
        kind,
        SyntaxKind::Plus
            | SyntaxKind::Minus
            | SyntaxKind::Star
            | SyntaxKind::Slash
            | SyntaxKind::Percent
            | SyntaxKind::Less
            | SyntaxKind::Greater
            | SyntaxKind::LessEqual
            | SyntaxKind::GreaterEqual
            | SyntaxKind::EqualEqual
            | SyntaxKind::NotEqual
            | SyntaxKind::AndAnd
            | SyntaxKind::OrOr
            | SyntaxKind::Dollar
            | SyntaxKind::Bang
            | SyntaxKind::Dot
    )
}

fn parse_int_literal(text: &str) -> i64 {
    if text.starts_with("0x") || text.starts_with("0X") {
        i64::from_str_radix(&text[2..], 16).unwrap_or(0)
    } else if text.starts_with("0o") || text.starts_with("0O") {
        i64::from_str_radix(&text[2..], 8).unwrap_or(0)
    } else if text.starts_with("0b") || text.starts_with("0B") {
        i64::from_str_radix(&text[2..], 2).unwrap_or(0)
    } else {
        text.parse::<i64>().unwrap_or(0)
    }
}

fn unescape_string(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('t') => out.push('\t'),
                Some('r') => out.push('\r'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some('0') => out.push('\0'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

fn unescape_char(s: &str) -> char {
    let mut chars = s.chars();
    match chars.next() {
        Some('\\') => match chars.next() {
            Some('n') => '\n',
            Some('t') => '\t',
            Some('r') => '\r',
            Some('\\') => '\\',
            Some('\'') => '\'',
            Some('0') => '\0',
            Some(c) => c,
            None => '\\',
        },
        Some(c) => c,
        None => '\0',
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(source: &str) -> Program {
        let mut parser = Parser::new(source);
        parser.parse_program()
    }

    #[test]
    fn parse_type_sig() {
        let prog = parse("add : I32 -> I32 -> I32");
        assert_eq!(prog.decls.len(), 1);
        match &prog.decls[0] {
            Decl::TypeSig { name, .. } => assert_eq!(name, "add"),
            other => panic!("expected TypeSig, got {:?}", other),
        }
    }

    #[test]
    fn parse_fun_decl() {
        let prog = parse("add x y = x + y");
        assert_eq!(prog.decls.len(), 1);
        match &prog.decls[0] {
            Decl::FunDecl { name, params, .. } => {
                assert_eq!(name, "add");
                assert_eq!(params.len(), 2);
            }
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_data_decl() {
        let prog = parse("data Color = Red | Green | Blue");
        assert_eq!(prog.decls.len(), 1);
        match &prog.decls[0] {
            Decl::DataDecl {
                name, constructors, ..
            } => {
                assert_eq!(name, "Color");
                assert_eq!(constructors.len(), 3);
                assert_eq!(constructors[0].name, "Red");
                assert_eq!(constructors[1].name, "Green");
                assert_eq!(constructors[2].name, "Blue");
            }
            other => panic!("expected DataDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_case_expr() {
        let source = "f c = match c\n  | Red -> 0\n  | Green -> 1\n  | Blue -> 2";
        let prog = parse(source);
        assert_eq!(prog.decls.len(), 1);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => match body {
                Expr::Case(_, arms, _) => {
                    assert_eq!(arms.len(), 3);
                }
                other => panic!("expected Case, got {:?}", other),
            },
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_let_expr() {
        let source = "f x = let y = x + 1 in y";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => match body {
                Expr::Let(binds, _, _) => {
                    assert_eq!(binds.len(), 1);
                    assert_eq!(binds[0].0, "y");
                }
                other => panic!("expected Let, got {:?}", other),
            },
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_lambda() {
        let source = "f = \\x y -> x + y";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => match body {
                Expr::Lambda(params, _, _) => {
                    assert_eq!(params.len(), 2);
                }
                other => panic!("expected Lambda, got {:?}", other),
            },
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_if_expr() {
        let source = "f x = if x == 0 then 1 else 2";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => {
                assert!(matches!(body, Expr::If(..)));
            }
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_full_program() {
        let source = "\
data Color = Red | Green | Blue

show : Color -> I32
show c = match c
  | Red   -> 0
  | Green -> 1
  | Blue  -> 2

add : I32 -> I32 -> I32
add x y = x + y

main : I32 -> I32
main x =
  let y = add x 1
  in show Red
";
        let prog = parse(source);
        // We expect: DataDecl, TypeSig, FunDecl, TypeSig, FunDecl, TypeSig, FunDecl
        assert!(
            prog.decls.len() >= 4,
            "expected at least 4 decls, got {}: {:#?}",
            prog.decls.len(),
            prog.decls
        );
    }

    #[test]
    fn parse_entry_point() {
        let source = "@vertex\nmain x = x + 1";
        let prog = parse(source);
        assert_eq!(prog.decls.len(), 1);
        match &prog.decls[0] {
            Decl::EntryPoint {
                attributes, name, ..
            } => {
                assert_eq!(attributes.len(), 1);
                assert_eq!(attributes[0].name, "vertex");
                assert_eq!(name, "main");
            }
            other => panic!("expected EntryPoint, got {:?}", other),
        }
    }

    #[test]
    fn parse_operator_section() {
        let source = "f = (+)";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => {
                assert!(matches!(body, Expr::OpSection(..)));
            }
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_tuple() {
        let source = "f = (1, 2, 3)";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => match body {
                Expr::Tuple(elems, _) => {
                    assert_eq!(elems.len(), 3);
                }
                other => panic!("expected Tuple, got {:?}", other),
            },
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_unit() {
        let source = "f = ()";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => {
                assert!(matches!(body, Expr::Tuple(elems, _) if elems.is_empty()));
            }
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_precedence() {
        // 1 + 2 * 3 should parse as 1 + (2 * 3)
        let source = "f = 1 + 2 * 3";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => match body {
                Expr::Infix(_, op, rhs, _) => {
                    assert_eq!(op, "+");
                    assert!(matches!(rhs.as_ref(), Expr::Infix(_, op2, _, _) if op2 == "*"));
                }
                other => panic!("expected Infix(+), got {:?}", other),
            },
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_dollar_right_assoc() {
        // f $ g $ x should parse as f $ (g $ x)
        let source = "r = f $ g $ x";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => match body {
                Expr::Infix(_lhs, op, rhs, _) => {
                    assert_eq!(op, "$");
                    // rhs should be another `$` application
                    assert!(matches!(rhs.as_ref(), Expr::Infix(_, op2, _, _) if op2 == "$"));
                }
                other => panic!("expected Infix($), got {:?}", other),
            },
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }

    #[test]
    fn parse_function_application() {
        let source = "f = add x y";
        let prog = parse(source);
        match &prog.decls[0] {
            Decl::FunDecl { body, .. } => {
                // add x y  =  (add x) y  =  App(App(Var(add), Var(x)), Var(y))
                match body {
                    Expr::App(lhs, rhs, _) => {
                        assert!(matches!(rhs.as_ref(), Expr::Var(name, _) if name == "y"));
                        assert!(matches!(lhs.as_ref(), Expr::App(..)));
                    }
                    other => panic!("expected App, got {:?}", other),
                }
            }
            other => panic!("expected FunDecl, got {:?}", other),
        }
    }
}
