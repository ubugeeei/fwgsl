use std::fmt;

/// Every token and CST node in fwgsl is tagged with a `SyntaxKind`.
///
/// The variants are laid out as `#[repr(u8)]` so they can be stored
/// compactly inside arena-allocated green nodes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum SyntaxKind {
    // ── Trivia ──────────────────────────────────────────────────────
    /// Contiguous horizontal whitespace (spaces / tabs).
    Whitespace = 0,
    /// A newline character (`\n` or `\r\n`).
    Newline,
    /// A line comment starting with `--`.
    LineComment,
    /// A block comment delimited by `{-` ... `-}`.
    BlockComment,

    // ── Layout (virtual tokens inserted by the layout resolver) ────
    LayoutBraceOpen,
    LayoutSemicolon,
    LayoutBraceClose,

    // ── Punctuation ────────────────────────────────────────────────
    /// `(`
    LParen,
    /// `)`
    RParen,
    /// `[`
    LBracket,
    /// `]`
    RBracket,
    /// `{`
    LBrace,
    /// `}`
    RBrace,
    /// `,`
    Comma,
    /// `;`
    Semicolon,
    /// `:`
    Colon,
    /// `::`
    ColonColon,
    /// `.`
    Dot,
    /// `..`
    DotDot,
    /// `->`
    Arrow,
    /// `=>`
    FatArrow,
    /// `\`
    Backslash,
    /// `@`
    At,
    /// `|`
    Pipe,
    /// `=`
    Equals,
    /// `_`
    Underscore,

    // ── Operators ──────────────────────────────────────────────────
    /// `+`
    Plus,
    /// `-`
    Minus,
    /// `*`
    Star,
    /// `/`
    Slash,
    /// `%`
    Percent,
    /// `<`
    Less,
    /// `>`
    Greater,
    /// `<=`
    LessEqual,
    /// `>=`
    GreaterEqual,
    /// `==`
    EqualEqual,
    /// `/=`
    NotEqual,
    /// `&&`
    AndAnd,
    /// `||`
    OrOr,
    /// `!`
    Bang,
    /// `$`
    Dollar,
    /// `` ` ``
    Backtick,
    /// `|>`
    PipeForward,
    /// `.` when used as function composition (resolved in the parser).
    Compose,
    /// `<-`
    LeftArrow,

    // ── Literals ───────────────────────────────────────────────────
    IntLiteral,
    FloatLiteral,
    StringLiteral,
    CharLiteral,

    // ── Identifiers ────────────────────────────────────────────────
    /// A lower-case or underscore-leading identifier.
    Ident,
    /// An identifier starting with an uppercase letter.
    UpperIdent,
    /// A symbolic operator wrapped in parentheses, e.g. `(+)`.
    Operator,

    // ── Keywords ───────────────────────────────────────────────────
    KwModule,
    KwWhere,
    KwImport,
    KwData,
    KwType,
    KwClass,
    KwInstance,
    KwLet,
    KwIn,
    KwCase,
    KwOf,
    KwMatch,
    KwIf,
    KwThen,
    KwElse,
    KwDo,
    KwForall,
    KwInfixl,
    KwInfixr,
    KwInfix,
    KwDeriving,

    // ── CST node kinds ─────────────────────────────────────────────
    SourceFile,
    ModuleDecl,
    ModuleHeader,
    ExportList,
    ExportItem,

    ImportDecl,
    ImportList,
    ImportItem,

    DataDecl,
    ConDecl,
    FieldDecl,

    /// Type alias declaration.
    TypeDecl,

    ClassDecl,
    InstanceDecl,
    MethodDecl,

    TypeSig,
    FunDecl,
    FunEquation,
    GuardedExpr,

    ParamPat,
    WhereBind,

    Attribute,
    AttrArg,

    // ── Expression nodes ───────────────────────────────────────────
    ExprApp,
    ExprInfix,
    ExprPrefix,
    ExprLambda,
    ExprLet,
    ExprLetBind,
    ExprCase,
    ExprCaseArm,
    ExprIf,
    ExprDo,
    ExprDoStmt,

    ExprParen,
    ExprTuple,
    ExprList,
    ExprRecord,
    ExprFieldAccess,
    ExprSection,

    ExprLit,
    ExprIdent,
    ExprOp,
    ExprCompose,
    ExprBacktick,

    // ── Type nodes ─────────────────────────────────────────────────
    TypeArrow,
    TypeApp,
    TypeCon,
    TypeVar,
    TypeParen,
    TypeTuple,
    TypeForall,
    TypeConstraint,

    // ── Pattern nodes ──────────────────────────────────────────────
    PatWild,
    PatVar,
    PatCon,
    PatLit,
    PatParen,
    PatTuple,
    PatRecord,
    PatAs,
    PatFieldPat,

    // ── Misc ───────────────────────────────────────────────────────
    QualifiedName,

    // ── Sentinel / error ───────────────────────────────────────────
    Error,
    Eof,
    Tombstone,
}

// ────────────────────────────────────────────────────────────────────
// Helper constants (first/last of each category) used by the query
// methods so they stay in sync with the enum order automatically.
// ────────────────────────────────────────────────────────────────────

impl SyntaxKind {
    const FIRST_KEYWORD: Self = SyntaxKind::KwModule;
    const LAST_KEYWORD: Self = SyntaxKind::KwDeriving;

    const FIRST_OPERATOR: Self = SyntaxKind::Plus;
    const LAST_OPERATOR: Self = SyntaxKind::PipeForward;

    /// Returns `true` for tokens the parser should skip when building
    /// the CST (whitespace, newlines, comments).
    #[inline]
    pub fn is_trivia(self) -> bool {
        matches!(
            self,
            SyntaxKind::Whitespace
                | SyntaxKind::Newline
                | SyntaxKind::LineComment
                | SyntaxKind::BlockComment
        )
    }

    /// Returns `true` for any `Kw*` variant.
    #[inline]
    pub fn is_keyword(self) -> bool {
        let v = self as u8;
        v >= Self::FIRST_KEYWORD as u8 && v <= Self::LAST_KEYWORD as u8
    }

    /// Returns `true` for literal tokens.
    #[inline]
    pub fn is_literal(self) -> bool {
        matches!(
            self,
            SyntaxKind::IntLiteral
                | SyntaxKind::FloatLiteral
                | SyntaxKind::StringLiteral
                | SyntaxKind::CharLiteral
        )
    }

    /// Returns `true` for operator tokens (`Plus` through `Bang`,
    /// plus `Dollar` and `Backtick`).
    #[inline]
    pub fn is_operator(self) -> bool {
        let v = self as u8;
        v >= Self::FIRST_OPERATOR as u8 && v <= Self::LAST_OPERATOR as u8
    }
}

// ────────────────────────────────────────────────────────────────────
// keyword_from_str
// ────────────────────────────────────────────────────────────────────

/// Map a source string to its keyword `SyntaxKind`, if any.
pub fn keyword_from_str(s: &str) -> Option<SyntaxKind> {
    match s {
        "module" => Some(SyntaxKind::KwModule),
        "where" => Some(SyntaxKind::KwWhere),
        "import" => Some(SyntaxKind::KwImport),
        "data" => Some(SyntaxKind::KwData),
        "type" => Some(SyntaxKind::KwType),
        "class" => Some(SyntaxKind::KwClass),
        "instance" => Some(SyntaxKind::KwInstance),
        "let" => Some(SyntaxKind::KwLet),
        "in" => Some(SyntaxKind::KwIn),
        "case" => Some(SyntaxKind::KwCase),
        "of" => Some(SyntaxKind::KwOf),
        "match" => Some(SyntaxKind::KwMatch),
        "if" => Some(SyntaxKind::KwIf),
        "then" => Some(SyntaxKind::KwThen),
        "else" => Some(SyntaxKind::KwElse),
        "do" => Some(SyntaxKind::KwDo),
        "forall" => Some(SyntaxKind::KwForall),
        "infixl" => Some(SyntaxKind::KwInfixl),
        "infixr" => Some(SyntaxKind::KwInfixr),
        "infix" => Some(SyntaxKind::KwInfix),
        "deriving" => Some(SyntaxKind::KwDeriving),
        _ => None,
    }
}

// ────────────────────────────────────────────────────────────────────
// From<SyntaxKind> for u8
// ────────────────────────────────────────────────────────────────────

impl From<SyntaxKind> for u8 {
    #[inline]
    fn from(kind: SyntaxKind) -> u8 {
        kind as u8
    }
}

// ────────────────────────────────────────────────────────────────────
// TryFrom<u8> for SyntaxKind
// ────────────────────────────────────────────────────────────────────

impl TryFrom<u8> for SyntaxKind {
    type Error = u8;

    fn try_from(value: u8) -> Result<Self, <SyntaxKind as TryFrom<u8>>::Error> {
        if value <= SyntaxKind::Tombstone as u8 {
            // SAFETY: `SyntaxKind` is `#[repr(u8)]` and all discriminants
            // from 0 to `Tombstone` are defined contiguously.
            Ok(unsafe { std::mem::transmute::<u8, SyntaxKind>(value) })
        } else {
            Err(value)
        }
    }
}

// ────────────────────────────────────────────────────────────────────
// Display
// ────────────────────────────────────────────────────────────────────

impl fmt::Display for SyntaxKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            // Trivia
            SyntaxKind::Whitespace => "whitespace",
            SyntaxKind::Newline => "newline",
            SyntaxKind::LineComment => "line comment",
            SyntaxKind::BlockComment => "block comment",

            // Layout
            SyntaxKind::LayoutBraceOpen => "layout '{'",
            SyntaxKind::LayoutSemicolon => "layout ';'",
            SyntaxKind::LayoutBraceClose => "layout '}'",

            // Punctuation
            SyntaxKind::LParen => "'('",
            SyntaxKind::RParen => "')'",
            SyntaxKind::LBracket => "'['",
            SyntaxKind::RBracket => "']'",
            SyntaxKind::LBrace => "'{'",
            SyntaxKind::RBrace => "'}'",
            SyntaxKind::Comma => "','",
            SyntaxKind::Semicolon => "';'",
            SyntaxKind::Colon => "':'",
            SyntaxKind::ColonColon => "'::'",
            SyntaxKind::Dot => "'.'",
            SyntaxKind::DotDot => "'..'",
            SyntaxKind::Arrow => "'->'",
            SyntaxKind::FatArrow => "'=>'",
            SyntaxKind::Backslash => "'\\'",
            SyntaxKind::At => "'@'",
            SyntaxKind::Pipe => "'|'",
            SyntaxKind::Equals => "'='",
            SyntaxKind::Underscore => "'_'",

            // Operators
            SyntaxKind::Plus => "'+'",
            SyntaxKind::Minus => "'-'",
            SyntaxKind::Star => "'*'",
            SyntaxKind::Slash => "'/'",
            SyntaxKind::Percent => "'%'",
            SyntaxKind::Less => "'<'",
            SyntaxKind::Greater => "'>'",
            SyntaxKind::LessEqual => "'<='",
            SyntaxKind::GreaterEqual => "'>='",
            SyntaxKind::EqualEqual => "'=='",
            SyntaxKind::NotEqual => "'/='",
            SyntaxKind::AndAnd => "'&&'",
            SyntaxKind::OrOr => "'||'",
            SyntaxKind::Bang => "'!'",
            SyntaxKind::Dollar => "'$'",
            SyntaxKind::Backtick => "'`'",
            SyntaxKind::Compose => "'.'",
            SyntaxKind::LeftArrow => "'<-'",
            SyntaxKind::PipeForward => "'|>'",

            // Literals
            SyntaxKind::IntLiteral => "integer literal",
            SyntaxKind::FloatLiteral => "float literal",
            SyntaxKind::StringLiteral => "string literal",
            SyntaxKind::CharLiteral => "char literal",

            // Identifiers
            SyntaxKind::Ident => "identifier",
            SyntaxKind::UpperIdent => "upper identifier",
            SyntaxKind::Operator => "operator",

            // Keywords
            SyntaxKind::KwModule => "'module'",
            SyntaxKind::KwWhere => "'where'",
            SyntaxKind::KwImport => "'import'",
            SyntaxKind::KwData => "'data'",
            SyntaxKind::KwType => "'type'",
            SyntaxKind::KwClass => "'class'",
            SyntaxKind::KwInstance => "'instance'",
            SyntaxKind::KwLet => "'let'",
            SyntaxKind::KwIn => "'in'",
            SyntaxKind::KwCase => "'case'",
            SyntaxKind::KwOf => "'of'",
            SyntaxKind::KwMatch => "'match'",
            SyntaxKind::KwIf => "'if'",
            SyntaxKind::KwThen => "'then'",
            SyntaxKind::KwElse => "'else'",
            SyntaxKind::KwDo => "'do'",
            SyntaxKind::KwForall => "'forall'",
            SyntaxKind::KwInfixl => "'infixl'",
            SyntaxKind::KwInfixr => "'infixr'",
            SyntaxKind::KwInfix => "'infix'",
            SyntaxKind::KwDeriving => "'deriving'",

            // CST node kinds
            SyntaxKind::SourceFile => "source file",
            SyntaxKind::ModuleDecl => "module declaration",
            SyntaxKind::ModuleHeader => "module header",
            SyntaxKind::ExportList => "export list",
            SyntaxKind::ExportItem => "export item",

            SyntaxKind::ImportDecl => "import declaration",
            SyntaxKind::ImportList => "import list",
            SyntaxKind::ImportItem => "import item",

            SyntaxKind::DataDecl => "data declaration",
            SyntaxKind::ConDecl => "constructor declaration",
            SyntaxKind::FieldDecl => "field declaration",

            SyntaxKind::TypeDecl => "type alias declaration",

            SyntaxKind::ClassDecl => "class declaration",
            SyntaxKind::InstanceDecl => "instance declaration",
            SyntaxKind::MethodDecl => "method declaration",

            SyntaxKind::TypeSig => "type signature",
            SyntaxKind::FunDecl => "function declaration",
            SyntaxKind::FunEquation => "function equation",
            SyntaxKind::GuardedExpr => "guarded expression",

            SyntaxKind::ParamPat => "parameter pattern",
            SyntaxKind::WhereBind => "where binding",

            SyntaxKind::Attribute => "attribute",
            SyntaxKind::AttrArg => "attribute argument",

            SyntaxKind::ExprApp => "function application",
            SyntaxKind::ExprInfix => "infix expression",
            SyntaxKind::ExprPrefix => "prefix expression",
            SyntaxKind::ExprLambda => "lambda expression",
            SyntaxKind::ExprLet => "let expression",
            SyntaxKind::ExprLetBind => "let binding",
            SyntaxKind::ExprCase => "case expression",
            SyntaxKind::ExprCaseArm => "case arm",
            SyntaxKind::ExprIf => "if expression",
            SyntaxKind::ExprDo => "do expression",
            SyntaxKind::ExprDoStmt => "do statement",

            SyntaxKind::ExprParen => "parenthesized expression",
            SyntaxKind::ExprTuple => "tuple expression",
            SyntaxKind::ExprList => "list expression",
            SyntaxKind::ExprRecord => "record expression",
            SyntaxKind::ExprFieldAccess => "field access",
            SyntaxKind::ExprSection => "section expression",

            SyntaxKind::ExprLit => "literal expression",
            SyntaxKind::ExprIdent => "identifier expression",
            SyntaxKind::ExprOp => "operator expression",
            SyntaxKind::ExprCompose => "compose expression",
            SyntaxKind::ExprBacktick => "backtick expression",

            SyntaxKind::TypeArrow => "function type",
            SyntaxKind::TypeApp => "type application",
            SyntaxKind::TypeCon => "type constructor",
            SyntaxKind::TypeVar => "type variable",
            SyntaxKind::TypeParen => "parenthesized type",
            SyntaxKind::TypeTuple => "tuple type",
            SyntaxKind::TypeForall => "forall type",
            SyntaxKind::TypeConstraint => "type constraint",

            SyntaxKind::PatWild => "wildcard pattern",
            SyntaxKind::PatVar => "variable pattern",
            SyntaxKind::PatCon => "constructor pattern",
            SyntaxKind::PatLit => "literal pattern",
            SyntaxKind::PatParen => "parenthesized pattern",
            SyntaxKind::PatTuple => "tuple pattern",
            SyntaxKind::PatRecord => "record pattern",
            SyntaxKind::PatAs => "as pattern",
            SyntaxKind::PatFieldPat => "field pattern",

            SyntaxKind::QualifiedName => "qualified name",

            SyntaxKind::Error => "error",
            SyntaxKind::Eof => "end of file",
            SyntaxKind::Tombstone => "tombstone",
        };
        f.write_str(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trivia_classification() {
        assert!(SyntaxKind::Whitespace.is_trivia());
        assert!(SyntaxKind::Newline.is_trivia());
        assert!(SyntaxKind::LineComment.is_trivia());
        assert!(SyntaxKind::BlockComment.is_trivia());
        assert!(!SyntaxKind::Ident.is_trivia());
    }

    #[test]
    fn keyword_classification() {
        assert!(SyntaxKind::KwModule.is_keyword());
        assert!(SyntaxKind::KwDeriving.is_keyword());
        assert!(SyntaxKind::KwLet.is_keyword());
        assert!(!SyntaxKind::Ident.is_keyword());
        assert!(!SyntaxKind::Plus.is_keyword());
    }

    #[test]
    fn literal_classification() {
        assert!(SyntaxKind::IntLiteral.is_literal());
        assert!(SyntaxKind::FloatLiteral.is_literal());
        assert!(SyntaxKind::StringLiteral.is_literal());
        assert!(SyntaxKind::CharLiteral.is_literal());
        assert!(!SyntaxKind::Ident.is_literal());
    }

    #[test]
    fn operator_classification() {
        assert!(SyntaxKind::Plus.is_operator());
        assert!(SyntaxKind::Minus.is_operator());
        assert!(SyntaxKind::Bang.is_operator());
        assert!(SyntaxKind::Dollar.is_operator());
        assert!(SyntaxKind::Backtick.is_operator());
        assert!(!SyntaxKind::Ident.is_operator());
        assert!(!SyntaxKind::Compose.is_operator());
    }

    #[test]
    fn keyword_from_str_works() {
        assert_eq!(keyword_from_str("module"), Some(SyntaxKind::KwModule));
        assert_eq!(keyword_from_str("where"), Some(SyntaxKind::KwWhere));
        assert_eq!(keyword_from_str("let"), Some(SyntaxKind::KwLet));
        assert_eq!(keyword_from_str("forall"), Some(SyntaxKind::KwForall));
        assert_eq!(keyword_from_str("deriving"), Some(SyntaxKind::KwDeriving));
        assert_eq!(keyword_from_str("notakeyword"), None);
        assert_eq!(keyword_from_str(""), None);
    }

    #[test]
    fn u8_round_trip() {
        let kind = SyntaxKind::KwLet;
        let raw: u8 = kind.into();
        let back = SyntaxKind::try_from(raw).unwrap();
        assert_eq!(kind, back);
    }

    #[test]
    fn u8_out_of_range() {
        assert!(SyntaxKind::try_from(255u8).is_err());
    }

    #[test]
    fn display_formatting() {
        assert_eq!(format!("{}", SyntaxKind::KwLet), "'let'");
        assert_eq!(format!("{}", SyntaxKind::Ident), "identifier");
        assert_eq!(format!("{}", SyntaxKind::Arrow), "'->'");
        assert_eq!(format!("{}", SyntaxKind::SourceFile), "source file");
    }
}
