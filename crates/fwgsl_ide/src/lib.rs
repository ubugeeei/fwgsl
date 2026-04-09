mod catalog;

use std::collections::{HashMap, HashSet};

pub use catalog::{
    all_completion_specs, completion_item_from_spec, lookup_completion_spec, spec_matches_context,
    CompletionContext, CompletionSpec,
};
use fwgsl_parser::lexer::Token;
use fwgsl_parser::parser::{Attribute, Decl, DoStmt, Expr, Pat, Program, Type};
use fwgsl_parser::{lex, Parser};
use fwgsl_semantic::SemanticAnalyzer;
use fwgsl_span::Span;
use fwgsl_syntax::SyntaxKind;
use fwgsl_typechecker::{InferEngine, Scheme};
use lsp_types::{
    CompletionItem, CompletionItemKind, Documentation, GotoDefinitionResponse, Hover,
    HoverContents, Location, MarkupContent, MarkupKind, Position, Range, Url,
};

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum Namespace {
    Value,
    Type,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum OccurrenceRole {
    Definition,
    Reference,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SymbolKind {
    Function,
    EntryPoint,
    Parameter,
    LocalBinding,
    PatternBinding,
    DataType,
    TypeAlias,
    Constructor,
    TypeParameter,
}

#[derive(Clone, Debug)]
struct Symbol {
    id: usize,
    name: String,
    namespace: Namespace,
    kind: SymbolKind,
    primary_span: Span,
    definition_spans: Vec<Span>,
    scope_span: Span,
    scope_depth: usize,
    visible_from: u32,
    container: Option<String>,
}

struct NewSymbol {
    name: String,
    namespace: Namespace,
    kind: SymbolKind,
    span: Span,
    scope_span: Span,
    scope_depth: usize,
    visible_from: u32,
    container: Option<String>,
}

#[derive(Clone, Debug)]
struct Occurrence {
    symbol_id: usize,
    span: Span,
    role: OccurrenceRole,
}

#[derive(Default)]
struct DocumentIndex {
    symbols: Vec<Symbol>,
    occurrences: Vec<Occurrence>,
}

struct IdeState<'a> {
    source: &'a str,
    analyzer: SemanticAnalyzer,
    index: DocumentIndex,
}

#[derive(Clone)]
struct ScopeFrame {
    span: Span,
    container: Option<String>,
    value_defs: HashMap<String, usize>,
    type_defs: HashMap<String, usize>,
}

impl ScopeFrame {
    fn new(span: Span, container: Option<String>) -> Self {
        Self {
            span,
            container,
            value_defs: HashMap::new(),
            type_defs: HashMap::new(),
        }
    }
}

impl DocumentIndex {
    fn push_symbol(&mut self, symbol: NewSymbol) -> usize {
        let id = self.symbols.len();
        self.symbols.push(Symbol {
            id,
            name: symbol.name,
            namespace: symbol.namespace,
            kind: symbol.kind,
            primary_span: symbol.span,
            definition_spans: vec![symbol.span],
            scope_span: symbol.scope_span,
            scope_depth: symbol.scope_depth,
            visible_from: symbol.visible_from,
            container: symbol.container,
        });
        self.push_occurrence(id, symbol.span, OccurrenceRole::Definition);
        id
    }

    fn add_definition_span(&mut self, symbol_id: usize, span: Span) {
        let symbol = &mut self.symbols[symbol_id];
        let is_new = !symbol.definition_spans.contains(&span);
        if is_new {
            symbol.definition_spans.push(span);
        }
        if symbol.primary_span.start > span.start {
            symbol.primary_span = span;
        }
        if is_new {
            self.push_occurrence(symbol_id, span, OccurrenceRole::Definition);
        }
    }

    fn push_occurrence(&mut self, symbol_id: usize, span: Span, role: OccurrenceRole) {
        self.occurrences.push(Occurrence {
            symbol_id,
            span,
            role,
        });
    }

    fn symbol_at_offset(&self, offset: u32) -> Option<&Occurrence> {
        self.occurrences
            .iter()
            .find(|occurrence| occurrence.span.start <= offset && offset < occurrence.span.end)
    }

    fn visible_symbols(
        &self,
        offset: u32,
        context: CompletionContext,
    ) -> impl Iterator<Item = &Symbol> {
        self.symbols.iter().filter(move |symbol| {
            if symbol.visible_from > offset {
                return false;
            }
            if !(symbol.scope_span.start <= offset && offset <= symbol.scope_span.end) {
                return false;
            }
            match context {
                CompletionContext::Attribute => false,
                CompletionContext::Type => symbol.namespace == Namespace::Type,
                CompletionContext::Value => true,
            }
        })
    }
}

struct IndexBuilder<'a> {
    source: &'a str,
    tokens: Vec<Token>,
    index: DocumentIndex,
    top_level_values: HashMap<String, usize>,
    top_level_types: HashMap<String, usize>,
}

impl<'a> IndexBuilder<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source,
            tokens: lex(source),
            index: DocumentIndex::default(),
            top_level_values: HashMap::new(),
            top_level_types: HashMap::new(),
        }
    }

    fn build(mut self, program: &Program) -> DocumentIndex {
        let whole_file = Span::new(0, self.source.len() as u32);
        self.collect_top_level(program, whole_file);
        self.walk_program(program, whole_file);
        self.index
    }

    fn collect_top_level(&mut self, program: &Program, whole_file: Span) {
        for decl in &program.decls {
            match decl {
                Decl::TypeSig { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_values.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::Function,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values.insert(name.clone(), id);
                        id
                    });
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::FunDecl { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_values.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::Function,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values.insert(name.clone(), id);
                        id
                    });
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::EntryPoint { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_values.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::EntryPoint,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values.insert(name.clone(), id);
                        id
                    });
                    self.index.symbols[symbol_id].kind = SymbolKind::EntryPoint;
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::DataDecl {
                    name,
                    constructors,
                    span,
                    ..
                } => {
                    let type_name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let type_id = self.top_level_types.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Type,
                            kind: SymbolKind::DataType,
                            span: type_name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_types.insert(name.clone(), id);
                        id
                    });
                    self.index.symbols[type_id].kind = SymbolKind::DataType;
                    self.index.add_definition_span(type_id, type_name_span);

                    for constructor in constructors {
                        let constructor_span = self
                            .first_name_span(&constructor.name, constructor.span)
                            .unwrap_or(constructor.span);
                        let constructor_id = self.index.push_symbol(NewSymbol {
                            name: constructor.name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::Constructor,
                            span: constructor_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values
                            .insert(constructor.name.clone(), constructor_id);
                    }
                }
                Decl::TypeAlias { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_types.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Type,
                            kind: SymbolKind::TypeAlias,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_types.insert(name.clone(), id);
                        id
                    });
                    self.index.symbols[symbol_id].kind = SymbolKind::TypeAlias;
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::ResourceDecl { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_values.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::Function,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values.insert(name.clone(), id);
                        id
                    });
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::BitfieldDecl { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_types.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Type,
                            kind: SymbolKind::TypeAlias,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_types.insert(name.clone(), id);
                        id
                    });
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::ConstDecl { name, span, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.top_level_values.get(name).copied().unwrap_or_else(|| {
                        let id = self.index.push_symbol(NewSymbol {
                            name: name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::Function,
                            span: name_span,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values.insert(name.clone(), id);
                        id
                    });
                    self.index.add_definition_span(symbol_id, name_span);
                }
                Decl::TraitDecl { name, span, methods, .. } => {
                    let name_span = self.first_name_span(name, *span).unwrap_or(*span);
                    let symbol_id = self.index.push_symbol(NewSymbol {
                        name: name.clone(),
                        namespace: Namespace::Type,
                        kind: SymbolKind::TypeAlias,
                        span: name_span,
                        scope_span: whole_file,
                        scope_depth: 0,
                        visible_from: 0,
                        container: Some(name.clone()),
                    });
                    self.top_level_types.insert(name.clone(), symbol_id);
                    for m in methods {
                        let mspan = self.first_name_span(&m.name, m.span).unwrap_or(m.span);
                        let mid = self.index.push_symbol(NewSymbol {
                            name: m.name.clone(),
                            namespace: Namespace::Value,
                            kind: SymbolKind::Function,
                            span: mspan,
                            scope_span: whole_file,
                            scope_depth: 0,
                            visible_from: 0,
                            container: Some(name.clone()),
                        });
                        self.top_level_values.insert(m.name.clone(), mid);
                    }
                }
                Decl::ImplDecl { .. } | Decl::ExternDecl { .. } => {
                    // Impl methods are lowered as mangled functions;
                    // Extern declarations are type-level only.
                    // No separate top-level symbols needed.
                }
                Decl::ModuleDecl { .. } | Decl::ImportDecl { .. } => {
                    // Module/import declarations don't define symbols.
                }
            }
        }
    }

    fn walk_program(&mut self, program: &Program, whole_file: Span) {
        let mut frames = vec![ScopeFrame::new(whole_file, None)];
        frames[0].value_defs = self.top_level_values.clone();
        frames[0].type_defs = self.top_level_types.clone();

        for decl in &program.decls {
            self.walk_decl(decl, &mut frames);
        }
    }

    fn walk_decl(&mut self, decl: &Decl, frames: &mut Vec<ScopeFrame>) {
        match decl {
            Decl::TypeSig { ty, .. } => {
                self.walk_type(ty, frames);
            }
            Decl::FunDecl {
                name,
                params,
                body,
                where_binds,
                span,
                ..
            } => {
                self.walk_callable(name, params, body, where_binds, *span, frames);
            }
            Decl::EntryPoint {
                attributes,
                name,
                params,
                body,
                span,
                ..
            } => {
                for attribute in attributes {
                    self.walk_attribute(attribute);
                }
                self.walk_callable(name, params, body, &[], *span, frames);
            }
            Decl::DataDecl {
                name,
                type_params,
                constructors,
                span,
                ..
            } => {
                frames.push(ScopeFrame::new(*span, Some(name.clone())));
                for type_param in type_params {
                    let type_param_span = self.first_name_span(type_param, *span).unwrap_or(*span);
                    let symbol_id = self.index.push_symbol(NewSymbol {
                        name: type_param.clone(),
                        namespace: Namespace::Type,
                        kind: SymbolKind::TypeParameter,
                        span: type_param_span,
                        scope_span: *span,
                        scope_depth: frames.len() - 1,
                        visible_from: span.start,
                        container: Some(name.clone()),
                    });
                    frames
                        .last_mut()
                        .expect("type scope")
                        .type_defs
                        .insert(type_param.clone(), symbol_id);
                }
                for constructor in constructors {
                    match &constructor.fields {
                        fwgsl_parser::parser::ConFields::Positional(fields) => {
                            for field in fields {
                                self.walk_type(field, frames);
                            }
                        }
                        fwgsl_parser::parser::ConFields::Record(fields) => {
                            for f in fields {
                                self.walk_type(&f.ty, frames);
                            }
                        }
                        fwgsl_parser::parser::ConFields::Empty => {}
                    }
                }
                frames.pop();
            }
            Decl::TypeAlias {
                name,
                params,
                ty,
                span,
                ..
            } => {
                frames.push(ScopeFrame::new(*span, Some(name.clone())));
                for type_param in params {
                    let type_param_span = self.first_name_span(type_param, *span).unwrap_or(*span);
                    let symbol_id = self.index.push_symbol(NewSymbol {
                        name: type_param.clone(),
                        namespace: Namespace::Type,
                        kind: SymbolKind::TypeParameter,
                        span: type_param_span,
                        scope_span: *span,
                        scope_depth: frames.len() - 1,
                        visible_from: span.start,
                        container: Some(name.clone()),
                    });
                    frames
                        .last_mut()
                        .expect("type scope")
                        .type_defs
                        .insert(type_param.clone(), symbol_id);
                }
                self.walk_type(ty, frames);
                frames.pop();
            }
            Decl::ResourceDecl { ty, .. } => {
                self.walk_type(ty, frames);
            }
            Decl::BitfieldDecl { base_ty, .. } => {
                self.walk_type(base_ty, frames);
            }
            Decl::ConstDecl { ty, value, .. } => {
                self.walk_type(ty, frames);
                self.walk_expr(value, frames);
            }
            Decl::TraitDecl { methods, .. } => {
                for m in methods {
                    self.walk_type(&m.ty, frames);
                }
            }
            Decl::ImplDecl { ty, methods, .. } => {
                self.walk_type(ty, frames);
                for m in methods {
                    self.walk_expr(&m.body, frames);
                }
            }
            Decl::ExternDecl { ty, .. } => {
                self.walk_type(ty, frames);
            }
            Decl::ModuleDecl { .. } | Decl::ImportDecl { .. } => {}
        }
    }

    fn walk_callable(
        &mut self,
        name: &str,
        params: &[Pat],
        body: &Expr,
        where_binds: &[(String, Expr)],
        span: Span,
        frames: &mut Vec<ScopeFrame>,
    ) {
        let container = Some(name.to_owned());
        frames.push(ScopeFrame::new(span, container.clone()));
        for param in params {
            self.define_pattern(param, span.start, frames, SymbolKind::Parameter);
        }

        let where_depth = frames.len() - 1;
        for (binding, expr) in where_binds {
            let binding_span = self
                .last_name_span_before(binding, span, expr.span().start)
                .unwrap_or(expr.span());
            let symbol_id = self.index.push_symbol(NewSymbol {
                name: binding.clone(),
                namespace: Namespace::Value,
                kind: SymbolKind::LocalBinding,
                span: binding_span,
                scope_span: span,
                scope_depth: where_depth,
                visible_from: span.start,
                container: container.clone(),
            });
            frames[where_depth]
                .value_defs
                .insert(binding.clone(), symbol_id);
        }

        for (_, expr) in where_binds {
            self.walk_expr(expr, frames);
        }
        self.walk_expr(body, frames);
        frames.pop();
    }

    fn walk_attribute(&mut self, _attribute: &Attribute) {}

    fn walk_expr(&mut self, expr: &Expr, frames: &mut Vec<ScopeFrame>) {
        match expr {
            Expr::Lit(_, _) | Expr::OpSection(_, _) => {}
            Expr::Var(name, span) | Expr::Con(name, span) => {
                if let Some(symbol_id) = self.resolve_value(name, frames) {
                    self.index
                        .push_occurrence(symbol_id, *span, OccurrenceRole::Reference);
                }
            }
            Expr::App(left, right, _) => {
                self.walk_expr(left, frames);
                self.walk_expr(right, frames);
            }
            Expr::Infix(left, _, right, _) => {
                self.walk_expr(left, frames);
                self.walk_expr(right, frames);
            }
            Expr::Lambda(params, body, span) => {
                let container = frames.last().and_then(|frame| frame.container.clone());
                frames.push(ScopeFrame::new(*span, container));
                for param in params {
                    self.define_pattern(param, span.start, frames, SymbolKind::Parameter);
                }
                self.walk_expr(body, frames);
                frames.pop();
            }
            Expr::Let(bindings, body, span) => {
                let container = frames.last().and_then(|frame| frame.container.clone());
                frames.push(ScopeFrame::new(*span, container));
                let depth = frames.len() - 1;
                for (binding, value) in bindings {
                    let binding_span = self
                        .last_name_span_before(binding, *span, value.span().start)
                        .unwrap_or(value.span());
                    let symbol_id = self.index.push_symbol(NewSymbol {
                        name: binding.clone(),
                        namespace: Namespace::Value,
                        kind: SymbolKind::LocalBinding,
                        span: binding_span,
                        scope_span: *span,
                        scope_depth: depth,
                        visible_from: span.start,
                        container: frames[depth].container.clone(),
                    });
                    frames[depth].value_defs.insert(binding.clone(), symbol_id);
                }
                for (_, value) in bindings {
                    self.walk_expr(value, frames);
                }
                self.walk_expr(body, frames);
                frames.pop();
            }
            Expr::Case(scrutinee, arms, _) => {
                self.walk_expr(scrutinee, frames);
                for (pattern, body) in arms {
                    let container = frames.last().and_then(|frame| frame.container.clone());
                    frames.push(ScopeFrame::new(body.span(), container));
                    self.define_pattern(
                        pattern,
                        body.span().start,
                        frames,
                        SymbolKind::PatternBinding,
                    );
                    self.walk_expr(body, frames);
                    frames.pop();
                }
            }
            Expr::If(condition, then_branch, else_branch, _) => {
                self.walk_expr(condition, frames);
                self.walk_expr(then_branch, frames);
                self.walk_expr(else_branch, frames);
            }
            Expr::Paren(inner, _) | Expr::Neg(inner, _) => {
                self.walk_expr(inner, frames);
            }
            Expr::Tuple(items, _) | Expr::VecLit(items, _) => {
                for item in items {
                    self.walk_expr(item, frames);
                }
            }
            Expr::Record(_, fields, _) => {
                for (_, value) in fields {
                    self.walk_expr(value, frames);
                }
            }
            Expr::FieldAccess(base, _, _) => {
                self.walk_expr(base, frames);
            }
            Expr::Index(base, index, _) => {
                self.walk_expr(base, frames);
                self.walk_expr(index, frames);
            }
            Expr::Do(statements, span) => {
                let container = frames.last().and_then(|frame| frame.container.clone());
                frames.push(ScopeFrame::new(*span, container));
                let depth = frames.len() - 1;
                for statement in statements {
                    match statement {
                        DoStmt::Bind(name, value, stmt_span)
                        | DoStmt::Let(name, value, stmt_span) => {
                            self.walk_expr(value, frames);
                            let binding_span = self
                                .last_name_span_before(name, *span, value.span().start)
                                .unwrap_or(*stmt_span);
                            let symbol_id = self.index.push_symbol(NewSymbol {
                                name: name.clone(),
                                namespace: Namespace::Value,
                                kind: SymbolKind::LocalBinding,
                                span: binding_span,
                                scope_span: *span,
                                scope_depth: depth,
                                visible_from: stmt_span.start,
                                container: frames[depth].container.clone(),
                            });
                            frames[depth].value_defs.insert(name.clone(), symbol_id);
                        }
                        DoStmt::Expr(value, _) => self.walk_expr(value, frames),
                    }
                }
                frames.pop();
            }
            Expr::Loop(loop_name, bindings, body, span) => {
                let container = frames.last().and_then(|frame| frame.container.clone());
                frames.push(ScopeFrame::new(*span, container));
                let depth = frames.len() - 1;
                // Register loop name as a local binding
                let loop_name_span = *span; // approximate
                let loop_sym = self.index.push_symbol(NewSymbol {
                    name: loop_name.clone(),
                    namespace: Namespace::Value,
                    kind: SymbolKind::LocalBinding,
                    span: loop_name_span,
                    scope_span: *span,
                    scope_depth: depth,
                    visible_from: span.start,
                    container: frames[depth].container.clone(),
                });
                frames[depth].value_defs.insert(loop_name.clone(), loop_sym);
                for (bind_name, init_expr) in bindings {
                    self.walk_expr(init_expr, frames);
                    let binding_span = self
                        .last_name_span_before(bind_name, *span, init_expr.span().start)
                        .unwrap_or(*span);
                    let symbol_id = self.index.push_symbol(NewSymbol {
                        name: bind_name.clone(),
                        namespace: Namespace::Value,
                        kind: SymbolKind::LocalBinding,
                        span: binding_span,
                        scope_span: *span,
                        scope_depth: depth,
                        visible_from: span.start,
                        container: frames[depth].container.clone(),
                    });
                    frames[depth].value_defs.insert(bind_name.clone(), symbol_id);
                }
                self.walk_expr(body, frames);
                frames.pop();
            }
            Expr::RecordUpdate(base, fields, _) => {
                self.walk_expr(base, frames);
                for (_, value) in fields {
                    self.walk_expr(value, frames);
                }
            }
        }
    }

    fn walk_type(&mut self, ty: &Type, frames: &mut Vec<ScopeFrame>) {
        match ty {
            Type::Con(name, span) | Type::Var(name, span) => {
                if let Some(symbol_id) = self.resolve_type(name, frames) {
                    self.index
                        .push_occurrence(symbol_id, *span, OccurrenceRole::Reference);
                }
            }
            Type::App(left, right, _) | Type::Arrow(left, right, _) => {
                self.walk_type(left, frames);
                self.walk_type(right, frames);
            }
            Type::Paren(inner, _) => self.walk_type(inner, frames),
            Type::Tuple(items, _) => {
                for item in items {
                    self.walk_type(item, frames);
                }
            }
            Type::Nat(_, _) | Type::Unit(_) => {}
        }
    }

    fn define_pattern(
        &mut self,
        pattern: &Pat,
        visible_from: u32,
        frames: &mut Vec<ScopeFrame>,
        kind: SymbolKind,
    ) {
        match pattern {
            Pat::Wild(_) | Pat::Lit(_, _) => {}
            Pat::Var(name, span) => {
                let depth = frames.len() - 1;
                let symbol_id = self.index.push_symbol(NewSymbol {
                    name: name.clone(),
                    namespace: Namespace::Value,
                    kind,
                    span: *span,
                    scope_span: frames[depth].span,
                    scope_depth: depth,
                    visible_from,
                    container: frames[depth].container.clone(),
                });
                frames[depth].value_defs.insert(name.clone(), symbol_id);
            }
            Pat::Con(name, fields, span) => {
                if let Some(symbol_id) = self.resolve_value(name, frames) {
                    self.index
                        .push_occurrence(symbol_id, *span, OccurrenceRole::Reference);
                }
                for field in fields {
                    self.define_pattern(field, visible_from, frames, kind);
                }
            }
            Pat::Paren(inner, _) => self.define_pattern(inner, visible_from, frames, kind),
            Pat::Tuple(items, _) => {
                for item in items {
                    self.define_pattern(item, visible_from, frames, kind);
                }
            }
            Pat::Record(name, fields, span) => {
                if let Some(symbol_id) = self.resolve_value(name, frames) {
                    self.index
                        .push_occurrence(symbol_id, *span, OccurrenceRole::Reference);
                }
                for (_, field) in fields {
                    if let Some(field_pattern) = field {
                        self.define_pattern(field_pattern, visible_from, frames, kind);
                    }
                }
            }
            Pat::As(name, inner, span) => {
                let depth = frames.len() - 1;
                let symbol_id = self.index.push_symbol(NewSymbol {
                    name: name.clone(),
                    namespace: Namespace::Value,
                    kind,
                    span: *span,
                    scope_span: frames[depth].span,
                    scope_depth: depth,
                    visible_from,
                    container: frames[depth].container.clone(),
                });
                frames[depth].value_defs.insert(name.clone(), symbol_id);
                self.define_pattern(inner, visible_from, frames, kind);
            }
            Pat::Or(alternatives, _) => {
                for alt in alternatives {
                    self.define_pattern(alt, visible_from, frames, kind);
                }
            }
        }
    }

    fn resolve_value(&self, name: &str, frames: &[ScopeFrame]) -> Option<usize> {
        for frame in frames.iter().rev() {
            if let Some(symbol_id) = frame.value_defs.get(name) {
                return Some(*symbol_id);
            }
        }
        self.top_level_values.get(name).copied()
    }

    fn resolve_type(&self, name: &str, frames: &[ScopeFrame]) -> Option<usize> {
        for frame in frames.iter().rev() {
            if let Some(symbol_id) = frame.type_defs.get(name) {
                return Some(*symbol_id);
            }
        }
        self.top_level_types.get(name).copied()
    }

    fn first_name_span(&self, name: &str, within: Span) -> Option<Span> {
        self.tokens
            .iter()
            .find(|token| {
                matches!(token.kind, SyntaxKind::Ident | SyntaxKind::UpperIdent)
                    && within.start <= token.span.start
                    && token.span.end <= within.end
                    && token.text(self.source) == name
            })
            .map(|token| token.span)
    }

    fn last_name_span_before(&self, name: &str, within: Span, before: u32) -> Option<Span> {
        self.tokens
            .iter()
            .rev()
            .find(|token| {
                matches!(token.kind, SyntaxKind::Ident | SyntaxKind::UpperIdent)
                    && within.start <= token.span.start
                    && token.span.end <= before
                    && token.text(self.source) == name
            })
            .map(|token| token.span)
    }
}

pub fn build_completions(source: &str, pos: Position) -> Vec<CompletionItem> {
    let prefix = completion_prefix(source, pos);
    let context = completion_context(source, pos, &prefix);
    let offset = position_to_offset(source, pos).unwrap_or(source.len()) as u32;
    let state = build_ide_state(source);

    let mut items = Vec::new();
    let mut seen = HashSet::new();

    for spec in all_completion_specs() {
        if !spec_matches_context(spec, context) {
            continue;
        }
        if !matches_prefix(spec.label, &prefix) {
            continue;
        }
        seen.insert(spec.label.to_owned());
        items.push(completion_item_from_spec(spec));
    }

    if context != CompletionContext::Attribute {
        for (label, scheme) in state.analyzer.env.iter() {
            if seen.contains(label) || !matches_prefix(label, &prefix) || !is_word_completion(label)
            {
                continue;
            }
            items.push(generic_builtin_completion_item(
                label,
                scheme,
                &state.analyzer.engine,
                builtin_completion_kind(label),
            ));
            seen.insert(label.to_owned());
        }
    }

    let mut visible_symbols = state
        .index
        .visible_symbols(offset, context)
        .filter(|symbol| matches_prefix(&symbol.name, &prefix))
        .collect::<Vec<_>>();

    visible_symbols.sort_by(|left, right| {
        right
            .scope_depth
            .cmp(&left.scope_depth)
            .then_with(|| right.visible_from.cmp(&left.visible_from))
            .then_with(|| left.name.cmp(&right.name))
    });

    let mut chosen = HashMap::<String, &Symbol>::new();
    for symbol in visible_symbols {
        chosen.entry(symbol.name.clone()).or_insert(symbol);
    }

    for symbol in chosen.into_values() {
        let kind = completion_kind_for_symbol(symbol);
        let documentation = symbol_markdown(&state, symbol);
        let detail = symbol_detail(&state, symbol);
        items.retain(|item| item.label != symbol.name);
        items.push(CompletionItem {
            label: symbol.name.clone(),
            kind: Some(kind),
            detail: Some(detail),
            documentation: Some(Documentation::MarkupContent(MarkupContent {
                kind: MarkupKind::Markdown,
                value: documentation,
            })),
            sort_text: Some(format!(
                "00-{}-{}",
                99usize.saturating_sub(symbol.scope_depth),
                symbol.name
            )),
            filter_text: Some(symbol.name.clone()),
            ..Default::default()
        });
    }

    items.sort_by(|left, right| {
        left.sort_text
            .as_deref()
            .cmp(&right.sort_text.as_deref())
            .then_with(|| left.label.cmp(&right.label))
    });
    items
}

pub fn build_hover(source: &str, pos: Position) -> Option<Hover> {
    let offset = position_to_offset(source, pos)? as u32;
    let tokens = lex(source);
    let (tok_index, tok) = tokens
        .iter()
        .enumerate()
        .find(|(_, token)| token.span.start <= offset && offset < token.span.end)?;
    let range = span_to_range(source, tok.span);
    let state = build_ide_state(source);

    match tok.kind {
        SyntaxKind::Ident | SyntaxKind::UpperIdent => {
            let name = tok.text(source);

            if previous_non_trivia_token(&tokens, tok_index)
                .is_some_and(|prev| prev.kind == SyntaxKind::At)
            {
                if let Some(spec) = lookup_completion_spec(name, CompletionContext::Attribute) {
                    return Some(spec_hover(&state, spec, range, Some(name)));
                }
            }

            if let Some(occurrence) = state.index.symbol_at_offset(offset) {
                let symbol = &state.index.symbols[occurrence.symbol_id];
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: symbol_markdown(&state, symbol),
                    }),
                    range: Some(range),
                });
            }

            if let Some(spec) = lookup_non_attribute_spec(name) {
                return Some(spec_hover(&state, spec, range, Some(name)));
            }

            if let Some(markdown) = generic_builtin_markdown(&state, name) {
                return Some(Hover {
                    contents: HoverContents::Markup(MarkupContent {
                        kind: MarkupKind::Markdown,
                        value: markdown,
                    }),
                    range: Some(range),
                });
            }

            None
        }
        kind if kind.is_keyword() => {
            let keyword = tok.text(source);
            lookup_non_attribute_spec(keyword).map(|spec| spec_hover(&state, spec, range, None))
        }
        SyntaxKind::At => Some(Hover {
            contents: HoverContents::Markup(MarkupContent {
                kind: MarkupKind::Markdown,
                value: "Use `@` to introduce a WGSL attribute such as `@compute`, `@vertex`, or `@workgroup_size(...)`.".to_owned(),
            }),
            range: Some(range),
        }),
        _ => None,
    }
}

pub fn build_goto_definition(
    uri: &Url,
    source: &str,
    pos: Position,
) -> Option<GotoDefinitionResponse> {
    let offset = position_to_offset(source, pos)? as u32;
    let state = build_ide_state(source);
    let occurrence = state.index.symbol_at_offset(offset)?;
    let symbol = &state.index.symbols[occurrence.symbol_id];
    let mut definition_spans = symbol.definition_spans.clone();
    definition_spans.sort_by(|left, right| {
        left.start
            .cmp(&right.start)
            .then_with(|| left.end.cmp(&right.end))
    });
    definition_spans.dedup();
    let locations = definition_spans
        .into_iter()
        .map(|span| Location {
            uri: uri.clone(),
            range: span_to_range(source, span),
        })
        .collect::<Vec<_>>();

    match locations.as_slice() {
        [] => None,
        [single] => Some(GotoDefinitionResponse::Scalar(single.clone())),
        _ => Some(GotoDefinitionResponse::Array(locations)),
    }
}

pub fn build_references(
    uri: &Url,
    source: &str,
    pos: Position,
    include_declaration: bool,
) -> Option<Vec<Location>> {
    let offset = position_to_offset(source, pos)? as u32;
    let state = build_ide_state(source);
    let occurrence = state.index.symbol_at_offset(offset)?;
    let mut locations = state
        .index
        .occurrences
        .iter()
        .filter(|candidate| candidate.symbol_id == occurrence.symbol_id)
        .filter(|candidate| include_declaration || candidate.role != OccurrenceRole::Definition)
        .map(|candidate| Location {
            uri: uri.clone(),
            range: span_to_range(source, candidate.span),
        })
        .collect::<Vec<_>>();

    locations.sort_by(|left, right| {
        left.range
            .start
            .line
            .cmp(&right.range.start.line)
            .then_with(|| left.range.start.character.cmp(&right.range.start.character))
    });
    locations.dedup_by(|left, right| left.range == right.range);

    if locations.is_empty() {
        None
    } else {
        Some(locations)
    }
}

fn build_ide_state(source: &str) -> IdeState<'_> {
    let mut parser = Parser::new(source);
    let mut program = parser.parse_program();

    // Prepend prelude declarations for type environment
    let prelude = fwgsl_parser::prelude_program();
    let mut combined = prelude.decls.clone();
    combined.append(&mut program.decls);
    program.decls = combined;

    let mut analyzer = SemanticAnalyzer::new();
    analyzer.analyze(&program);
    let index = IndexBuilder::new(source).build(&program);

    IdeState {
        source,
        analyzer,
        index,
    }
}

fn completion_prefix(source: &str, pos: Position) -> String {
    let offset = position_to_offset(source, pos).unwrap_or(source.len());
    let mut start = offset;

    while start > 0 {
        let Some(ch) = source[..start].chars().next_back() else {
            break;
        };
        if !is_completion_word_char(ch) {
            break;
        }
        start -= ch.len_utf8();
    }

    source[start..offset].to_owned()
}

fn completion_context(source: &str, pos: Position, prefix: &str) -> CompletionContext {
    let offset = position_to_offset(source, pos).unwrap_or(source.len());
    let before_cursor = &source[..offset];
    let before_prefix = &before_cursor[..before_cursor.len().saturating_sub(prefix.len())];

    if before_prefix
        .chars()
        .rev()
        .find(|ch| !ch.is_whitespace())
        .is_some_and(|ch| ch == '@')
    {
        return CompletionContext::Attribute;
    }

    if prefix.chars().next().is_some_and(char::is_uppercase) {
        return CompletionContext::Type;
    }

    let line_start = before_cursor.rfind('\n').map_or(0, |index| index + 1);
    let line = &before_cursor[line_start..];
    let last_colon = line.rfind(':');
    let last_equals = line.rfind('=');

    if last_colon.is_some() && last_equals.is_none_or(|equals| last_colon > Some(equals)) {
        return CompletionContext::Type;
    }

    CompletionContext::Value
}

fn is_completion_word_char(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || matches!(ch, '_' | '\'' | '$')
}

fn matches_prefix(candidate: &str, prefix: &str) -> bool {
    prefix.is_empty() || candidate.starts_with(prefix)
}

fn is_word_completion(label: &str) -> bool {
    label
        .chars()
        .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '\'' | '$'))
}

fn completion_kind_for_symbol(symbol: &Symbol) -> CompletionItemKind {
    match symbol.kind {
        SymbolKind::Function | SymbolKind::EntryPoint => CompletionItemKind::FUNCTION,
        SymbolKind::Parameter | SymbolKind::LocalBinding | SymbolKind::PatternBinding => {
            CompletionItemKind::VARIABLE
        }
        SymbolKind::Constructor => CompletionItemKind::CONSTRUCTOR,
        SymbolKind::DataType | SymbolKind::TypeAlias | SymbolKind::TypeParameter => {
            CompletionItemKind::TYPE_PARAMETER
        }
    }
}

fn builtin_completion_kind(label: &str) -> CompletionItemKind {
    if label.chars().next().is_some_and(char::is_lowercase) {
        CompletionItemKind::FUNCTION
    } else {
        CompletionItemKind::VARIABLE
    }
}

fn generic_builtin_completion_item(
    label: &str,
    scheme: &Scheme,
    engine: &InferEngine,
    kind: CompletionItemKind,
) -> CompletionItem {
    let signature = format_scheme(engine, scheme);
    CompletionItem {
        label: label.to_owned(),
        kind: Some(kind),
        detail: Some(format!("builtin : {}", signature)),
        documentation: Some(Documentation::MarkupContent(MarkupContent {
            kind: MarkupKind::Markdown,
            value: generic_builtin_markdown_text(label, &signature),
        })),
        sort_text: Some(format!("04-{}", label)),
        filter_text: Some(label.to_owned()),
        ..Default::default()
    }
}

fn generic_builtin_markdown(state: &IdeState<'_>, label: &str) -> Option<String> {
    state.analyzer.env.lookup(label).map(|scheme| {
        generic_builtin_markdown_text(label, &format_scheme(&state.analyzer.engine, scheme))
    })
}

fn generic_builtin_markdown_text(label: &str, signature: &str) -> String {
    let mut sections = vec![format!("```fwgsl\n{} : {}\n```", label, signature)];
    sections.push("Builtin prelude symbol.".to_owned());
    sections.push("Available everywhere without an explicit import.".to_owned());
    sections.join("\n\n")
}

fn symbol_detail(state: &IdeState<'_>, symbol: &Symbol) -> String {
    match symbol.kind {
        SymbolKind::Function => state
            .analyzer
            .env
            .lookup(&symbol.name)
            .map(|scheme| {
                format!(
                    "binding : {}",
                    format_scheme(&state.analyzer.engine, scheme)
                )
            })
            .unwrap_or_else(|| "binding".to_owned()),
        SymbolKind::EntryPoint => state
            .analyzer
            .env
            .lookup(&symbol.name)
            .map(|scheme| {
                format!(
                    "entry point : {}",
                    format_scheme(&state.analyzer.engine, scheme)
                )
            })
            .unwrap_or_else(|| "entry point".to_owned()),
        SymbolKind::Parameter => "parameter".to_owned(),
        SymbolKind::LocalBinding | SymbolKind::PatternBinding => state
            .analyzer
            .env
            .lookup(&symbol.name)
            .map(|scheme| format!("local : {}", format_scheme(&state.analyzer.engine, scheme)))
            .unwrap_or_else(|| "local binding".to_owned()),
        SymbolKind::Constructor => state
            .analyzer
            .env
            .lookup(&symbol.name)
            .map(|scheme| {
                format!(
                    "constructor : {}",
                    format_scheme(&state.analyzer.engine, scheme)
                )
            })
            .unwrap_or_else(|| "constructor".to_owned()),
        SymbolKind::DataType => "data type".to_owned(),
        SymbolKind::TypeAlias => "type alias".to_owned(),
        SymbolKind::TypeParameter => "type parameter".to_owned(),
    }
}

fn symbol_markdown(state: &IdeState<'_>, symbol: &Symbol) -> String {
    let mut sections = Vec::new();

    if let Some(signature) = symbol_signature(state, symbol) {
        sections.push(format!("```fwgsl\n{} : {}\n```", symbol.name, signature));
    } else if let Some(excerpt) = declaration_excerpt(state, symbol) {
        sections.push(format!("```fwgsl\n{}\n```", excerpt));
    } else {
        sections.push(format!("**`{}`**", symbol.name));
    }

    sections.push(symbol_summary(state, symbol));

    let references = state
        .index
        .occurrences
        .iter()
        .filter(|occurrence| occurrence.symbol_id == symbol.id)
        .count();
    sections.push(format!(
        "{} reference{} in this document.",
        references,
        if references == 1 { "" } else { "s" }
    ));

    sections.join("\n\n")
}

fn symbol_signature(state: &IdeState<'_>, symbol: &Symbol) -> Option<String> {
    match symbol.kind {
        SymbolKind::DataType | SymbolKind::TypeAlias | SymbolKind::TypeParameter => None,
        _ => state
            .analyzer
            .env
            .lookup(&symbol.name)
            .map(|scheme| format_scheme(&state.analyzer.engine, scheme)),
    }
}

fn symbol_summary(state: &IdeState<'_>, symbol: &Symbol) -> String {
    match symbol.kind {
        SymbolKind::Function => "Top-level binding from this document.".to_owned(),
        SymbolKind::EntryPoint => "Shader entry point from this document.".to_owned(),
        SymbolKind::Parameter => match &symbol.container {
            Some(container) => format!("Parameter of `{}`.", container),
            None => "Function parameter.".to_owned(),
        },
        SymbolKind::LocalBinding => match &symbol.container {
            Some(container) => format!("Local binding inside `{}`.", container),
            None => "Local binding.".to_owned(),
        },
        SymbolKind::PatternBinding => match &symbol.container {
            Some(container) => format!("Pattern-bound name inside `{}`.", container),
            None => "Pattern-bound name.".to_owned(),
        },
        SymbolKind::DataType => {
            if let Some(data_type) = state.analyzer.data_types.get(&symbol.name) {
                if data_type.constructors.is_empty() {
                    "User-defined data type.".to_owned()
                } else {
                    format!(
                        "User-defined data type with constructors: {}.",
                        data_type.constructors.join(", ")
                    )
                }
            } else {
                "User-defined data type.".to_owned()
            }
        }
        SymbolKind::TypeAlias => "Named type alias from this document.".to_owned(),
        SymbolKind::Constructor => state
            .analyzer
            .constructors
            .get(&symbol.name)
            .map(|constructor| format!("Constructor for `{}`.", constructor.type_name))
            .unwrap_or_else(|| "Data constructor.".to_owned()),
        SymbolKind::TypeParameter => match &symbol.container {
            Some(container) => format!("Type parameter scoped to `{}`.", container),
            None => "Type parameter.".to_owned(),
        },
    }
}

fn declaration_excerpt(state: &IdeState<'_>, symbol: &Symbol) -> Option<String> {
    let line_index = compute_line_starts(state.source)
        .binary_search(&symbol.primary_span.start)
        .unwrap_or_else(|index| index.saturating_sub(1));
    state
        .source
        .lines()
        .nth(line_index)
        .map(|line| line.trim().to_owned())
        .filter(|line| !line.is_empty())
}

fn previous_non_trivia_token(tokens: &[Token], index: usize) -> Option<&Token> {
    tokens[..index]
        .iter()
        .rev()
        .find(|token| !token.kind.is_trivia())
}

fn lookup_non_attribute_spec(label: &str) -> Option<&'static CompletionSpec> {
    lookup_completion_spec(label, CompletionContext::Value)
        .or_else(|| lookup_completion_spec(label, CompletionContext::Type))
}

fn spec_hover(
    state: &IdeState<'_>,
    spec: &CompletionSpec,
    range: Range,
    symbol_name: Option<&str>,
) -> Hover {
    let signature = symbol_name
        .and_then(|name| state.analyzer.env.lookup(name))
        .map(|scheme| format_scheme(&state.analyzer.engine, scheme));
    let mut sections = Vec::new();
    if let Some(signature) = signature {
        sections.push(format!("```fwgsl\n{} : {}\n```", spec.label, signature));
    }
    sections.push(format!("**`{}`**", spec.label));
    sections.push(format!("_{}_", spec.detail));
    sections.push(spec.documentation.to_owned());

    Hover {
        contents: HoverContents::Markup(MarkupContent {
            kind: MarkupKind::Markdown,
            value: sections.join("\n\n"),
        }),
        range: Some(range),
    }
}

fn format_scheme(engine: &InferEngine, scheme: &Scheme) -> String {
    let ty = engine.finalize(&scheme.ty);
    if scheme.vars.is_empty() {
        format!("{}", ty)
    } else {
        let vars = scheme
            .vars
            .iter()
            .map(|var| format!("t{}", var))
            .collect::<Vec<_>>()
            .join(" ");
        format!("forall {}. {}", vars, ty)
    }
}

pub fn compute_line_starts(source: &str) -> Vec<u32> {
    let mut starts = vec![0u32];
    for (i, b) in source.bytes().enumerate() {
        if b == b'\n' {
            starts.push((i + 1) as u32);
        }
    }
    starts
}

pub fn offset_to_line_col(line_starts: &[u32], offset: u32) -> (u32, u32) {
    let line = match line_starts.binary_search(&offset) {
        Ok(i) => i,
        Err(i) => i.saturating_sub(1),
    };
    let col = offset - line_starts[line];
    (line as u32, col)
}

pub fn position_to_offset(source: &str, pos: Position) -> Option<usize> {
    let line_starts = compute_line_starts(source);
    let line = pos.line as usize;
    if line >= line_starts.len() {
        return None;
    }
    let line_start = line_starts[line] as usize;
    let offset = line_start + pos.character as usize;
    if offset <= source.len() {
        Some(offset)
    } else {
        None
    }
}

pub fn span_to_range(source: &str, span: Span) -> Range {
    let line_starts = compute_line_starts(source);
    let (start_line, start_col) = offset_to_line_col(&line_starts, span.start);
    let (end_line, end_col) = offset_to_line_col(&line_starts, span.end);
    Range::new(
        Position::new(start_line, start_col),
        Position::new(end_line, end_col),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completions_include_generic_prefixed_builtin_docs() {
        let items = build_completions("le", Position::new(0, 2));
        let item = items.iter().find(|item| item.label == "length").unwrap();
        assert!(item
            .detail
            .as_deref()
            .unwrap_or_default()
            .contains("builtin"));
        match &item.documentation {
            Some(Documentation::MarkupContent(markup)) => {
                assert!(markup.value.contains("length"));
                assert!(markup.value.contains("Builtin prelude symbol"));
            }
            other => panic!("expected builtin docs, got {other:?}"),
        }
    }

    #[test]
    fn hover_on_prefixed_builtin_includes_signature() {
        let source = "main x = length x";
        let hover = build_hover(source, Position::new(0, 10)).unwrap();
        match hover.contents {
            HoverContents::Markup(markup) => {
                assert!(markup.value.contains("length"));
                assert!(markup.value.contains("Builtin prelude symbol"));
                assert!(markup.value.contains("```fwgsl"));
            }
            other => panic!("expected markup hover, got {other:?}"),
        }
    }

    #[test]
    fn goto_definition_returns_signature_and_implementation() {
        let source = "double : I32 -> I32\ndouble x = x * 2\nmain = double 2";
        let uri = Url::parse("file:///test.fwgsl").unwrap();
        let response = build_goto_definition(&uri, source, Position::new(2, 9)).unwrap();
        match response {
            GotoDefinitionResponse::Array(locations) => {
                assert_eq!(locations.len(), 2);
                assert_eq!(locations[0].range.start.line, 0);
                assert_eq!(locations[1].range.start.line, 1);
            }
            other => panic!("expected multiple locations, got {other:?}"),
        }
    }

    #[test]
    fn references_include_definition_and_uses() {
        let source = "double : I32 -> I32\ndouble x = x * 2\nmain = double (double 2)";
        let uri = Url::parse("file:///test.fwgsl").unwrap();
        let references = build_references(&uri, source, Position::new(2, 9), true).unwrap();
        assert_eq!(references.len(), 4);
        assert_eq!(references[0].range.start.line, 0);
        assert_eq!(references[1].range.start.line, 1);
        assert_eq!(references[2].range.start.line, 2);
        assert_eq!(references[3].range.start.line, 2);
    }
}
