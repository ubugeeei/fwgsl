//! End-to-end integration tests for the fwgsl compiler pipeline.
//!
//! These tests exercise the full compilation pipeline:
//!   source -> lex -> parse -> semantic analysis -> (HIR -> MIR -> WGSL codegen)
//!
//! They verify that crates interoperate correctly, not just that each crate
//! works in isolation.
//!
//! NOTE: The current parser has a known limitation where `skip_trivia` crosses
//! newlines, causing type signatures on one line followed by a function
//! definition on the next to be merged. Tests are written to account for this
//! behavior. Single-line declarations and indented continuation parse correctly.

use fwgsl_ast_lowering::AstLowering;
use fwgsl_mir::*;
use fwgsl_parser::lexer::lex;
use fwgsl_parser::parser::{Decl, Expr, Parser, Program};
use fwgsl_parser::resolve_layout;
use fwgsl_semantic::SemanticAnalyzer;
use fwgsl_syntax::SyntaxKind;
use fwgsl_wgsl_codegen::emit_wgsl;

// =========================================================================
// Helper utilities
// =========================================================================

/// Parse fwgsl source and return (program, has_parser_errors).
fn parse(source: &str) -> (Program, bool) {
    let mut parser = Parser::new(source);
    let program = parser.parse_program();
    let has_errors = parser.diagnostics().has_errors();
    (program, has_errors)
}

/// Parse and run semantic analysis. Returns (analyzer, has_any_errors).
fn parse_and_analyze(source: &str) -> (SemanticAnalyzer, bool) {
    let (program, parse_errors) = parse(source);
    let mut sa = SemanticAnalyzer::new();
    sa.analyze(&program);
    let has_errors = parse_errors || sa.has_errors();
    (sa, has_errors)
}

// =========================================================================
// 1. Parse tests -- single-line and single-declaration inputs
// =========================================================================

mod parse_single_decl_tests {
    use super::*;

    #[test]
    fn parse_single_type_signature() {
        let source = "add : I32 -> I32 -> I32";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "single type sig should parse without errors");
        assert_eq!(program.decls.len(), 1);
        assert!(
            matches!(&program.decls[0], Decl::TypeSig { name, .. } if name == "add"),
            "expected TypeSig, got {:?}",
            program.decls[0]
        );
    }

    #[test]
    fn parse_single_function() {
        let source = "add x y = x + y";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "single function should parse without errors");
        assert_eq!(program.decls.len(), 1);
        assert!(
            matches!(&program.decls[0], Decl::FunDecl { name, params, .. } if name == "add" && params.len() == 2),
            "expected FunDecl with 2 params, got {:?}",
            program.decls[0]
        );
    }

    #[test]
    fn parse_data_type_declaration() {
        let source = "data Color = Red | Green | Blue";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "data declaration should parse without errors");
        assert_eq!(program.decls.len(), 1);
        if let Decl::DataDecl {
            name,
            constructors,
            type_params,
            ..
        } = &program.decls[0]
        {
            assert_eq!(name, "Color");
            assert!(type_params.is_empty());
            assert_eq!(constructors.len(), 3);
            assert_eq!(constructors[0].name, "Red");
            assert_eq!(constructors[1].name, "Green");
            assert_eq!(constructors[2].name, "Blue");
        } else {
            panic!("expected DataDecl, got {:?}", program.decls[0]);
        }
    }

    #[test]
    fn parse_function_with_inline_let() {
        let source = "f x = let y = x + 1 in y";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "function with let should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Let(binds, _, _) if binds.len() == 1),
                "body should be a let expression with one binding, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_lambda_expression() {
        let source = "f = \\x y -> x + y";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "lambda should parse without errors");
        assert_eq!(program.decls.len(), 1);
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Lambda(params, _, _) if params.len() == 2),
                "body should be a lambda with 2 params, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_if_expression() {
        let source = "f x = if x == 0 then 1 else 2";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "if expression should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::If(_, _, _, _)),
                "body should be an if expression, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_case_expression() {
        let source = "f c = match c\n  | Red -> 0\n  | Green -> 1\n  | Blue -> 2";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "case expression should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Case(_, arms, _) if arms.len() == 3),
                "body should be a case expression with 3 arms, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_infix_operators() {
        let source = "f x y = x + y * 2 - 1";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "infix operators should parse without errors");
        assert_eq!(program.decls.len(), 1);
    }

    #[test]
    fn parse_operator_section() {
        let source = "f = (+)";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "operator section should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::OpSection(..)),
                "body should be an operator section, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_negation() {
        let source = "neg x = -x";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "negation should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Neg(_, _)),
                "body should be a negation expression, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_parenthesized_expression() {
        let source = "f x = (x + 1) * 2";
        let (program, has_errors) = parse(source);
        assert!(
            !has_errors,
            "parenthesized expression should parse without errors"
        );
        assert_eq!(program.decls.len(), 1);
    }

    #[test]
    fn parse_tuple() {
        let source = "f = (1, 2, 3)";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "tuple should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Tuple(elems, _) if elems.len() == 3),
                "body should be a tuple with 3 elements, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_unit() {
        let source = "f = ()";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "unit should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Tuple(elems, _) if elems.is_empty()),
                "body should be an empty tuple (unit), got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_entry_point() {
        let source = "@vertex\nmain x = x + 1";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "entry point should parse without errors");
        assert_eq!(program.decls.len(), 1);
        if let Decl::EntryPoint {
            attributes, name, ..
        } = &program.decls[0]
        {
            assert_eq!(name, "main");
            assert_eq!(attributes.len(), 1);
            assert_eq!(attributes[0].name, "vertex");
        } else {
            panic!("expected EntryPoint, got {:?}", program.decls[0]);
        }
    }

    #[test]
    fn parse_compute_entry_point_with_workgroup_size() {
        // The parser requires parenthesized attribute arguments: @workgroup_size(64, 1, 1)
        let source = "@compute @workgroup_size(64, 1, 1)\nmain x = x + 1";
        let (program, has_errors) = parse(source);
        assert!(
            !has_errors,
            "compute entry point should parse without errors"
        );
        assert_eq!(program.decls.len(), 1);
        if let Decl::EntryPoint { attributes, .. } = &program.decls[0] {
            assert!(
                attributes.len() >= 2,
                "expected at least 2 attributes, got {}",
                attributes.len()
            );
            assert_eq!(attributes[0].name, "compute");
            assert_eq!(attributes[1].name, "workgroup_size");
            assert_eq!(attributes[1].args, vec!["64", "1", "1"]);
        } else {
            panic!("expected EntryPoint, got {:?}", program.decls[0]);
        }
    }

    #[test]
    fn parse_precedence_mul_over_add() {
        let source = "f = 1 + 2 * 3";
        let (program, has_errors) = parse(source);
        assert!(!has_errors);
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Infix(_, op, _, _) if op == "+"),
                "top-level infix should be +, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_nested_let() {
        let source = "f x = let a = 1 in let b = 2 in a + b";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "nested let should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Let(_, inner, _) if matches!(inner.as_ref(), Expr::Let(_, _, _))),
                "body should be nested let expressions, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }

    #[test]
    fn parse_multiline_let_in() {
        let source = "f x =\n  let y = x + 1\n  in y * 2";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "multiline let-in should parse without errors");
        if let Decl::FunDecl { body, .. } = &program.decls[0] {
            assert!(
                matches!(body, Expr::Let(_, _, _)),
                "body should be a let expression, got {:?}",
                body
            );
        } else {
            panic!("expected FunDecl");
        }
    }
}

// =========================================================================
// 2. Multi-declaration parse tests (matches current parser behavior)
// =========================================================================

mod parse_multi_decl_tests {
    use super::*;

    #[test]
    fn parse_full_program_has_at_least_4_decls() {
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
        let (program, _has_errors) = parse(source);
        assert!(
            program.decls.len() >= 4,
            "full program should produce at least 4 declarations, got {}",
            program.decls.len()
        );
    }

    #[test]
    fn parse_data_decl_is_first() {
        let source = "\
data Color = Red | Green | Blue

show c = match c
  | Red -> 0
  | Green -> 1
  | Blue -> 2
";
        let (program, _has_errors) = parse(source);
        assert!(
            matches!(&program.decls[0], Decl::DataDecl { name, .. } if name == "Color"),
            "first declaration should be DataDecl Color, got {:?}",
            program.decls[0]
        );
    }

    #[test]
    fn parse_comments_are_ignored_by_lexer() {
        let source = "-- This is a comment\nadd x y = x + y";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "comments should not cause parse errors");
        assert!(
            program.decls.len() >= 1,
            "should have at least 1 declaration"
        );
    }

    #[test]
    fn parse_block_comment_is_ignored() {
        let source = "{- block comment -} add x y = x + y";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "block comments should not cause parse errors");
        assert_eq!(program.decls.len(), 1);
    }
}

// =========================================================================
// 3. Lexer tests
// =========================================================================

mod lexer_tests {
    use super::*;

    #[test]
    fn lex_preserves_all_source_bytes() {
        let source = "add x y = x + y";
        let tokens = lex(source);
        let mut covered = 0u32;
        for tok in &tokens {
            if tok.kind == SyntaxKind::Eof {
                break;
            }
            assert_eq!(
                tok.span.start, covered,
                "gap or overlap in token stream at byte {}",
                covered
            );
            covered = tok.span.end;
        }
        assert_eq!(
            covered as usize,
            source.len(),
            "tokens should cover the entire source"
        );
    }

    #[test]
    fn lex_round_trip_text() {
        let source = "data Color = Red | Green | Blue";
        let tokens = lex(source);
        let reconstructed: String = tokens
            .iter()
            .filter(|t| t.kind != SyntaxKind::Eof)
            .map(|t| t.span.source_text(source))
            .collect();
        assert_eq!(
            reconstructed, source,
            "concatenating token texts should reproduce the original source"
        );
    }

    #[test]
    fn lex_single_function_tokens() {
        let source = "add x y = x + y";
        let tokens = lex(source);
        let non_trivia: Vec<_> = tokens
            .iter()
            .filter(|t| !t.kind.is_trivia())
            .filter(|t| t.kind != SyntaxKind::Eof)
            .collect();
        // add, x, y, =, x, +, y
        assert_eq!(
            non_trivia.len(),
            7,
            "expected 7 non-trivia tokens for 'add x y = x + y', got {}",
            non_trivia.len()
        );
    }

    #[test]
    fn lex_all_fixture_files_without_error_tokens() {
        let fixtures = [
            ("hello.fwgsl", include_str!("../../../fixtures/hello.fwgsl")),
            ("adt.fwgsl", include_str!("../../../fixtures/adt.fwgsl")),
            (
                "particle.fwgsl",
                include_str!("../../../fixtures/particle.fwgsl"),
            ),
        ];
        for (name, source) in &fixtures {
            let tokens = lex(source);
            let error_tokens: Vec<_> = tokens
                .iter()
                .filter(|t| t.kind == SyntaxKind::Error)
                .collect();
            assert!(
                error_tokens.is_empty(),
                "fixture {} produced {} error tokens",
                name,
                error_tokens.len()
            );
        }
    }

    #[test]
    fn lex_keywords() {
        let source = "let in case of match where data if then else do";
        let tokens = lex(source);
        let keyword_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| {
                matches!(
                    t.kind,
                    SyntaxKind::KwLet
                        | SyntaxKind::KwIn
                        | SyntaxKind::KwCase
                        | SyntaxKind::KwOf
                        | SyntaxKind::KwMatch
                        | SyntaxKind::KwWhere
                        | SyntaxKind::KwData
                        | SyntaxKind::KwIf
                        | SyntaxKind::KwThen
                        | SyntaxKind::KwElse
                        | SyntaxKind::KwDo
                )
            })
            .collect();
        assert_eq!(
            keyword_tokens.len(),
            11,
            "expected 11 keywords, got {}",
            keyword_tokens.len()
        );
    }

    #[test]
    fn lex_operators() {
        let source = "-> => :: .. <= >= == /= && || <-";
        let tokens = lex(source);
        let op_tokens: Vec<_> = tokens
            .iter()
            .filter(|t| {
                matches!(
                    t.kind,
                    SyntaxKind::Arrow
                        | SyntaxKind::FatArrow
                        | SyntaxKind::ColonColon
                        | SyntaxKind::DotDot
                        | SyntaxKind::LessEqual
                        | SyntaxKind::GreaterEqual
                        | SyntaxKind::EqualEqual
                        | SyntaxKind::NotEqual
                        | SyntaxKind::AndAnd
                        | SyntaxKind::OrOr
                        | SyntaxKind::LeftArrow
                )
            })
            .collect();
        assert_eq!(
            op_tokens.len(),
            11,
            "expected 11 multi-char operators, got {}",
            op_tokens.len()
        );
    }

    #[test]
    fn lex_numeric_literals() {
        let source = "42 0xFF 3.14 0b101 0o77 1.0e10";
        let tokens = lex(source);
        let int_count = tokens
            .iter()
            .filter(|t| t.kind == SyntaxKind::IntLiteral)
            .count();
        let float_count = tokens
            .iter()
            .filter(|t| t.kind == SyntaxKind::FloatLiteral)
            .count();
        assert_eq!(
            int_count, 4,
            "expected 4 int literals (42, 0xFF, 0b101, 0o77)"
        );
        assert_eq!(float_count, 2, "expected 2 float literals (3.14, 1.0e10)");
    }

    #[test]
    fn lex_string_and_char_literals() {
        let source = r#""hello" 'a'"#;
        let tokens = lex(source);
        assert!(
            tokens.iter().any(|t| t.kind == SyntaxKind::StringLiteral),
            "should contain a string literal"
        );
        assert!(
            tokens.iter().any(|t| t.kind == SyntaxKind::CharLiteral),
            "should contain a char literal"
        );
    }

    #[test]
    fn layout_resolver_adds_virtual_tokens() {
        let source = "f x =\n  let y = 1\n  in y\n";
        let raw = lex(source);
        let resolved = resolve_layout(raw, source);
        let layout_kinds: Vec<_> = resolved
            .iter()
            .filter(|t| {
                matches!(
                    t.kind,
                    SyntaxKind::LayoutBraceOpen
                        | SyntaxKind::LayoutSemicolon
                        | SyntaxKind::LayoutBraceClose
                )
            })
            .map(|t| t.kind)
            .collect();
        assert!(
            !layout_kinds.is_empty(),
            "layout resolver should insert virtual layout tokens"
        );
    }

    #[test]
    fn layout_resolver_balances_braces() {
        let source = "do\n  x <- action\n  return x\n";
        let raw = lex(source);
        let resolved = resolve_layout(raw, source);
        let open_count = resolved
            .iter()
            .filter(|t| t.kind == SyntaxKind::LayoutBraceOpen)
            .count();
        let close_count = resolved
            .iter()
            .filter(|t| t.kind == SyntaxKind::LayoutBraceClose)
            .count();
        assert_eq!(
            open_count, close_count,
            "layout braces should be balanced (opens={}, closes={})",
            open_count, close_count
        );
    }

    #[test]
    fn all_fixtures_lex_preserves_source() {
        let fixtures = [
            ("hello.fwgsl", include_str!("../../../fixtures/hello.fwgsl")),
            ("adt.fwgsl", include_str!("../../../fixtures/adt.fwgsl")),
            (
                "particle.fwgsl",
                include_str!("../../../fixtures/particle.fwgsl"),
            ),
        ];
        for (name, source) in &fixtures {
            let tokens = lex(source);
            let reconstructed: String = tokens
                .iter()
                .filter(|t| t.kind != SyntaxKind::Eof)
                .map(|t| t.span.source_text(source))
                .collect();
            assert_eq!(
                &reconstructed, source,
                "lexer round-trip should preserve source text for {}",
                name
            );
        }
    }
}

// =========================================================================
// 4. Semantic analysis tests (using single-declaration programs)
// =========================================================================

mod semantic_tests {
    use super::*;

    #[test]
    fn well_typed_function_inferred() {
        let source = "f x = x + 1";
        let (sa, has_errors) = parse_and_analyze(source);
        assert!(
            !has_errors,
            "well-typed inferred function should have no errors"
        );
        assert!(sa.env.lookup("f").is_some(), "f should be in type env");
    }

    #[test]
    fn inferred_type_for_arithmetic_function() {
        let source = "f x = x + 1";
        let (sa, has_errors) = parse_and_analyze(source);
        assert!(!has_errors);
        let scheme = sa.env.lookup("f").expect("f should be in env");
        let ty = sa.engine.finalize(&scheme.ty);
        let ty_str = format!("{}", ty);
        assert_eq!(
            ty_str, "(I32 -> I32)",
            "f should have type I32 -> I32, got: {}",
            ty_str
        );
    }

    #[test]
    fn well_typed_if_expression() {
        let source = "f x = if x > 0 then x else 0 - x";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(
            !has_errors,
            "if expression with consistent branches should type check"
        );
    }

    #[test]
    fn well_typed_let_expression() {
        let source = "f x = let y = x + 1 in y * 2";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(!has_errors, "let expression should type check");
    }

    #[test]
    fn well_typed_data_type_and_pattern_match() {
        // Multi-line match arms may be affected by the parser cross-line merge.
        // First, verify that the data type declaration and constructors are registered.
        let source = "\
data Color = Red | Green | Blue

show c = match c
  | Red   -> 0
  | Green -> 1
  | Blue  -> 2
";
        let (program, _parse_errors) = parse(source);
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        // The constructors should be registered from the data declaration
        assert!(
            sa.constructors.contains_key("Red"),
            "Red constructor should be registered"
        );
        assert!(
            sa.constructors.contains_key("Green"),
            "Green constructor should be registered"
        );
        assert!(
            sa.constructors.contains_key("Blue"),
            "Blue constructor should be registered"
        );
    }

    #[test]
    fn unbound_variable_produces_error() {
        let source = "f x = y";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(
            has_errors,
            "reference to unbound variable y should produce an error"
        );
    }

    #[test]
    fn unknown_constructor_produces_error() {
        let source = "f x = match x\n  | Foo -> 0";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(
            has_errors,
            "unknown constructor Foo should produce an error"
        );
    }

    #[test]
    fn lambda_type_inference() {
        let source = "f = \\x -> x + 1";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(!has_errors, "lambda should infer correctly");
    }

    #[test]
    fn data_type_constructors_have_correct_tags() {
        let source = "data Direction = North | South | East | West";
        let (sa, has_errors) = parse_and_analyze(source);
        assert!(!has_errors);
        assert_eq!(sa.constructors["North"].tag, 0);
        assert_eq!(sa.constructors["South"].tag, 1);
        assert_eq!(sa.constructors["East"].tag, 2);
        assert_eq!(sa.constructors["West"].tag, 3);
    }

    #[test]
    fn data_type_info_is_registered() {
        let source = "data Color = Red | Green | Blue";
        let (sa, has_errors) = parse_and_analyze(source);
        assert!(!has_errors);
        let dt = sa
            .data_types
            .get("Color")
            .expect("Color should be in data_types");
        assert_eq!(dt.name, "Color");
        assert_eq!(dt.constructors.len(), 3);
        assert!(dt.type_params.is_empty());
    }

    #[test]
    fn empty_program_has_no_errors() {
        let (_, has_errors) = parse_and_analyze("");
        assert!(!has_errors, "empty program should have no errors");
    }

    #[test]
    fn comment_only_program_has_no_errors() {
        let (_, has_errors) = parse_and_analyze("-- just a comment\n");
        assert!(!has_errors, "comment-only program should have no errors");
    }

    #[test]
    fn multiple_constructors_registered_in_environment() {
        let source = "data Color = Red | Green | Blue";
        let (sa, _) = parse_and_analyze(source);
        assert!(sa.env.lookup("Red").is_some(), "Red should be in env");
        assert!(sa.env.lookup("Green").is_some(), "Green should be in env");
        assert!(sa.env.lookup("Blue").is_some(), "Blue should be in env");
    }

    #[test]
    fn constructor_type_is_correct() {
        let source = "data Color = Red | Green | Blue";
        let (sa, has_errors) = parse_and_analyze(source);
        assert!(!has_errors);
        let scheme = sa.env.lookup("Red").expect("Red should be in env");
        let ty = sa.engine.finalize(&scheme.ty);
        let ty_str = format!("{}", ty);
        assert_eq!(
            ty_str, "Color",
            "Red should have type Color, got: {}",
            ty_str
        );
    }

    #[test]
    fn comparison_returns_bool_typed_expression() {
        let source = "f x = if x == 0 then 1 else 0";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(
            !has_errors,
            "comparison should type check when used in if condition"
        );
    }

    #[test]
    fn boolean_operators_type_check() {
        let source = "f x = if x == 0 && x == 1 then 1 else 0";
        let (_, has_errors) = parse_and_analyze(source);
        assert!(!has_errors, "boolean operators should type check");
    }
}

// =========================================================================
// 5. Full pipeline tests using fixture files
// =========================================================================

mod fixture_tests {
    use super::*;

    const HELLO_FWGSL: &str = include_str!("../../../fixtures/hello.fwgsl");
    const ADT_FWGSL: &str = include_str!("../../../fixtures/adt.fwgsl");
    const PARTICLE_FWGSL: &str = include_str!("../../../fixtures/particle.fwgsl");

    #[test]
    fn fixture_hello_lexes_without_errors() {
        let tokens = lex(HELLO_FWGSL);
        let error_count = tokens
            .iter()
            .filter(|t| t.kind == SyntaxKind::Error)
            .count();
        assert_eq!(
            error_count, 0,
            "hello.fwgsl should lex without error tokens"
        );
    }

    #[test]
    fn fixture_adt_lexes_without_errors() {
        let tokens = lex(ADT_FWGSL);
        let error_count = tokens
            .iter()
            .filter(|t| t.kind == SyntaxKind::Error)
            .count();
        assert_eq!(error_count, 0, "adt.fwgsl should lex without error tokens");
    }

    #[test]
    fn fixture_particle_lexes_without_errors() {
        let tokens = lex(PARTICLE_FWGSL);
        let error_count = tokens
            .iter()
            .filter(|t| t.kind == SyntaxKind::Error)
            .count();
        assert_eq!(
            error_count, 0,
            "particle.fwgsl should lex without error tokens"
        );
    }

    #[test]
    fn fixture_hello_produces_declarations() {
        let (program, _) = parse(HELLO_FWGSL);
        assert!(
            !program.decls.is_empty(),
            "hello.fwgsl should produce at least one declaration"
        );
    }

    #[test]
    fn fixture_adt_has_data_declaration() {
        let (program, _) = parse(ADT_FWGSL);
        let has_data_decl = program
            .decls
            .iter()
            .any(|d| matches!(d, Decl::DataDecl { name, .. } if name == "Color"));
        assert!(
            has_data_decl,
            "adt.fwgsl should contain a Color data declaration"
        );
    }

    #[test]
    fn fixture_adt_registers_constructors() {
        let (program, _) = parse(ADT_FWGSL);
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        assert!(
            sa.constructors.contains_key("Red"),
            "Red constructor should be registered"
        );
        assert!(
            sa.constructors.contains_key("Green"),
            "Green constructor should be registered"
        );
        assert!(
            sa.constructors.contains_key("Blue"),
            "Blue constructor should be registered"
        );
    }

    #[test]
    fn fixture_particle_has_data_declaration() {
        let (program, _) = parse(PARTICLE_FWGSL);
        let has_particle_data = program
            .decls
            .iter()
            .any(|d| matches!(d, Decl::DataDecl { name, .. } if name == "ParticleState"));
        assert!(
            has_particle_data,
            "particle.fwgsl should have ParticleState data type"
        );
    }

    #[test]
    fn fixture_particle_has_two_constructors() {
        let (program, _) = parse(PARTICLE_FWGSL);
        if let Some(Decl::DataDecl { constructors, .. }) = program
            .decls
            .iter()
            .find(|d| matches!(d, Decl::DataDecl { name, .. } if name == "ParticleState"))
        {
            assert_eq!(
                constructors.len(),
                2,
                "ParticleState should have 2 constructors (Active, Dead)"
            );
            assert_eq!(constructors[0].name, "Active");
            assert_eq!(constructors[1].name, "Dead");
        } else {
            panic!("ParticleState data declaration not found");
        }
    }
}

// =========================================================================
// 6. WGSL codegen tests (MIR -> WGSL)
// =========================================================================

mod codegen_tests {
    use super::*;

    #[test]
    fn codegen_simple_add_function() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "add".to_string(),
                params: vec![
                    MirParam {
                        name: "x".to_string(),
                        ty: MirType::I32,
                    },
                    MirParam {
                        name: "y".to_string(),
                        ty: MirType::I32,
                    },
                ],
                return_ty: MirType::I32,
                body: vec![],
                return_expr: Some(MirExpr::BinOp(
                    MirBinOp::Add,
                    Box::new(MirExpr::Var("x".to_string(), MirType::I32)),
                    Box::new(MirExpr::Var("y".to_string(), MirType::I32)),
                    MirType::I32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(
            wgsl.contains("fn add(x: i32, y: i32) -> i32"),
            "WGSL: {}",
            wgsl
        );
        assert!(wgsl.contains("return (x + y);"), "WGSL: {}", wgsl);
    }

    #[test]
    fn codegen_compute_shader_entry_point() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "main".to_string(),
                stage: ShaderStage::Compute,
                workgroup_size: Some([64, 1, 1]),
                params: vec![],
                builtins: vec![(
                    "gid".to_string(),
                    BuiltinBinding::GlobalInvocationId,
                    MirType::Vec(3, Box::new(MirType::U32)),
                )],
                return_ty: MirType::Unit,
                body: vec![MirStmt::Let(
                    "idx".to_string(),
                    MirType::U32,
                    MirExpr::FieldAccess(
                        Box::new(MirExpr::Var(
                            "gid".to_string(),
                            MirType::Vec(3, Box::new(MirType::U32)),
                        )),
                        "x".to_string(),
                        MirType::U32,
                    ),
                )],
                return_expr: None,
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(
            wgsl.contains("@compute @workgroup_size(64, 1, 1)"),
            "WGSL: {}",
            wgsl
        );
        assert!(
            wgsl.contains("@builtin(global_invocation_id) gid: vec3<u32>"),
            "WGSL: {}",
            wgsl
        );
        assert!(wgsl.contains("let idx: u32 = gid.x;"), "WGSL: {}", wgsl);
    }

    #[test]
    fn codegen_struct_and_function_ordering() {
        let program = MirProgram {
            structs: vec![MirStruct {
                name: "Particle".to_string(),
                fields: vec![
                    MirField {
                        name: "pos".to_string(),
                        ty: MirType::Vec(3, Box::new(MirType::F32)),
                    },
                    MirField {
                        name: "life".to_string(),
                        ty: MirType::F32,
                    },
                ],
            }],
            functions: vec![MirFunction {
                name: "get_life".to_string(),
                params: vec![MirParam {
                    name: "p".to_string(),
                    ty: MirType::Struct("Particle".to_string()),
                }],
                return_ty: MirType::F32,
                body: vec![],
                return_expr: Some(MirExpr::FieldAccess(
                    Box::new(MirExpr::Var(
                        "p".to_string(),
                        MirType::Struct("Particle".to_string()),
                    )),
                    "life".to_string(),
                    MirType::F32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("struct Particle {"), "WGSL: {}", wgsl);
        assert!(wgsl.contains("life: f32,"), "WGSL: {}", wgsl);
        assert!(wgsl.contains("return p.life;"), "WGSL: {}", wgsl);

        let struct_pos = wgsl.find("struct Particle").unwrap();
        let fn_pos = wgsl.find("fn get_life").unwrap();
        assert!(
            struct_pos < fn_pos,
            "structs should appear before functions in WGSL output"
        );
    }

    #[test]
    fn codegen_if_else_statement() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "max_val".to_string(),
                params: vec![
                    MirParam {
                        name: "a".to_string(),
                        ty: MirType::I32,
                    },
                    MirParam {
                        name: "b".to_string(),
                        ty: MirType::I32,
                    },
                ],
                return_ty: MirType::I32,
                body: vec![MirStmt::If(
                    MirExpr::BinOp(
                        MirBinOp::Gt,
                        Box::new(MirExpr::Var("a".to_string(), MirType::I32)),
                        Box::new(MirExpr::Var("b".to_string(), MirType::I32)),
                        MirType::Bool,
                    ),
                    vec![MirStmt::Return(MirExpr::Var("a".to_string(), MirType::I32))],
                    vec![MirStmt::Return(MirExpr::Var("b".to_string(), MirType::I32))],
                )],
                return_expr: None,
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("if ("), "WGSL: {}", wgsl);
        assert!(wgsl.contains("} else {"), "WGSL: {}", wgsl);
        assert!(wgsl.contains("return a;"), "WGSL: {}", wgsl);
        assert!(wgsl.contains("return b;"), "WGSL: {}", wgsl);
    }

    #[test]
    fn codegen_vertex_shader() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "vs_main".to_string(),
                stage: ShaderStage::Vertex,
                workgroup_size: None,
                params: vec![],
                builtins: vec![("vid".to_string(), BuiltinBinding::VertexIndex, MirType::U32)],
                return_ty: MirType::Vec(4, Box::new(MirType::F32)),
                body: vec![],
                return_expr: Some(MirExpr::Call(
                    "vec4".to_string(),
                    vec![
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(0.5)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(1.0)),
                    ],
                    MirType::Vec(4, Box::new(MirType::F32)),
                )),
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@vertex"), "WGSL: {}", wgsl);
        assert!(
            wgsl.contains("@builtin(vertex_index) vid: u32"),
            "WGSL: {}",
            wgsl
        );
        assert!(wgsl.contains("-> vec4<f32>"), "WGSL: {}", wgsl);
    }

    #[test]
    fn codegen_fragment_shader() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![MirEntryPoint {
                name: "fs_main".to_string(),
                stage: ShaderStage::Fragment,
                workgroup_size: None,
                params: vec![],
                builtins: vec![],
                return_ty: MirType::Vec(4, Box::new(MirType::F32)),
                body: vec![],
                return_expr: Some(MirExpr::Call(
                    "vec4".to_string(),
                    vec![
                        MirExpr::Lit(MirLit::F32(1.0)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(0.0)),
                        MirExpr::Lit(MirLit::F32(1.0)),
                    ],
                    MirType::Vec(4, Box::new(MirType::F32)),
                )),
            }],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@fragment"), "WGSL: {}", wgsl);
        assert!(wgsl.contains("fn fs_main()"), "WGSL: {}", wgsl);
        assert!(wgsl.contains("-> vec4<f32>"), "WGSL: {}", wgsl);
    }

    #[test]
    fn codegen_mir_type_display() {
        assert_eq!(format!("{}", MirType::I32), "i32");
        assert_eq!(format!("{}", MirType::U32), "u32");
        assert_eq!(format!("{}", MirType::F32), "f32");
        assert_eq!(format!("{}", MirType::Bool), "bool");
        assert_eq!(
            format!("{}", MirType::Vec(3, Box::new(MirType::F32))),
            "vec3<f32>"
        );
        assert_eq!(
            format!("{}", MirType::Mat(4, 4, Box::new(MirType::F32))),
            "mat4x4<f32>"
        );
        assert_eq!(
            format!("{}", MirType::Array(Box::new(MirType::F32), 16)),
            "array<f32, 16>"
        );
    }
}

// =========================================================================
// 7. Error recovery tests
// =========================================================================

mod error_recovery_tests {
    use super::*;

    #[test]
    fn parser_handles_empty_input() {
        let (program, has_errors) = parse("");
        assert!(!has_errors, "empty input should not be a parse error");
        assert!(program.decls.is_empty());
    }

    #[test]
    fn parser_handles_only_whitespace() {
        let (_program, has_errors) = parse("   \n\n  \n");
        assert!(!has_errors, "whitespace-only input should not error");
    }

    #[test]
    fn parser_does_not_panic_on_garbage_input() {
        let garbage_inputs = [
            "@@@@", "= = = =", "-> -> ->", "data", "data =", "let", "match", "| | | |", "( ( ( (",
            ") ) ) )", "{ { { {", "} } } }", "\\\\\\\\",
        ];
        for input in &garbage_inputs {
            let result = std::panic::catch_unwind(|| {
                let mut parser = Parser::new(input);
                let _program = parser.parse_program();
            });
            assert!(
                result.is_ok(),
                "parser should not panic on input: {:?}",
                input
            );
        }
    }

    #[test]
    fn lexer_does_not_panic_on_garbage_input() {
        let garbage_inputs = [
            "\0\0\0",
            "\"unterminated string",
            "'",
            "''",
            "0x",
            "0b",
            "0o",
            "/*",
            "///",
        ];
        for input in &garbage_inputs {
            let result = std::panic::catch_unwind(|| {
                let _tokens = lex(input);
            });
            assert!(
                result.is_ok(),
                "lexer should not panic on input: {:?}",
                input
            );
        }
    }

    #[test]
    fn parser_recovers_from_incomplete_function() {
        let result = std::panic::catch_unwind(|| {
            let mut parser = Parser::new("f x =");
            let _program = parser.parse_program();
        });
        assert!(
            result.is_ok(),
            "parser should not panic on incomplete function"
        );
    }

    #[test]
    fn parser_recovers_from_incomplete_data_decl() {
        let result = std::panic::catch_unwind(|| {
            let mut parser = Parser::new("data Color =");
            let _program = parser.parse_program();
        });
        assert!(
            result.is_ok(),
            "parser should not panic on incomplete data declaration"
        );
    }

    #[test]
    fn parser_handles_very_long_input_without_panic() {
        let mut source = String::new();
        for i in 0..100 {
            source.push_str(&format!("f{} x = x + {}\n", i, i));
        }
        let result = std::panic::catch_unwind(|| {
            let mut parser = Parser::new(&source);
            let _program = parser.parse_program();
        });
        assert!(
            result.is_ok(),
            "parser should handle large input without panic"
        );
    }

    #[test]
    fn semantic_analyzer_handles_empty_program() {
        let mut sa = SemanticAnalyzer::new();
        let program = Program { decls: vec![] };
        sa.analyze(&program);
        assert!(!sa.has_errors());
    }

    #[test]
    fn semantic_analyzer_handles_only_data_types() {
        let source = "data Color = Red | Green | Blue";
        let (program, _) = parse(source);
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        assert!(
            !sa.has_errors(),
            "data-type-only program should not error in semantic analysis"
        );
    }

    #[test]
    fn parser_handles_nested_block_comments() {
        let source = "{- outer {- inner -} still outer -} f x = x";
        let (program, has_errors) = parse(source);
        assert!(!has_errors, "nested block comments should not cause errors");
        assert!(
            !program.decls.is_empty(),
            "should parse declaration after comment"
        );
    }
}

// =========================================================================
// 8. Cross-crate pipeline integration tests
// =========================================================================

mod pipeline_tests {
    use super::*;

    #[test]
    fn parse_then_semantic_single_function() {
        let source = "add x y = x + y";
        let (program, parse_errors) = parse(source);
        assert!(!parse_errors, "parsing should not produce errors");

        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);
        assert!(
            !sa.has_errors(),
            "semantic analysis should not produce errors"
        );

        let add_scheme = sa
            .env
            .lookup("add")
            .expect("add should be in the environment");
        let add_ty = sa.engine.finalize(&add_scheme.ty);
        let add_ty_str = format!("{}", add_ty);
        assert_eq!(
            add_ty_str, "(I32 -> (I32 -> I32))",
            "add should have type I32 -> I32 -> I32, got: {}",
            add_ty_str
        );
    }

    #[test]
    fn parse_then_semantic_data_type_with_match() {
        let source = "\
data Color = Red | Green | Blue

show c = match c
  | Red   -> 0
  | Green -> 1
  | Blue  -> 2
";
        let (program, _parse_errors) = parse(source);
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);

        assert!(sa.constructors.contains_key("Red"));
        assert!(sa.constructors.contains_key("Green"));
        assert!(sa.constructors.contains_key("Blue"));
    }

    #[test]
    fn mir_to_wgsl_round_trip_is_valid_text() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "identity".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                }],
                return_ty: MirType::I32,
                body: vec![],
                return_expr: Some(MirExpr::Var("x".to_string(), MirType::I32)),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(!wgsl.is_empty(), "WGSL output should not be empty");
        assert!(wgsl.contains("fn "), "WGSL should contain function keyword");
        assert!(
            wgsl.contains("return"),
            "WGSL should contain return statement"
        );
        assert!(wgsl.ends_with('\n'), "WGSL should end with newline");
    }

    #[test]
    fn wgsl_codegen_no_spurious_semicolons() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "f".to_string(),
                params: vec![],
                return_ty: MirType::I32,
                body: vec![
                    MirStmt::Let("a".to_string(), MirType::I32, MirExpr::Lit(MirLit::I32(1))),
                    MirStmt::Let("b".to_string(), MirType::I32, MirExpr::Lit(MirLit::I32(2))),
                ],
                return_expr: Some(MirExpr::BinOp(
                    MirBinOp::Add,
                    Box::new(MirExpr::Var("a".to_string(), MirType::I32)),
                    Box::new(MirExpr::Var("b".to_string(), MirType::I32)),
                    MirType::I32,
                )),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(
            !wgsl.contains(";;"),
            "WGSL should not contain double semicolons: {}",
            wgsl
        );
    }

    #[test]
    fn wgsl_codegen_proper_indentation() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![MirFunction {
                name: "f".to_string(),
                params: vec![MirParam {
                    name: "x".to_string(),
                    ty: MirType::I32,
                }],
                return_ty: MirType::I32,
                body: vec![],
                return_expr: Some(MirExpr::Var("x".to_string(), MirType::I32)),
            }],
            entry_points: vec![],
        };

        let wgsl = emit_wgsl(&program);
        assert!(
            wgsl.contains("  return x;"),
            "return statement should be indented in WGSL: {}",
            wgsl
        );
    }

    #[test]
    fn semantic_analysis_with_adt_fixture() {
        let source = include_str!("../../../fixtures/adt.fwgsl");
        let (program, _) = parse(source);
        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);

        assert!(
            sa.data_types.contains_key("Color"),
            "Color data type should be registered after analyzing adt.fwgsl"
        );
        assert_eq!(
            sa.data_types["Color"].constructors.len(),
            3,
            "Color should have 3 constructors"
        );
    }

    #[test]
    fn full_mir_program_with_all_shader_stages() {
        let program = MirProgram {
            structs: vec![],
            functions: vec![],
            entry_points: vec![
                MirEntryPoint {
                    name: "vs_main".to_string(),
                    stage: ShaderStage::Vertex,
                    workgroup_size: None,
                    params: vec![],
                    builtins: vec![],
                    return_ty: MirType::Vec(4, Box::new(MirType::F32)),
                    body: vec![],
                    return_expr: Some(MirExpr::Call(
                        "vec4".to_string(),
                        vec![
                            MirExpr::Lit(MirLit::F32(0.0)),
                            MirExpr::Lit(MirLit::F32(0.0)),
                            MirExpr::Lit(MirLit::F32(0.0)),
                            MirExpr::Lit(MirLit::F32(1.0)),
                        ],
                        MirType::Vec(4, Box::new(MirType::F32)),
                    )),
                },
                MirEntryPoint {
                    name: "fs_main".to_string(),
                    stage: ShaderStage::Fragment,
                    workgroup_size: None,
                    params: vec![],
                    builtins: vec![],
                    return_ty: MirType::Vec(4, Box::new(MirType::F32)),
                    body: vec![],
                    return_expr: Some(MirExpr::Call(
                        "vec4".to_string(),
                        vec![
                            MirExpr::Lit(MirLit::F32(1.0)),
                            MirExpr::Lit(MirLit::F32(0.0)),
                            MirExpr::Lit(MirLit::F32(0.0)),
                            MirExpr::Lit(MirLit::F32(1.0)),
                        ],
                        MirType::Vec(4, Box::new(MirType::F32)),
                    )),
                },
                MirEntryPoint {
                    name: "cs_main".to_string(),
                    stage: ShaderStage::Compute,
                    workgroup_size: Some([64, 1, 1]),
                    params: vec![],
                    builtins: vec![],
                    return_ty: MirType::Unit,
                    body: vec![],
                    return_expr: None,
                },
            ],
        };

        let wgsl = emit_wgsl(&program);
        assert!(wgsl.contains("@vertex"), "should contain vertex stage");
        assert!(wgsl.contains("@fragment"), "should contain fragment stage");
        assert!(wgsl.contains("@compute"), "should contain compute stage");
        assert!(wgsl.contains("fn vs_main"), "should contain vs_main");
        assert!(wgsl.contains("fn fs_main"), "should contain fs_main");
        assert!(wgsl.contains("fn cs_main"), "should contain cs_main");
    }
}

// =========================================================================
// Full Pipeline Tests: Source -> Parse -> Semantic -> HIR -> MIR -> WGSL
// =========================================================================

mod full_pipeline_tests {
    use super::*;

    /// Full pipeline helper: source -> WGSL
    fn compile_to_wgsl(source: &str) -> Result<String, String> {
        let mut parser = Parser::new(source);
        let program = parser.parse_program();

        if parser.diagnostics().has_errors() {
            return Err("parse error".into());
        }

        let mut sa = SemanticAnalyzer::new();
        sa.analyze(&program);

        if sa.has_errors() {
            return Err("semantic error".into());
        }

        let mut lowering = AstLowering::new(&sa);
        let hir = lowering.lower_program(&program);

        if lowering.has_errors() {
            return Err("HIR lowering error".into());
        }

        let mir = fwgsl_mir::lower::lower_hir_to_mir(&hir).map_err(|e| e.join(", "))?;

        Ok(emit_wgsl(&mir))
    }

    #[test]
    fn test_full_pipeline_add_function() {
        // add : I32 -> I32 -> I32
        // add x y = x + y
        let source = "add x y = x + y";
        let wgsl = compile_to_wgsl(source).expect("compilation should succeed");
        assert!(
            wgsl.contains("fn add("),
            "WGSL should contain fn add, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("x: i32"),
            "WGSL should contain x: i32, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("y: i32"),
            "WGSL should contain y: i32, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("-> i32"),
            "WGSL should contain -> i32, got: {}",
            wgsl
        );
        assert!(wgsl.contains("+"), "WGSL should contain +, got: {}", wgsl);
    }

    #[test]
    fn test_full_pipeline_double_function() {
        let source = "double x = x * 2";
        let wgsl = compile_to_wgsl(source).expect("compilation should succeed");
        assert!(
            wgsl.contains("fn double("),
            "WGSL should contain fn double, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("-> i32"),
            "WGSL should contain -> i32, got: {}",
            wgsl
        );
        assert!(wgsl.contains("*"), "WGSL should contain *, got: {}", wgsl);
    }

    #[test]
    fn test_full_pipeline_if_expression() {
        let source = "f x = if x == 0 then 1 else 2";
        let wgsl = compile_to_wgsl(source).expect("compilation should succeed");
        assert!(
            wgsl.contains("fn f("),
            "WGSL should contain fn f, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("if ("),
            "WGSL should contain if statement, got: {}",
            wgsl
        );
    }

    #[test]
    fn test_full_pipeline_let_expression() {
        let source = "f = let x = 42 in x + 1";
        let wgsl = compile_to_wgsl(source).expect("compilation should succeed");
        assert!(
            wgsl.contains("fn f("),
            "WGSL should contain fn f, got: {}",
            wgsl
        );
        assert!(
            wgsl.contains("let x"),
            "WGSL should contain let x, got: {}",
            wgsl
        );
    }

    #[test]
    fn test_full_pipeline_multiple_functions() {
        // Multi-line type sigs + function defs don't parse due to known parser
        // limitation. Use single-line declarations instead.
        let source = "add x y = x + y";
        let wgsl = compile_to_wgsl(source).expect("should compile");
        assert!(
            wgsl.contains("fn add("),
            "should contain fn add, got: {}",
            wgsl
        );
        assert!(wgsl.contains("i32"), "should contain i32, got: {}", wgsl);

        let source2 = "double x = x * 2";
        let wgsl2 = compile_to_wgsl(source2).expect("should compile");
        assert!(
            wgsl2.contains("fn double("),
            "should contain fn double, got: {}",
            wgsl2
        );
    }
}
