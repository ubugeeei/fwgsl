//! Snapshot tests for the fwgsl lexer, layout resolver, and parser.

use fwgsl_parser::layout::resolve_layout;
use fwgsl_parser::lexer::lex;
use fwgsl_parser::parser::Parser;

const EXAMPLE_PROGRAM: &str = "\
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

// ── Lexer ────────────────────────────────────────────────────────────────

#[test]
fn snapshot_lex_tokens() {
    let tokens = lex(EXAMPLE_PROGRAM);
    let summary: Vec<String> = tokens
        .iter()
        .filter(|t| !t.kind.is_trivia())
        .map(|t| {
            let text = t.span.source_text(EXAMPLE_PROGRAM);
            format!("{:?} {:?}", t.kind, text)
        })
        .collect();
    insta::assert_debug_snapshot!(summary);
}

// ── Layout ───────────────────────────────────────────────────────────────

#[test]
fn snapshot_layout_tokens() {
    let tokens = lex(EXAMPLE_PROGRAM);
    let resolved = resolve_layout(tokens, EXAMPLE_PROGRAM);
    let layout_tokens: Vec<String> = resolved
        .iter()
        .filter(|t| {
            matches!(
                t.kind,
                fwgsl_syntax::SyntaxKind::LayoutBraceOpen
                    | fwgsl_syntax::SyntaxKind::LayoutSemicolon
                    | fwgsl_syntax::SyntaxKind::LayoutBraceClose
            )
        })
        .map(|t| format!("{:?} at byte {}", t.kind, t.span.start))
        .collect();
    insta::assert_debug_snapshot!(layout_tokens);
}

// ── Parser ───────────────────────────────────────────────────────────────

#[test]
fn snapshot_parsed_program() {
    let mut parser = Parser::new(EXAMPLE_PROGRAM);
    let program = parser.parse_program();
    insta::assert_debug_snapshot!(program);
}

// ── Individual expression tests ─────────────────────────────────────────

#[test]
fn snapshot_parse_lambda() {
    let mut parser = Parser::new("f = \\x y -> x + y");
    let program = parser.parse_program();
    insta::assert_debug_snapshot!(program);
}

#[test]
fn snapshot_parse_if() {
    let mut parser = Parser::new("f x = if x == 0 then 1 else 2");
    let program = parser.parse_program();
    insta::assert_debug_snapshot!(program);
}

#[test]
fn snapshot_parse_entry_point() {
    let mut parser = Parser::new("@vertex\nmain x = x + 1");
    let program = parser.parse_program();
    insta::assert_debug_snapshot!(program);
}
