///|
/// Concrete syntax tree (CST) for fwgsl.
///
/// The Rust `fwgsl_cst` crate is currently a placeholder and the parser
/// emits an AST directly. This package mirrors that placement and will
/// grow alongside the formatter / lint roadmap in `README.md`.
///
/// A minimal `GreenNode` / `GreenToken` pair is provided so dependent
/// packages can already start to model "kind plus span" CST handles.

///|
/// A CST token: a `SyntaxKind` together with its source span.
pub(all) struct GreenToken {
  kind : @syntax.SyntaxKind
  span : @span.Span
} derive(Show, Eq)

///|
pub fn GreenToken::new(
  kind : @syntax.SyntaxKind,
  span : @span.Span
) -> GreenToken {
  GreenToken::{ kind, span }
}

///|
/// A CST node: a tagged subtree of `GreenNode | GreenToken` children.
///
/// Children are stored in source order. This is intentionally minimal:
/// the full red-green tree (with parent pointers) will be layered on top
/// when the formatter/lint work begins.
pub(all) struct GreenNode {
  kind : @syntax.SyntaxKind
  span : @span.Span
  children : Array[GreenChild]
} derive(Show)

///|
/// A child of a `GreenNode`.
pub(all) enum GreenChild {
  Node(GreenNode)
  Token(GreenToken)
} derive(Show)

///|
pub fn GreenNode::new(
  kind : @syntax.SyntaxKind,
  span : @span.Span,
  children : Array[GreenChild]
) -> GreenNode {
  GreenNode::{ kind, span, children }
}

///|
/// Span of a child, dispatching on whether it is a node or a token.
pub fn GreenChild::span(self : GreenChild) -> @span.Span {
  match self {
    Node(n) => n.span
    Token(t) => t.span
  }
}
