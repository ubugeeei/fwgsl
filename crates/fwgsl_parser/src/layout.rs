//! Layout resolver for fwgsl.
//!
//! Implements Haskell 2010-style indentation-sensitive layout rules by
//! inserting virtual `LayoutBraceOpen`, `LayoutSemicolon`, and
//! `LayoutBraceClose` tokens into the token stream.

use fwgsl_span::Span;
use fwgsl_syntax::SyntaxKind;

use crate::lexer::Token;

/// Insert virtual layout tokens into the token stream.
///
/// The layout algorithm is based on the Haskell 2010 report:
/// - After `where`, `let`, `of`, `do`, if the next non-trivia token is not
///   an explicit `{`, insert `LayoutBraceOpen` at the column of that token.
/// - Track an indentation stack. On each newline, compare the next token's
///   column to the top of the stack:
///   - Equal → insert `LayoutSemicolon`
///   - Less  → pop and insert `LayoutBraceClose`, repeat
/// - At EOF, close all open layout contexts.
pub fn resolve_layout(tokens: Vec<Token>, source: &str) -> Vec<Token> {
    let mut resolver = LayoutResolver::new(tokens, source);
    resolver.resolve();
    resolver.output
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

/// Pre-computed line-start offsets for column calculation.
struct LineMap {
    /// Byte offsets at which each line starts (line_starts[0] == 0).
    line_starts: Vec<usize>,
}

impl LineMap {
    fn build(source: &str) -> Self {
        let mut starts = vec![0usize];
        for (i, b) in source.bytes().enumerate() {
            if b == b'\n' {
                starts.push(i + 1);
            }
        }
        Self {
            line_starts: starts,
        }
    }

    /// 0-based column for a byte offset.
    fn column(&self, offset: u32) -> u32 {
        let off = offset as usize;
        // Binary search for the line containing this offset.
        let line = match self.line_starts.binary_search(&off) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        (off - self.line_starts[line]) as u32
    }
}

struct LayoutResolver {
    tokens: Vec<Token>,
    pos: usize,
    output: Vec<Token>,
    /// Stack of indentation columns for open layout contexts.
    indent_stack: Vec<u32>,
    line_map: LineMap,
    /// Suppress the next semicolon insertion (right after a LayoutBraceOpen).
    suppress_next_semi: bool,
}

impl LayoutResolver {
    fn new(tokens: Vec<Token>, source: &str) -> Self {
        Self {
            tokens,
            pos: 0,
            output: Vec::new(),
            indent_stack: Vec::new(),
            line_map: LineMap::build(source),
            suppress_next_semi: false,
        }
    }

    fn at_end(&self) -> bool {
        self.pos >= self.tokens.len()
    }

    fn peek_kind(&self) -> SyntaxKind {
        if self.at_end() {
            SyntaxKind::Eof
        } else {
            self.tokens[self.pos].kind
        }
    }

    /// Find the next non-trivia token index from current position.
    fn next_non_trivia_index(&self, from: usize) -> Option<usize> {
        let mut i = from;
        while i < self.tokens.len() {
            if !self.tokens[i].kind.is_trivia() {
                return Some(i);
            }
            i += 1;
        }
        None
    }

    fn make_virtual_token(kind: SyntaxKind, offset: u32) -> Token {
        Token::new(kind, Span::new(offset, offset))
    }

    fn resolve(&mut self) {
        // Insert a top-level layout context if the file does not start with
        // `module` (which would trigger its own `where` layout). This mimics
        // Haskell 2010's implicit `module Main where {…}` wrapping.
        if let Some(first_idx) = self.next_non_trivia_index(0) {
            let first_kind = self.tokens[first_idx].kind;
            if first_kind != SyntaxKind::KwModule && first_kind != SyntaxKind::Eof {
                let col = self.line_map.column(self.tokens[first_idx].span.start);
                let offset = self.tokens[first_idx].span.start;
                self.indent_stack.push(col);
                // Do NOT suppress the first semicolon for the top-level context,
                // since there is no keyword before it (unlike `where`/`let`/`do`).
                self.suppress_next_semi = false;
                self.output.push(Self::make_virtual_token(
                    SyntaxKind::LayoutBraceOpen,
                    offset,
                ));
            }
        }

        while !self.at_end() {
            let kind = self.peek_kind();

            match kind {
                SyntaxKind::Newline => {
                    // Emit the newline
                    self.output.push(self.tokens[self.pos].clone());
                    self.pos += 1;

                    // Look at the next non-trivia token to determine layout actions
                    if let Some(next_idx) = self.next_non_trivia_index(self.pos) {
                        let next_kind = self.tokens[next_idx].kind;
                        if next_kind == SyntaxKind::Eof {
                            continue;
                        }
                        let col = self.line_map.column(self.tokens[next_idx].span.start);
                        let offset = self.tokens[next_idx].span.start;

                        // Compare against the indent stack
                        while let Some(&top) = self.indent_stack.last() {
                            if col == top {
                                if self.suppress_next_semi {
                                    self.suppress_next_semi = false;
                                } else {
                                    self.output.push(Self::make_virtual_token(
                                        SyntaxKind::LayoutSemicolon,
                                        offset,
                                    ));
                                }
                                break;
                            } else if col < top {
                                self.indent_stack.pop();
                                self.output.push(Self::make_virtual_token(
                                    SyntaxKind::LayoutBraceClose,
                                    offset,
                                ));
                                // Continue loop to check further stack entries
                            } else {
                                // col > top: nothing to do
                                break;
                            }
                        }
                    }
                }

                // Layout-triggering keywords
                SyntaxKind::KwWhere | SyntaxKind::KwLet | SyntaxKind::KwOf | SyntaxKind::KwDo => {
                    // Emit the keyword
                    self.output.push(self.tokens[self.pos].clone());
                    self.pos += 1;

                    // Find next non-trivia token
                    if let Some(next_idx) = self.next_non_trivia_index(self.pos) {
                        let next_kind = self.tokens[next_idx].kind;
                        if next_kind != SyntaxKind::LBrace {
                            let col = self.line_map.column(self.tokens[next_idx].span.start);
                            let offset = self.tokens[next_idx].span.start;
                            self.indent_stack.push(col);
                            self.suppress_next_semi = true;
                            self.output.push(Self::make_virtual_token(
                                SyntaxKind::LayoutBraceOpen,
                                offset,
                            ));
                        }
                    }
                }

                SyntaxKind::Eof => {
                    // Close all open layout contexts before Eof
                    let offset = self.tokens[self.pos].span.start;
                    while self.indent_stack.pop().is_some() {
                        self.output.push(Self::make_virtual_token(
                            SyntaxKind::LayoutBraceClose,
                            offset,
                        ));
                    }
                    self.output.push(self.tokens[self.pos].clone());
                    self.pos += 1;
                }

                _ => {
                    // When we emit a real (non-trivia) token after a LayoutBraceOpen
                    // *on the same line*, the suppress flag is no longer needed because
                    // the next newline should produce a LayoutSemicolon normally.
                    if !kind.is_trivia() && self.suppress_next_semi {
                        self.suppress_next_semi = false;
                    }
                    self.output.push(self.tokens[self.pos].clone());
                    self.pos += 1;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex;

    fn layout_kinds(source: &str) -> Vec<SyntaxKind> {
        let tokens = lex(source);
        let resolved = resolve_layout(tokens, source);
        resolved.iter().map(|t| t.kind).collect()
    }

    #[test]
    fn layout_where() {
        let source = "f x = x where\n  g = 1\n  h = 2\n";
        let kinds = layout_kinds(source);

        // After `where`, we expect LayoutBraceOpen
        assert!(
            kinds.contains(&SyntaxKind::LayoutBraceOpen),
            "expected LayoutBraceOpen, got {:?}",
            kinds
        );
        // Between the bindings at the same indent, we expect LayoutSemicolon
        assert!(
            kinds.contains(&SyntaxKind::LayoutSemicolon),
            "expected LayoutSemicolon, got {:?}",
            kinds
        );
    }

    #[test]
    fn layout_let_in() {
        let source = "let\n  x = 1\n  y = 2\nin x";
        let kinds = layout_kinds(source);
        assert!(kinds.contains(&SyntaxKind::LayoutBraceOpen));
        assert!(kinds.contains(&SyntaxKind::LayoutSemicolon));
        assert!(kinds.contains(&SyntaxKind::LayoutBraceClose));
    }

    #[test]
    fn layout_closes_at_eof() {
        let source = "do\n  x <- action";
        let kinds = layout_kinds(source);
        // Should have LayoutBraceOpen and at least one LayoutBraceClose at EOF
        let open_count = kinds
            .iter()
            .filter(|k| **k == SyntaxKind::LayoutBraceOpen)
            .count();
        let close_count = kinds
            .iter()
            .filter(|k| **k == SyntaxKind::LayoutBraceClose)
            .count();
        assert_eq!(open_count, close_count);
    }

    #[test]
    fn explicit_braces_not_overridden() {
        let source = "let { x = 1; y = 2 } in x";
        let kinds = layout_kinds(source);
        // The top-level layout context always inserts one LayoutBraceOpen.
        // The `let` keyword should NOT insert an additional one because
        // an explicit `{` follows it.
        let layout_open_count = kinds
            .iter()
            .filter(|k| **k == SyntaxKind::LayoutBraceOpen)
            .count();
        assert_eq!(
            layout_open_count, 1,
            "should only have the top-level layout brace, got {:?}",
            kinds
        );
    }
}
