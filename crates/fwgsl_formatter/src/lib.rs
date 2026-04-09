//! Token-stream based formatter for fwgsl.
//!
//! Operates on the raw lexed tokens (not the layout-resolved stream), so that
//! comments and original structure are preserved. Re-emits tokens with
//! canonical whitespace and indentation.

use fwgsl_parser::lexer::Token;
use fwgsl_parser::lex;
use fwgsl_syntax::SyntaxKind;

/// Formatting configuration.
#[derive(Debug, Clone)]
pub struct FormatConfig {
    /// Number of spaces per indentation level.
    pub indent_width: usize,
    /// Maximum line width (soft limit — no auto-wrapping yet).
    pub max_width: usize,
}

impl Default for FormatConfig {
    fn default() -> Self {
        FormatConfig {
            indent_width: 2,
            max_width: 100,
        }
    }
}

/// Format a fwgsl source string with the given configuration.
pub fn format(source: &str, config: &FormatConfig) -> String {
    let tokens = lex(source);
    let mut engine = FormatEngine::new(source, &tokens, config);
    engine.run();
    engine.output
}

/// Format a fwgsl source string with default configuration.
pub fn format_default(source: &str) -> String {
    format(source, &FormatConfig::default())
}

// ---------------------------------------------------------------------------
// Format engine
// ---------------------------------------------------------------------------

struct FormatEngine<'a> {
    source: &'a str,
    tokens: &'a [Token],
    #[allow(dead_code)]
    config: &'a FormatConfig,
    pos: usize,
    output: String,
    /// Current column in the output (0-based).
    col: usize,
    /// True if we're at the start of a line (only whitespace so far).
    at_line_start: bool,
    /// Number of consecutive blank lines emitted.
    blank_lines: usize,
}

impl<'a> FormatEngine<'a> {
    fn new(source: &'a str, tokens: &'a [Token], config: &'a FormatConfig) -> Self {
        FormatEngine {
            source,
            tokens,
            config,
            pos: 0,
            output: String::with_capacity(source.len()),
            col: 0,
            at_line_start: true,
            blank_lines: 0,
        }
    }

    fn text(&self, tok: &Token) -> &'a str {
        tok.span.source_text(self.source)
    }

    #[allow(dead_code)]
    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.pos)
    }

    #[allow(dead_code)]
    fn peek_kind(&self) -> SyntaxKind {
        self.peek().map_or(SyntaxKind::Eof, |t| t.kind)
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos];
        self.pos += 1;
        tok
    }

    fn emit_str(&mut self, s: &str) {
        for ch in s.chars() {
            if ch == '\n' {
                self.col = 0;
                self.at_line_start = true;
            } else {
                self.col += 1;
                self.at_line_start = false;
            }
        }
        self.output.push_str(s);
    }

    fn emit_newline(&mut self) {
        self.output.push('\n');
        self.col = 0;
        self.at_line_start = true;
    }

    fn emit_space(&mut self) {
        self.output.push(' ');
        self.col += 1;
        self.at_line_start = false;
    }

    #[allow(dead_code)]
    fn emit_spaces(&mut self, n: usize) {
        for _ in 0..n {
            self.output.push(' ');
        }
        self.col += n;
        if n > 0 {
            self.at_line_start = false;
        }
    }

    /// Compute the column of a byte offset in the source.
    #[allow(dead_code)]
    fn source_column(&self, offset: u32) -> usize {
        let off = offset as usize;
        let line_start = self.source[..off].rfind('\n').map_or(0, |i| i + 1);
        off - line_start
    }

    fn run(&mut self) {
        // The formatter preserves the original indentation and structure,
        // but normalizes whitespace within lines:
        // - Single space between tokens on the same line
        // - Spaces around binary operators
        // - No trailing whitespace
        // - Normalize blank lines between top-level declarations to exactly 1
        // - Preserve comment placement

        while self.pos < self.tokens.len() {
            let tok = &self.tokens[self.pos];
            match tok.kind {
                SyntaxKind::Newline => {
                    self.advance();
                    self.trim_trailing_whitespace();
                    self.emit_newline();
                    self.blank_lines += 1;
                }
                SyntaxKind::Whitespace => {
                    // Whitespace at line start = indentation. Preserve it.
                    // Whitespace mid-line = normalize to single space (handled per-token below).
                    let text = self.text(tok);
                    if self.at_line_start {
                        // Count blank lines: if we just saw newlines, check for top-level gaps
                        if self.blank_lines > 1 {
                            // Normalize multiple blank lines to exactly one
                            // (we already emitted one newline; the extra ones were counted)
                            // Remove extra blank lines from output
                            self.collapse_blank_lines();
                        }
                        self.blank_lines = 0;
                        // Preserve original indentation
                        self.advance();
                        self.emit_str(text);
                    } else {
                        // Mid-line whitespace — skip it; we insert canonical spacing
                        // between tokens in the main loop.
                        self.advance();
                    }
                }
                SyntaxKind::LineComment => {
                    if self.blank_lines > 1 {
                        self.collapse_blank_lines();
                    }
                    self.blank_lines = 0;
                    if !self.at_line_start && self.col > 0 {
                        // Trailing comment — ensure single space before it
                        self.ensure_single_space();
                    }
                    let text = self.text(tok);
                    self.advance();
                    self.emit_str(text);
                }
                SyntaxKind::BlockComment => {
                    if self.blank_lines > 1 {
                        self.collapse_blank_lines();
                    }
                    self.blank_lines = 0;
                    if !self.at_line_start {
                        self.ensure_single_space();
                    }
                    let text = self.text(tok);
                    self.advance();
                    self.emit_str(text);
                }
                SyntaxKind::Eof => break,
                _ => {
                    if self.blank_lines > 1 {
                        self.collapse_blank_lines();
                    }
                    self.blank_lines = 0;

                    // For non-trivia tokens, ensure proper spacing
                    if !self.at_line_start {
                        self.emit_inter_token_space(tok);
                    }

                    let text = self.text(tok);
                    self.advance();
                    self.emit_str(text);
                }
            }
        }

        // Final cleanup
        self.trim_trailing_whitespace();
        // Ensure file ends with exactly one newline
        if !self.output.ends_with('\n') {
            self.output.push('\n');
        }
        // Remove trailing blank lines (keep exactly one \n at end)
        while self.output.ends_with("\n\n") {
            self.output.pop();
        }
    }

    /// Emit appropriate spacing between the previous token and the next one.
    fn emit_inter_token_space(&mut self, next: &Token) {
        let kind = next.kind;

        // No space before certain punctuation
        if matches!(
            kind,
            SyntaxKind::RParen
                | SyntaxKind::RBracket
                | SyntaxKind::RBrace
                | SyntaxKind::Comma
                | SyntaxKind::Semicolon
        ) {
            return;
        }

        // No space after opening brackets (already handled since we're looking at 'next')
        if self.last_emitted_kind() == Some(SyntaxKind::LParen)
            || self.last_emitted_kind() == Some(SyntaxKind::LBracket)
            || self.last_emitted_kind() == Some(SyntaxKind::LBrace)
        {
            return;
        }

        // No space between `@` and attribute name
        if self.last_emitted_kind() == Some(SyntaxKind::At) {
            return;
        }

        // No space before/after `.` (field access, composition)
        if kind == SyntaxKind::Dot || self.last_emitted_kind() == Some(SyntaxKind::Dot) {
            return;
        }

        // Single space for everything else
        self.ensure_single_space();
    }

    /// Look back at the last non-whitespace character to determine the previous token kind.
    fn last_emitted_kind(&self) -> Option<SyntaxKind> {
        // Walk backwards from the current position to find the last emitted real token.
        if self.pos < 2 {
            return None;
        }
        for i in (0..self.pos - 1).rev() {
            let k = self.tokens[i].kind;
            if !k.is_trivia() {
                return Some(k);
            }
        }
        None
    }

    fn ensure_single_space(&mut self) {
        if !self.output.ends_with(' ') && !self.output.ends_with('\n') {
            self.emit_space();
        }
    }

    /// Remove trailing spaces from the last line in the output.
    fn trim_trailing_whitespace(&mut self) {
        while self.output.ends_with(' ') || self.output.ends_with('\t') {
            self.output.pop();
        }
    }

    /// Collapse multiple blank lines to at most one.
    fn collapse_blank_lines(&mut self) {
        // Remove extra trailing newlines, keep at most 2 (which = 1 blank line)
        while self.output.ends_with("\n\n\n") {
            // Find the position to truncate
            let len = self.output.len();
            self.output.truncate(len - 1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_preserves_simple_program() {
        let source = "double x = x * 2\n";
        let result = format_default(source);
        assert_eq!(result, "double x = x * 2\n");
    }

    #[test]
    fn format_normalizes_trailing_whitespace() {
        let source = "f x = x   \n";
        let result = format_default(source);
        assert_eq!(result, "f x = x\n");
    }

    #[test]
    fn format_ensures_final_newline() {
        let source = "f x = x";
        let result = format_default(source);
        assert_eq!(result, "f x = x\n");
    }

    #[test]
    fn format_collapses_multiple_blank_lines() {
        let source = "f x = x\n\n\n\ng y = y\n";
        let result = format_default(source);
        assert_eq!(result, "f x = x\n\ng y = y\n");
    }

    #[test]
    fn format_preserves_comments() {
        let source = "-- this is a comment\nf x = x\n";
        let result = format_default(source);
        assert_eq!(result, "-- this is a comment\nf x = x\n");
    }

    #[test]
    fn format_preserves_indentation() {
        let source = "main x =\n  let y = x\n  in y + 1\n";
        let result = format_default(source);
        assert_eq!(result, "main x =\n  let y = x\n  in y + 1\n");
    }

    #[test]
    fn format_normalizes_extra_spaces() {
        let source = "f   x   =   x  +  1\n";
        let result = format_default(source);
        assert_eq!(result, "f x = x + 1\n");
    }

    #[test]
    fn format_no_space_inside_parens() {
        let source = "f ( x ) = ( x + 1 )\n";
        let result = format_default(source);
        assert_eq!(result, "f (x) = (x + 1)\n");
    }

    #[test]
    fn format_preserves_data_decl() {
        let source = "data Color = Red | Green | Blue\n";
        let result = format_default(source);
        assert_eq!(result, "data Color = Red | Green | Blue\n");
    }

    #[test]
    fn format_preserves_type_sig() {
        let source = "add : I32 -> I32 -> I32\n";
        let result = format_default(source);
        assert_eq!(result, "add : I32 -> I32 -> I32\n");
    }

    #[test]
    fn format_no_space_around_dot() {
        let source = "f x = x . y\n";
        let result = format_default(source);
        assert_eq!(result, "f x = x.y\n");
    }

    #[test]
    fn format_attribute_no_space() {
        let source = "@ compute\n";
        let result = format_default(source);
        assert_eq!(result, "@compute\n");
    }

    #[test]
    fn format_idempotent() {
        let source = "-- Example\nf x = x + 1\n\ng : I32 -> I32\ng y = y * 2\n";
        let first = format_default(source);
        let second = format_default(&first);
        assert_eq!(first, second, "formatter is not idempotent");
    }

    #[test]
    fn format_list_literal() {
        let source = "v = [ 1.0 , 2.0 , 3.0 ]\n";
        let result = format_default(source);
        assert_eq!(result, "v = [1.0, 2.0, 3.0]\n");
    }

    #[test]
    fn format_record() {
        let source = "p = Particle { x = 1.0 , y = 2.0 }\n";
        let result = format_default(source);
        assert_eq!(result, "p = Particle {x = 1.0, y = 2.0}\n");
    }

    #[test]
    fn format_hello_example() {
        let source = "-- Minimal end-to-end example.\n-- Demonstrates arithmetic, function calls, and let-bindings.\n\ndouble x = x * 2\n\nmain x =\n  let y = double x\n  in y + 1\n";
        let result = format_default(source);
        assert_eq!(result, source);
    }
}
