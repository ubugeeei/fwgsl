//! Hand-written lexer for fwgsl (Oxc-style, not regex-based).
//!
//! Takes a `&str` source and produces `Vec<Token>` where each token carries
//! a `SyntaxKind` and a `Span`.

use fwgsl_span::Span;
use fwgsl_syntax::{keyword_from_str, SyntaxKind};

/// A single token produced by the lexer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Token {
    pub kind: SyntaxKind,
    pub span: Span,
}

impl Token {
    /// Create a new token.
    #[inline]
    pub fn new(kind: SyntaxKind, span: Span) -> Self {
        Self { kind, span }
    }

    /// Extract the source text for this token.
    #[inline]
    pub fn text<'a>(&self, source: &'a str) -> &'a str {
        self.span.source_text(source)
    }
}

/// Lex the entire source into a token stream, ending with `Eof`.
pub fn lex(source: &str) -> Vec<Token> {
    let mut lexer = Lexer::new(source);
    lexer.lex_all();
    lexer.tokens
}

// ---------------------------------------------------------------------------
// Internal lexer state
// ---------------------------------------------------------------------------

struct Lexer<'a> {
    source: &'a [u8],
    pos: usize,
    tokens: Vec<Token>,
}

impl<'a> Lexer<'a> {
    fn new(source: &'a str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
            tokens: Vec::new(),
        }
    }

    // -- Helpers -----------------------------------------------------------

    #[inline]
    fn at_end(&self) -> bool {
        self.pos >= self.source.len()
    }

    #[inline]
    fn peek(&self) -> u8 {
        if self.at_end() {
            0
        } else {
            self.source[self.pos]
        }
    }

    #[inline]
    fn peek_at(&self, offset: usize) -> u8 {
        let idx = self.pos + offset;
        if idx >= self.source.len() {
            0
        } else {
            self.source[idx]
        }
    }

    #[inline]
    fn advance(&mut self) -> u8 {
        let b = self.source[self.pos];
        self.pos += 1;
        b
    }

    #[inline]
    fn emit(&mut self, kind: SyntaxKind, start: usize) {
        self.tokens
            .push(Token::new(kind, Span::new(start as u32, self.pos as u32)));
    }

    // -- Top-level loop ----------------------------------------------------

    fn lex_all(&mut self) {
        while !self.at_end() {
            self.lex_token();
        }
        // Emit Eof
        self.emit(SyntaxKind::Eof, self.pos);
    }

    fn lex_token(&mut self) {
        let start = self.pos;
        let b = self.peek();

        match b {
            // Newlines
            b'\n' => {
                self.advance();
                self.emit(SyntaxKind::Newline, start);
            }
            b'\r' => {
                self.advance();
                if self.peek() == b'\n' {
                    self.advance();
                }
                self.emit(SyntaxKind::Newline, start);
            }

            // Horizontal whitespace
            b' ' | b'\t' => {
                self.advance();
                while !self.at_end() && matches!(self.peek(), b' ' | b'\t') {
                    self.advance();
                }
                self.emit(SyntaxKind::Whitespace, start);
            }

            // Comments or operators starting with `-`
            b'-' => {
                if self.peek_at(1) == b'-' {
                    self.lex_line_comment(start);
                } else if self.peek_at(1) == b'>' {
                    self.advance();
                    self.advance();
                    self.emit(SyntaxKind::Arrow, start);
                } else {
                    self.advance();
                    self.emit(SyntaxKind::Minus, start);
                }
            }

            // Block comment `{-` or LBrace
            b'{' => {
                if self.peek_at(1) == b'-' {
                    self.lex_block_comment(start);
                } else {
                    self.advance();
                    self.emit(SyntaxKind::LBrace, start);
                }
            }

            // String literal
            b'"' => self.lex_string(start),

            // Char literal
            b'\'' => self.lex_char(start),

            // Digits
            b'0'..=b'9' => self.lex_number(start),

            // Identifiers (lower / underscore)
            b'a'..=b'z' => self.lex_ident(start),
            b'_' => {
                // Check if followed by ident continuation chars
                if is_ident_continue(self.peek_at(1)) {
                    self.lex_ident(start);
                } else {
                    self.advance();
                    self.emit(SyntaxKind::Underscore, start);
                }
            }

            // Upper identifiers
            b'A'..=b'Z' => self.lex_upper_ident(start),

            // Multi-char operators and single-char punctuation
            b'=' => {
                self.advance();
                if self.peek() == b'>' {
                    self.advance();
                    self.emit(SyntaxKind::FatArrow, start);
                } else if self.peek() == b'=' {
                    self.advance();
                    self.emit(SyntaxKind::EqualEqual, start);
                } else {
                    self.emit(SyntaxKind::Equals, start);
                }
            }
            b':' => {
                self.advance();
                if self.peek() == b':' {
                    self.advance();
                    self.emit(SyntaxKind::ColonColon, start);
                } else {
                    self.emit(SyntaxKind::Colon, start);
                }
            }
            b'.' => {
                self.advance();
                if self.peek() == b'.' {
                    self.advance();
                    self.emit(SyntaxKind::DotDot, start);
                } else {
                    self.emit(SyntaxKind::Dot, start);
                }
            }
            b'<' => {
                self.advance();
                if self.peek() == b'=' {
                    self.advance();
                    self.emit(SyntaxKind::LessEqual, start);
                } else if self.peek() == b'-' {
                    self.advance();
                    self.emit(SyntaxKind::LeftArrow, start);
                } else {
                    self.emit(SyntaxKind::Less, start);
                }
            }
            b'>' => {
                self.advance();
                if self.peek() == b'=' {
                    self.advance();
                    self.emit(SyntaxKind::GreaterEqual, start);
                } else {
                    self.emit(SyntaxKind::Greater, start);
                }
            }
            b'/' => {
                self.advance();
                if self.peek() == b'=' {
                    self.advance();
                    self.emit(SyntaxKind::NotEqual, start);
                } else {
                    self.emit(SyntaxKind::Slash, start);
                }
            }
            b'&' => {
                self.advance();
                if self.peek() == b'&' {
                    self.advance();
                    self.emit(SyntaxKind::AndAnd, start);
                } else {
                    // Single `&` is not a valid fwgsl token, emit error
                    self.emit(SyntaxKind::Error, start);
                }
            }
            b'|' => {
                self.advance();
                if self.peek() == b'|' {
                    self.advance();
                    self.emit(SyntaxKind::OrOr, start);
                } else if self.peek() == b'>' {
                    self.advance();
                    self.emit(SyntaxKind::PipeForward, start);
                } else {
                    self.emit(SyntaxKind::Pipe, start);
                }
            }

            // Single-char punctuation / operators
            b'(' => {
                self.advance();
                self.emit(SyntaxKind::LParen, start);
            }
            b')' => {
                self.advance();
                self.emit(SyntaxKind::RParen, start);
            }
            b'[' => {
                self.advance();
                self.emit(SyntaxKind::LBracket, start);
            }
            b']' => {
                self.advance();
                self.emit(SyntaxKind::RBracket, start);
            }
            b'}' => {
                self.advance();
                self.emit(SyntaxKind::RBrace, start);
            }
            b',' => {
                self.advance();
                self.emit(SyntaxKind::Comma, start);
            }
            b';' => {
                self.advance();
                self.emit(SyntaxKind::Semicolon, start);
            }
            b'\\' => {
                self.advance();
                self.emit(SyntaxKind::Backslash, start);
            }
            b'@' => {
                self.advance();
                self.emit(SyntaxKind::At, start);
            }
            b'+' => {
                self.advance();
                self.emit(SyntaxKind::Plus, start);
            }
            b'*' => {
                self.advance();
                self.emit(SyntaxKind::Star, start);
            }
            b'%' => {
                self.advance();
                self.emit(SyntaxKind::Percent, start);
            }
            b'!' => {
                self.advance();
                self.emit(SyntaxKind::Bang, start);
            }
            b'$' => {
                self.advance();
                // $ followed by alpha/underscore → builtin identifier (e.g. $vec4, $sin)
                if self.peek().is_ascii_alphabetic() || self.peek() == b'_' {
                    while self.peek().is_ascii_alphanumeric() || self.peek() == b'_' {
                        self.advance();
                    }
                    // Allow primed identifiers: $foo'
                    while self.peek() == b'\'' {
                        self.advance();
                    }
                    self.emit(SyntaxKind::Ident, start);
                } else {
                    self.emit(SyntaxKind::Dollar, start);
                }
            }
            b'`' => {
                self.advance();
                self.emit(SyntaxKind::Backtick, start);
            }

            // Unknown / error: advance one UTF-8 codepoint
            _ => {
                self.advance_utf8_char();
                self.emit(SyntaxKind::Error, start);
            }
        }
    }

    // -- Sub-lexers --------------------------------------------------------

    fn lex_line_comment(&mut self, start: usize) {
        // Skip `--`
        self.advance();
        self.advance();
        while !self.at_end() && self.peek() != b'\n' && self.peek() != b'\r' {
            self.advance();
        }
        self.emit(SyntaxKind::LineComment, start);
    }

    fn lex_block_comment(&mut self, start: usize) {
        // Skip `{-`
        self.advance();
        self.advance();
        let mut depth: u32 = 1;
        while !self.at_end() && depth > 0 {
            if self.peek() == b'{' && self.peek_at(1) == b'-' {
                self.advance();
                self.advance();
                depth += 1;
            } else if self.peek() == b'-' && self.peek_at(1) == b'}' {
                self.advance();
                self.advance();
                depth -= 1;
            } else {
                self.advance();
            }
        }
        self.emit(SyntaxKind::BlockComment, start);
    }

    fn lex_string(&mut self, start: usize) {
        self.advance(); // skip opening `"`
        while !self.at_end() {
            match self.peek() {
                b'"' => {
                    self.advance();
                    self.emit(SyntaxKind::StringLiteral, start);
                    return;
                }
                b'\\' => {
                    self.advance(); // skip backslash
                    if !self.at_end() {
                        self.advance(); // skip escaped char
                    }
                }
                b'\n' | b'\r' => {
                    // Unterminated string at newline
                    break;
                }
                _ => {
                    self.advance();
                }
            }
        }
        // Unterminated string
        self.emit(SyntaxKind::Error, start);
    }

    fn lex_char(&mut self, start: usize) {
        self.advance(); // skip opening `'`

        // Handle the character content
        if self.at_end() {
            self.emit(SyntaxKind::Error, start);
            return;
        }

        if self.peek() == b'\\' {
            // Escape sequence
            self.advance(); // skip `\`
            if !self.at_end() {
                self.advance(); // skip escaped char
            }
        } else if self.peek() == b'\'' {
            // Empty char literal ''
            self.advance();
            self.emit(SyntaxKind::Error, start);
            return;
        } else {
            // Normal char -- advance one UTF-8 codepoint
            self.advance_utf8_char();
        }

        if !self.at_end() && self.peek() == b'\'' {
            self.advance();
            self.emit(SyntaxKind::CharLiteral, start);
        } else {
            // If the next char is an ident-continuation character, this might
            // be a primed identifier like x'. Let's not consume further; emit
            // what we have as Error.
            self.emit(SyntaxKind::Error, start);
        }
    }

    fn lex_number(&mut self, start: usize) {
        let first = self.advance();

        // Check for 0x, 0o, 0b prefixes
        if first == b'0' && !self.at_end() {
            match self.peek() {
                b'x' | b'X' => {
                    self.advance();
                    while !self.at_end() && is_hex_digit(self.peek()) {
                        self.advance();
                    }
                    self.emit(SyntaxKind::IntLiteral, start);
                    return;
                }
                b'o' | b'O' => {
                    self.advance();
                    while !self.at_end() && is_oct_digit(self.peek()) {
                        self.advance();
                    }
                    self.emit(SyntaxKind::IntLiteral, start);
                    return;
                }
                b'b' | b'B' => {
                    self.advance();
                    while !self.at_end() && is_bin_digit(self.peek()) {
                        self.advance();
                    }
                    self.emit(SyntaxKind::IntLiteral, start);
                    return;
                }
                _ => {}
            }
        }

        // Consume remaining digits
        while !self.at_end() && self.peek().is_ascii_digit() {
            self.advance();
        }

        // Check for `.` followed by digit (float)
        if !self.at_end() && self.peek() == b'.' && self.peek_at(1).is_ascii_digit() {
            self.advance(); // skip `.`
            while !self.at_end() && self.peek().is_ascii_digit() {
                self.advance();
            }
            // Check for exponent
            if !self.at_end() && (self.peek() == b'e' || self.peek() == b'E') {
                self.advance();
                if !self.at_end() && (self.peek() == b'+' || self.peek() == b'-') {
                    self.advance();
                }
                while !self.at_end() && self.peek().is_ascii_digit() {
                    self.advance();
                }
            }
            self.emit(SyntaxKind::FloatLiteral, start);
        } else {
            self.emit(SyntaxKind::IntLiteral, start);
        }
    }

    fn lex_ident(&mut self, start: usize) {
        self.advance(); // first char already validated
        while !self.at_end() && is_ident_continue(self.peek()) {
            self.advance();
        }
        let text = std::str::from_utf8(&self.source[start..self.pos]).unwrap_or("");
        if let Some(kw) = keyword_from_str(text) {
            self.emit(kw, start);
        } else {
            self.emit(SyntaxKind::Ident, start);
        }
    }

    fn lex_upper_ident(&mut self, start: usize) {
        self.advance(); // first char already validated
        while !self.at_end() && is_ident_continue(self.peek()) {
            self.advance();
        }
        self.emit(SyntaxKind::UpperIdent, start);
    }

    /// Advance past a single UTF-8 codepoint.
    fn advance_utf8_char(&mut self) {
        if self.at_end() {
            return;
        }
        let b = self.source[self.pos];
        let len = utf8_char_len(b);
        for _ in 0..len {
            if !self.at_end() {
                self.pos += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Character classification helpers
// ---------------------------------------------------------------------------

#[inline]
fn is_ident_continue(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_' || b == b'\''
}

#[inline]
fn is_hex_digit(b: u8) -> bool {
    b.is_ascii_hexdigit()
}

#[inline]
fn is_oct_digit(b: u8) -> bool {
    matches!(b, b'0'..=b'7')
}

#[inline]
fn is_bin_digit(b: u8) -> bool {
    b == b'0' || b == b'1'
}

#[inline]
fn utf8_char_len(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if first < 0xE0 {
        2
    } else if first < 0xF0 {
        3
    } else {
        4
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(source: &str) -> Vec<SyntaxKind> {
        lex(source).iter().map(|t| t.kind).collect()
    }

    #[test]
    fn lex_empty() {
        let tokens = lex("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].kind, SyntaxKind::Eof);
    }

    #[test]
    fn lex_whitespace_and_newline() {
        let tokens = lex("  \n\t");
        assert_eq!(
            kinds("  \n\t"),
            vec![
                SyntaxKind::Whitespace,
                SyntaxKind::Newline,
                SyntaxKind::Whitespace,
                SyntaxKind::Eof,
            ]
        );
        let _ = tokens;
    }

    #[test]
    fn lex_line_comment() {
        let tokens = lex("-- hello\nx");
        assert_eq!(tokens[0].kind, SyntaxKind::LineComment);
        assert_eq!(tokens[1].kind, SyntaxKind::Newline);
        assert_eq!(tokens[2].kind, SyntaxKind::Ident);
    }

    #[test]
    fn lex_block_comment_nested() {
        let tokens = lex("{- outer {- inner -} still outer -}");
        assert_eq!(tokens[0].kind, SyntaxKind::BlockComment);
        assert_eq!(tokens[1].kind, SyntaxKind::Eof);
    }

    #[test]
    fn lex_integers() {
        assert_eq!(kinds("42"), vec![SyntaxKind::IntLiteral, SyntaxKind::Eof]);
        assert_eq!(kinds("0xFF"), vec![SyntaxKind::IntLiteral, SyntaxKind::Eof]);
        assert_eq!(kinds("0o77"), vec![SyntaxKind::IntLiteral, SyntaxKind::Eof]);
        assert_eq!(
            kinds("0b101"),
            vec![SyntaxKind::IntLiteral, SyntaxKind::Eof]
        );
    }

    #[test]
    fn lex_float() {
        assert_eq!(
            kinds("3.14"),
            vec![SyntaxKind::FloatLiteral, SyntaxKind::Eof]
        );
        assert_eq!(
            kinds("1.0e10"),
            vec![SyntaxKind::FloatLiteral, SyntaxKind::Eof]
        );
        assert_eq!(
            kinds("2.5e-3"),
            vec![SyntaxKind::FloatLiteral, SyntaxKind::Eof]
        );
    }

    #[test]
    fn lex_string() {
        let tokens = lex(r#""hello world""#);
        assert_eq!(tokens[0].kind, SyntaxKind::StringLiteral);
    }

    #[test]
    fn lex_char() {
        let tokens = lex("'a'");
        assert_eq!(tokens[0].kind, SyntaxKind::CharLiteral);
    }

    #[test]
    fn lex_keywords() {
        assert_eq!(
            kinds("let in case of match where data"),
            vec![
                SyntaxKind::KwLet,
                SyntaxKind::Whitespace,
                SyntaxKind::KwIn,
                SyntaxKind::Whitespace,
                SyntaxKind::KwCase,
                SyntaxKind::Whitespace,
                SyntaxKind::KwOf,
                SyntaxKind::Whitespace,
                SyntaxKind::KwMatch,
                SyntaxKind::Whitespace,
                SyntaxKind::KwWhere,
                SyntaxKind::Whitespace,
                SyntaxKind::KwData,
                SyntaxKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_identifiers() {
        assert_eq!(
            kinds("foo Bar _x"),
            vec![
                SyntaxKind::Ident,
                SyntaxKind::Whitespace,
                SyntaxKind::UpperIdent,
                SyntaxKind::Whitespace,
                SyntaxKind::Ident,
                SyntaxKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_underscore_alone() {
        assert_eq!(
            kinds("_ x"),
            vec![
                SyntaxKind::Underscore,
                SyntaxKind::Whitespace,
                SyntaxKind::Ident,
                SyntaxKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_operators() {
        assert_eq!(
            kinds("-> => :: .. <= >= == /= && || <-"),
            vec![
                SyntaxKind::Arrow,
                SyntaxKind::Whitespace,
                SyntaxKind::FatArrow,
                SyntaxKind::Whitespace,
                SyntaxKind::ColonColon,
                SyntaxKind::Whitespace,
                SyntaxKind::DotDot,
                SyntaxKind::Whitespace,
                SyntaxKind::LessEqual,
                SyntaxKind::Whitespace,
                SyntaxKind::GreaterEqual,
                SyntaxKind::Whitespace,
                SyntaxKind::EqualEqual,
                SyntaxKind::Whitespace,
                SyntaxKind::NotEqual,
                SyntaxKind::Whitespace,
                SyntaxKind::AndAnd,
                SyntaxKind::Whitespace,
                SyntaxKind::OrOr,
                SyntaxKind::Whitespace,
                SyntaxKind::LeftArrow,
                SyntaxKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_single_char_punct() {
        assert_eq!(
            kinds("( ) [ ] { } , ; : . = + - * / % < > | ! $ ` \\ @"),
            vec![
                SyntaxKind::LParen,
                SyntaxKind::Whitespace,
                SyntaxKind::RParen,
                SyntaxKind::Whitespace,
                SyntaxKind::LBracket,
                SyntaxKind::Whitespace,
                SyntaxKind::RBracket,
                SyntaxKind::Whitespace,
                SyntaxKind::LBrace,
                SyntaxKind::Whitespace,
                SyntaxKind::RBrace,
                SyntaxKind::Whitespace,
                SyntaxKind::Comma,
                SyntaxKind::Whitespace,
                SyntaxKind::Semicolon,
                SyntaxKind::Whitespace,
                SyntaxKind::Colon,
                SyntaxKind::Whitespace,
                SyntaxKind::Dot,
                SyntaxKind::Whitespace,
                SyntaxKind::Equals,
                SyntaxKind::Whitespace,
                SyntaxKind::Plus,
                SyntaxKind::Whitespace,
                SyntaxKind::Minus,
                SyntaxKind::Whitespace,
                SyntaxKind::Star,
                SyntaxKind::Whitespace,
                SyntaxKind::Slash,
                SyntaxKind::Whitespace,
                SyntaxKind::Percent,
                SyntaxKind::Whitespace,
                SyntaxKind::Less,
                SyntaxKind::Whitespace,
                SyntaxKind::Greater,
                SyntaxKind::Whitespace,
                SyntaxKind::Pipe,
                SyntaxKind::Whitespace,
                SyntaxKind::Bang,
                SyntaxKind::Whitespace,
                SyntaxKind::Dollar,
                SyntaxKind::Whitespace,
                SyntaxKind::Backtick,
                SyntaxKind::Whitespace,
                SyntaxKind::Backslash,
                SyntaxKind::Whitespace,
                SyntaxKind::At,
                SyntaxKind::Eof,
            ]
        );
    }

    #[test]
    fn lex_primed_ident() {
        // x' is a valid Haskell-style identifier
        let tokens = lex("x'");
        assert_eq!(tokens[0].kind, SyntaxKind::Ident);
        assert_eq!(tokens[0].span.source_text("x'"), "x'");
    }

    #[test]
    fn lex_small_program() {
        let source = "add x y = x + y";
        let tokens = lex(source);
        let k: Vec<_> = tokens.iter().map(|t| t.kind).collect();
        assert_eq!(
            k,
            vec![
                SyntaxKind::Ident, // add
                SyntaxKind::Whitespace,
                SyntaxKind::Ident, // x
                SyntaxKind::Whitespace,
                SyntaxKind::Ident, // y
                SyntaxKind::Whitespace,
                SyntaxKind::Equals,
                SyntaxKind::Whitespace,
                SyntaxKind::Ident, // x
                SyntaxKind::Whitespace,
                SyntaxKind::Plus,
                SyntaxKind::Whitespace,
                SyntaxKind::Ident, // y
                SyntaxKind::Eof,
            ]
        );
    }
}
