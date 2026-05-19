///|
/// Hand-written lexer for fwgsl (Oxc-style, not regex-based).
///
/// The Rust crate operates on `&[u8]` byte slices. MoonBit's `String` is
/// UTF-16, so this port works on code-unit offsets instead of byte
/// offsets. For ASCII source (which is everything fwgsl currently
/// accepts inside tokens) the two are identical, so spans remain
/// byte-compatible with the Rust version for ASCII-only programs.

///|
/// A single token produced by the lexer.
pub(all) struct Token {
  kind : @syntax.SyntaxKind
  span : @span.Span
} derive(Show, Eq)

///|
/// Create a new token.
pub fn Token::new(kind : @syntax.SyntaxKind, span : @span.Span) -> Token {
  Token::{ kind, span }
}

///|
/// Extract the source text for this token.
pub fn Token::text(self : Token, source : String) -> String {
  self.span.source_text(source)
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

///|
/// Lex the entire source into a token stream, ending with `Eof`.
pub fn lex(source : String) -> Array[Token] {
  let lexer = Lexer::new(source)
  lexer.lex_all()
  lexer.tokens
}

// ---------------------------------------------------------------------------
// Internal lexer state
// ---------------------------------------------------------------------------

///|
struct Lexer {
  source : String
  source_len : Int
  mut pos : Int
  tokens : Array[Token]
}

///|
fn Lexer::new(source : String) -> Lexer {
  Lexer::{
    source,
    source_len: source.length(),
    pos: 0,
    tokens: Array::new(),
  }
}

// -- Helpers ---------------------------------------------------------------

///|
fn Lexer::at_end(self : Lexer) -> Bool {
  self.pos >= self.source_len
}

///|
/// Peek at the code unit at the current position. Returns 0 at EOF.
fn Lexer::peek(self : Lexer) -> Int {
  if self.at_end() {
    0
  } else {
    self.source.unsafe_charcode_at(self.pos)
  }
}

///|
fn Lexer::peek_at(self : Lexer, offset : Int) -> Int {
  let idx = self.pos + offset
  if idx >= self.source_len {
    0
  } else {
    self.source.unsafe_charcode_at(idx)
  }
}

///|
fn Lexer::advance(self : Lexer) -> Int {
  let c = self.source.unsafe_charcode_at(self.pos)
  self.pos = self.pos + 1
  c
}

///|
fn Lexer::emit(self : Lexer, kind : @syntax.SyntaxKind, start : Int) -> Unit {
  let span = @span.Span::new(start.reinterpret_as_uint(), self.pos.reinterpret_as_uint())
  self.tokens.push(Token::new(kind, span))
}

// -- Top-level loop --------------------------------------------------------

///|
fn Lexer::lex_all(self : Lexer) -> Unit {
  while !self.at_end() {
    self.lex_token()
  }
  self.emit(@syntax.SyntaxKind::Eof, self.pos)
}

///|
fn Lexer::lex_token(self : Lexer) -> Unit {
  let start = self.pos
  let b = self.peek()
  match b {
    // Newlines
    0x0A => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Newline, start)
    }
    0x0D => {
      let _ = self.advance()
      if self.peek() == 0x0A {
        let _ = self.advance()
      }
      self.emit(@syntax.SyntaxKind::Newline, start)
    }
    // Horizontal whitespace: space or tab
    0x20 | 0x09 => {
      let _ = self.advance()
      while !self.at_end() {
        let c = self.peek()
        if c == 0x20 || c == 0x09 {
          let _ = self.advance()
        } else {
          break
        }
      }
      self.emit(@syntax.SyntaxKind::Whitespace, start)
    }
    // '-' starts comments, arrows, or minus
    0x2D => {
      if self.peek_at(1) == 0x2D {
        self.lex_line_comment(start)
      } else if self.peek_at(1) == 0x3E {
        let _ = self.advance()
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::Arrow, start)
      } else {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::Minus, start)
      }
    }
    // '{' starts block comments or is a brace
    0x7B => {
      if self.peek_at(1) == 0x2D {
        self.lex_block_comment(start)
      } else {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::LBrace, start)
      }
    }
    // '"' starts a string literal
    0x22 => self.lex_string(start)
    // '\'' starts a char literal
    0x27 => self.lex_char(start)
    // '=' '<' '>' '/' '&' '|' multi-char operators
    0x3D => {
      let _ = self.advance()
      if self.peek() == 0x3E {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::FatArrow, start)
      } else if self.peek() == 0x3D {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::EqualEqual, start)
      } else {
        self.emit(@syntax.SyntaxKind::Equals, start)
      }
    }
    0x3A => {
      let _ = self.advance()
      if self.peek() == 0x3A {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::ColonColon, start)
      } else {
        self.emit(@syntax.SyntaxKind::Colon, start)
      }
    }
    0x2E => {
      let _ = self.advance()
      if self.peek() == 0x2E {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::DotDot, start)
      } else {
        self.emit(@syntax.SyntaxKind::Dot, start)
      }
    }
    0x3C => {
      let _ = self.advance()
      if self.peek() == 0x3D {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::LessEqual, start)
      } else if self.peek() == 0x2D {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::LeftArrow, start)
      } else {
        self.emit(@syntax.SyntaxKind::Less, start)
      }
    }
    0x3E => {
      let _ = self.advance()
      if self.peek() == 0x3D {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::GreaterEqual, start)
      } else {
        self.emit(@syntax.SyntaxKind::Greater, start)
      }
    }
    0x2F => {
      let _ = self.advance()
      if self.peek() == 0x3D {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::NotEqual, start)
      } else {
        self.emit(@syntax.SyntaxKind::Slash, start)
      }
    }
    0x26 => {
      let _ = self.advance()
      if self.peek() == 0x26 {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::AndAnd, start)
      } else {
        // Single `&` is not a valid fwgsl token, emit error
        self.emit(@syntax.SyntaxKind::Error, start)
      }
    }
    0x7C => {
      let _ = self.advance()
      if self.peek() == 0x7C {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::OrOr, start)
      } else if self.peek() == 0x3E {
        let _ = self.advance()
        self.emit(@syntax.SyntaxKind::PipeForward, start)
      } else {
        self.emit(@syntax.SyntaxKind::Pipe, start)
      }
    }
    // Single-char punctuation / operators
    0x28 => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::LParen, start)
    }
    0x29 => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::RParen, start)
    }
    0x5B => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::LBracket, start)
    }
    0x5D => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::RBracket, start)
    }
    0x7D => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::RBrace, start)
    }
    0x2C => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Comma, start)
    }
    0x3B => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Semicolon, start)
    }
    0x5C => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Backslash, start)
    }
    0x40 => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::At, start)
    }
    0x2B => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Plus, start)
    }
    0x2A => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Star, start)
    }
    0x25 => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Percent, start)
    }
    0x21 => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Bang, start)
    }
    0x24 => {
      // '$' followed by alpha/underscore is a builtin identifier ($vec4, $sin).
      let _ = self.advance()
      let next = self.peek()
      if is_ascii_alpha(next) || next == 0x5F {
        while !self.at_end() {
          let c = self.peek()
          if is_ascii_alnum(c) || c == 0x5F {
            let _ = self.advance()
          } else {
            break
          }
        }
        while self.peek() == 0x27 {
          let _ = self.advance()
        }
        self.emit(@syntax.SyntaxKind::Ident, start)
      } else {
        self.emit(@syntax.SyntaxKind::Dollar, start)
      }
    }
    0x60 => {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::Backtick, start)
    }
    _ => {
      if is_ascii_digit(b) {
        self.lex_number(start)
      } else if is_ascii_lower(b) {
        self.lex_ident(start)
      } else if b == 0x5F {
        if is_ident_continue(self.peek_at(1)) {
          self.lex_ident(start)
        } else {
          let _ = self.advance()
          self.emit(@syntax.SyntaxKind::Underscore, start)
        }
      } else if is_ascii_upper(b) {
        self.lex_upper_ident(start)
      } else {
        self.advance_utf8_char()
        self.emit(@syntax.SyntaxKind::Error, start)
      }
    }
  }
}

// -- Sub-lexers ------------------------------------------------------------

///|
fn Lexer::lex_line_comment(self : Lexer, start : Int) -> Unit {
  let _ = self.advance() // first '-'
  let _ = self.advance() // second '-'
  while !self.at_end() {
    let c = self.peek()
    if c == 0x0A || c == 0x0D {
      break
    }
    let _ = self.advance()
  }
  self.emit(@syntax.SyntaxKind::LineComment, start)
}

///|
fn Lexer::lex_block_comment(self : Lexer, start : Int) -> Unit {
  let _ = self.advance() // '{'
  let _ = self.advance() // '-'
  let mut depth = 1
  while !self.at_end() && depth > 0 {
    if self.peek() == 0x7B && self.peek_at(1) == 0x2D {
      let _ = self.advance()
      let _ = self.advance()
      depth = depth + 1
    } else if self.peek() == 0x2D && self.peek_at(1) == 0x7D {
      let _ = self.advance()
      let _ = self.advance()
      depth = depth - 1
    } else {
      let _ = self.advance()
    }
  }
  self.emit(@syntax.SyntaxKind::BlockComment, start)
}

///|
fn Lexer::lex_string(self : Lexer, start : Int) -> Unit {
  let _ = self.advance() // opening '"'
  while !self.at_end() {
    let c = self.peek()
    if c == 0x22 {
      let _ = self.advance()
      self.emit(@syntax.SyntaxKind::StringLiteral, start)
      return
    } else if c == 0x5C {
      let _ = self.advance()
      if !self.at_end() {
        let _ = self.advance()
      }
    } else if c == 0x0A || c == 0x0D {
      break
    } else {
      let _ = self.advance()
    }
  }
  self.emit(@syntax.SyntaxKind::Error, start)
}

///|
fn Lexer::lex_char(self : Lexer, start : Int) -> Unit {
  let _ = self.advance() // opening '\''
  if self.at_end() {
    self.emit(@syntax.SyntaxKind::Error, start)
    return
  }
  if self.peek() == 0x5C {
    let _ = self.advance() // '\'
    if !self.at_end() {
      let _ = self.advance()
    }
  } else if self.peek() == 0x27 {
    let _ = self.advance()
    self.emit(@syntax.SyntaxKind::Error, start)
    return
  } else {
    self.advance_utf8_char()
  }
  if !self.at_end() && self.peek() == 0x27 {
    let _ = self.advance()
    self.emit(@syntax.SyntaxKind::CharLiteral, start)
  } else {
    self.emit(@syntax.SyntaxKind::Error, start)
  }
}

///|
fn Lexer::lex_number(self : Lexer, start : Int) -> Unit {
  let first = self.advance()
  if first == 0x30 && !self.at_end() {
    let next = self.peek()
    if next == 0x78 || next == 0x58 {
      let _ = self.advance()
      while !self.at_end() && is_hex_digit(self.peek()) {
        let _ = self.advance()
      }
      self.emit(@syntax.SyntaxKind::IntLiteral, start)
      return
    }
    if next == 0x6F || next == 0x4F {
      let _ = self.advance()
      while !self.at_end() && is_oct_digit(self.peek()) {
        let _ = self.advance()
      }
      self.emit(@syntax.SyntaxKind::IntLiteral, start)
      return
    }
    if next == 0x62 || next == 0x42 {
      let _ = self.advance()
      while !self.at_end() && is_bin_digit(self.peek()) {
        let _ = self.advance()
      }
      self.emit(@syntax.SyntaxKind::IntLiteral, start)
      return
    }
  }
  while !self.at_end() && is_ascii_digit(self.peek()) {
    let _ = self.advance()
  }
  if !self.at_end() &&
    self.peek() == 0x2E &&
    is_ascii_digit(self.peek_at(1)) {
    let _ = self.advance() // '.'
    while !self.at_end() && is_ascii_digit(self.peek()) {
      let _ = self.advance()
    }
    if !self.at_end() &&
      (self.peek() == 0x65 || self.peek() == 0x45) {
      let _ = self.advance()
      if !self.at_end() &&
        (self.peek() == 0x2B || self.peek() == 0x2D) {
        let _ = self.advance()
      }
      while !self.at_end() && is_ascii_digit(self.peek()) {
        let _ = self.advance()
      }
    }
    self.emit(@syntax.SyntaxKind::FloatLiteral, start)
  } else {
    self.emit(@syntax.SyntaxKind::IntLiteral, start)
  }
}

///|
fn Lexer::lex_ident(self : Lexer, start : Int) -> Unit {
  let _ = self.advance() // first char already validated
  while !self.at_end() && is_ident_continue(self.peek()) {
    let _ = self.advance()
  }
  let text = self.source.substring(start=start, end=self.pos)
  match @syntax.keyword_from_str(text) {
    Some(kw) => self.emit(kw, start)
    None => self.emit(@syntax.SyntaxKind::Ident, start)
  }
}

///|
fn Lexer::lex_upper_ident(self : Lexer, start : Int) -> Unit {
  let _ = self.advance() // first char already validated
  while !self.at_end() && is_ident_continue(self.peek()) {
    let _ = self.advance()
  }
  self.emit(@syntax.SyntaxKind::UpperIdent, start)
}

///|
/// Advance past a single UTF-16 surrogate pair or BMP code unit.
fn Lexer::advance_utf8_char(self : Lexer) -> Unit {
  if self.at_end() {
    return
  }
  let c = self.source.unsafe_charcode_at(self.pos)
  self.pos = self.pos + 1
  // High surrogate: consume the trailing low surrogate too.
  if c >= 0xD800 && c <= 0xDBFF && !self.at_end() {
    self.pos = self.pos + 1
  }
}

// ---------------------------------------------------------------------------
// Character classification helpers (ASCII)
// ---------------------------------------------------------------------------

///|
fn is_ascii_digit(c : Int) -> Bool {
  c >= 0x30 && c <= 0x39
}

///|
fn is_ascii_lower(c : Int) -> Bool {
  c >= 0x61 && c <= 0x7A
}

///|
fn is_ascii_upper(c : Int) -> Bool {
  c >= 0x41 && c <= 0x5A
}

///|
fn is_ascii_alpha(c : Int) -> Bool {
  is_ascii_lower(c) || is_ascii_upper(c)
}

///|
fn is_ascii_alnum(c : Int) -> Bool {
  is_ascii_digit(c) || is_ascii_alpha(c)
}

///|
fn is_hex_digit(c : Int) -> Bool {
  is_ascii_digit(c) ||
  (c >= 0x61 && c <= 0x66) ||
  (c >= 0x41 && c <= 0x46)
}

///|
fn is_oct_digit(c : Int) -> Bool {
  c >= 0x30 && c <= 0x37
}

///|
fn is_bin_digit(c : Int) -> Bool {
  c == 0x30 || c == 0x31
}

///|
fn is_ident_continue(c : Int) -> Bool {
  is_ascii_alnum(c) || c == 0x5F || c == 0x27
}
