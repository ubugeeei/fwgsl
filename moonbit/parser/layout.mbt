///|
/// Layout resolver for fwgsl.
///
/// Implements Haskell 2010-style indentation-sensitive layout rules by
/// inserting virtual `LayoutBraceOpen`, `LayoutSemicolon`, and
/// `LayoutBraceClose` tokens into the token stream.

///|
/// Insert virtual layout tokens into the token stream.
///
/// The layout algorithm is based on the Haskell 2010 report:
/// - After `where`, `let`, `of`, `do`, if the next non-trivia token is not
///   an explicit `{`, insert `LayoutBraceOpen` at the column of that token.
/// - Track an indentation stack. On each newline, compare the next token's
///   column to the top of the stack:
///   - Equal -> insert `LayoutSemicolon`
///   - Less  -> pop and insert `LayoutBraceClose`, repeat
/// - At EOF, close all open layout contexts.
pub fn resolve_layout(tokens : Array[Token], source : String) -> Array[Token] {
  let r = LayoutResolver::new(tokens, source)
  r.resolve()
  r.output
}

// ---------------------------------------------------------------------------
// Internal state
// ---------------------------------------------------------------------------

///|
/// Pre-computed line-start offsets for column calculation.
struct LineMap {
  line_starts : Array[Int]
}

///|
fn LineMap::build(source : String) -> LineMap {
  let starts : Array[Int] = [0]
  let len = source.length()
  let mut i = 0
  while i < len {
    if source.unsafe_charcode_at(i) == 0x0A {
      starts.push(i + 1)
    }
    i = i + 1
  }
  LineMap::{ line_starts: starts }
}

///|
/// 0-based column for a code-unit offset.
fn LineMap::column(self : LineMap, offset : UInt) -> UInt {
  let off = offset.reinterpret_as_int()
  // Binary search for the line containing this offset.
  let starts = self.line_starts
  let mut lo = 0
  let mut hi = starts.length()
  while lo < hi {
    let mid = (lo + hi) / 2
    if starts[mid] <= off {
      lo = mid + 1
    } else {
      hi = mid
    }
  }
  // lo - 1 is the line index whose start <= off.
  let line = lo - 1
  (off - starts[line]).reinterpret_as_uint()
}

///|
struct LayoutResolver {
  tokens : Array[Token]
  mut pos : Int
  output : Array[Token]
  indent_stack : Array[UInt]
  line_map : LineMap
  mut suppress_next_semi : Bool
}

///|
fn LayoutResolver::new(
  tokens : Array[Token],
  source : String
) -> LayoutResolver {
  LayoutResolver::{
    tokens,
    pos: 0,
    output: Array::new(),
    indent_stack: Array::new(),
    line_map: LineMap::build(source),
    suppress_next_semi: false,
  }
}

///|
fn LayoutResolver::at_end(self : LayoutResolver) -> Bool {
  self.pos >= self.tokens.length()
}

///|
fn LayoutResolver::peek_kind(self : LayoutResolver) -> @syntax.SyntaxKind {
  if self.at_end() {
    @syntax.SyntaxKind::Eof
  } else {
    self.tokens[self.pos].kind
  }
}

///|
fn LayoutResolver::next_non_trivia_index(
  self : LayoutResolver,
  from : Int
) -> Int? {
  let mut i = from
  let len = self.tokens.length()
  while i < len {
    if !self.tokens[i].kind.is_trivia() {
      return Some(i)
    }
    i = i + 1
  }
  None
}

///|
fn make_virtual_token(kind : @syntax.SyntaxKind, offset : UInt) -> Token {
  Token::new(kind, @span.Span::new(offset, offset))
}

///|
fn LayoutResolver::resolve(self : LayoutResolver) -> Unit {
  // Insert a top-level layout context if the file does not start with
  // `module` (which would trigger its own `where` layout). This mimics
  // Haskell 2010's implicit `module Main where {…}` wrapping.
  match self.next_non_trivia_index(0) {
    Some(first_idx) => {
      let first_kind = self.tokens[first_idx].kind
      if first_kind != @syntax.SyntaxKind::KwModule &&
        first_kind != @syntax.SyntaxKind::Eof {
        let col = self.line_map.column(self.tokens[first_idx].span.start)
        let offset = self.tokens[first_idx].span.start
        self.indent_stack.push(col)
        self.suppress_next_semi = false
        self.output.push(
          make_virtual_token(@syntax.SyntaxKind::LayoutBraceOpen, offset),
        )
      }
    }
    None => ()
  }
  while !self.at_end() {
    let kind = self.peek_kind()
    match kind {
      Newline => {
        self.output.push(self.tokens[self.pos])
        self.pos = self.pos + 1
        match self.next_non_trivia_index(self.pos) {
          Some(next_idx) => {
            let next_kind = self.tokens[next_idx].kind
            if next_kind != @syntax.SyntaxKind::Eof {
              let col = self.line_map.column(self.tokens[next_idx].span.start)
              let offset = self.tokens[next_idx].span.start
              let mut keep_going = true
              while keep_going && self.indent_stack.length() > 0 {
                let top = self.indent_stack[self.indent_stack.length() - 1]
                if col == top {
                  if self.suppress_next_semi {
                    self.suppress_next_semi = false
                  } else {
                    self.output.push(
                      make_virtual_token(@syntax.SyntaxKind::LayoutSemicolon, offset),
                    )
                  }
                  keep_going = false
                } else if col < top {
                  let _ = self.indent_stack.pop()
                  self.output.push(
                    make_virtual_token(@syntax.SyntaxKind::LayoutBraceClose, offset),
                  )
                } else {
                  keep_going = false
                }
              }
            }
          }
          None => ()
        }
      }
      KwWhere | KwLet | KwOf | KwDo => {
        self.output.push(self.tokens[self.pos])
        self.pos = self.pos + 1
        match self.next_non_trivia_index(self.pos) {
          Some(next_idx) => {
            let next_kind = self.tokens[next_idx].kind
            if next_kind != @syntax.SyntaxKind::LBrace {
              let col = self.line_map.column(self.tokens[next_idx].span.start)
              let offset = self.tokens[next_idx].span.start
              self.indent_stack.push(col)
              self.suppress_next_semi = true
              self.output.push(
                make_virtual_token(@syntax.SyntaxKind::LayoutBraceOpen, offset),
              )
            }
          }
          None => ()
        }
      }
      Eof => {
        let offset = self.tokens[self.pos].span.start
        while self.indent_stack.length() > 0 {
          let _ = self.indent_stack.pop()
          self.output.push(
            make_virtual_token(@syntax.SyntaxKind::LayoutBraceClose, offset),
          )
        }
        self.output.push(self.tokens[self.pos])
        self.pos = self.pos + 1
      }
      _ => {
        if !kind.is_trivia() && self.suppress_next_semi {
          self.suppress_next_semi = false
        }
        self.output.push(self.tokens[self.pos])
        self.pos = self.pos + 1
      }
    }
  }
}
