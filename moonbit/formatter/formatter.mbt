///|
/// Token-stream formatter for fwgsl source.
///
/// The CST-based formatter is part of the longer-term roadmap. Until
/// the CST is wired up, this lexer-driven pass already covers the
/// useful day-to-day cases: it canonicalises whitespace, collapses
/// runs of blank lines, and trims trailing whitespace while leaving
/// every comment, literal, and identifier byte-for-byte intact.
///
/// The output preserves the original ordering of tokens. The only
/// edits are:
///
/// - Multiple consecutive horizontal spaces between two non-trivia
///   tokens are collapsed to a single space, except when the original
///   source contained no whitespace at all (we keep that
///   `name(...)` shape).
/// - Trailing horizontal whitespace at the end of a line is dropped.
/// - Runs of more than two consecutive `\n` characters are collapsed
///   to two — the common "one blank line between decls" shape.
/// - The output always ends with exactly one `\n` if the input had
///   any tokens at all.

///|
pub fn format(source : String) -> String {
  let tokens = @parser.lex(source)
  let buf : Array[String] = Array::new()
  let mut last_was_space = false
  let mut last_was_newline = false
  let mut consecutive_newlines = 0
  for tok in tokens {
    if tok.kind == @syntax.SyntaxKind::Eof {
      break
    }
    let text = tok.span.source_text(source)
    if tok.kind == @syntax.SyntaxKind::Whitespace {
      if not(last_was_space) && not(last_was_newline) {
        buf.push(" ")
        last_was_space = true
      }
      continue
    }
    if tok.kind == @syntax.SyntaxKind::Newline {
      if consecutive_newlines < 2 {
        // Drop any trailing space we may have emitted before this
        // newline — `foo \n` becomes `foo\n`.
        if last_was_space && buf.length() > 0 {
          let _ = buf.pop()
        }
        buf.push("\n")
        consecutive_newlines = consecutive_newlines + 1
      }
      last_was_space = false
      last_was_newline = true
      continue
    }
    buf.push(text)
    last_was_space = false
    last_was_newline = false
    consecutive_newlines = 0
  }
  let mut out = ""
  for s in buf {
    out = out + s
  }
  // Always end with a single trailing newline if there was any content.
  if out.length() == 0 {
    out
  } else if out.unsafe_charcode_at(out.length() - 1) != 0x0A {
    out + "\n"
  } else {
    // Trim any extra trailing newlines so we end on exactly one.
    let mut end = out.length()
    while end > 1 && out.unsafe_charcode_at(end - 1) == 0x0A &&
      out.unsafe_charcode_at(end - 2) == 0x0A {
      end = end - 1
    }
    out.substring(start=0, end=end)
  }
}
