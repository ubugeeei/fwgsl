//! Token-stream based formatter for fwgsl.
//!
//! Operates on the raw lexed tokens (not the layout-resolved stream), so that
//! comments and original structure are preserved. Re-emits tokens with
//! canonical whitespace and indentation.

use fwgsl_parser::lex;
use fwgsl_parser::lexer::Token;
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
    align_record_fields(&mut engine.output);
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
    /// Nesting depth of type-parameter angle brackets (`Vec<..>`).
    type_angle_depth: usize,
    /// The last mid-line whitespace that was skipped (for alignment preservation).
    last_skipped_ws: Option<&'a str>,
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
            type_angle_depth: 0,
            last_skipped_ws: None,
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
                    self.last_skipped_ws = None;
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
                        // Store the original whitespace for alignment preservation.
                        let ws_text = self.text(tok);
                        self.last_skipped_ws = Some(ws_text);
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
                    self.last_skipped_ws = None;

                    // Track type-parameter angle brackets (after spacing decision)
                    if tok.kind == SyntaxKind::Less
                        && self.last_emitted_kind() == Some(SyntaxKind::UpperIdent)
                    {
                        self.type_angle_depth += 1;
                    } else if tok.kind == SyntaxKind::Greater && self.type_angle_depth > 0 {
                        self.type_angle_depth -= 1;
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
        let prev = self.last_emitted_kind();

        // No space before certain punctuation
        if matches!(
            kind,
            SyntaxKind::RParen | SyntaxKind::RBracket | SyntaxKind::Comma | SyntaxKind::Semicolon
        ) {
            return;
        }

        // No space after opening brackets (already handled since we're looking at 'next')
        if prev == Some(SyntaxKind::LParen) || prev == Some(SyntaxKind::LBracket) {
            return;
        }

        // No space between `@` and attribute name
        if prev == Some(SyntaxKind::At) {
            return;
        }

        // No space before/after `.` (field access, composition)
        if kind == SyntaxKind::Dot || prev == Some(SyntaxKind::Dot) {
            return;
        }

        // No space after `\` (lambda)
        if prev == Some(SyntaxKind::Backslash) {
            return;
        }

        // No space between attribute/function name and `(` — e.g. `@builtin(...)`,
        // `@workgroup_size(...)`, `@location(0)`, `@interpolate(flat)`
        if kind == SyntaxKind::LParen
            && matches!(prev, Some(SyntaxKind::Ident | SyntaxKind::UpperIdent))
            && self.is_prev_attribute_name()
        {
            return;
        }

        // No space between `storage` keyword and `(` for `storage(read_write)`
        if kind == SyntaxKind::LParen && prev == Some(SyntaxKind::KwStorage) {
            return;
        }

        // No space before `[` when preceded by an identifier or `)` (array indexing)
        if kind == SyntaxKind::LBracket
            && matches!(
                prev,
                Some(
                    SyntaxKind::Ident
                        | SyntaxKind::UpperIdent
                        | SyntaxKind::RParen
                        | SyntaxKind::RBracket
                )
            )
        {
            return;
        }

        // No space after unary `-` or `!` when they appear as prefix operators.
        // Detect prefix context: the `-` or `!` follows `(`, `=`, `let`, `in`,
        // `then`, `else`, `->`, a comma, or is at statement start.
        if (prev == Some(SyntaxKind::Minus) || prev == Some(SyntaxKind::Bang))
            && self.is_prev_unary_context()
        {
            return;
        }

        // Type parameter angle brackets: no space around `<`, `>`, or after `,`
        // when inside `Type<...>`.
        if kind == SyntaxKind::Less && prev == Some(SyntaxKind::UpperIdent) {
            return; // no space before `<` in `Vec<`
        }
        if kind == SyntaxKind::Greater && self.type_angle_depth > 0 {
            return; // no space before `>` in `...>`
        }
        if prev == Some(SyntaxKind::Less) && self.type_angle_depth > 0 {
            return; // no space after `<` in `<4, ...`
        }

        // Preserve original multi-space padding before alignment-sensitive tokens.
        // This keeps author-intentional column alignment in let/where `=`, record `:`,
        // match `->`, const `:`, and binding `:` / `@group`.
        if let Some(ws) = self.last_skipped_ws {
            if ws.len() > 1 {
                let is_alignment_target = match kind {
                    // `=` alignment only preserved on indented lines (let/where bindings,
                    // record construction fields) — not top-level function defs.
                    SyntaxKind::Equals => self.line_is_indented(),
                    // `:`, `->`, `@` alignment preserved everywhere (const, extern, match arms).
                    SyntaxKind::Colon | SyntaxKind::Arrow | SyntaxKind::At => true,
                    // Ident after `)` — preserve attribute-to-field-name padding in records
                    SyntaxKind::Ident | SyntaxKind::UpperIdent
                        if prev == Some(SyntaxKind::RParen)
                            && self.line_has_attribute() =>
                    {
                        true
                    }
                    _ => false,
                };
                if is_alignment_target {
                    self.emit_str(ws);
                    return;
                }
            }
        }

        // Single space for everything else
        self.ensure_single_space();
    }

    /// Check if the previous token is an attribute name (follows `@`).
    fn is_prev_attribute_name(&self) -> bool {
        if self.pos < 2 {
            return false;
        }
        // Walk back from current pos to find the identifier, then check if `@` precedes it
        for i in (0..self.pos).rev() {
            let k = self.tokens[i].kind;
            if k.is_trivia() {
                continue;
            }
            if k == SyntaxKind::Ident || k == SyntaxKind::UpperIdent {
                // Now check if the token before this ident is `@`
                for j in (0..i).rev() {
                    let k2 = self.tokens[j].kind;
                    if k2.is_trivia() {
                        continue;
                    }
                    return k2 == SyntaxKind::At;
                }
                return false;
            }
            return false;
        }
        false
    }

    /// Check if the previous `-` or `!` is in a unary (prefix) context.
    fn is_prev_unary_context(&self) -> bool {
        if self.pos < 2 {
            return true; // at the start, it must be unary
        }
        // Find the token before the `-` or `!`
        let op_idx = self.pos - 1; // the `-` or `!` token
        for i in (0..op_idx).rev() {
            let k = self.tokens[i].kind;
            if k.is_trivia() {
                continue;
            }
            // Prefix context: after `(`, `[`, `=`, `,`, `->`, `||`, `&&`, keywords, operators
            return matches!(
                k,
                SyntaxKind::LParen
                    | SyntaxKind::LBracket
                    | SyntaxKind::Equals
                    | SyntaxKind::Comma
                    | SyntaxKind::Arrow
                    | SyntaxKind::Pipe
                    | SyntaxKind::OrOr
                    | SyntaxKind::AndAnd
                    | SyntaxKind::KwLet
                    | SyntaxKind::KwIn
                    | SyntaxKind::KwThen
                    | SyntaxKind::KwElse
                    | SyntaxKind::KwIf
                    | SyntaxKind::KwMatch
                    | SyntaxKind::KwCase
                    | SyntaxKind::KwOf
            );
        }
        true
    }

    /// Look back at the last non-whitespace character to determine the previous token kind.
    fn last_emitted_kind(&self) -> Option<SyntaxKind> {
        // Walk backwards from the current position to find the last emitted real token.
        if self.pos == 0 {
            return None;
        }
        for i in (0..self.pos).rev() {
            let k = self.tokens[i].kind;
            if !k.is_trivia() {
                return Some(k);
            }
        }
        None
    }

    /// Check if the current output line contains an `@` (attribute context).
    fn line_has_attribute(&self) -> bool {
        let line_start = self.output.rfind('\n').map_or(0, |i| i + 1);
        self.output[line_start..].contains('@')
    }

    /// Check if the current output line is indented (starts with whitespace).
    fn line_is_indented(&self) -> bool {
        if let Some(nl) = self.output.rfind('\n') {
            let line_start = nl + 1;
            self.output[line_start..]
                .bytes()
                .next()
                .map_or(false, |b| b == b' ' || b == b'\t')
        } else {
            // No newline yet — check from start
            self.output
                .bytes()
                .next()
                .map_or(false, |b| b == b' ' || b == b'\t')
        }
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

// ---------------------------------------------------------------------------
// Post-processing: align `:` in record field blocks
// ---------------------------------------------------------------------------

/// Align colons in consecutive record field lines that share the same indentation.
///
/// A "field line" matches the pattern: `<indent><name> : <type>...`
/// Groups of consecutive field lines with identical indent get their `:`
/// aligned to the longest name in the group.
fn align_record_fields(output: &mut String) {
    let lines: Vec<&str> = output.split('\n').collect();
    let mut result = Vec::with_capacity(lines.len());
    let mut i = 0;

    while i < lines.len() {
        // Try to start a group of field lines
        if let Some((indent, _name_end)) = parse_field_line(lines[i]) {
            let mut group_indices = vec![i];
            i += 1;
            // Collect consecutive field lines with the same indent
            while i < lines.len() {
                if let Some((ind2, _)) = parse_field_line(lines[i]) {
                    if ind2 == indent {
                        group_indices.push(i);
                        i += 1;
                        continue;
                    }
                }
                break;
            }
            if group_indices.len() > 1 {
                // Find max name_end column in the group
                let max_name_end = group_indices
                    .iter()
                    .map(|&idx| parse_field_line(lines[idx]).unwrap().1)
                    .max()
                    .unwrap();
                // Rewrite each line with aligned colon
                for &idx in &group_indices {
                    let (_, name_end) = parse_field_line(lines[idx]).unwrap();
                    let line = lines[idx];
                    // Everything up to end of field name
                    let before = &line[..name_end];
                    // Find `: <rest>` after the name
                    let after_name = &line[name_end..];
                    let colon_rel = after_name.find(':').unwrap();
                    let from_colon = &after_name[colon_rel..]; // ": <type>,..."
                    let padding = max_name_end - name_end + 1; // +1 for the space before `:`
                    let mut aligned = String::with_capacity(line.len() + padding);
                    aligned.push_str(before);
                    for _ in 0..padding {
                        aligned.push(' ');
                    }
                    aligned.push_str(from_colon);
                    result.push(aligned);
                }
            } else {
                // Single field line — no alignment needed
                for &idx in &group_indices {
                    result.push(lines[idx].to_string());
                }
            }
        } else {
            result.push(lines[i].to_string());
            i += 1;
        }
    }

    *output = result.join("\n");
}

/// Parse a line as a record field declaration.
/// Returns `(indent_len, name_end_col)` where `name_end_col` is the column
/// (from line start) where the field name ends. The caller pads between
/// `name_end_col` and `:` to align colons across a group.
/// Matches: `<indent>[<@attr(...)> ]<ident> : ...`
fn parse_field_line(line: &str) -> Option<(usize, usize)> {
    let bytes = line.as_bytes();
    // Measure leading whitespace (must have some indent for a record field)
    let indent = bytes
        .iter()
        .take_while(|&&b| b == b' ' || b == b'\t')
        .count();
    if indent == 0 || indent >= bytes.len() {
        return None;
    }
    // Work with byte offset from line start
    let mut pos = indent;
    // Skip optional attributes like @builtin(...) @location(0) @interpolate(flat)
    while bytes.get(pos) == Some(&b'@') {
        pos += 1;
        // Skip ident
        let ident_start = pos;
        while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
            pos += 1;
        }
        if pos == ident_start {
            return None;
        }
        // Optionally skip (...)
        if pos < bytes.len() && bytes[pos] == b'(' {
            let close = line[pos..].find(')')?;
            pos += close + 1;
        }
        // Skip whitespace after attribute
        while pos < bytes.len() && (bytes[pos] == b' ' || bytes[pos] == b'\t') {
            pos += 1;
        }
    }
    // Must start with an identifier character
    if pos >= bytes.len() || !(bytes[pos].is_ascii_alphabetic() || bytes[pos] == b'_') {
        return None;
    }
    // Measure name
    let name_start = pos;
    while pos < bytes.len() && (bytes[pos].is_ascii_alphanumeric() || bytes[pos] == b'_') {
        pos += 1;
    }
    if pos == name_start {
        return None;
    }
    let name_end = pos;
    // Must be followed by whitespace then `: `
    if pos >= bytes.len() || bytes[pos] != b' ' {
        return None;
    }
    // Skip whitespace to find `:`
    while pos < bytes.len() && bytes[pos] == b' ' {
        pos += 1;
    }
    if pos >= bytes.len() || bytes[pos] != b':' {
        return None;
    }
    // `:` must be followed by ` `
    if pos + 1 >= bytes.len() || bytes[pos + 1] != b' ' {
        return None;
    }
    Some((indent, name_end))
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
    fn format_field_access_in_expr() {
        let source = "f p = p.x + p.y\n";
        let result = format_default(source);
        assert_eq!(result, "f p = p.x + p.y\n");
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
        assert_eq!(result, "p = Particle { x = 1.0, y = 2.0 }\n");
    }

    #[test]
    fn format_hello_example() {
        let source = "-- Minimal end-to-end example.\n-- Demonstrates arithmetic, function calls, and let-bindings.\n\ndouble x = x * 2\n\nmain x =\n  let y = double x\n  in y + 1\n";
        let result = format_default(source);
        assert_eq!(result, source);
    }

    #[test]
    fn format_aligns_record_field_colons() {
        let source =
            "data Particle = Particle {\n  x : F32,\n  y : F32,\n  vx : F32,\n  vy : F32,\n}\n";
        let expected =
            "data Particle = Particle {\n  x  : F32,\n  y  : F32,\n  vx : F32,\n  vy : F32,\n}\n";
        let result = format_default(source);
        assert_eq!(result, expected);
    }

    #[test]
    fn format_field_alignment_idempotent() {
        let source =
            "data Particle = Particle {\n  x  : F32,\n  y  : F32,\n  vx : F32,\n  vy : F32,\n}\n";
        let result = format_default(source);
        let result2 = format_default(&result);
        assert_eq!(result, result2, "field alignment is not idempotent");
    }

    #[test]
    fn format_record_update_idempotent() {
        let source = "nudgeX dx p = p { x = p.x + dx }\n";
        let result = format_default(source);
        assert_eq!(result, source);
    }

    #[test]
    fn format_type_params_no_spaces() {
        let source = "f : Vec < 4 , F32 > -> F32\n";
        let result = format_default(source);
        assert_eq!(result, "f : Vec<4, F32> -> F32\n");
    }

    #[test]
    fn format_nested_type_params() {
        let source = "f : Array < Vec < 3 , F32 > , 64 > -> F32\n";
        let result = format_default(source);
        assert_eq!(result, "f : Array<Vec<3, F32>, 64> -> F32\n");
    }

    #[test]
    fn format_type_params_idempotent() {
        let source = "f : Vec<4, F32> -> F32\n";
        let result = format_default(source);
        let result2 = format_default(&result);
        assert_eq!(result, result2, "type params formatting is not idempotent");
    }

    #[test]
    fn format_attribute_args_no_space() {
        let source = "@builtin(position) foo : Vec<4, F32>\n";
        let result = format_default(source);
        assert_eq!(result, "@builtin(position) foo : Vec<4, F32>\n");
    }

    #[test]
    fn format_workgroup_size_no_space() {
        let source = "@compute @workgroup_size(64, 1, 1)\n";
        let result = format_default(source);
        assert_eq!(result, "@compute @workgroup_size(64, 1, 1)\n");
    }

    #[test]
    fn format_array_index_no_space() {
        let source = "f x = buf[idx]\n";
        let result = format_default(source);
        assert_eq!(result, "f x = buf[idx]\n");
    }

    #[test]
    fn format_unary_negation_no_space() {
        let source = "f x = (-x)\n";
        let result = format_default(source);
        assert_eq!(result, "f x = (-x)\n");
    }

    #[test]
    fn format_unary_not_no_space() {
        let source = "f x = !x\n";
        let result = format_default(source);
        assert_eq!(result, "f x = !x\n");
    }

    #[test]
    fn format_binary_minus_has_space() {
        let source = "f x = x - 1\n";
        let result = format_default(source);
        assert_eq!(result, "f x = x - 1\n");
    }

    #[test]
    fn format_lambda_no_space_after_backslash() {
        let source = "f = (\\x -> x + 1)\n";
        let result = format_default(source);
        assert_eq!(result, "f = (\\x -> x + 1)\n");
    }

    #[test]
    fn format_negative_literal() {
        let source = "f = Fp64 (-a.high) (-a.low)\n";
        let result = format_default(source);
        assert_eq!(result, "f = Fp64 (-a.high) (-a.low)\n");
    }

    #[test]
    fn format_match_negation() {
        let source = "  | 4 | 8 -> -1.0\n";
        let result = format_default(source);
        assert_eq!(result, "  | 4 | 8 -> -1.0\n");
    }

    #[test]
    fn format_multiple_attributes() {
        let source = "  @location(4) @interpolate(flat) cap_type : U32\n";
        let result = format_default(source);
        assert_eq!(result, "  @location(4) @interpolate(flat) cap_type : U32\n");
    }

    #[test]
    fn format_preserves_let_binding_alignment() {
        let source = "  let x    = 1\n      yLong = 2\n";
        let result = format_default(source);
        assert_eq!(result, "  let x    = 1\n      yLong = 2\n");
    }

    #[test]
    fn format_preserves_const_colon_alignment() {
        let source = "const FOO      : I32 = 1\nconst BAR_LONG : I32 = 2\n";
        let result = format_default(source);
        assert_eq!(result, "const FOO      : I32 = 1\nconst BAR_LONG : I32 = 2\n");
    }

    #[test]
    fn format_preserves_match_arrow_alignment() {
        let source = "  | Red   -> 1\n  | Green -> 2\n";
        let result = format_default(source);
        assert_eq!(result, "  | Red   -> 1\n  | Green -> 2\n");
    }

    #[test]
    fn format_preserves_binding_decl_alignment() {
        let source = "@group(0) @binding(0) uniform frame      : FrameData\n@group(0) @binding(1) uniform capFlags   : CapFlags\n@group(1) @binding(0) storage prims      : Array<Prim>\n";
        let result = format_default(source);
        assert_eq!(result, "@group(0) @binding(0) uniform frame      : FrameData\n@group(0) @binding(1) uniform capFlags   : CapFlags\n@group(1) @binding(0) storage prims      : Array<Prim>\n");
    }

    #[test]
    fn format_preserves_record_field_attr_padding() {
        let source = "  @builtin(position)              clip_pos : Vec<4, F32>,\n  @location(0)                    dist     : F32,\n";
        let result = format_default(source);
        assert_eq!(result, "  @builtin(position)              clip_pos : Vec<4, F32>,\n  @location(0)                    dist     : F32,\n");
    }

    #[test]
    fn format_normalizes_toplevel_equals() {
        // Top-level function definitions should NOT preserve alignment padding
        let source = "f   x   =   x + 1\n";
        let result = format_default(source);
        assert_eq!(result, "f x = x + 1\n");
    }

    #[test]
    fn format_unary_not_after_or() {
        let source = "  x = !a || !b\n";
        let result = format_default(source);
        assert_eq!(result, "  x = !a || !b\n");
    }

    #[test]
    fn format_alignment_idempotent_let_block() {
        let source = "  let x    = 1\n      yLong = 2\n";
        let first = format_default(source);
        let second = format_default(&first);
        assert_eq!(first, second, "let binding alignment is not idempotent");
    }
}
