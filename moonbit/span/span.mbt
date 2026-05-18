///|
/// Span and atom types for fwgsl source tracking.
///
/// Provides lightweight source location tracking (`Span`) and a thin
/// wrapper for interned-style string atoms (`Atom`) used throughout the
/// compiler pipeline.

///|
/// A span representing a byte range in source code.
///
/// Both `start` and `end` are byte offsets. The range is `[start, end)`.
pub(all) struct Span {
  start : UInt
  end : UInt
} derive(Eq, Hash, Show)

///|
/// A zero-length span at position 0, used as a default/placeholder.
pub let span_zero : Span = Span::{ start: 0, end: 0 }

///|
/// Create a new span from byte offsets.
pub fn Span::new(start : UInt, end : UInt) -> Span {
  Span::{ start, end }
}

///|
/// Merge two spans, producing a span that covers both.
///
/// The resulting span starts at the minimum start and ends at the maximum end.
pub fn Span::merge(self : Span, other : Span) -> Span {
  let start = if self.start < other.start { self.start } else { other.start }
  let end = if self.end > other.end { self.end } else { other.end }
  Span::{ start, end }
}

///|
/// Check whether this span contains the given byte offset.
pub fn Span::contains(self : Span, offset : UInt) -> Bool {
  self.start <= offset && offset < self.end
}

///|
/// Extract the source text covered by this span.
pub fn Span::source_text(self : Span, source : String) -> String {
  source.substring(start=self.start.reinterpret_as_int(), end=self.end.reinterpret_as_int())
}

///|
/// An interned string atom.
///
/// MoonBit does not need the inline-small-string optimisation from the Rust
/// `CompactString`; under wasm-gc the underlying `String` is already a managed
/// reference. `Atom` exists as a distinct type to keep the API parity with the
/// Rust crate and to leave room for a future deduplicating intern table.
pub(all) struct Atom {
  value : String
} derive(Eq, Hash, Show, Compare)

///|
/// Create a new atom from a string slice.
pub fn Atom::new(s : String) -> Atom {
  Atom::{ value: s }
}

///|
/// Returns the atom as a string slice.
pub fn Atom::as_str(self : Atom) -> String {
  self.value
}

///|
/// A node paired with its source span.
pub(all) struct Spanned[T] {
  node : T
  span : Span
}

///|
/// Create a new spanned node.
pub fn Spanned::new[T](node : T, span : Span) -> Spanned[T] {
  Spanned::{ node, span }
}
