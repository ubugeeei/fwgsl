//! Span and atom types for fwgsl source tracking.
//!
//! Provides lightweight source location tracking (`Span`) and
//! interned string atoms (`Atom`) used throughout the compiler pipeline.

use std::fmt;

use compact_str::CompactString;

/// A span representing a byte range in source code.
///
/// Both `start` and `end` are byte offsets. The range is `[start, end)`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Default)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

/// A zero-length span at position 0, used as a default/placeholder.
pub const SPAN_ZERO: Span = Span { start: 0, end: 0 };

impl Span {
    /// Create a new span from byte offsets.
    #[inline]
    pub fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    /// Merge two spans, producing a span that covers both.
    ///
    /// The resulting span starts at the minimum start and ends at the maximum end.
    #[inline]
    pub fn merge(self, other: Span) -> Span {
        Span {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Check whether this span contains the given byte offset.
    #[inline]
    pub fn contains(self, offset: u32) -> bool {
        self.start <= offset && offset < self.end
    }

    /// Extract the source text covered by this span.
    #[inline]
    pub fn source_text<'a>(&self, source: &'a str) -> &'a str {
        &source[self.start as usize..self.end as usize]
    }
}

/// An interned string atom backed by `CompactString`.
///
/// Small strings (up to 24 bytes on 64-bit) are stored inline
/// without heap allocation.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Atom(CompactString);

impl Atom {
    /// Create a new atom from a string slice.
    #[inline]
    pub fn new(s: &str) -> Self {
        Self(CompactString::new(s))
    }

    /// Returns the atom as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for Atom {
    #[inline]
    fn from(s: &str) -> Self {
        Self(CompactString::new(s))
    }
}

impl From<String> for Atom {
    #[inline]
    fn from(s: String) -> Self {
        Self(CompactString::from(s))
    }
}

impl fmt::Display for Atom {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl AsRef<str> for Atom {
    #[inline]
    fn as_ref(&self) -> &str {
        self.0.as_str()
    }
}

/// A node paired with its source span.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Spanned<T> {
    pub node: T,
    pub span: Span,
}

impl<T> Spanned<T> {
    /// Create a new spanned node.
    #[inline]
    pub fn new(node: T, span: Span) -> Self {
        Self { node, span }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_new() {
        let span = Span::new(10, 20);
        assert_eq!(span.start, 10);
        assert_eq!(span.end, 20);
    }

    #[test]
    fn test_span_merge() {
        let a = Span::new(5, 15);
        let b = Span::new(10, 25);
        let merged = a.merge(b);
        assert_eq!(merged.start, 5);
        assert_eq!(merged.end, 25);
    }

    #[test]
    fn test_span_contains() {
        let span = Span::new(10, 20);
        assert!(span.contains(10));
        assert!(span.contains(15));
        assert!(span.contains(19));
        assert!(!span.contains(20));
        assert!(!span.contains(9));
    }

    #[test]
    fn test_span_source_text() {
        let source = "hello world";
        let span = Span::new(6, 11);
        assert_eq!(span.source_text(source), "world");
    }

    #[test]
    fn test_span_zero() {
        assert_eq!(SPAN_ZERO, Span::new(0, 0));
    }

    #[test]
    fn test_atom_from_str() {
        let atom = Atom::from("hello");
        assert_eq!(atom.as_str(), "hello");
        assert_eq!(format!("{}", atom), "hello");
    }

    #[test]
    fn test_atom_from_string() {
        let atom = Atom::from(String::from("world"));
        assert_eq!(atom.as_ref(), "world");
    }

    #[test]
    fn test_atom_ord() {
        let a = Atom::from("apple");
        let b = Atom::from("banana");
        assert!(a < b);
    }

    #[test]
    fn test_spanned() {
        let spanned = Spanned::new(42, Span::new(0, 5));
        assert_eq!(spanned.node, 42);
        assert_eq!(spanned.span, Span::new(0, 5));
    }
}
