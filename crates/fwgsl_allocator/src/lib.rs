//! Arena allocator for fwgsl, wrapping bumpalo.
//!
//! Inspired by Oxc's arena allocation strategy for high-performance
//! compilation and toolchain operations.

use bumpalo::Bump;

/// Arena allocator backed by bumpalo's bump allocator.
///
/// All AST nodes and intermediate data structures are allocated
/// in this arena, enabling fast allocation and bulk deallocation.
pub struct Allocator {
    bump: Bump,
}

/// An immutable reference to an arena-allocated value.
///
/// This is a zero-cost abstraction over `&'a T` that semantically
/// indicates the value lives in the arena.
pub type Box<'a, T> = &'a T;

/// A growable vector allocated in the arena.
pub type Vec<'a, T> = bumpalo::collections::Vec<'a, T>;

impl Allocator {
    /// Create a new arena allocator.
    #[inline]
    pub fn new() -> Self {
        Self { bump: Bump::new() }
    }

    /// Allocate a value in the arena, returning an immutable reference.
    #[inline]
    pub fn alloc<T>(&self, val: T) -> Box<'_, T> {
        self.bump.alloc(val)
    }

    /// Create a new empty vector in the arena.
    #[inline]
    pub fn vec<T>(&self) -> Vec<'_, T> {
        bumpalo::collections::Vec::new_in(&self.bump)
    }

    /// Returns a reference to the underlying bumpalo `Bump` allocator.
    #[inline]
    pub fn bump(&self) -> &Bump {
        &self.bump
    }
}

impl Default for Allocator {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_and_read() {
        let allocator = Allocator::new();
        let val: Box<'_, i32> = allocator.alloc(42);
        assert_eq!(*val, 42);
    }

    #[test]
    fn test_vec() {
        let allocator = Allocator::new();
        let mut v: Vec<'_, i32> = allocator.vec();
        v.push(1);
        v.push(2);
        v.push(3);
        assert_eq!(v.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn test_default() {
        let allocator = Allocator::default();
        let val = allocator.alloc("hello");
        assert_eq!(*val, "hello");
    }

    #[test]
    fn test_multiple_allocations() {
        let allocator = Allocator::new();
        let a = allocator.alloc(1u32);
        let b = allocator.alloc(2u32);
        let c = allocator.alloc(3u32);
        assert_eq!(*a, 1);
        assert_eq!(*b, 2);
        assert_eq!(*c, 3);
    }
}
