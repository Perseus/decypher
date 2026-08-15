//! Shared typed arena storage.

use std::fmt;
use std::marker::PhantomData;

/// A compact, type-safe index into an [`Arena`].
///
/// The type parameter identifies the kind of value stored in the arena, so an
/// ID for one value type cannot be passed to an arena for another value type.
pub struct ArenaId<T> {
    index: usize,
    marker: PhantomData<fn() -> T>,
}

impl<T> ArenaId<T> {
    /// A sentinel used while constructing self-referential arena values.
    pub const INVALID: Self = Self::from_raw(usize::MAX);

    const fn from_raw(index: usize) -> Self {
        Self {
            index,
            marker: PhantomData,
        }
    }

    /// Return the underlying arena index.
    pub const fn index(self) -> usize {
        self.index
    }
}

impl<T> Copy for ArenaId<T> {}

impl<T> Clone for ArenaId<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> PartialEq for ArenaId<T> {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index
    }
}

impl<T> Eq for ArenaId<T> {}

impl<T> std::hash::Hash for ArenaId<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.index.hash(state);
    }
}

impl<T> fmt::Debug for ArenaId<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ArenaId").field(&self.index).finish()
    }
}

impl<T> From<ArenaId<T>> for usize {
    fn from(id: ArenaId<T>) -> Self {
        id.index
    }
}

/// A growable collection with stable, typed indices.
///
/// Values are allocated in insertion order and remain at the same index for
/// the lifetime of the arena.
pub struct Arena<T> {
    entries: Vec<T>,
}

impl<T> Arena<T> {
    /// Create an empty arena.
    pub const fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Store a value and return its typed ID.
    pub fn alloc(&mut self, value: T) -> ArenaId<T> {
        let id = ArenaId::from_raw(self.entries.len());
        self.entries.push(value);
        id
    }

    /// Return the value referenced by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is outside this arena.
    pub fn get(&self, id: ArenaId<T>) -> &T {
        &self.entries[id.index]
    }

    /// Return the mutable value referenced by `id`.
    ///
    /// # Panics
    ///
    /// Panics if `id` is outside this arena.
    pub fn get_mut(&mut self, id: ArenaId<T>) -> &mut T {
        &mut self.entries[id.index]
    }

    /// Iterate over IDs and values in allocation order.
    pub fn iter(&self) -> impl Iterator<Item = (ArenaId<T>, &T)> {
        self.entries
            .iter()
            .enumerate()
            .map(|(index, value)| (ArenaId::from_raw(index), value))
    }

    /// Return the number of allocated values.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return whether the arena has no allocated values.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl<T: fmt::Debug> fmt::Debug for Arena<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Arena")
            .field("len", &self.entries.len())
            .finish()
    }
}

impl<T: Clone> Clone for Arena<T> {
    fn clone(&self) -> Self {
        Self {
            entries: self.entries.clone(),
        }
    }
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn allocates_and_iterates_in_order() {
        let mut arena = Arena::new();
        let first = arena.alloc("first");
        let second = arena.alloc("second");

        assert_eq!(first.index(), 0);
        assert_eq!(second.index(), 1);
        assert_eq!(arena.get(first), &"first");
        assert_eq!(arena.get(second), &"second");
        assert_eq!(
            arena
                .iter()
                .map(|(id, value)| (id.index(), *value))
                .collect::<Vec<_>>(),
            vec![(0, "first"), (1, "second")]
        );
    }

    #[test]
    fn supports_mutation_by_typed_id() {
        let mut arena = Arena::new();
        let id = arena.alloc(1);

        *arena.get_mut(id) = 2;

        assert_eq!(arena.get(id), &2);
    }
}
