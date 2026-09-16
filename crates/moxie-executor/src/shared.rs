//! Private, single-threaded shared ownership with fallible construction.
//!
//! Arena ranges must keep their physical allocation alive independently of the
//! arena handle. `Rc::new` cannot refuse host allocation on stable Rust. This
//! deliberately smaller handle has no weak references, raw-pointer API, COW or
//! cross-thread support. A one-element Vec supplies fallible allocation; the last
//! handle reconstructs that exact Vec. CUDA ownership and accounting stay in
//! DeviceArena, not here.

use std::cell::Cell;
use std::marker::PhantomData;
use std::ops::Deref;
use std::ptr::NonNull;
use std::rc::Rc;

struct Entry<T> {
    owners: Cell<usize>,
    value: T,
}

pub(crate) struct Shared<T> {
    entry: NonNull<Entry<T>>,
    capacity: usize,
    // Own T for drop checking; retain Rc's !Send / !Sync even for Send T.
    marker: PhantomData<Rc<T>>,
}

impl<T> Shared<T> {
    pub(crate) fn try_new(value: T) -> moxie_types::Result<Self> {
        let mut storage = moxie_memory::fallible::with_capacity(1)?;
        storage.push(Entry {
            owners: Cell::new(1),
            value,
        });
        let result = Self {
            entry: NonNull::new(storage.as_mut_ptr()).expect("one live entry"),
            capacity: storage.capacity(),
            marker: PhantomData,
        };
        std::mem::forget(storage);
        Ok(result)
    }

    pub(crate) fn get_mut(this: &mut Self) -> Option<&mut T> {
        // SAFETY: each handle owns one counted reference to the live Entry.
        // Only a unique handle may expose &mut T. No weak/raw handles exist;
        // &mut Self prevents another borrow from this handle during the result.
        unsafe {
            if this.entry.as_ref().owners.get() == 1 {
                Some(&mut this.entry.as_mut().value)
            } else {
                None
            }
        }
    }

    pub(crate) fn ptr_eq(a: &Self, b: &Self) -> bool {
        a.entry == b.entry
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        // SAFETY: self keeps the Entry alive, and this type is !Send / !Sync.
        let count = unsafe { &self.entry.as_ref().owners };
        let next = count
            .get()
            .checked_add(1)
            .expect("shared owner count overflow");
        count.set(next);
        Self {
            entry: self.entry,
            capacity: self.capacity,
            marker: PhantomData,
        }
    }
}

impl<T> Deref for Shared<T> {
    type Target = T;
    fn deref(&self) -> &T {
        // SAFETY: self retains the entry, and get_mut requires unique ownership
        // plus exclusive borrowing. The reference cannot outlive this handle.
        unsafe { &self.entry.as_ref().value }
    }
}

impl<T> Drop for Shared<T> {
    fn drop(&mut self) {
        // SAFETY: this is a counted live reference. Counts change only on this
        // thread. At one owner no other handle or borrowed T can survive self;
        // reconstruct the original allocation with its original length/capacity.
        unsafe {
            let count = self.entry.as_ref().owners.get();
            if count == 1 {
                drop(Vec::from_raw_parts(self.entry.as_ptr(), 1, self.capacity));
            } else {
                self.entry.as_ref().owners.set(count - 1);
            }
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("Shared").field(&**self).finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sharing_retains_one_value_until_the_last_owner_and_restores_unique_access() {
        struct CountDrop<'a>(&'a Cell<usize>, usize);
        impl Drop for CountDrop<'_> {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }
        let drops = Cell::new(0);
        let mut a = Shared::try_new(CountDrop(&drops, 7)).unwrap();
        let b = a.clone();
        let c = b.clone();
        assert!(Shared::ptr_eq(&a, &c));
        assert!(Shared::get_mut(&mut a).is_none());
        drop(b);
        drop(c);
        assert_eq!(drops.get(), 0);
        Shared::get_mut(&mut a).unwrap().1 = 9;
        assert_eq!(a.1, 9);
        drop(a);
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn pointer_identity_distinguishes_equal_values_and_zst_values_drop() {
        let a = Shared::try_new(1u32).unwrap();
        let b = Shared::try_new(1u32).unwrap();
        assert!(!Shared::ptr_eq(&a, &b));
        let zst = Shared::try_new(()).unwrap();
        let other = zst.clone();
        drop(zst);
        assert_eq!(*other, ());
    }
}
