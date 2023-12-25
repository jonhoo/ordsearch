extern crate ordsearch;
extern crate tango_bench;

use common::{search_benchmarks, FromSortedVec};
use ordsearch::OrderedCollection;
use tango_bench::{tango_benchmarks, tango_main};

mod common;

impl<T: Ord> FromSortedVec for OrderedCollection<T> {
    type Item = T;
    fn from_sorted_vec(v: Vec<T>) -> Self {
        OrderedCollection::from_sorted_iter(v)
    }
}

fn search_ord<T: Copy + Ord>(haystack: &impl AsRef<OrderedCollection<T>>, needle: &T) -> Option<T> {
    haystack.as_ref().find_gte(*needle).copied()
}

tango_benchmarks!(
    search_benchmarks(search_ord::<u8>),
    search_benchmarks(search_ord::<u16>),
    search_benchmarks(search_ord::<u32>),
    search_benchmarks(search_ord::<u64>)
);
tango_main!(common::SETTINGS);
