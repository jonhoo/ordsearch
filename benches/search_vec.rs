extern crate tango_bench;

use common::{search_benchmarks, Sample};
use tango_bench::{tango_benchmarks, tango_main};

mod common;

fn search_vec<T: Copy + Ord>(haystack: &Sample<Vec<T>>, needle: &T) -> Option<T> {
    let haystack = haystack.as_ref();
    haystack
        .binary_search(needle)
        .ok()
        .and_then(|idx| haystack.get(idx))
        .copied()
}

tango_benchmarks!(
    search_benchmarks(search_vec::<u8>),
    search_benchmarks(search_vec::<u16>),
    search_benchmarks(search_vec::<u32>),
    search_benchmarks(search_vec::<u64>)
);

tango_main!(common::SETTINGS);
