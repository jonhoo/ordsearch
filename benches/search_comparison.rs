extern crate criterion;
extern crate num_traits;
extern crate ordsearch;
extern crate tango_bench;

use criterion::{
    criterion_group, criterion_main, measurement::WallTime, AxisScale, BatchSize, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration,
};
use ordsearch::OrderedCollection;
use std::{
    any::type_name,
    collections::BTreeSet,
    convert::TryFrom,
    iter,
    sync::atomic::{AtomicUsize, Ordering},
    time::Duration,
};

/// Because benchmarks are builded with linker flag -rdynamic there should be library entry point defined
/// in all benchmarks. On macOS linker is able to strip all tango_*() FFI functions, because the corresponding
/// module tango_bench::cli is not used. On Linux it is not possible to strip them, so we need to define
/// dummy entry point. This is only needed when two harnesses are used.
#[cfg(target_os = "linux")]
mod linker_fix {
    tango_bench::tango_benchmarks!([]);
}

const WARM_UP_TIME: Duration = Duration::from_millis(500);
const MEASUREMENT_TIME: Duration = Duration::from_millis(1000);
const DUPLICATION_FACTOR: usize = 16;

criterion_main!(benches);

criterion_group!(
    benches,
    benchmarks_for::<u8, { u8::MAX as usize }>,
    benchmarks_for::<u16, { u16::MAX as usize }>,
    benchmarks_for::<u32, { u32::MAX as usize }>,
    benchmarks_for::<u64, { u64::MAX as usize }>,
    benchmarks_for::<u128, { u64::MAX as usize }>,
);

fn benchmarks_for<T, const MAX: usize>(c: &mut Criterion)
where
    T: TryFrom<usize> + Ord + Copy,
{
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let sizes = [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536,
    ];

    {
        let groupname = format!("Search {}", type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config.clone());

        for i in sizes {
            search_bench_case::<MAX, T, _>(
                "sorted_vec",
                make_sorted_vec,
                search_sorted_vec,
                &mut group,
                i,
                false,
            );
            search_bench_case::<MAX, T, _>(
                "btreeset",
                make_btreeset,
                search_btreeset,
                &mut group,
                i,
                false,
            );
            search_bench_case::<MAX, T, _>(
                "ordsearch",
                make_this,
                search_this,
                &mut group,
                i,
                false,
            );
        }
        group.finish();
    }

    {
        let groupname = format!("Search (with duplicates) {}", type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config.clone());

        for i in sizes {
            search_bench_case::<MAX, T, _>(
                "sorted_vec",
                make_sorted_vec,
                search_sorted_vec,
                &mut group,
                i,
                true,
            );
            search_bench_case::<MAX, T, _>(
                "btreeset",
                make_btreeset,
                search_btreeset,
                &mut group,
                i,
                true,
            );
            search_bench_case::<MAX, T, _>(
                "ordsearch",
                make_this,
                search_this,
                &mut group,
                i,
                true,
            );
        }
        group.finish();
    }

    {
        let groupname = format!("Construction {}", type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config.clone());

        for i in sizes {
            construction_bench_case::<MAX, T, _>(
                "sorted_vec",
                make_sorted_vec,
                &mut group,
                i,
                false,
            );
            construction_bench_case::<MAX, T, _>("btreeset", make_btreeset, &mut group, i, false);
            construction_bench_case::<MAX, T, _>("ordsearch", make_this, &mut group, i, false);
        }
        group.finish();
    }

    {
        let groupname = format!("Construction (with duplicates) {}", type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config);

        for i in sizes {
            construction_bench_case::<MAX, T, _>(
                "sorted_vec",
                make_sorted_vec,
                &mut group,
                i,
                true,
            );
            construction_bench_case::<MAX, T, _>("btreeset", make_btreeset, &mut group, i, true);
            construction_bench_case::<MAX, T, _>("ordsearch", make_this, &mut group, i, true);
        }
        group.finish();
    }
}

fn search_bench_case<const MAX: usize, T, Coll>(
    name: &str,
    setup_fun: impl Fn(Vec<T>) -> Coll,
    search_fun: impl Fn(&Coll, T) -> Option<&T>,
    group: &mut BenchmarkGroup<WallTime>,
    size: usize,
    duplicates: bool,
) where
    T: TryFrom<usize> + Ord + Copy,
{
    group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &size| {
        // increasing sequence of even numbers, bounded by MAX
        let iter = (0usize..)
            // Generate only even numbers to provide a ~50% hit ratio in the benchmark
            .map(|i| (i * 2) % MAX)
            .map(|i| T::try_from(i).ok().unwrap());

        let v: Vec<T> = if duplicates {
            iter.flat_map(|i| iter::repeat(i).take(DUPLICATION_FACTOR))
                .take(size)
                .collect()
        } else {
            iter.take(size).collect()
        };

        // to generate ~50% hit ratio generated number should be in the range 0..(size * 2) because
        // test payload contains only even numbers: [0, 2, ..., 2 * size]
        let mut r = pseudorandom_iter::<T>(MAX.min(size * 2));
        let c = setup_fun(v);
        b.iter(|| search_fun(&c, r.next().unwrap()))
    });
}

fn construction_bench_case<const MAX: usize, T, Coll>(
    name: &str,
    setup_fun: impl Fn(Vec<T>) -> Coll,
    group: &mut BenchmarkGroup<WallTime>,
    size: usize,
    duplicates: bool,
) where
    T: TryFrom<usize> + Ord + Copy,
{
    group.bench_with_input(BenchmarkId::new(name, size), &size, |b, &size| {
        let v: Vec<T> = if duplicates {
            pseudorandom_iter(MAX)
                .flat_map(|i| iter::repeat(i).take(DUPLICATION_FACTOR))
                .take(size)
                .collect()
        } else {
            pseudorandom_iter(MAX).take(size).collect()
        };

        b.iter_batched(|| v.clone(), &setup_fun, BatchSize::SmallInput);
    });
}

fn make_this<T: Ord>(mut v: Vec<T>) -> OrderedCollection<T> {
    v.sort_unstable();
    OrderedCollection::from_sorted_iter(v)
}

fn search_this<T: Ord>(c: &OrderedCollection<T>, x: T) -> Option<&T> {
    c.find_gte(x)
}

fn make_btreeset<T: Ord>(v: Vec<T>) -> BTreeSet<T> {
    use std::iter::FromIterator;
    BTreeSet::from_iter(v)
}

fn search_btreeset<T: Ord>(c: &BTreeSet<T>, x: T) -> Option<&T> {
    use std::collections::Bound;
    c.range((Bound::Included(x), Bound::Unbounded)).next()
}

fn make_sorted_vec<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort_unstable();
    v
}

#[allow(clippy::ptr_arg)]
fn search_sorted_vec<T: Ord>(c: &Vec<T>, x: T) -> Option<&T> {
    c.binary_search(&x).ok().map(|i| &c[i])
}

fn pseudorandom_iter<T>(max: usize) -> impl Iterator<Item = T>
where
    T: TryFrom<usize>,
{
    static SEED: AtomicUsize = AtomicUsize::new(0);
    let mut seed = SEED.fetch_add(1, Ordering::SeqCst);

    iter::from_fn(move || {
        // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);

        // High 32 bits have much higher period
        let value = (seed >> 32) % max;
        Some(T::try_from(value).ok().unwrap())
    })
}
