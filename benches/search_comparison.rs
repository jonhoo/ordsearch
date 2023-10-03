extern crate criterion;
extern crate num_traits;
extern crate ordsearch;

use criterion::{
    black_box, criterion_group, criterion_main, measurement::WallTime, AxisScale, BenchmarkGroup,
    BenchmarkId, Criterion, PlotConfiguration,
};
use ordsearch::OrderedCollection;
use std::{collections::BTreeSet, convert::TryFrom, time::Duration};

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
    T: TryFrom<usize>
        + Ord
        + std::ops::Rem<Output = T>
        + num_traits::ops::wrapping::WrappingMul
        + Clone,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    let plot_config = PlotConfiguration::default().summary_scale(AxisScale::Logarithmic);

    let sizes = [
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4069, 8192, 16384, 32768, 65536,
    ];

    {
        let groupname = format!("Search {}", std::any::type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config.clone());

        for i in sizes.iter() {
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
        let groupname = format!("Search (with duplicates) {}", std::any::type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config.clone());

        for i in sizes.iter() {
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
        let groupname = format!("Construction {}", std::any::type_name::<T>());
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config.clone());

        for i in sizes.iter() {
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
        let groupname = format!(
            "Construction (with duplicates) {}",
            std::any::type_name::<T>()
        );
        let mut group = c.benchmark_group(groupname);
        group
            .warm_up_time(WARM_UP_TIME)
            .measurement_time(MEASUREMENT_TIME)
            .plot_config(plot_config);

        for i in sizes.iter() {
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
    size: &usize,
    duplicates: bool,
) where
    T: TryFrom<usize> + Ord + Clone,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    group.bench_with_input(BenchmarkId::new(name, size), size, |b, &size| {
        // increasing sequence of even numbers, bounded by MAX
        let iter = (0usize..)
            // Generate only even numbers to provide a ~50% hit ratio in the benchmark
            .map(|i| (i * 2) % MAX)
            .map(|i| T::try_from(i).unwrap());

        let v: Vec<T> = if duplicates {
            iter
                // Repeat each items 16 times
                .flat_map(|i| std::iter::repeat(i).take(DUPLICATION_FACTOR))
                .take(size)
                .collect()
        } else {
            iter.take(size).collect()
        };

        let mut r = pseudorandom_iter::<T>((size * 2) % MAX);
        let c = setup_fun(v);
        b.iter(|| {
            let x = r.next().unwrap();
            let _res = black_box(search_fun(&c, x));
        })
    });
}

fn construction_bench_case<const MAX: usize, T, Coll>(
    name: &str,
    setup_fun: impl Fn(Vec<T>) -> Coll,
    group: &mut BenchmarkGroup<WallTime>,
    size: &usize,
    duplicates: bool,
) where
    T: TryFrom<usize> + Ord + Clone,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    group.bench_with_input(BenchmarkId::new(name, size), size, |b, &size| {
        let v: Vec<T> = if duplicates {
            pseudorandom_iter(MAX)
                .flat_map(|i| std::iter::repeat(i).take(DUPLICATION_FACTOR))
                .take(size)
                .collect()
        } else {
            pseudorandom_iter(MAX).take(size).collect()
        };

        b.iter_batched(
            || v.clone(),
            |v| {
                let _res = black_box(setup_fun(v));
            },
            criterion::BatchSize::SmallInput,
        );
    });
}

fn make_this<T: Ord>(mut v: Vec<T>) -> OrderedCollection<T> {
    v.sort_unstable();
    OrderedCollection::from_sorted_iter(v.into_iter())
}

fn search_this<T: Ord>(c: &OrderedCollection<T>, x: T) -> Option<&T> {
    c.find_gte(x).map(|v| &*v)
}

fn make_btreeset<T: Ord>(v: Vec<T>) -> BTreeSet<T> {
    use std::iter::FromIterator;
    BTreeSet::from_iter(v.into_iter())
}

fn search_btreeset<T: Ord>(c: &BTreeSet<T>, x: T) -> Option<&T> {
    use std::collections::Bound;
    c.range((Bound::Included(x), Bound::Unbounded))
        .next()
        .map(|v| &*v)
}

fn make_sorted_vec<T: Ord>(mut v: Vec<T>) -> Vec<T> {
    v.sort_unstable();
    v
}

fn search_sorted_vec<'a, T: Ord>(c: &'a Vec<T>, x: T) -> Option<&'a T> {
    c.binary_search(&x).ok().map(|i| &c[i])
}

fn pseudorandom_iter<T>(max: usize) -> impl Iterator<Item = T>
where
    T: TryFrom<usize>,
    <T as TryFrom<usize>>::Error: core::fmt::Debug,
{
    let mut seed = 0usize;
    std::iter::from_fn(move || {
        // LCG constants from https://en.wikipedia.org/wiki/Numerical_Recipes.
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        let r = seed % max;

        Some(T::try_from(r).unwrap())
    })
}
