# ordsearch

[![Crates.io](https://img.shields.io/crates/v/ordsearch.svg)](https://crates.io/crates/ordsearch)
[![Documentation](https://docs.rs/ordsearch/badge.svg)](https://docs.rs/ordsearch/)
[![Build Status](https://travis-ci.org/jonhoo/ordsearch.svg?branch=master)](https://travis-ci.org/jonhoo/ordsearch)

This crate provides a data structure for approximate lookups in ordered collections.

More concretely, given a set `A` of `n` values, and a query value `x`, this library provides an
efficient mechanism for finding the smallest value in `A` that is greater than or equal to `x`.
In particular, this library caters to the important case where there are many such queries to
the same array, `A`.

This library is constructed from the best solution identified in [Array Layouts for
Comparison-Based Searching](https://arxiv.org/abs/1509.05053) by Paul-Virak Khuong and Pat
Morin. For more information, see the paper, [their
website](http://cglab.ca/~morin/misc/arraylayout-v2/), and the [C++ implementation
repository](https://github.com/patmorin/arraylayout).

## Current implementation

At the time of writing, this implementation uses a branch-free search over an
Eytzinger-arranged array with masked prefetching based on the [C++
implementation](https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L253)
written by the authors of the aforementioned paper. This is the recommended algorithm from the
paper, and what the authors suggested in
https://github.com/patmorin/arraylayout/issues/3#issuecomment-338472755.

Note that prefetching is *only* enabled with the (non-default) `nightly` feature due to
https://github.com/aweinstock314/prefetch/issues/1. Suggestions for workarounds welcome.

## Performance

The included benchmarks can be run with

```console,ignore
$ cargo +nightly bench --features nightly
```

This will benchmark both construction and search with different number of values, and
differently sized values -- look for the line that aligns closest with your data. The general
trend is that `ordsearch` is faster when `n` is smaller and `T` is larger as long as you
compile with
[`target-cpu=native`](https://github.com/jonhoo/ordsearch/issues/2#issuecomment-390441137) and
[`lto=thin`](https://github.com/jonhoo/ordsearch/issues/2#issuecomment-390446671). The
performance gain seems to be best on Intel processors, and is smaller since the (relatively)
recent improvement to [SliceExt::binary_search
performance](https://github.com/rust-lang/rust/pull/45333).

Below are [summarized](https://github.com/BurntSushi/cargo-benchcmp) results from an Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30GHz CPU run with:

```console
$ rustc +nightly --version
rustc 1.73.0-nightly (33a2c2487 2023-07-12)
$ env CARGO_INCREMENTAL=0 RUSTFLAGS='-C target-cpu=native' cargo +nightly bench --features nightly
```

Compared to binary search over a sorted vector:

```diff,ignore
 name           sorted_vec ns/iter  this ns/iter  diff ns/iter   diff %  speedup
+u32::l1        46                  15                     -31  -67.39%   x 3.07
+u32::l1_dup    30                  14                     -16  -53.33%   x 2.14
+u32::l2        66                  26                     -40  -60.61%   x 2.54
+u32::l2_dup    44                  27                     -17  -38.64%   x 1.63
+u32::l3        129                 37                     -92  -71.32%   x 3.49
+u32::l3_dup    109                 37                     -72  -66.06%   x 2.95
+u8::l1         30                  16                     -14  -46.67%   x 1.88
+u8::l1_dup     20                  14                      -6  -30.00%   x 1.43
-u8::l2         26                  29                       3   11.54%   x 0.90
-u8::l2_dup     19                  26                       7   36.84%   x 0.73
-u8::l3         13                  36                      23  176.92%   x 0.36
-u8::l3_dup     17                  31                      14   82.35%   x 0.55
+usize::l1      49                  13                     -36  -73.47%   x 3.77
+usize::l1_dup  30                  15                     -15  -50.00%   x 2.00
+usize::l2      68                  25                     -43  -63.24%   x 2.72
+usize::l2_dup  47                  27                     -20  -42.55%   x 1.74
+usize::l3      155                 52                    -103  -66.45%   x 2.98
+usize::l3_dup  157                 61                     -96  -61.15%   x 2.57
```

Compared to a `BTreeSet`:

```diff,ignore
 name           btreeset ns/iter  this ns/iter  diff ns/iter   diff %  speedup
+u32::l1        49                15                     -34  -69.39%   x 3.27
+u32::l1_dup    30                14                     -16  -53.33%   x 2.14
+u32::l2        61                26                     -35  -57.38%   x 2.35
+u32::l2_dup    43                27                     -16  -37.21%   x 1.59
+u32::l3        125               37                     -88  -70.40%   x 3.38
+u32::l3_dup    83                37                     -46  -55.42%   x 2.24
+u8::l1         34                16                     -18  -52.94%   x 2.12
+u8::l1_dup     24                14                     -10  -41.67%   x 1.71
+u8::l2         30                29                      -1   -3.33%   x 1.03
+u8::l2_dup     27                26                      -1   -3.70%   x 1.04
-u8::l3         23                36                      13   56.52%   x 0.64
-u8::l3_dup     26                31                       5   19.23%   x 0.84
+usize::l1      48                13                     -35  -72.92%   x 3.69
+usize::l1_dup  31                15                     -16  -51.61%   x 2.07
+usize::l2      63                25                     -38  -60.32%   x 2.52
+usize::l2_dup  42                27                     -15  -35.71%   x 1.56
+usize::l3      166               52                    -114  -68.67%   x 3.19
+usize::l3_dup  80                61                     -19  -23.75%   x 1.31
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128
