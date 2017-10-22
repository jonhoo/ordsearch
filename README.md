# ordsearch

[![Crates.io](https://img.shields.io/crates/v/ordsearch.svg)](https://crates.io/crates/ordsearch)
[![Documentation](https://docs.rs/ordsearch/badge.svg)](https://docs.rs/ordsearch/)
[![Build Status](https://travis-ci.org/jonhoo/ordsearch.svg?branch=master)](https://travis-ci.org/jonhoo/ordsearch)

This crate provides an efficient data structure for approximate lookups in ordered collections.

More conretely, given a set `A` of `n` values, and a query value `x`, this library provides an
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

These will benchmark both construction and search with different number of values, and
differently sized values. `search_common` is likely the metric you want to pay the most
attention to: it is the cost of looking up among 1024 `usize` values.
[Summarized](https://github.com/BurntSushi/cargo-benchcmp) results from my laptop (an X1 Carbon
with i7-5600U @ 2.60GHz) are given below.

Compared to binary search over a sorted vector:

```text,ignore
name                   sorted_vec:: ns/iter  this:: ns/iter  diff ns/iter   diff %  speedup
search_common          67                    35                       -32  -47.76%   x 1.91
search_few_u32         42                    21                       -21  -50.00%   x 2.00
search_few_u8          42                    21                       -21  -50.00%   x 2.00
search_many_u32        115                   98                       -17  -14.78%   x 1.17
search_many_u8         52                    70                        18   34.62%   x 0.74
construction_few_u32   3,365                 3,924                    559   16.61%   x 0.86
construction_few_u8    3,483                 4,125                    642   18.43%   x 0.84
construction_many_u32  2,338,019             2,619,590            281,571   12.04%   x 0.89
construction_many_u8   1,174,587             1,481,139            306,552   26.10%   x 0.79
```

Compared to a `BTreeSet`:

```text,ignore
name                   btreeset:: ns/iter  this:: ns/iter  diff ns/iter   diff %  speedup
search_common          94                  35                       -59  -62.77%   x 2.69
search_few_u32         70                  21                       -49  -70.00%   x 3.33
search_few_u8          64                  21                       -43  -67.19%   x 3.05
search_many_u32        182                 98                       -84  -46.15%   x 1.86
search_many_u8         69                  70                         1    1.45%   x 0.99
construction_few_u32   9,598               3,924                 -5,674  -59.12%   x 2.45
construction_few_u8    8,410               4,125                 -4,285  -50.95%   x 2.04
construction_many_u32  10,416,444          2,619,590         -7,796,854  -74.85%   x 3.98
construction_many_u8   3,643,919           1,481,139         -2,162,780  -59.35%   x 2.46
```

## Future work

 - [ ] Implement aligned operation: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L204
 - [ ] Implement deep prefetching for large `T`: https://github.com/patmorin/arraylayout/blob/3f20174a2a0ab52c6f37f2ea87d087307f19b5ee/src/eytzinger_array.h#L128

