#!/usr/bin/env bash

# This scripts generate plot.svg in the current directory.
#
# To use it properly gnuplot should be installed on the system
#  1. run benchmarks
#     $ cargo +nightly bench --bench=search_comparison --features=nightly "^Search"
#  2. run this script
#     $ ./plots/generate-plot.sh

RELATIVE_PATH=$(dirname "$0")

# Running all commands in the current directory
cd "$RELATIVE_PATH"
gnuplot ./plot.gnuplot