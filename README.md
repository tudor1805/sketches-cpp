# ddsketch

[![Build Status](https://travis-ci.com/tudor1805/sketches-cpp.svg?branch=master)](https://travis-ci.com/github/tudor1805/sketches-cpp/builds)
[![Coverage Status](https://coveralls.io/repos/github/tudor1805/sketches-cpp/badge.svg?branch=master)](https://coveralls.io/github/tudor1805/sketches-cpp?branch=master)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/f857296d43db467cbff1d498650427b1)](https://www.codacy.com/gh/tudor1805/sketches-cpp/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=tudor1805/sketches-cpp&amp;utm_campaign=Badge_Grade)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This repo contains a C++14 port of the implementation for the distributed quantile sketch algorithm DDSketch [[1]](#References).

The port is based on the Python implementation - [reference-commit](https://github.com/DataDog/sketches-py/commit/4ea7ffafd91747d0b868488d7d6836f6473f5e83)

DDSketch has relative-error guarantees for any quantile `q` in `[0, 1]`. That is if the true value of the qth-quantile is `x` then DDSketch returns a value `y` such that `|x-y| / x < e` where `e` is the relative error parameter. (The default here is set to 0.01)

DDSketch is also fully mergeable, meaning that multiple sketches from distributed systems can be combined in a central node.

The original implementation can be found here:
*   [sketches-java](https://github.com/DataDog/sketches-java)
*   [sketches-py](https://github.com/DataDog/sketches-py)
*   [sketches-go](https://github.com/DataDog/sketches-go)

## Installation

The **ddsketch.h** header needs to be copied and included into the application you are building.

As the implementation uses some Modern C++ features, you will need to compile your application with `-std=c++14`

## Usage
    #include "ddsketch.h"

    constexpr auto kDesiredRelativeAccuracy = 0.01;
    ddsketch::DDSketch sketch(kDesiredRelativeAccuracy);

    for (auto value = 1; value <= 100; ++value) {
        sketch.add(value);
    }

    const auto quantiles = {
        0.01, 0.05, 0.10, 0.20, 0.25,
        0.40, 0.50, 0.60, 0.75, 0.85,
        0.95, 0.96, 0.97, 0.98, 0.99
    };

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    for (const auto quantile : quantiles) {
        const auto computed_quantile = sketch.get_quantile_value(quantile);

        std::cout << "Quantile: " << quantile << "\n"
                  << "Computed Quantile Value: " << computed_quantile << "\n";
    }

## Build

The build system uses [CMake](https://cmake.org/).
You can compile the example application by running the following commands:

    mkdir build && cd build

    cmake ../ -DBUILD_EXAMPLES=ON
    cmake --build .

    examples/DDSketch_Examples

## Performance

Below, we will attempt to benchmark the insertion rate of the algorithm

Given the fact that **ddsketch** only computes a logarithm for every input value, the insert operations are quite fast.

| CPU      | Avg Insert Rate | Ns per Insert |
| :---        |    :----:   |          ---: |
| [Intel i7-7820HQ](https://www.intel.com/content/www/us/en/products/processors/core/i7-processors/i7-7820hq.html)  (Virtual-Machine)   | 8.5 million       | 117  |
| [AMD EPYC](https://www.amd.com/en/products/cpu/amd-epyc-7302)   | 11.4 million | 84     |

## License

Apache 2.0

### References
Charles Masson and Jee E Rim and Homin K. Lee. DDSketch: A fast and fully-mergeable quantile sketch with relative-error guarantees. PVLDB, 12(12): 2195-2205, 2019
