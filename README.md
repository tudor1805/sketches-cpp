# ddsketch

This repo contains a C++14 port of the implementation for the distributed quantile sketch algorithm DDSketch [1].

DDSketch has relative-error guarantees for any quantile q in [0, 1]. That is if the true value of the qth-quantile is `x` then DDSketch returns a value `y` such that `|x-y| / x < e` where `e` is the relative error parameter. (The default here is set to 0.01.)

DDSketch is also fully mergeable, meaning that multiple sketches from distributed systems can be combined in a central node.

The original implementation can be found here:
* [sketches-java](https://github.com/DataDog/sketches-java)
* [sketches-py](https://github.com/DataDog/sketches-py)
* [sketches-go](https://github.com/DataDog/sketches-go)

### Installation

The **ddsketch.h** header needs to be copied and included into the application you are building.

### Usage
    #include "ddsketch.h"

    constexpr auto kDesiredRelativeAccuracy = 0.01;
    ddsketch::DDSketch sketch(kDesiredRelativeAccuracy);

    for (auto value = 1; value <= 100; ++value) sketch.add(value);

    constexpr auto quantiles = {
        0.01, 0.05, 0.10, 0.20, 0.25,
        0.40, 0.50, 0.60, 0.75, 0.85,
        0.95, 0.96, 0.97, 0.98, 0.99
    };

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    for (const auto quantile : quantiles) {
        auto computed_quantile = sketch.get_quantile_value(quantile);
        std::cout << "Quantile: " << quantile << "\n"
                  << "Computed Value: " << computed_quantile << "\n";
    }

### Performance

TODO

License
----

Apache 2.0

### References
[1] Charles Masson and Jee E Rim and Homin K. Lee. DDSketch: A fast and fully-mergeable quantile sketch with relative-error guarantees. PVLDB, 12(12): 2195-2205, 2019.

