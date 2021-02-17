/*
 * Unless explicitly stated otherwise all files in this repository are licensed
 * under the Apache License 2.0.
 */
#include <iostream>

#include "../include/test/datasets.h"
#include "../include/ddsketch/ddsketch.h"

namespace test = ddsketch::test;

static void example_test_distributions() {
    std::vector<std::unique_ptr<test::GenericDataSet>> datasets;
    constexpr size_t kDatasetSize = 10;

    datasets.emplace_back(std::make_unique<test::UniformForward>());
    datasets.emplace_back(std::make_unique<test::UniformBackward>());
    datasets.emplace_back(std::make_unique<test::NegativeUniformForward>());
    datasets.emplace_back(std::make_unique<test::NegativeUniformBackward>());
    datasets.emplace_back(std::make_unique<test::NumberLineBackward>());
    datasets.emplace_back(std::make_unique<test::UniformZoomIn>());
    datasets.emplace_back(std::make_unique<test::UniformZoomOut>());
    datasets.emplace_back(std::make_unique<test::UniformSqrt>());
    datasets.emplace_back(std::make_unique<test::Constant>());
    datasets.emplace_back(std::make_unique<test::Exponential>());
    datasets.emplace_back(std::make_unique<test::Lognormal>());
    datasets.emplace_back(std::make_unique<test::Normal>());
    datasets.emplace_back(std::make_unique<test::Laplace>());
    datasets.emplace_back(std::make_unique<test::Bimodal>());
    datasets.emplace_back(std::make_unique<test::Mixed>());
    datasets.emplace_back(std::make_unique<test::Trimodal>());
    datasets.emplace_back(std::make_unique<test::Integers>());

    for (auto& dataset : datasets) {
        dataset->populate(kDatasetSize);
    }

    for (const auto& dataset : datasets) {
        std::cout << *dataset << "\n";
    }
}

static void example_uniform_forward_ddsketch() {
    constexpr size_t kDatasetSize = 10;
    constexpr test::DataValue kDesiredQuantile = 0.80;
    constexpr auto kDesiredRank = 4;
    test::UniformForward uniform_forward;

    uniform_forward.populate(kDatasetSize);
    std::cout << uniform_forward << "\n";

    std::cout << "Quantile: "
              << uniform_forward.quantile(kDesiredQuantile) << "\n";

    std::cout << "Rank: " << uniform_forward.rank(kDesiredRank) << "\n";
}


static void example_basic_ddsketch() {
    constexpr auto kDesiredRelativeAccuracy = 0.01;
    constexpr auto kTotalNumbers = 100;

    ddsketch::DDSketch sketch(kDesiredRelativeAccuracy);

    for (auto value = 1; value <= kTotalNumbers; ++value) {
        sketch.add(value);
    }

    const auto quantiles = {
        0.01, 0.05, 0.10, 0.20, 0.25,
        0.40, 0.50, 0.60, 0.75, 0.85,
        0.95, 0.96, 0.97, 0.98, 0.99
    };

    std::cout.precision(std::numeric_limits<double>::max_digits10);

    for (const auto quantile : quantiles) {
        auto computed_quantile = sketch.get_quantile_value(quantile);
        std::cout << "Quantile: " << quantile
                  << "\nComputed Value: " << computed_quantile << "\n\n";
    }
}

int main() {
    example_test_distributions();

    example_basic_ddsketch();

    example_uniform_forward_ddsketch();

    return 0;
}
