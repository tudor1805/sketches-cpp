/*
 * Unless explicitly stated otherwise all files in this repository are licensed
 * under the Apache License 2.0.
 */
#include <iostream>

#include "../includes/test/datasets.h"
#include "../includes/ddsketch/ddsketch.h"

static void example_test_distributions() {
    using namespace ddsketch::test;

    std::vector<std::unique_ptr<GenericDataSet>> datasets;
    constexpr size_t kDatasetSize = 10;

    datasets.emplace_back(std::make_unique<UniformForward>());
    datasets.emplace_back(std::make_unique<UniformBackward>());
    datasets.emplace_back(std::make_unique<NegativeUniformForward>());
    datasets.emplace_back(std::make_unique<NegativeUniformBackward>());
    datasets.emplace_back(std::make_unique<NumberLineBackward>());
    datasets.emplace_back(std::make_unique<UniformZoomIn>());
    datasets.emplace_back(std::make_unique<UniformZoomOut>());
    datasets.emplace_back(std::make_unique<UniformSqrt>());
    datasets.emplace_back(std::make_unique<Constant>());
    datasets.emplace_back(std::make_unique<Exponential>());
    datasets.emplace_back(std::make_unique<Lognormal>());
    datasets.emplace_back(std::make_unique<Normal>());
    datasets.emplace_back(std::make_unique<Laplace>());
    datasets.emplace_back(std::make_unique<Bimodal>());
    datasets.emplace_back(std::make_unique<Mixed>());
    datasets.emplace_back(std::make_unique<Trimodal>());
    datasets.emplace_back(std::make_unique<Integers>());

    for (auto& dataset : datasets)
        dataset->populate(kDatasetSize);

    for (const auto& dataset : datasets)
        std::cout << *dataset << "\n";

    constexpr DataValue kDesiredQuantile = 0.80;
    constexpr Index kDesiredRank = 4;
    UniformForward uniform_forward;
    uniform_forward.populate(kDatasetSize);

    std::cout << uniform_forward << "\n";

    std::cout << "Quantile: "
              << uniform_forward.quantile(kDesiredQuantile) << "\n";

    std::cout << "Rank: " << uniform_forward.rank(kDesiredRank) << "\n";
}

static void example_basic_ddsketch() {
    constexpr auto kDesiredRelativeAccuracy = 0.05;

    ddsketch::DDSketch sketch(kDesiredRelativeAccuracy);

    for (auto value = 1; value <= 100; ++value)
        sketch.add(value);

    constexpr auto quantiles = {
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

    return 0;
}
