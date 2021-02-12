/*
 * Unless explicitly stated otherwise all files in this repository are licensed
 * under the Apache License 2.0.
 */

#include <iostream>
#include <vector>
#include <map>

#include "../includes/ddsketch/ddsketch.h"
#include "../includes/test/datasets.h"

#include "gtest/gtest.h"

using namespace ddsketch;

namespace ddsketch { namespace test {

using StoreValue = int64_t;
using StoreValues = std::vector<int64_t>;
using StoreValueList = std::vector<StoreValues>;

#define EXPECT_ALMOST_EQ(a, b) EXPECT_NEAR((a), (b), 0.000001)

template <typename Mapping>
class MappingTest : public ::testing::Test {
 protected:
    static Mapping create_mapping(RealValue relative_accuracy,
                                  RealValue offset) {
        return Mapping(relative_accuracy, offset);
    }

    /* Helper method to calculate the relative error */
    static RealValue relative_error(RealValue expected_min,
                                    RealValue expected_max,
                                    RealValue actual) {
        if (expected_min < 0 || expected_max < 0 || actual < 0)
            throw std::invalid_argument("Arguments should be positive numbers");

        if (expected_min <= actual && actual <= expected_max)
            return 0.0;

        if (expected_min == 0 && expected_max == 0) {
            if (actual == 0)
                return 0.0;
            else
                return std::numeric_limits<RealValue>::max();
        }

        if (actual < expected_min)
            return (expected_min - actual) / expected_min;

        return (actual - expected_max) / expected_max;
    }

    /* Calculate relative accuracy of a mapping on a large range of values */
    static RealValue test_value_rel_acc(Mapping& mapping) {
        auto value_mult = 2.0 - std::sqrt(2) * 1.0e-1;
        auto max_relative_acc = 0.0;
        auto value = mapping.min_possible();

        while (value < mapping.max_possible() / value_mult)  {
            value *= value_mult;
            auto map_val = mapping.value(mapping.key(value));
            auto rel_err = relative_error(value, value, map_val);

            EXPECT_LT(rel_err, mapping.relative_accuracy());
            max_relative_acc = std::max(max_relative_acc, rel_err);
        }

        max_relative_acc = std::max(
            max_relative_acc,
            relative_error(
                mapping.max_possible(),
                mapping.max_possible(),
                mapping.value(mapping.key(mapping.max_possible()))));

       return max_relative_acc;
    }

    /* Test the mapping on a large range of relative accuracies */
    static void test_relative_accuracy() {
        RealValue rel_acc_mult = 1 - std::sqrt(2) * 1.0e-1;
        RealValue min_rel_acc = 1.0e-8;
        RealValue rel_acc = 1 - 1.0e-3;

        while (rel_acc >= min_rel_acc) {
            auto mapping = create_mapping(rel_acc, 0.0);
            auto max_rel_acc = test_value_rel_acc(mapping);
            EXPECT_LT(max_rel_acc, mapping.relative_accuracy());
            rel_acc *= rel_acc_mult;
        }
    }

    static void test_offsets() {
        static constexpr RealValue kRelativeAccuracy = 0.01;
        std::vector<RealValue> offsets = {0, 1, -12.23, 7768.3};

        for (auto offset : offsets) {
            auto mapping = create_mapping(kRelativeAccuracy, offset);
            EXPECT_EQ(mapping.key(1), static_cast<int>(offset));
        }
    }
};

class LogarithmicMappingTest
    : public MappingTest<LogarithmicMapping> {
};

class LinearlyInterpolatedMappingTest
    : public MappingTest<LinearlyInterpolatedMapping> {
};

class CubicallyInterpolatedMappingTest
    : public MappingTest<CubicallyInterpolatedMapping> {
};

TEST_F(LogarithmicMappingTest, TestRelativeAccuracy) {
    test_relative_accuracy();
}

TEST_F(LogarithmicMappingTest, TestOffsets) {
    test_offsets();
}

TEST_F(LinearlyInterpolatedMappingTest, TestRelativeAccuracy) {
    test_relative_accuracy();
}

TEST_F(LinearlyInterpolatedMappingTest, TestOffsets) {
    test_offsets();
}

TEST_F(CubicallyInterpolatedMappingTest, TestRelativeAccuracy) {
    test_relative_accuracy();
}

TEST_F(CubicallyInterpolatedMappingTest, TestOffsets) {
    test_offsets();
}

class Counter {
 public:
    using KeyValueContainer = std::map<StoreValue, StoreValue>;
    using iterator = typename KeyValueContainer::iterator;
    using const_iterator = typename KeyValueContainer::const_iterator;

    iterator begin() {
         return map_.begin();
    }

    iterator end() {
        return map_.end();
    }

    const_iterator begin() const {
        return map_.begin();
    }

    const_iterator end() const {
        return map_.end();
    }

    explicit Counter(const StoreValues& values) {
        for (const auto& value : values) {
            /* The element does not exist */
            if (map_.find(value) == map_.end())
                map_[value] = 1;
            else
                ++map_[value];
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const Counter& counter) {
        os << "{ ";

        for (const auto &pair : counter.map_)
            os << pair.first << ":" << pair.second << " ";

        os << "}";

        return os;
    }

    StoreValue sum_values() const {
        return std::accumulate(
            std::begin(map_),
            std::end(map_),
            0,
            [] (auto partial_sum, const KeyValueContainer::value_type& p) {
                return partial_sum + p.second;
            });
    }

    StoreValue max_value() const {
        return std::accumulate(
            std::begin(map_),
            std::end(map_),
            std::numeric_limits<StoreValue>::min(),
            [] (auto partial_max, const KeyValueContainer::value_type& p) {
                return std::max(partial_max, p.first);
            });
    }

    StoreValue min_value() const {
        return std::accumulate(
            std::begin(map_),
            std::end(map_),
            std::numeric_limits<StoreValue>::max(),
            [] (auto partial_min, const KeyValueContainer::value_type& p) {
                return std::min(partial_min, p.first);
            });
    }

    StoreValue operator[] (Index key) {
        return map_[key];
    }

 public:
    KeyValueContainer map_;
};

template <typename ConcreteStore>
class StoreTest : public ::testing::Test {
 protected:
    StoreTest() = default;

    virtual void test_values(const ConcreteStore& store,
                             const StoreValues& values) = 0;

    virtual void test_store(const StoreValues& values) = 0;
    virtual void test_merging(const StoreValueList& values_list) = 0;

    /* Test no values */
    void test_empty() {
        test_store({});
    }

    /* Test a constant stream of values */
    void test_constant() {
        constexpr auto kNumValues = 10000;
        constexpr RealValue zero_value = 0;

        StoreValues constant_values(kNumValues, zero_value);

        test_store(constant_values);
    }

    /* Test a stream of increasing values */
    void test_increasing_linearly() {
        constexpr auto kNumValues = 10000;

        StoreValues values(kNumValues);

        std::generate(
            values.begin(),
            values.end(),
            [n = 0] () mutable {
                return n++;
            });

        test_store(values);
    }

    /* Test a stream of decreasing values */
    void test_decreasing_linearly() {
        constexpr auto kNumValues = 10000;

        StoreValues values(kNumValues);

        std::generate(
            values.begin(),
            values.end(),
            [n = kNumValues - 1] () mutable {
                return n--;
            });

        test_store(values);
    }

    /* Test a stream of values increasing exponentially */
    void test_increasing_exponentially() {
        constexpr auto kNumValues = 16;

        StoreValues values(kNumValues);

        std::generate(
            values.begin(),
            values.end(),
            [power = 0] () mutable {
                return std::pow(2, power++);
            });

        test_store(values);
    }

    /* Test a stream of values decreasing exponentially */
    void test_decreasing_exponentially() {
        constexpr auto kNumValues = 16;

        StoreValues values(kNumValues);

        std::generate(
            values.begin(),
            values.end(),
            [power = kNumValues - 1] () mutable {
                return std::pow(2, power--);
            });

        test_store(values);
    }

    /* Test bin counts for positive and negative numbers */
    void test_bin_counts() {
        StoreValues values;

        for (auto x = 0; x < 10; ++x) {
            for (auto i = 0; i < 2 * x; ++i)
                values.push_back(x);
        }
        test_store(values);

        values.clear();
        for (auto x = 0; x < 10; ++x) {
            for (auto i = 0; i < 2 * x; ++i)
                values.push_back(-x);
        }
        test_store(values);
    }

    /* Test extreme values */
    virtual void test_extreme_values() {
        test_store({kExtremeMax});
        test_store({kExtremeMin});
        test_store({0, kExtremeMin});
        test_store({0, kExtremeMax});
        test_store({kExtremeMin, kExtremeMax});
        test_store({kExtremeMax, kExtremeMin});
    }

    /* Test merging empty stores */
    void test_merging_empty() {
        test_merging({{}, {}});
    }

    /* Test merging stores with values that are fare apart */
    void test_merging_far_apart() {
        constexpr auto kBigValue = 10000;

        test_merging({{-kBigValue}, {kBigValue}});
        test_merging({{kBigValue}, {-kBigValue}});
        test_merging({{kBigValue}, {-kBigValue}, {0}});
        test_merging({{kBigValue, 0}, {-kBigValue}, {0}});
    }

    /* Test merging stores with the same constants */
    void test_merging_constant() {
        test_merging({{2, 2}, {2, 2, 2}, {2}});
        test_merging({{-8, -8}, {-8}});
    }

    /* Test merging stores with extreme values */
    virtual void test_merging_extreme_values() {
        test_merging({{0}, {kExtremeMin}});
        test_merging({{0}, {kExtremeMax}});
        test_merging({{kExtremeMin}, {0}});
        test_merging({{kExtremeMax}, {0}});
        test_merging({{kExtremeMin}, {kExtremeMin}});
        test_merging({{kExtremeMax}, {kExtremeMax}});
        test_merging({{kExtremeMin}, {kExtremeMax}});
        test_merging({{kExtremeMax}, {kExtremeMin}});
        test_merging({{0}, {kExtremeMin, kExtremeMax}});
        test_merging({{kExtremeMin, kExtremeMax}, {0}});
    }

    /* Test copying empty stores */
    void test_copying_empty() {
    }

    /* Test copying stores */
    void test_copying_non_empty() {
    }

    virtual ~StoreTest() = default;

    StoreValues flatten(const StoreValueList& values_list) {
        StoreValues result;

        /* Reserve the total size in advance */
        auto total_size =
            std::accumulate(
                std::begin(values_list),
                std::end(values_list),
                0,
                [] (auto partial_sum, const auto& store_values) {
                    return partial_sum + store_values.size();
                });
        result.reserve(total_size);

        for (const auto& store_values : values_list)
            result.insert(
                result.end(), store_values.begin(), store_values.end());

        return result;
    }

    static constexpr auto kExtremeMax = std::numeric_limits<StoreValue>::max();
    static constexpr auto kExtremeMin = std::numeric_limits<StoreValue>::min();
};

class DenseStoreTest : public StoreTest<DenseStore> {
 protected:
    /* Test that key_at_rank properly handles decimal ranks */
    void test_key_at_rank() {
        auto store = DenseStore();

        store.add(4);
        store.add(10);
        store.add(100);

        EXPECT_EQ(store.key_at_rank(0), 4);
        EXPECT_EQ(store.key_at_rank(1), 10);
        EXPECT_EQ(store.key_at_rank(2), 100);
        EXPECT_EQ(store.key_at_rank(0, false), 4);
        EXPECT_EQ(store.key_at_rank(1, false), 10);
        EXPECT_EQ(store.key_at_rank(2, false), 100);
        EXPECT_EQ(store.key_at_rank(0.5), 4);
        EXPECT_EQ(store.key_at_rank(1.5), 10);
        EXPECT_EQ(store.key_at_rank(2.5), 100);
        EXPECT_EQ(store.key_at_rank(-0.5, false), 4);
        EXPECT_EQ(store.key_at_rank(0.5, false), 10);
        EXPECT_EQ(store.key_at_rank(1.5, false), 100);
    }

    void test_values(const DenseStore& store,
                     const StoreValues& values) override {
        auto counter = Counter(values);

        auto expected_total_count = counter.sum_values();
        EXPECT_EQ(expected_total_count, store.bins().sum());

        if (expected_total_count == 0) {
            EXPECT_EQ(store.bins().has_only_zeros(), true);
        } else {
            EXPECT_EQ(store.bins().has_only_zeros(), false);

            auto idx = 0;
            for (const auto& item : store.bins()) {
                if (item != 0)
                    EXPECT_EQ(counter[idx + store.offset()], item);
                ++idx;
            }
        }
    }

    void test_store(const StoreValues& store_values) override {
        auto store = DenseStore();

        for (const auto& value : store_values)
            store.add(value);

        test_values(store, store_values);
    }

    void test_merging(const StoreValueList& store_values_list) override {
        auto store = DenseStore();

        for (const auto& store_values : store_values_list) {
            auto intermediate_store = DenseStore();

            for (const auto& value : store_values)
                intermediate_store.add(value);

            store.merge(intermediate_store);
        }

        test_values(store, flatten(store_values_list));
    }

    void test_extreme_values() override {
        /*
         * DenseStore is not meant to be used with values that are extremely
         * far from one another as it would allocate an excessively large array
         */
    }

    void test_merging_extreme_values() override {
        /*
         * DenseStore is not meant to be used with values that are extremely
         * far from one another as it would allocate an excessively large array
         */
    }
};

TEST_F(DenseStoreTest, TestEmpty) {
    test_empty();
}

TEST_F(DenseStoreTest, TestConstant) {
    test_constant();
}

TEST_F(DenseStoreTest, TestIncreasingLinearly) {
    test_increasing_linearly();
}

TEST_F(DenseStoreTest, TestDecreasingLinearly) {
    test_decreasing_linearly();
}

TEST_F(DenseStoreTest, TestIncreasingExponentially) {
    test_increasing_exponentially();
}

TEST_F(DenseStoreTest, TestDecreasingExponentially) {
    test_decreasing_exponentially();
}

TEST_F(DenseStoreTest, TestBinCounts) {
    test_bin_counts();
}

TEST_F(DenseStoreTest, TestExtremeValues) {
    test_extreme_values();
}

TEST_F(DenseStoreTest, TestMergingEmpty) {
    test_merging_empty();
}

TEST_F(DenseStoreTest, TestMergingConstant) {
    test_merging_constant();
}

TEST_F(DenseStoreTest, TestMergingExtremeValues) {
    test_merging_extreme_values();
}

TEST_F(DenseStoreTest, TestCopyingEmpty) {
    test_copying_empty();
}

TEST_F(DenseStoreTest, TestCopyingNonEmpty) {
    test_copying_non_empty();
}

TEST_F(DenseStoreTest, TestKeyAtRank) {
    test_key_at_rank();
}

class CollapsingLowestDenseStoreTest
    : public StoreTest<CollapsingLowestDenseStore> {
 protected:
    void test_values(const CollapsingLowestDenseStore& store,
                     const StoreValues& values) {
        auto counter = Counter(values);

        auto expected_total_count = counter.sum_values();
        EXPECT_EQ(expected_total_count, store.bins().sum());

        if (expected_total_count == 0) {
            EXPECT_EQ(store.bins().has_only_zeros(), true);
        } else {
            EXPECT_EQ(store.bins().has_only_zeros(), false);


            auto max_index = counter.max_value();
            auto min_storable_index =
                std::max(
                    std::numeric_limits<StoreValue>::min(),
                    max_index - store.bin_limit() + 1);
            auto normalized_values =
                normalize_smaller_values(values, min_storable_index);
            auto normalized_counter = Counter(normalized_values);

            auto idx = 0;
            for (const auto& sbin : store.bins()) {
                if (sbin != 0)
                    EXPECT_EQ(normalized_counter[idx + store.offset()], sbin);
                ++idx;
            }
        }
    }

    void test_store(const StoreValues& values) override {
        auto test_bin_limits = {1, 20, 1000};

        for (const auto bin_limit : test_bin_limits) {
            auto store = CollapsingLowestDenseStore(bin_limit);

            for (const auto& value : values)
                store.add(value);

            test_values(store, values);
        }
    }

    void test_merging(const StoreValueList& store_values_list) override {
        auto test_bin_limits = {1, 20, 1000};

        for (const auto bin_limit : test_bin_limits) {
            auto store = CollapsingLowestDenseStore(bin_limit);

            for (const auto& store_values : store_values_list) {
                auto intermediate_store = CollapsingLowestDenseStore(bin_limit);

                for (const auto& value : store_values)
                    intermediate_store.add(value);

                store.merge(intermediate_store);
            }

            test_values(store, flatten(store_values_list));
        }
    }

    static StoreValues normalize_smaller_values(const StoreValues& store_values,
                                                Index min_storable_index) {
        auto result = StoreValues();

        std::transform(
            store_values.begin(),
            store_values.end(),
            std::back_inserter(result),
            [min_storable_index](StoreValue value) -> StoreValue {
                return std::max(value, min_storable_index);
            });

        return result;
    }
};

TEST_F(CollapsingLowestDenseStoreTest, TestEmpty) {
    test_empty();
}

TEST_F(CollapsingLowestDenseStoreTest, TestConstant) {
    test_constant();
}

TEST_F(CollapsingLowestDenseStoreTest, TestIncreasingLinearly) {
    test_increasing_linearly();
}

TEST_F(CollapsingLowestDenseStoreTest, TestDecreasingLinearly) {
    test_decreasing_linearly();
}

TEST_F(CollapsingLowestDenseStoreTest, TestIncreasingExponentially) {
    test_increasing_exponentially();
}

TEST_F(CollapsingLowestDenseStoreTest, TestDecreasingExponentially) {
    test_decreasing_exponentially();
}

TEST_F(CollapsingLowestDenseStoreTest, TestBinCounts) {
    test_bin_counts();
}

TEST_F(CollapsingLowestDenseStoreTest, TestMergingEmpty) {
    test_merging_empty();
}

TEST_F(CollapsingLowestDenseStoreTest, TestMergingConstant) {
    test_merging_constant();
}

TEST_F(CollapsingLowestDenseStoreTest, TestCopyingEmpty) {
    test_copying_empty();
}

TEST_F(CollapsingLowestDenseStoreTest, TestCopyingNonEmpty) {
    test_copying_non_empty();
}

class CollapsingHighestDenseStoreTest
    : public StoreTest<CollapsingHighestDenseStore> {
 private:
    void test_values(const CollapsingHighestDenseStore& store,
                     const StoreValues& values) {
        auto counter = Counter(values);

        auto expected_total_count = counter.sum_values();
        EXPECT_EQ(expected_total_count, store.bins().sum());

        if (expected_total_count == 0) {
            EXPECT_EQ(store.bins().has_only_zeros(), true);
        } else {
            EXPECT_EQ(store.bins().has_only_zeros(), false);

            auto min_index = counter.min_value();
            auto max_storable_index =
                std::min(
                    std::numeric_limits<StoreValue>::max(),
                    min_index + store.bin_limit() - 1);

            auto normalized_values =
                normalize_bigger_values(values, max_storable_index);

            auto normalized_counter = Counter(normalized_values);

            auto idx = 0;
            for (const auto& item : store.bins()) {
                if (item != 0)
                    EXPECT_EQ(normalized_counter[idx + store.offset()], item);
                ++idx;
            }
        }
    }

    void test_store(const StoreValues& values) override {
        auto test_bin_limits = {20};

        for (const auto bin_limit : test_bin_limits) {
            auto store = CollapsingHighestDenseStore(bin_limit);

            for (const auto& value : values)
                store.add(value);

            test_values(store, values);
        }
    }

    void test_merging(const StoreValueList& store_values_list) override {
        auto test_bin_limits = {1, 20, 1000};

        for (const auto bin_limit : test_bin_limits) {
            auto store = CollapsingHighestDenseStore(bin_limit);

            for (const auto& store_values : store_values_list) {
                auto intermediate_store =
                    CollapsingHighestDenseStore(bin_limit);

                for (const auto& value : store_values)
                    intermediate_store.add(value);

                store.merge(intermediate_store);
            }

            test_values(store, flatten(store_values_list));
        }
    }

    static StoreValues normalize_bigger_values(const StoreValues& store_values,
                                               Index max_storable_index) {
        auto result = StoreValues();

        std::transform(
            store_values.begin(),
            store_values.end(),
            std::back_inserter(result),
            [max_storable_index](StoreValue value) -> StoreValue {
                return std::min(value, max_storable_index);
            });

        return result;
    }
};

TEST_F(CollapsingHighestDenseStoreTest, TestConstant) {
    test_constant();
}

TEST_F(CollapsingHighestDenseStoreTest, TestIncreasingLinearly) {
    test_increasing_linearly();
}

TEST_F(CollapsingHighestDenseStoreTest, TestDecreasingLinearly) {
    test_decreasing_linearly();
}

TEST_F(CollapsingHighestDenseStoreTest, TestIncreasingExponentially) {
    test_increasing_exponentially();
}

TEST_F(CollapsingHighestDenseStoreTest, TestDecreasingExponentially) {
    test_decreasing_exponentially();
}

TEST_F(CollapsingHighestDenseStoreTest, TestBinCounts) {
    test_bin_counts();
}

TEST_F(CollapsingHighestDenseStoreTest, TestMergingEmpty) {
    test_merging_empty();
}

TEST_F(CollapsingHighestDenseStoreTest, TestMergingConstant) {
    test_merging_constant();
}

TEST_F(CollapsingHighestDenseStoreTest, TestCopyingEmpty) {
    test_copying_empty();
}

TEST_F(CollapsingHighestDenseStoreTest, TestCopyingNonEmpty) {
    test_copying_non_empty();
}

template <typename ConcreteDDSketch>
class SketchSummary {
 public:
    using SketchContainer = std::vector<RealValue>;
    using iterator = typename SketchContainer::iterator;
    using const_iterator = typename SketchContainer::const_iterator;

    SketchSummary(ConcreteDDSketch& sketch,
                  const std::vector<RealValue>& quantiles) {
        for (const auto& quantile : quantiles) {
            auto computed_quantile = sketch.get_quantile_value(quantile);
            summary_.push_back(computed_quantile);
        }

        summary_.push_back(sketch.sum());
        summary_.push_back(sketch.avg());
        summary_.push_back(sketch.num_values());
    }

    void assert_almost_equal(const SketchSummary<ConcreteDDSketch>& other) {
        std::equal(summary_.begin(),
                   summary_.end(),
                   other.summary_.begin(),
                   [](const RealValue& left, const RealValue& right) {
                       EXPECT_ALMOST_EQ(left, right);
                       return left == right;
                   });
    }

 private:
    SketchContainer summary_;
};

template <typename ConcreteDDSketch>
class BaseDDSketchTest : public ::testing::Test, CRTP<ConcreteDDSketch> {
 protected:
    virtual ~BaseDDSketchTest() = default;

    /* Create a new DDSketch of the appropriate type */
    virtual ConcreteDDSketch create_ddsketch() = 0;

    void evaluate_sketch_accuracy(ConcreteDDSketch& sketch,
                                  const DataSet<RealValue>& dataset,
                                  RealValue eps,
                                  bool summary_stats = true) {
    auto test_quantiles =
        {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0};

         for (const auto quantile : test_quantiles) {
             auto sketch_q = sketch.get_quantile_value(quantile);
             auto data_q = dataset.quantile(quantile);
             auto err = std::abs(sketch_q - data_q);

             EXPECT_TRUE(err - eps * std::abs(data_q) <= 1.0e-15);
         }

         EXPECT_EQ(sketch.num_values(), dataset.len());

         if (summary_stats) {
             EXPECT_ALMOST_EQ(sketch.sum(), dataset.sum());
             EXPECT_ALMOST_EQ(sketch.avg(), dataset.avg());
         }
    }

    /* Test DDSketch on values from various distributions */
    void test_distributions() {
        const auto& test_datasets = get_datasets();

        for (auto& dataset : test_datasets) {
            for (const auto size : {3, 5, 10, 100, 1000}) {
                dataset->populate(size);

                auto sketch = create_ddsketch();
                for (const auto& value : *dataset)
                    sketch.add(value);

                evaluate_sketch_accuracy(
                    sketch, *dataset, kTestRelativeAccuracy);
            }
        }
    }

    /* Test DDSketch on adding integer weighted values */
    void test_add_multiple() {
        auto dataset = Integers();
        dataset.populate(1000);

        StoreValues dataset_values;
        for (const auto& value : dataset)
            dataset_values.push_back(value);

        auto sketch = create_ddsketch();
        for (const auto& pair : Counter(dataset_values)) {
            auto value = pair.first;
            auto count = pair.second;

            sketch.add(value, count);
        }

        evaluate_sketch_accuracy(sketch, dataset, kTestRelativeAccuracy);
    }

    /* Test DDSketch on adding decimal weighted values */
    void test_add_decimal() {
        auto sketch = create_ddsketch();

        for (auto value = 0; value < 100; ++value)
            sketch.add(value, 1.1);

        sketch.add(100, 110.0);

        auto data_median = 99;
        auto sketch_median = sketch.get_quantile_value(0.5);
        auto err = std::abs(sketch_median - data_median);

        EXPECT_TRUE(
            err - kTestRelativeAccuracy * std::abs(data_median) <= 1.0e-15);
        EXPECT_ALMOST_EQ(sketch.num_values(), 110 * 2);
        EXPECT_ALMOST_EQ(sketch.sum(), 5445 + 11000);
        EXPECT_ALMOST_EQ(sketch.avg(), 74.75);
    }

    /* Test merging equal-sized DDSketches */
    void test_merge_equal() {
        std::vector<std::pair<RealValue, RealValue>> normal_parameters =
             {{35, 1}, {1, 3}, {15, 2}, {40, 0.5}};

        for (const auto size : {3, 5, 10, 100, 1000}) {
            auto dataset = EmptyDataSet();
            auto target_sketch = create_ddsketch();

            for (const auto& params : normal_parameters) {
                auto generator = Normal(params.first, params.second);
                generator.populate(size);

                auto sketch = create_ddsketch();
                for (const auto& value : generator) {
                    sketch.add(value);
                    dataset.add(value);
                }
                target_sketch.merge(sketch);

                evaluate_sketch_accuracy(
                    target_sketch, dataset, kTestRelativeAccuracy);
            }

            evaluate_sketch_accuracy(
                target_sketch, dataset, kTestRelativeAccuracy);
        }
    }

    /* Test merging variable-sized DDSketches */
    void test_merge_unequal() {
        constexpr auto num_tests = 20;

         std::random_device random_device;
         std::mt19937_64 generator(random_device());
         std::uniform_real_distribution<DataValue> random(0, 1);

        for (auto test_id = 0; test_id < num_tests; ++test_id) {
            for (const auto size : {3, 5, 10, 100, 1000}) {
                auto dataset = Lognormal();
                dataset.populate(size);

                auto sketch1 = create_ddsketch();
                auto sketch2 = create_ddsketch();

                for (const auto& value : dataset) {
                    if (random(generator) > 0.7)
                        sketch1.add(value);
                    else
                        sketch2.add(value);
                }

                sketch1.merge(sketch2);

                evaluate_sketch_accuracy(
                    sketch1, dataset, kTestRelativeAccuracy);
            }
        }
    }

    /* Test merging DDSketches of different distributions */
    void test_merge_mixed() {
       constexpr auto num_tests = 20;

        std::vector<std::unique_ptr<GenericDataSet>> test_datasets;
        test_datasets.emplace_back(std::make_unique<Normal>());
        test_datasets.emplace_back(std::make_unique<Exponential>());
        test_datasets.emplace_back(std::make_unique<Laplace>());
        test_datasets.emplace_back(std::make_unique<Bimodal>());

        std::random_device random_device;
        std::mt19937_64 generator(random_device());
        std::uniform_real_distribution<RealValue> random(0, 500);

        for (auto test_id = 0; test_id < num_tests; ++test_id) {
            auto merged_dataset = EmptyDataSet();
            auto merged_sketch = create_ddsketch();

            for (auto& dataset : test_datasets) {
                Index dataset_size = static_cast<Index>(random(generator));
                dataset->populate(dataset_size);
                auto sketch = create_ddsketch();

                for (const auto& value : *dataset) {
                    sketch.add(value);
                    merged_dataset.add(value);
                }

                merged_sketch.merge(sketch);
                evaluate_sketch_accuracy(
                    merged_sketch, merged_dataset, kTestRelativeAccuracy);
            }
        }
    }

    /* Test that merge() calls do not modify the argument sketch */
    void test_consistent_merge() {
        std::vector<RealValue> kTestQuantiles =
            {0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 1.0};

        auto sketch1 = create_ddsketch();
        auto sketch2 = create_ddsketch();

        auto dataset1 = Normal();
        dataset1.populate(100);

        for (const auto& value : dataset1)
            sketch1.add(value);

        sketch1.merge(sketch2);

        /* Sketch2 is still empty */
        EXPECT_EQ(sketch2.num_values(), 0);

        auto dataset2 = Normal();
        dataset2.populate(50);

        for (const auto& value : dataset2)
            sketch2.add(value);

        auto sketch2_summary =
            SketchSummary<ConcreteDDSketch>(sketch2, kTestQuantiles);
        sketch1.merge(sketch2);

        auto dataset3 = Normal();
        dataset3.populate(10);

        for (const auto& value : dataset3)
            sketch1.add(value);

        /* Changes to sketch1 does not affect sketch2 after merge */
        sketch2_summary =
            SketchSummary<ConcreteDDSketch>(sketch2, kTestQuantiles);

        auto sketch2_summary_tmp =
            SketchSummary<ConcreteDDSketch>(sketch2, kTestQuantiles);
        sketch2_summary.assert_almost_equal(sketch2_summary_tmp);

        auto sketch3 = create_ddsketch();
        sketch3.merge(sketch2);

        /* Merging to an empty sketch does not change sketch2 */
        sketch2_summary_tmp =
            SketchSummary<ConcreteDDSketch>(sketch2, kTestQuantiles);
        sketch2_summary.assert_almost_equal(sketch2_summary_tmp);
    }

    auto get_datasets() {
        std::vector<std::unique_ptr<GenericDataSet>> test_datasets;
        test_datasets.emplace_back(std::make_unique<UniformForward>());
        test_datasets.emplace_back(std::make_unique<UniformBackward>());
        test_datasets.emplace_back(std::make_unique<UniformZoomIn>());
        test_datasets.emplace_back(std::make_unique<UniformZoomOut>());
        test_datasets.emplace_back(std::make_unique<UniformSqrt>());
        test_datasets.emplace_back(std::make_unique<Constant>());
        test_datasets.emplace_back(std::make_unique<NegativeUniformBackward>());
        test_datasets.emplace_back(std::make_unique<NegativeUniformForward>());
        test_datasets.emplace_back(std::make_unique<NumberLineBackward>());
        test_datasets.emplace_back(std::make_unique<NumberLineForward>());
        test_datasets.emplace_back(std::make_unique<Exponential>());
        test_datasets.emplace_back(std::make_unique<Lognormal>());
        test_datasets.emplace_back(std::make_unique<Normal>());
        test_datasets.emplace_back(std::make_unique<Laplace>());
        test_datasets.emplace_back(std::make_unique<Bimodal>());
        test_datasets.emplace_back(std::make_unique<Trimodal>());
        test_datasets.emplace_back(std::make_unique<Mixed>());
        test_datasets.emplace_back(std::make_unique<Integers>());

        return test_datasets;
    }

    static constexpr RealValue kTestRelativeAccuracy = 0.05;
    static constexpr Index kTestBinLimit = 1024;
};

class DDSketchTest : public BaseDDSketchTest<DDSketch> {
 protected:
    DDSketch create_ddsketch() {
        return DDSketch(kTestRelativeAccuracy);
    }
};

TEST_F(DDSketchTest, TestDistributions) {
    test_distributions();
}

TEST_F(DDSketchTest, TestAddMultiple) {
    test_add_multiple();
}

TEST_F(DDSketchTest, TestAddDecimal) {
    test_add_decimal();
}

TEST_F(DDSketchTest, TestMergeEqual) {
     test_merge_equal();
}

TEST_F(DDSketchTest, TestMergeUnequal) {
    test_merge_unequal();
}

TEST_F(DDSketchTest, TestMergeMixed) {
     test_merge_mixed();
}

TEST_F(DDSketchTest, TestConsistentMerge) {
    test_consistent_merge();
}

class TestLogCollapsingLowestDenseDDSketch
    : public BaseDDSketchTest<LogCollapsingLowestDenseDDSketch> {
 protected:
    LogCollapsingLowestDenseDDSketch create_ddsketch() override {
        return LogCollapsingLowestDenseDDSketch(
                                kTestRelativeAccuracy, kTestBinLimit);
    }
};

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestDistributions) {
    test_distributions();
}

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestAddMultiple) {
    test_add_multiple();
}

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestAddDecimal) {
    test_add_decimal();
}

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestMergeEqual) {
    test_merge_equal();
}

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestMergeUnequal) {
    test_merge_unequal();
}

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestMergeMixed) {
    test_merge_mixed();
}

TEST_F(TestLogCollapsingLowestDenseDDSketch, TestConsistentMerge) {
    test_consistent_merge();
}

class TestLogCollapsingHighestDenseDDSketch
    : public BaseDDSketchTest<LogCollapsingHighestDenseDDSketch> {
 protected:
    LogCollapsingHighestDenseDDSketch create_ddsketch() override {
        return LogCollapsingHighestDenseDDSketch(
                                kTestRelativeAccuracy, kTestBinLimit);
    }
};

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestDistributions) {
    test_distributions();
}

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestAddMultiple) {
    test_add_multiple();
}

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestAddDecimal) {
    test_add_decimal();
}

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestMergeEqual) {
    test_merge_equal();
}

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestMergeUnequal) {
    test_merge_unequal();
}

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestMergeMixed) {
    test_merge_mixed();
}

TEST_F(TestLogCollapsingHighestDenseDDSketch, TestConsistentMerge) {
    test_consistent_merge();
}

}  // namespace test
}  // namespace ddsketch

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}

