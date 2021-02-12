/*
 * Unless explicitly stated otherwise all files in this repository are licensed
 * under the Apache License 2.0.
 */
#ifndef INCLUDES_TEST_DATASETS_H_
#define INCLUDES_TEST_DATASETS_H_

#include <sstream>
#include <vector>
#include <string>
#include <memory>
#include <limits>
#include <random>
#include <functional>
#include <algorithm>

namespace ddsketch { namespace test {

using IntegerValue = int64_t;
using Index  = IntegerValue;
using DataValue = double;

template <typename Value = DataValue>
class DataSet {
 public:
    using DataSetValueContainer = std::vector<Value>;
    using iterator = typename DataSetValueContainer::iterator;
    using const_iterator = typename DataSetValueContainer::const_iterator;

    iterator begin() {
        return data_.begin();
    }

    iterator end() {
        return data_.end();
    }

    const_iterator begin() const {
        return data_.begin();
    }

    const_iterator end() const {
        return data_.end();
    }

    std::string to_string() const {
        std::ostringstream str_repr;

        str_repr << "Distribution: " << name() << " " <<
                    "Size: " << std::to_string(data_.size()) << "\n";

        str_repr << "[ ";

        for (const auto& value : data_)
            str_repr << value << " ";

        str_repr << " ]";

        str_repr << "\n";

        return str_repr.str();
    }

    auto len() const {
        return data_.size();
    }

    Index rank(Value value) const {
        auto tmp_data = data_;
        auto index = 0;

        sort(tmp_data.begin(), tmp_data.end(), std::less<Value>());

        auto it = std::lower_bound(tmp_data.begin(), tmp_data.end(), value);

        if (it == tmp_data.end())
            index = tmp_data.size() - 1;
        else
            index = std::distance(tmp_data.begin(), it);

        return index;
    }

    Value quantile(Value quantile) const {
        auto tmp_data = data_;

        sort(tmp_data.begin(), tmp_data.end(), std::less<Value>());
        Index item_rank = quantile * (data_.size() - 1);

        return tmp_data[item_rank];
    }

    Value sum() const {
        auto sum_of_elems =
            std::accumulate(
                data_.begin(), data_.end(), Value(0.0));

        return sum_of_elems;
    }

    Value avg() const {
        return sum() / len();
    }

    /* Populate data_ with values */
    virtual void populate(int size) = 0;

    /* Name of DataSet */
    virtual std::string name() const = 0;

    friend std::ostream& operator<<(std::ostream& os, const DataSet& dataset) {
        os << dataset.to_string();
        return os;
    }

    virtual ~DataSet() = default;

 protected:
    DataSet() = default;

    DataSetValueContainer data_;
};

class EmptyDataSet : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Empty";
    }

    void populate(int) override {
        /* Do Nothing */
    }

    void add(DataValue val) {
        data_.push_back(val);
    }

    void add_all(const DataSetValueContainer& values) {
        data_.insert(data_.begin(), values.begin(), values.end());
    }
};

class UniformForward : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Uniform_Forward";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [n = 0] () mutable {
                return n++;
            });
    }
};

class UniformBackward : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Uniform_Backward";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [n = size] () mutable {
                return n--;
            });
    }
};

class NegativeUniformForward : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Negative_Uniform_Forward";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [n = size] () mutable {
                return -n--;
            });
    }
};

class NegativeUniformBackward : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Negative_Uniform_Backward";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [n = 0] () mutable {
                return -n++;
            });
    }
};

class NumberLineForward : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Number_Line_Forward";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [n = -size / 2 + 1] () mutable {
                return -n++;
            });
    }
};

class NumberLineBackward : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Number_Line_Backward";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [n = size / 2] () mutable {
                return n--;
            });
    }
};

class UniformZoomIn : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Uniform_Zoom_In";
    }

    void populate(int size) override {
        data_.resize(size);
        auto idx = 0;

        for (int item = 0; item < size / 2; ++item) {
            data_[idx++] = item;
            data_[idx++] = size - item - 1;
        }

        if (size % 2)
            data_[idx++] = size / 2;
    }
};

class UniformZoomOut : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Uniform_Zoom_Out";
    }

    void populate(int size) override {
        data_.resize(size);

        auto idx = 0;

        if (size % 2) {
            data_[idx++] = size / 2;

            int half = size / 2;
            for (int item = 1; item < half + 1; ++item) {
                data_[idx++] = half + item;
                data_[idx++] = half - item;
            }
        } else {
            auto half = std::ceil(size / 2.0) - 0.5;
            auto upper_limit = static_cast<int>(half + 0.5);
            for (int item = 0; item < upper_limit; ++item) {
                data_[idx++] = static_cast<int>(half + item + 0.5);
                data_[idx++] = static_cast<int>(half - item - 0.5);
            }
        }
    }
};

class UniformSqrt : public DataSet<DataValue> {
 public:
    std::string name() const override {
        return "Uniform_Sqrt";
    }

    void populate(int size) override {
        data_.resize(size);

        auto idx = 0;
        auto t = static_cast<int>(std::sqrt(2 * size));
        auto initial_item = 0;
        auto initial_skip = 1;
        auto emitted = 0;
        auto i = 0;

        while (emitted < size) {
            auto item = initial_item;
            auto skip = initial_skip;

            for (int j = 0; j < t - i; ++j) {
                if (item < size) {
                    data_[idx++] = item;
                    emitted += 1;
                }
                item += skip;
                skip += 1;
            }

            if (t - i > 1) {
                initial_skip += 1;
                initial_item += initial_skip;
                i += 1;
            } else {
                initial_item += 1;
            }
        }
    }
};

class Constant : public DataSet<DataValue> {
 public:
    explicit Constant(DataValue constant = 42.0)
        : constant_(constant) {
    }

    std::string name() const override {
        return "Constant";
    }

    void populate(int size) override {
        data_.resize(size);

        std::generate(
            data_.begin(),
            data_.end(),
            [this] () { return constant_; });
    }

 private:
    DataValue constant_;
};

class Exponential : public DataSet<DataValue> {
 public:
    explicit Exponential(DataValue alpha = 100)
        : alpha_(alpha) {
        if (alpha <= 0)
            throw std::invalid_argument("Argument should be a positive number");
    }

    std::string name() const override {
        return "Exponential";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::exponential_distribution<DataValue> exponential(alpha_);

        std::generate(
            data_.begin(),
            data_.end(),
            [this, generator, exponential] () mutable {
                return exponential(generator);
            });
    }

 private:
    DataValue alpha_;
};

class Lognormal : public DataSet<DataValue> {
 public:
    Lognormal(DataValue mean = 0.0,
              DataValue sigma = 1.0,
              DataValue scale = 100.0)
        : mean_(mean), sigma_(sigma), scale_(scale) {
        if (scale <= 0)
            throw std::invalid_argument("Scale should be a positive number");
    }

    std::string name() const override {
        return "Lognormal";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::lognormal_distribution<DataValue> lognormal(mean_, sigma_);

        std::generate(
            data_.begin(),
            data_.end(),
            [this, generator, lognormal] () mutable {
                return lognormal(generator) / scale_;
            });
    }

 private:
    DataValue mean_;
    DataValue sigma_;
    DataValue scale_;
};

class Normal : public DataSet<DataValue> {
 public:
    explicit Normal(DataValue loc = 37.4,
                    DataValue scale = 1.0)
        : loc_(loc),
          scale_(scale) {
    }

    std::string name() const override {
        return "Normal";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::normal_distribution<DataValue> normal(loc_, scale_);

        std::generate(
            data_.begin(),
            data_.end(),
            [generator, normal] () mutable {
                return normal(generator);
            });
    }

 private:
    DataValue loc_;
    DataValue scale_;
};

class Laplace : public DataSet<DataValue> {
 public:
    explicit Laplace(DataValue loc = 11278.0,
                     DataValue scale = 100.0)
        : loc_(loc),
          scale_(scale) {
    }

    std::string name() const override {
        return "Laplace";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::uniform_real_distribution<DataValue> random(0.0, 1.0);

        std::generate(
            data_.begin(),
            data_.end(),
            [this, generator, random] () mutable {
                auto laplace_x = -std::log(1.0 - random(generator)) * scale_;

                if (random(generator) < 0.5)
                    laplace_x = -laplace_x;

                return laplace_x + loc_;
            });
    }

 private:
    DataValue loc_;
    DataValue scale_;
};

class Bimodal : public DataSet<DataValue> {
 public:
    explicit Bimodal(DataValue right_loc = 17.3,
                     DataValue left_loc = -2.0,
                     DataValue left_std = 3.0)
        :  right_loc_(right_loc),
           left_loc_(left_loc),
           left_std_(left_std) {
    }

    std::string name() const override {
        return "Bimodal";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::uniform_real_distribution<DataValue> random(0, 1);
        std::normal_distribution<DataValue> normal(left_loc_, left_std_);

        std::generate(
            data_.begin(),
            data_.end(),
            [this, generator, random, normal] () mutable {
                if (random(generator) > 0.5) {
                    auto laplace_x = -std::log(1.0 - random(generator));

                    if (random(generator) < 0.5)
                        laplace_x = -laplace_x;

                    return laplace_x + right_loc_;
                } else {
                    return normal(generator);
                }
            });
    }

 private:
    DataValue right_loc_;
    DataValue left_loc_;
    DataValue left_std_;
};

class Mixed : public DataSet<DataValue> {
 public:
    explicit Mixed(DataValue mean = 0.0,
                   DataValue sigma = 0.25,
                   DataValue scale_factor = 0.1,
                   DataValue loc = 10.0,
                   DataValue scale = 0.5,
                   DataValue ratio = 0.9)
        : mean_(mean),
          sigma_(sigma),
          scale_factor_(scale_factor),
          loc_(loc),
          scale_(scale),
          ratio_(ratio) {
    }

    std::string name() const override {
        return "Mixed";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::uniform_real_distribution<DataValue> random(0, 1);
        std::lognormal_distribution<DataValue> lognormal(mean_, sigma_);
        std::normal_distribution<DataValue> normal(loc_, scale_);

        std::generate(
            data_.begin(),
            data_.end(),
            [this, generator, random, lognormal, normal] () mutable {
                if (random(generator) < ratio_)
                    return scale_factor_ * lognormal(generator);
                else
                    return normal(generator);
            });
    }

 private:
    DataValue mean_;
    DataValue sigma_;
    DataValue scale_factor_;
    DataValue loc_;
    DataValue scale_;
    DataValue ratio_;
};

class Trimodal : public DataSet<DataValue> {
 public:
    explicit Trimodal(DataValue right_loc = 17.3,
                      DataValue left_loc = 5.0,
                      DataValue left_std = 1.0,
                      DataValue exp_scale = 0.01)
        : right_loc_(right_loc),
          left_loc_(left_loc),
          left_std_(left_std),
          exp_scale_(exp_scale) {
    }

    std::string name() const override {
        return "Trimodal";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937_64 generator(random_device());

        std::uniform_real_distribution<DataValue> random(0, 1);
        std::normal_distribution<DataValue> normal(left_loc_, left_std_);
        std::exponential_distribution<DataValue> exponential(exp_scale_);

        std::generate(
            data_.begin(),
            data_.end(),
            [this, generator, random, normal, exponential] () mutable {
                auto random_value = random(generator);

                if (random_value > 2.0 / 3.0) {
                    auto laplace_x = -std::log(1.0 - random(generator));

                    if (random(generator) < 0.5)
                        laplace_x = -laplace_x;

                    return laplace_x + right_loc_;
                } else if (random_value > 1.0 / 3.0) {
                    return normal(generator);
                } else {
                    return exponential(generator);
                }
            });
    }

 private:
    DataValue right_loc_;
    DataValue left_loc_;
    DataValue left_std_;
    DataValue exp_scale_;
};

class Integers : public DataSet<DataValue> {
 public:
    explicit Integers(DataValue loc = 4.3,
                      DataValue scale = 5.0)
        : loc_(loc),
          scale_(scale) {
    }

    std::string name() const override {
        return "Integers";
    }

    void populate(int size) override {
        data_.resize(size);

        std::random_device random_device;
        std::mt19937 generator(random_device());
        std::normal_distribution<DataValue> distribution(loc_, scale_);

        std::generate(
            data_.begin(),
            data_.end(),
            [generator, distribution] () mutable {
                return static_cast<IntegerValue>(distribution(generator));
            });
    }

 private:
    DataValue loc_;
    DataValue scale_;
};

using GenericDataSet = DataSet<DataValue>;

}  // namespace test
}  // namespace ddsketch

#endif  // INCLUDES_TEST_DATASETS_H_
