/*
 * Unless explicitly stated otherwise all files in this repository are licensed
 * under the Apache License 2.0.
 */

#ifndef INCLUDES_DDSKETCH_DDSKETCH_H_
#define INCLUDES_DDSKETCH_DDSKETCH_H_

#include <string>
#include <deque>
#include <sstream>
#include <algorithm>
#include <limits>
#include <numeric>

/*
 * A quantile sketch with relative-error guarantees. This sketch computes
 * quantile values with an approximation error that is relative to the actual
 * quantile value. It works on both negative and non-negative input values.
 *
 * For instance, using DDSketch with a relative accuracy guarantee set to 1%, if
 * the expected quantile value is 100, the computed quantile value is guaranteed to
 * be between 99 and 101. If the expected quantile value is 1000, the computed
 * quantile value is guaranteed to be between 990 and 1010.
 * DDSketch works by mapping floating-point input values to bins and counting the
 * number of values for each bin. The underlying structure that keeps track of bin
 * counts is store.
 *
 * The memory size of the sketch depends on the range that is covered by the input
 * values: the larger that range, the more bins are needed to keep track of the
 * input values. As a rough estimate, if working on durations with a relative
 * accuracy of 2%, about 2kB (275 bins) are needed to cover values between 1
 * millisecond and 1 minute, and about 6kB (802 bins) to cover values between 1
 * nanosecond and 1 day.
 *
 * The size of the sketch can be have a fail-safe upper-bound by using collapsing
 * stores. As shown in http://www.vldb.org/pvldb/vol12/p2195-masson.pdf
 * the likelihood of a store collapsing when using the default bound is vanishingly
 * small for most data.
 *
 * DDSketch implementations are also available in:
 *     https://github.com/DataDog/sketches-go/
 *     https://github.com/DataDog/sketches-py/
 *     https://github.com/DataDog/sketches-js/
 */

namespace ddsketch {

using RealValue = double;
using Index = int64_t;

static constexpr Index kChunkSize = 128;

template <typename BinItem>
class BinList {
 public:
    using Container = std::deque<BinItem>;
    using iterator = typename Container::iterator;
    using const_iterator = typename Container::const_iterator;
    using reference = BinItem&;
    using const_reference = const BinItem&;

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

    BinList() = default;

    explicit BinList(size_t size) {
        initialize_with_zeros(size);
    }

    BinList(const BinList<BinItem>& bins) : data_(bins.data_) {
    }

    friend std::ostream& operator<<(std::ostream& os, const BinList& bins) {
        for (auto & elem : bins)
            os << elem << " ";

        return os;
    }

    size_t size() const {
        return data_.size();
    }

    reference operator[] (int idx) {
        return data_[idx];
    }

    const_reference operator[] (int idx) const {
        return data_[idx];
    }

    reference first() {
        return data_[0];
    }

    reference last() {
        return data_[size() - 1];
    }

    void insert(BinItem elem) {
        data_.push_back(elem);
    }

    BinItem collapsed_count(int start_idx, int end_idx) const {
        if (index_outside_bounds(start_idx) || index_outside_bounds(end_idx))
            throw std::invalid_argument("Indexes out of bounds");

        return std::accumulate(
                   data_.begin() + start_idx,
                   data_.begin() + end_idx,
                   typename decltype(data_)::value_type(0));
    }

    bool has_only_zeros() const {
        auto non_zero_item =
            std::find_if(
                    data_.begin(),
                    data_.end(),
                    [](const auto& item) {
                        return item != 0;
                    });

        return non_zero_item == data_.end();
    }

    BinItem sum() const {
        return collapsed_count(0, data_.size());
    }

    void initialize_with_zeros(size_t num_zeros) {
        auto trailing_zeros = Container(num_zeros, 0);

        data_ = trailing_zeros;
    }

    void extend_front_with_zeros(size_t count) {
       auto trailing_zeros = Container(count, 0);

       data_.insert(
           data_.begin(),
           trailing_zeros.begin(),
           trailing_zeros.end());
    }

    void extend_back_with_zeros(size_t count) {
       auto trailing_zeros = Container(count, 0);

       data_.insert(
           data_.end(),
           trailing_zeros.begin(),
           trailing_zeros.end());
    }

    void remove_trailing_elements(size_t count) {
        data_.erase(data_.end() - count, data_.end());
    }

    void remove_leading_elements(size_t count) {
        data_.erase(data_.begin(), data_.begin() + count);
    }

    void replace_range_with_zeros(int start_idx,
                                  int end_idx,
                                  size_t num_zeros) {
        auto zeros = Container(num_zeros, 0);

        data_.erase(data_.begin() + start_idx, data_.begin() + end_idx);
        data_.insert(data_.begin() + start_idx, zeros.begin(), zeros.end());
    }

 private:
    bool index_outside_bounds(size_t idx) const {
        return idx > size();
    }

    Container data_;
};

template <typename T>
struct CRTP {
    T& underlying() {
        return static_cast<T&>(*this);
    }

    T const& underlying() const {
        return static_cast<T const&>(*this);
    }
};

/* The basic specification of a store */
template <class ConcreteStore>
class BaseStore : CRTP<ConcreteStore> {
 public:
    /* Copy the input store into this one */
    void copy(const ConcreteStore &store) {
        this->underlying()->copy(store);
    }

    /* The number of bins */
    Index length() {
        return this->underlying()->length();
    }

    bool is_empty() {
        return this->underlying()->is_empty();
    }

    /*
     * Updates the counter at the specified index key,
     * growing the number of bins if necessary.
     */
    void add(RealValue key, RealValue weight) {
        this->underlying()->add(key, weight);
    }

    void add(RealValue key) {
        this->underlying()->add(key, 1.0);
    }

    /*
     * Return the key for the value at given rank
     *
     *  E.g., if the non-zero bins are [1, 1] for keys a, b with no offset
     *
     *  if (lower == true) {
     *       key_at_rank(x) = a for x in [0, 1)
     *       key_at_rank(x) = b for x in [1, 2)
     *  }
     *  if (lower == false) {
     *       key_at_rank(x) = a for x in (-1, 0]
     *       key_at_rank(x) = b for x in (0, 1]
     *  }
     */
    Index key_at_rank(RealValue rank, bool lower = false) const {
        return this->underlying()->key_at_rank(rank, lower);
    }

    /*
     * Merge another store into this one. This should be equivalent as running the
     * add operations that have been run on the other store on this one.
     */
    void merge(const ConcreteStore& store) {
        return this->underlying()->merge(store);
    }

 protected:
     BaseStore() = default;
    ~BaseStore() = default;
};

/*
 * A dense store that keeps all the bins between the bin for the min_key
 * and the bin for the max_key.
 */
template <class ConcreteStore = void>
class BaseDenseStore : public BaseStore<BaseDenseStore<ConcreteStore>> {
 public:
    explicit BaseDenseStore(Index chunk_size = kChunkSize)
    : count_(0),
      min_key_(std::numeric_limits<Index>::max()),
      max_key_(std::numeric_limits<Index>::min()),
      chunk_size_(chunk_size),
      offset_(0) {
    }

    std::string to_string() const {
        std::ostringstream repr;

        repr <<  "{";

        Index i = 0;
        for (const auto& sbin : bins_)
            repr << i++ + offset_ << ": " << sbin << ", ";

        repr << "}, ";

        repr << "min_key:" << min_key_
             << ", max_key:" << max_key_
             << ", offset:" << offset_;

        return repr.str();
    }

    void copy(const BaseDenseStore& store) {
        count_ = store.count_;
        min_key_ = store.min_key_;
        max_key_ = store.max_key_;
        offset_ = store.offset_;
        bins_ = store.bins_;
    }

    const BinList<RealValue>& bins() const {
        return bins_;
    }

    Index offset() const {
        return offset_;
    }

    RealValue count() const {
        return count_;
    }

    Index length() const {
        return bins_.size();
    }

    bool is_empty() const {
        return length() == kEmptyStoreLength;
    }

    void add(Index key, RealValue weight = 1.0) {
        Index idx = get_index(key);

        bins_[idx] += weight;
        count_ += weight;
    }

    Index key_at_rank(RealValue rank, bool lower = true) const {
        auto running_ct = 0.0;

        auto idx = 0;
        for (const auto bin_ct : bins_) {
            running_ct += bin_ct;
            if ((lower && running_ct > rank) ||
                (!lower && running_ct >= rank + 1))
                return idx + offset_;
            ++idx;
        }

        return max_key_;
    }

    void merge(const BaseDenseStore& store) {
        if (store.count_ == 0)
            return;

        if (count_ == 0) {
            copy(store);
            return;
        }

        if (store.min_key_ < min_key_ || store.max_key_ > max_key_)
            extend_range(store.min_key_, store.max_key_);

        for (auto key = store.min_key_; key <= store.max_key_ ; ++key)
            bins_[key - offset_] += store.bins_[key - store.offset_];

        count_ += store.count_;
    }

 protected:
    virtual Index get_new_length(Index new_min_key, Index new_max_key) {
        auto desired_length = new_max_key - new_min_key + 1;
        auto num_chunks = std::ceil((1.0 * desired_length) / chunk_size_);

        return chunk_size_ * num_chunks;
    }

    /*
     * Adjust the bins, the offset, the min_key, and max_key, without resizing
     * the bins, in order to try making it fit the specified range
     */
    virtual void adjust(Index new_min_key, Index new_max_key) {
        center_bins(new_min_key, new_max_key);

        min_key_ = new_min_key;
        max_key_ = new_max_key;
    }

    /* Shift the bins; this changes the offset */
    void shift_bins(Index shift) {
        if (shift > 0) {
            bins_.remove_trailing_elements(shift);
            bins_.extend_front_with_zeros(shift);
        } else {
            auto abs_shift = std::abs(shift);

            bins_.remove_leading_elements(abs_shift);
            bins_.extend_back_with_zeros(abs_shift);
        }

        offset_ -= shift;
    }

    /* Center the bins; this changes the offset */
    void center_bins(Index new_min_key, Index new_max_key) {
        auto middle_key = new_min_key + (new_max_key - new_min_key + 1) / 2;

        shift_bins(offset_ + length() / 2 - middle_key);
    }

    /* Grow the bins as necessary and call _adjust */
    void extend_range(Index key, Index second_key) {
        auto new_min_key = std::min({key, second_key, min_key_});
        auto new_max_key = std::max({key, second_key, max_key_});

        if (is_empty()) {
            /* Initialize bins */
            auto new_length = get_new_length(new_min_key, new_max_key);
            bins_.initialize_with_zeros(new_length);
            offset_ = new_min_key;
            adjust(new_min_key, new_max_key);
        } else if (new_min_key >= min_key_ &&
                   new_max_key < offset_ + length()) {
            /* No need to change the range; just update min/max keys */
            min_key_ = new_min_key;
            max_key_ = new_max_key;
        } else {
            /* Grow the bins */
            Index new_length = get_new_length(new_min_key, new_max_key);

            if (new_length > length())
                bins_.extend_back_with_zeros(new_length - length());

            adjust(new_min_key, new_max_key);
        }
    }

    void extend_range(Index key) {
        extend_range(key, key);
    }

    /* Calculate the bin index for the key, extending the range if necessary */
    virtual Index get_index(Index key) {
        if (key < min_key_ || key > max_key_)
            extend_range(key);

        return key - offset_;
    }

 public:
    RealValue count_; /* The sum of the counts for the bins */
    Index min_key_;   /* The minimum key bin */
    Index max_key_;   /* The maximum key bin */

    /* The number of bins to grow by */
    Index chunk_size_;

    /* The difference btw the keys and the index in which they are stored */
    Index offset_;
    BinList<RealValue> bins_;

 private:
    static constexpr size_t kEmptyStoreLength = 0;
};

using DenseStore = BaseDenseStore<>;

/*
 * A dense store that keeps all the bins between the bin for the min_key and the
 * bin for the max_key, but collapsing the left-most bins if the number of bins
 * exceeds the bin_limit
 */
class CollapsingLowestDenseStore
    : public BaseDenseStore<CollapsingLowestDenseStore> {
 public:
    explicit CollapsingLowestDenseStore(Index bin_limit,
                                        Index chunk_size = kChunkSize)
    : BaseDenseStore(chunk_size),
      bin_limit_(bin_limit),
      is_collapsed_(false) {
    }

    Index bin_limit() const {
        return bin_limit_;
    }

    void copy(const CollapsingLowestDenseStore& store) {
        count_ = store.count_;
        min_key_ = store.min_key_;
        max_key_ = store.max_key_;
        offset_ = store.offset_;
        bins_ = store.bins_;

        bin_limit_ = store.bin_limit_;
        is_collapsed_ = store.is_collapsed_;
    }

    void merge(const CollapsingLowestDenseStore& store) {
        if (store.count_ == 0)
            return;

        if (count_ == 0) {
            copy(store);
            return;
        }

        if (store.min_key_ < min_key_ || store.max_key_ > max_key_)
            extend_range(store.min_key_, store.max_key_);

        auto collapse_start_idx = store.min_key_ - store.offset_;

        auto collapse_end_idx =
                std::min(min_key_, store.max_key_ + 1) - store.offset_;

        if (collapse_end_idx > collapse_start_idx) {
            auto collapsed_count =
                bins_.collapsed_count(
                    collapse_start_idx, collapse_end_idx);

            bins_.first() += collapsed_count;
        } else {
            collapse_end_idx = collapse_start_idx;
        }

        for (auto key = collapse_end_idx + store.offset_;
                       key <= store.max_key_; ++key)
            bins_[key - offset_] += store.bins_[key - store.offset_];

        count_ += store.count_;
    }

 private:
    Index get_new_length(Index new_min_key, Index new_max_key) override {
        auto desired_length = new_max_key - new_min_key + 1;
        Index num_chunks = std::ceil((1.0 * desired_length) / chunk_size_);

        return std::min(chunk_size_ * num_chunks, bin_limit_);
    }

    /* Calculate the bin index for the key, extending the range if necessary */
    Index get_index(Index key) override {
        if (key < min_key_) {
            if (is_collapsed_)
                return 0;

            extend_range(key);

            if (is_collapsed_)
                return 0;
        } else if (key > max_key_) {
            extend_range(key);
        }

        return key - offset_;
    }

    /*
     * Override. Adjust the bins, the offset, the min_key, and max_key,
     * without resizing the bins, in order to try making it fit the specified
     * range. Collapse to the left if necessary
     */
    void adjust(Index new_min_key, Index new_max_key) override {
        if (new_max_key - new_min_key + 1 > length()) {
            /*
             * The range of keys is too wide.
             * The lowest bins need to be collapsed
             */
            new_min_key = new_max_key - length() + 1;

            if (new_min_key >= max_key_) {
                /* Put everything in the first bin */
                offset_ = new_min_key;
                min_key_ = new_min_key;

                bins_.initialize_with_zeros(length());
                bins_.first() = count_;
            } else {
                auto shift = offset_ - new_min_key;

                if (shift < 0) {
                    auto collapse_start_index = min_key_ - offset_;
                    auto collapse_end_index = new_min_key - offset_;

                    auto collapsed_count =
                        bins_.collapsed_count(
                            collapse_start_index,
                            collapse_end_index);

                    bins_.replace_range_with_zeros(
                        collapse_start_index,
                        collapse_end_index,
                        new_min_key - min_key_);

                    bins_[collapse_end_index] += collapsed_count;

                    min_key_ = new_min_key;

                    /* Shift the buckets to make room for new_max_key */
                    shift_bins(shift);
                } else {
                    min_key_ = new_min_key;

                    /* Shift the buckets to make room for new_min_key */
                    shift_bins(shift);
                }
              }

            max_key_ = new_max_key;
            is_collapsed_ = true;
        } else {
            center_bins(new_min_key, new_max_key);

            min_key_ = new_min_key;
            max_key_ = new_max_key;
        }
    }

 private:
    Index bin_limit_;  /* The maximum number of bins */
    bool is_collapsed_;
};

/*
 * A dense store that keeps all the bins between the bin for the min_key and the
 * bin for the max_key, but collapsing the right-most bins if the number of bins
 * exceeds the bin_limit
 */
class CollapsingHighestDenseStore
    : public BaseDenseStore<CollapsingHighestDenseStore> {
 public:
    explicit CollapsingHighestDenseStore(Index bin_limit,
                                         Index chunk_size = kChunkSize)
    : BaseDenseStore(chunk_size),
      bin_limit_(bin_limit),
      is_collapsed_(false) {
    }

    Index bin_limit() const {
        return bin_limit_;
    }

    void copy(const CollapsingHighestDenseStore& store) {
        count_ = store.count_;
        min_key_ = store.min_key_;
        max_key_ = store.max_key_;
        offset_ = store.offset_;
        bins_ = store.bins_;

        bin_limit_ = store.bin_limit_;
        is_collapsed_ = store.is_collapsed_;
    }

    void merge(const CollapsingHighestDenseStore& store) {
        if (store.count_ == 0)
            return;

        if (count_ == 0) {
            copy(store);
            return;
        }

        if (store.min_key_ < min_key_ || store.max_key_ > max_key_)
            extend_range(store.min_key_, store.max_key_);

        auto collapse_end_idx = store.max_key_ - store.offset_ + 1;
        auto collapse_start_idx =
            std::max(max_key_ + 1, store.min_key_) - store.offset_;

        if (collapse_end_idx > collapse_start_idx) {
             auto collapsed_count =
                    bins_.collapsed_count(
                        collapse_start_idx, collapse_end_idx);
            bins_.last() += collapsed_count;
        } else {
            collapse_start_idx = collapse_end_idx;
        }

        for (auto key = store.min_key_;
                 key < collapse_start_idx + store.offset_; ++key)
            bins_[key - offset_] += store.bins_[key - store.offset_];

        count_ += store.count_;
    }

 private:
    Index get_new_length(Index new_min_key, Index new_max_key) override {
        auto desired_length = new_max_key - new_min_key + 1;
        Index num_chunks = std::ceil((1.0 * desired_length) / chunk_size_);

        return std::min(chunk_size_ * num_chunks, bin_limit_);
    }

    /* Calculate the bin index for the key, extending the range if necessary */
    Index get_index(Index key) override {
        if (key > max_key_) {
            if (is_collapsed_)
                return length() - 1;

            extend_range(key);

            if (is_collapsed_)
                return length() - 1;
        } else if (key < min_key_) {
            extend_range(key);
        }

        return key - offset_;
    }

    /*
     * Override. Adjust the bins, the offset, the min_key, and max_key, without
     * resizing the bins, in order to try making it fit the specified range.
     * Collapse to the left if necessary.
     */
    void adjust(Index new_min_key, Index new_max_key) override {
        if (new_max_key - new_min_key + 1 > length()) {
            /*
             * The range of keys is too wide.
             * The lowest bins need to be collapsed
             */
            new_max_key = new_min_key + length() - 1;

            if (new_max_key <= min_key_) {
                /* Put everything in the last bin */
                offset_ = new_min_key;
                max_key_ = new_max_key;

                bins_ = BinList<RealValue>(length());
                bins_.last() = count_;
            } else {
                auto shift = offset_ - new_min_key;

                if (shift > 0) {
                    auto collapse_start_index = new_max_key - offset_ + 1;
                    auto collapse_end_index = max_key_ - offset_ + 1;

                    auto collapsed_count =
                        bins_.collapsed_count(
                            collapse_start_index, collapse_end_index);

                    bins_.replace_range_with_zeros(
                        collapse_start_index,
                        collapse_end_index,
                        max_key_ - new_max_key);

                    bins_[collapse_start_index - 1] += collapsed_count;

                    max_key_ = new_max_key;

                    /* Shift the buckets to make room for new_max_key */
                    shift_bins(shift);
                } else {
                    max_key_ = new_max_key;

                    /* Shift the buckets to make room for new_min_key */
                    shift_bins(shift);
                }
            }

            min_key_ = new_min_key;
            is_collapsed_ = true;
        } else {
            center_bins(new_min_key, new_max_key);

            min_key_ = new_min_key;
            max_key_ = new_max_key;
        }
    }

 private:
    Index bin_limit_;    /* The maximum number of bins */
    bool is_collapsed_;
};

/*
 * Thrown when an argument is misspecified
 */
class IllegalArgumentException : public std::exception {
 public:
    const char* what() const throw() {
        return message_.c_str();
    }

    explicit IllegalArgumentException(const std::string& message)
        : message_(message) {
    }

 private:
    std::string message_;
};

/*
 * Thrown when trying to merge two sketches with different
 * relative_accuracy parameters
 */
class UnequalSketchParametersException : public std::exception {
 public:
    const char* what() const throw() {
        return "Cannot merge two DDSketches with different parameters";
    }
};

/*
 * A mapping between values and integer indices that imposes relative accuracy
 * guarantees. Specifically, for any value `minIndexableValue() < value <
 * maxIndexableValue` implementations of `KeyMapping` must be such that
 * `value(key(v))` is close to `v` with a relative error that is less than
 * `relative_accuracy`.
 *
 * In implementations of KeyMapping, there is generally a trade-off between the
 * cost of computing the key and the number of keys that are required to cover a
 * given range of values (memory optimality). The most memory-optimal mapping is
 * the LogarithmicMapping, but it requires the costly evaluation of the logarithm
 * when computing the index. Other mappings can approximate the logarithmic
 *mapping, while being less computationally costly.
 */
class KeyMapping {
 public:
    /*
     * Args:
     *       value
     * Returns:
     *       The key specifying the bucket for value
     */
    Index key(RealValue value) {
        return static_cast<Index>(std::ceil(log_gamma(value)) + offset_);
    }

    /*
     * Args:
     *       key
     * Returns:
     *       The value represented by the bucket specified by the key
     */
    RealValue value(Index key) {
        return pow_gamma(key - offset_) * (2.0 / (1 + gamma_));
    }

    RealValue relative_accuracy() const {
        return relative_accuracy_;
    }

    RealValue min_possible() const {
        return min_possible_;
    }

    RealValue max_possible() const {
        return max_possible_;
    }

    RealValue gamma() const {
        return gamma_;
    }

 private:
    static RealValue adjust_accuracy(RealValue relative_accuracy) {
        if (relative_accuracy <= 0.0 || relative_accuracy >= 1.0)
            return kDefaultRelativeAccuracy;

        return relative_accuracy;
    }

    /*
     * The accuracy guarantee.
     * Referred to as alpha in the paper. (0. < alpha < 1.)
     */
    static constexpr RealValue kDefaultRelativeAccuracy = 0.01;

 protected:
    explicit KeyMapping(RealValue relative_accuracy,
                        RealValue offset = 0.0) {
        if (relative_accuracy <= 0.0 || relative_accuracy >= 1.0)
            throw IllegalArgumentException(
                "Relative accuracy must be between 0 and 1");

        relative_accuracy_ = relative_accuracy;
        offset_ = offset;

        auto gamma_mantissa = 2 * relative_accuracy / (1 - relative_accuracy);

        gamma_ = 1.0 + gamma_mantissa;
        multiplier_ = 1.0 / std::log1p(gamma_mantissa);

        min_possible_ = std::numeric_limits<RealValue>::min() * gamma_;
        max_possible_ = std::numeric_limits<RealValue>::max() / gamma_;
    }

    /* Return (an approximation of) the logarithm of the value base gamma */
    virtual RealValue log_gamma(RealValue value) = 0;

    /* Return (an approximation of) gamma to the power value */
    virtual RealValue pow_gamma(RealValue value) = 0;

    /*
     * The accuracy guarantee.
     * referred to as alpha in the paper (0. < alpha < 1.)
     */
    RealValue relative_accuracy_;

    /* An offset that can be used to shift all bin keys */
    RealValue offset_;

    /*
     * The base for the exponential buckets.
     * gamma = (1 + alpha) / (1 - alpha)
     */
    RealValue gamma_;

    /* The smallest value the sketch can distinguish from 0 */
    RealValue min_possible_;

    /* The largest value the sketch can handle */
    RealValue max_possible_;

    /*
     * Used for calculating log_gamma(value).
     * Initially multiplier = 1 / log(gamma)
     */
    RealValue multiplier_;
};

/*
 * A memory-optimal KeyMapping, i.e, given a targeted relative accuracy,
 * it requires the least number of keys to cover a given range of values.
 * This is done by logarithmically mapping floating-point values to integers.
 */
class LogarithmicMapping : public KeyMapping {
 public:
    explicit LogarithmicMapping(RealValue relative_accuracy,
                                RealValue offset = 0.0) :
        KeyMapping(relative_accuracy, offset) {
        multiplier_ *= std::log(2.0);
    }

 private:
    RealValue log_gamma(RealValue value) override {
        return std::log2(value) * multiplier_;
    }

    RealValue pow_gamma(RealValue value) override {
        return std::exp2(value / multiplier_);
    }
};

/*
 * A fast KeyMapping that approximates the memory-optimal one
 * (LogarithmicMapping) by extracting the floor value of the logarithm to the
 * base 2 from the binary representations of floating-point values and
 * linearly interpolating the logarithm in-between.
 */
class LinearlyInterpolatedMapping : public KeyMapping {
 public:
    explicit LinearlyInterpolatedMapping(RealValue relative_accuracy,
                                         RealValue offset = 0.0) :
        KeyMapping(relative_accuracy, offset) {
    }

 private:
    /*
     * Approximates log2 by s + f
     * where v = (s+1) * 2 ** f  for s in [0, 1)
     * frexp(v) returns m and e s.t.
     * v = m * 2 ** e ; (m in [0.5, 1) or 0.0)
     * so we adjust m and e accordingly
     */
    static RealValue log2_approx(RealValue value) {
        int exponent;

        auto mantissa = std::frexp(value, &exponent);
        auto significand = 2.0 * mantissa - 1;
        return significand + (exponent - 1);
    }

    /* Inverse of log2_approx */
    static RealValue exp2_approx(RealValue value) {
        auto exponent = std::floor(value) + 1;
        auto mantissa = (value - exponent + 2) / 2.0;
        return std::ldexp(mantissa, exponent);
    }

    RealValue log_gamma(RealValue value) override {
        return log2_approx(value) * multiplier_;
    }

    RealValue pow_gamma(RealValue value) override {
        return exp2_approx(value / multiplier_);
    }
};

/*
 * A fast KeyMapping that approximates the memory-optimal LogarithmicMapping by
 * extracting the floor value of the logarithm to the base 2 from the binary
 * representations of floating-point values and cubically interpolating the
 * logarithm in-between.
 *
 * More detailed documentation of this method can be found in:
 *     https://github.com/DataDog/sketches-java/
 */
class CubicallyInterpolatedMapping : public KeyMapping {
 public:
    explicit CubicallyInterpolatedMapping(RealValue relative_accuracy,
                                          RealValue offset = 0.0) :
        KeyMapping(relative_accuracy, offset) {
        multiplier_ /= C_;
    }

 private:
    /* Approximates log2 using a cubic polynomial */
    static RealValue cubic_log2_approx(RealValue value) {
        int exponent;
        auto mantissa = std::frexp(value, &exponent);
        auto significand = 2.0 * mantissa - 1;

        return
            ((A_ * significand + B_) * significand + C_) * significand +
            (exponent - 1);
    }

    /* Derived from Cardano's formula */
    static RealValue cubic_exp2_approx(RealValue value) {
        auto exponent =  std::floor(value);
        auto delta_0 = B_ * B_ - 3 * A_ * C_;
        auto delta_1 = (
              2  * B_ * B_ * B_
            - 9  * A_ * B_ * C_
            - 27 * A_ * A_ * (value - exponent));

        auto cardano = std::cbrt(
            (delta_1 -
                std::sqrt(delta_1 * delta_1 -
                              4 * delta_0 * delta_0 * delta_0)) / 2.0);

        auto significand_plus_one =
            -(B_ + cardano + delta_0 / cardano) / (3 * A_) + 1;

        auto mantissa = significand_plus_one / 2.0;

        return std::ldexp(mantissa, exponent + 1);
    }

    RealValue log_gamma(RealValue value) override {
        return cubic_log2_approx(value) * multiplier_;
    }

    RealValue pow_gamma(RealValue value) override {
        return cubic_exp2_approx(value / multiplier_);
    }

    static constexpr RealValue A_ = 6.0 / 35;
    static constexpr RealValue B_ = -3.0 / 5;
    static constexpr RealValue C_ = 10.0 / 7;
};

/*
 * Base implementation of DDSketch.
 * Concrete implementations, derive from this class
 */
template <typename Store, class Mapping>
class BaseDDSketch {
 public:
    BaseDDSketch(const Mapping& mapping,
                 const Store& store,
                 const Store& negative_store) :
        mapping_(mapping),
        store_(store),
        negative_store_(negative_store),
        zero_count_(0.0),
        count_(0.0),
        min_(std::numeric_limits<RealValue>::max()),
        max_(std::numeric_limits<RealValue>::min()),
        sum_(0.0) {
    }

    static std::string name() {
        return "DDSketch";
    }

    RealValue num_values() const {
        return count_;
    }

    RealValue sum() const {
        return sum_;
    }

    RealValue avg() const {
        return sum_ / count_;
    }

    /* Add a value to the sketch */
    void add(RealValue val, RealValue weight = 1.0) {
        if (weight <= 0.0)
            throw IllegalArgumentException("Weight must be positive");

        if (val > mapping_.min_possible()) {
            store_.add(mapping_.key(val), weight);
        } else if (val < -mapping_.min_possible()) {
            negative_store_.add(mapping_.key(-val), weight);
        } else {
            zero_count_ += weight;
        }

        /* Keep track of summary stats */
        count_ += weight;
        sum_ += val * weight;

        if (val < min_)
            min_ = val;

        if (val > max_)
            max_ = val;
    }

    /*
     * The approximate value at the specified quantile
     *   Args:
     *       quantile 0 <= q <=1
     *   Returns:
     *       The value at the specified quantile or NaN if the sketch is empty
     */
    RealValue get_quantile_value(RealValue quantile) {
        RealValue quantile_value;

        if (quantile < 0 || quantile > 1 || count_ == 0)
            return std::nan("");

        auto rank = quantile * (count_ - 1);

        if (rank < negative_store_.count()) {
            auto reversed_rank = negative_store_.count() - rank - 1;
            auto key = negative_store_.key_at_rank(reversed_rank, false);
            quantile_value = -mapping_.value(key);
        } else if (rank < zero_count_ + negative_store_.count()) {
            return 0;
        } else {
            auto key = store_.key_at_rank(
                            rank - zero_count_ - negative_store_.count());
            quantile_value = mapping_.value(key);
        }

        return quantile_value;
    }

    /*
     *  Merges the other sketch into this one.
     *
     *  After this operation, this sketch encodes the values that were
     *  added to both this and the input sketch.
     */
    void merge(const BaseDDSketch& sketch) {
        if (!mergeable(sketch))
            throw UnequalSketchParametersException();

        if (sketch.count_ == 0)
            return;

        if (count_ == 0) {
            copy(sketch);
            return;
        }

        /* Merge the stores */
        store_.merge(sketch.store_);
        negative_store_.merge(sketch.negative_store_);
        zero_count_ += sketch.zero_count_;

        /* Merge summary stats */
        count_ += sketch.count_;
        sum_ += sketch.sum_;

        if (sketch.min_ < min_)
            min_ = sketch.min_;

        if (sketch.max_ > max_)
            max_ = sketch.max_;
    }

    /* Two sketches can be merged only if their gammas are equal */
    bool mergeable(const BaseDDSketch<Store, Mapping>& other) const {
        return mapping_.gamma() == other.mapping_.gamma();
    }

    /* Copy the input sketch into this one */
    void copy(const BaseDDSketch& sketch) {
        store_.copy(sketch.store_);
        negative_store_.copy(sketch.negative_store_);
        zero_count_ = sketch.zero_count_;
        min_ = sketch.min_;
        max_ = sketch.max_;
        count_ = sketch.count_;
        sum_ = sketch.sum_;
    }

 protected:
     static Index adjust_bin_limit(Index bin_limit) {
        if (bin_limit <= 0)
            return kDefaultBinLimit;

        return bin_limit;
    }

    Mapping mapping_;       /* Map btw values and store bins */
    Store store_;           /* Storage for positive values */
    Store negative_store_;  /* Storage for negative values */
    RealValue zero_count_;  /* The count of zero values */

    RealValue count_;       /* The number of values seen by the sketch */
    RealValue min_;         /* The minimum value seen by the sketch */
    RealValue max_;         /* The maximum value seen by the sketch */
    RealValue sum_;         /* The sum of the values seen by the sketch */

    static constexpr Index kDefaultBinLimit = 2048;
};

/*
 * The default implementation of BaseDDSketch, with optimized memory usage at
 * the cost of lower ingestion speed, using an unlimited number of bins.
 * The number of bins will not exceed a reasonable number unless the data is
 * distributed with tails heavier than any subexponential.
 * (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
 */
class DDSketch : public BaseDDSketch<DenseStore, LogarithmicMapping> {
 public:
    explicit DDSketch(RealValue relative_accuracy)
        : BaseDDSketch<DenseStore, LogarithmicMapping>(
            LogarithmicMapping(relative_accuracy),
            DenseStore(),
            DenseStore()) {
    }
};

/*
 * Implementation of BaseDDSketch with optimized memory usage at the cost of
 * lower ingestion speed, using a limited number of bins. When the maximum
 * number of bins is reached, bins with lowest indices are collapsed, which
 * causes the relative accuracy to be lost on the lowest quantiles. For the
 * default bin limit, collapsing is unlikely to occur unless the data is
 * distributed with tails heavier than any subexponential.
 * (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
 */
class LogCollapsingLowestDenseDDSketch
    : public BaseDDSketch<CollapsingLowestDenseStore, LogarithmicMapping> {
 public:
    explicit LogCollapsingLowestDenseDDSketch(RealValue relative_accuracy,
                                              Index bin_limit)
        : BaseDDSketch<CollapsingLowestDenseStore, LogarithmicMapping>(
            LogarithmicMapping(relative_accuracy),
            CollapsingLowestDenseStore(adjust_bin_limit(bin_limit)),
            CollapsingLowestDenseStore(adjust_bin_limit(bin_limit))) {
    }
};

/*
 * Implementation of BaseDDSketch with optimized memory usage at the cost of
 * lower ingestion speed, using a limited number of bins. When the maximum
 * number of bins is reached, bins with highest indices are collapsed, which
 * causes the relative accuracy to be lost on the highest quantiles. For the
 * default bin limit, collapsing is unlikely to occur unless the data is
 * distributed with tails heavier than any
 * subexponential.
 * (cf. http://www.vldb.org/pvldb/vol12/p2195-masson.pdf)
 */
class LogCollapsingHighestDenseDDSketch
    : public BaseDDSketch<CollapsingHighestDenseStore, LogarithmicMapping> {
 public:
    LogCollapsingHighestDenseDDSketch(RealValue relative_accuracy,
                                      Index bin_limit)
        : BaseDDSketch<CollapsingHighestDenseStore, LogarithmicMapping>(
            LogarithmicMapping(relative_accuracy),
            CollapsingHighestDenseStore(adjust_bin_limit(bin_limit)),
            CollapsingHighestDenseStore(adjust_bin_limit(bin_limit))) {
    }
};

}  // namespace ddsketch

#endif  // INCLUDES_DDSKETCH_DDSKETCH_H_
