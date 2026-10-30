/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ISAAC_OS_COMMON_POSE_TREE_DETAIL_UNIQUE_INDEX_MAP_HPP
#define ISAAC_OS_COMMON_POSE_TREE_DETAIL_UNIQUE_INDEX_MAP_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>

#include <holoscan/core/expected.hpp>

namespace isaac::os::pose_tree {

// Provides a basic fixed capacity container for tracking objects and assigning a unique id to every
// object inserted. It provides constant time insert/delete/access for all elements with the
// generated uid. This container guarantees that uids are never reused, and will return an
// Unexpected Error if it runs out of capacity.
template <typename T>
class UniqueIndexMap {
 private:
  // number of bits to use for holding the version number
  static constexpr uint64_t kVersionSize = 32;
  // number of bits to use for holding the index into the underlying container
  static constexpr uint64_t kIndexSize = 32;
  // bitmask for Version number
  static constexpr uint64_t kVersionMask = ((1ULL << kVersionSize) - 1ULL) << kIndexSize;
  // bitmask for Index number
  static constexpr uint64_t kIndexMask = (1ULL << kIndexSize) - 1ULL;

  static_assert(kIndexSize + kVersionSize == 64, "Invalid UID bit length");
  static_assert((kVersionMask ^ kIndexMask) == 0xFFFFFFFFFFFFFFFF,
                "bitmasks must cover full size of uint64_t");

  // Storage for one T, carrying T's size and alignment but none of its lifetime. An array of these
  // is raw memory holding no objects until `emplace` constructs one, which is what lets a slot
  // stand empty without requiring T to be default constructible.
  struct alignas(T) Slot {
    std::byte storage[sizeof(T)];
  };

 public:
  // Custom Error codes for construction and access of UniqueIndexMap map.
  enum struct Error {
    kArgumentInvalid,  // Invalid construction/initialization parameter
    kOutOfMemory,      // Could not allocate memory during initialization
    kContainerFull,    // UniqueIndexMap is full
    kNotFound          // UID does not exits in UniqueIndexMap
  };

  // Expected type for UniqueIndexMap which uses class specific Error type.
  template <typename U>
  using Expected = holoscan::expected<U, Error>;

  // Both moves below are declared noexcept, which holds only while destroying a T cannot throw:
  // move assignment destroys what it is replacing. Asserted rather than assumed, so a T that broke
  // it would be a compile error here instead of a call to std::terminate somewhere else.
  static_assert(std::is_nothrow_destructible_v<T>,
                "UniqueIndexMap requires a type whose destructor does not throw");

  UniqueIndexMap() = default;
  UniqueIndexMap(const UniqueIndexMap<T>&) = delete;
  // Takes the other map's storage and leaves it empty; nothing here allocates or can throw.
  UniqueIndexMap(UniqueIndexMap<T>&& other) noexcept
      : uids_{std::move(other.uids_)},
        slots_{std::move(other.slots_)},
        capacity_{other.capacity_},
        size_{other.size_},
        next_{other.next_} {
    other.reset_values();
  }

  UniqueIndexMap<T>& operator=(const UniqueIndexMap<T>& other) = delete;
  // Releases what this map holds and takes the other's. Releasing runs destructors and frees
  // storage, neither of which throws.
  UniqueIndexMap<T>& operator=(UniqueIndexMap<T>&& other) noexcept {
    deinitialize();
    uids_ = std::move(other.uids_);
    slots_ = std::move(other.slots_);
    capacity_ = other.capacity_;
    size_ = other.size_;
    next_ = other.next_;

    other.reset_values();
    return *this;
  }

  ~UniqueIndexMap() { deinitialize(); }

  // Destroys the contained objects and releases the storage allocated in initialize. Noexcept
  // because the moves above say so.
  void deinitialize() noexcept {
    for (uint64_t i = 0; i < capacity_; ++i) {
      if (GetIndex(uids_[i]) == i) {
        at(i)->~T();
      }
    }

    reset_values();
  }

  // Allocate memory up to the provided capacity and initialize values for implicit free list.
  // Returns Unexpected<Error> if max_capacity is too large or if the memory is not available.
  // Re-initializing a map destroys whatever it already holds.
  Expected<void> initialize(uint64_t max_capacity) {
    if (max_capacity > (1ULL << kIndexSize)) {
      return holoscan::make_unexpected(Error::kArgumentInvalid);
    }

    // Both arrays are built into locals and handed over only once both have succeeded, so a failed
    // second allocation leaves the map exactly as it was rather than half replaced. The map
    // reports allocation failure rather than propagating it, hence the catch.
    std::unique_ptr<uint64_t[]> uids;
    std::unique_ptr<Slot[]> slots;
    try {
      // Neither array is read before it is written: the loop below fills the uids, and a slot is
      // raw storage until `emplace` constructs into it. Value initializing them would be wasted
      // work proportional to the capacity.
      uids = std::make_unique_for_overwrite<uint64_t[]>(max_capacity);
      slots = std::make_unique_for_overwrite<Slot[]>(max_capacity);
    } catch (const std::bad_alloc&) {
      return holoscan::make_unexpected(Error::kOutOfMemory);
    }

    for (uint64_t i = 0; i < max_capacity; ++i) {
      uids[i] = i + 1;
    }

    deinitialize();
    uids_ = std::move(uids);
    slots_ = std::move(slots);
    capacity_ = max_capacity;
    return {};
  }

  // Check if there exists an object with the provided UID. Returns true if the uid exists in the
  // allocated range and if the version numbers match.
  bool valid(uint64_t uid) const {
    const uint64_t idx = GetIndex(uid);
    return (idx < capacity_) && (uids_[idx] == uid);
  }

  // Insert a new object into the container and return the generated UID via move constructor.
  // Return Unexpected<Error> if the object cannot be inserted.
  Expected<uint64_t> insert(T&& object) { return emplace(static_cast<T&&>(object)); }

  // Insert a new object into the container and return the generated UID via copy constructor.
  // Return Unexpected<Error> if the object cannot be inserted.
  Expected<uint64_t> insert(const T& object) { return emplace(object); }

  // Construct a new object with the provided arguments directly in the container, and return the
  // generated UID via copy. Return Unexpected<Error> if the object cannot be inserted.
  template <typename... Args>
  Expected<uint64_t> emplace(Args&&... args) {
    if (size_ == capacity_) {
      return holoscan::make_unexpected(Error::kOutOfMemory);
    }

    if (GetVersion(next_) == (1ULL << kVersionSize) - 1ULL) {
      return holoscan::make_unexpected(Error::kContainerFull);
    }

    const uint64_t idx = GetIndex(next_);
    const uint64_t temp = next_;
    next_ = uids_[idx];
    uids_[idx] = temp + (1ULL << kIndexSize);
    ::new (static_cast<void*>(&slots_[idx])) T{static_cast<Args&&>(args)...};
    ++size_;
    return uids_[idx];
  }

  // Erase the object tracked by UID. Return Unexpected<Error> if UID is not valid.
  Expected<void> erase(uint64_t uid) {
    if (!valid(uid)) {
      return holoscan::make_unexpected(Error::kNotFound);
    }

    uint64_t idx = GetIndex(uid);
    uids_[idx] = next_;
    next_ = uid;
    --size_;
    at(idx)->~T();

    return {};
  }

  // Get a pointer to the object tracked by UID. Returns an error if UID is not valid.
  Expected<T*> try_get(uint64_t uid) {
    if (!valid(uid)) {
      return holoscan::make_unexpected(Error::kNotFound);
    }
    return at(GetIndex(uid));
  }

  // Get a pointer to the object tracked by UID. Returns an error if UID is not valid.
  Expected<const T*> try_get(uint64_t uid) const {
    if (!valid(uid)) {
      return holoscan::make_unexpected(Error::kNotFound);
    }
    return at(GetIndex(uid));
  }

  // Performs a linear search and returns the UID of the object.
  // Returns an error if object is not found
  Expected<uint64_t> find(const T& object) const {
    for (uint64_t i = 0; i < capacity_; i++) {
      if (valid(uids_[i]) && *at(i) == object) {
        return uids_[i];
      }
    }
    return holoscan::make_unexpected(Error::kNotFound);
  }

  // Maximum capacity of the unique index map.
  inline uint64_t capacity() const { return capacity_; }

  // Current number of elements in the container.
  inline uint64_t size() const { return size_; }

 private:
  static uint64_t GetVersion(uint64_t uid) { return uid >> kIndexSize; }
  static uint64_t GetIndex(uint64_t uid) { return uid & kIndexMask; }

  // The object occupying `idx`, which the caller must already know is occupied. std::launder is
  // what makes reading through this pointer defined: the T was created by placement new into bytes
  // the compiler otherwise knows only as a Slot.
  T* at(uint64_t idx) { return std::launder(reinterpret_cast<T*>(slots_[idx].storage)); }
  const T* at(uint64_t idx) const {
    return std::launder(reinterpret_cast<const T*>(slots_[idx].storage));
  }

  // Returns the map to its empty state without destroying anything, releasing the storage it owns.
  // Callers that hold live objects must destroy them first; see deinitialize.
  void reset_values() noexcept {
    uids_.reset();
    slots_.reset();
    capacity_ = 0;
    size_ = 0;
    next_ = 0;
  }

  std::unique_ptr<uint64_t[]> uids_;  // ids with implicit list of free elements
  std::unique_ptr<Slot[]> slots_;     // raw storage for the objects
  uint64_t capacity_{0};              // maximum number of concurrent objects
  uint64_t size_{0};                  // current number of objects
  uint64_t next_{0};                  // next "free" element in the container
};

}  // namespace isaac::os::pose_tree

#endif  // ISAAC_OS_COMMON_POSE_TREE_DETAIL_UNIQUE_INDEX_MAP_HPP
