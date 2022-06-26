/**
* Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
* Copyright 2021, 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef GRAPH_COMPILE_CACHE_POLICY_COMPILE_CACHE_STATE_H
#define GRAPH_COMPILE_CACHE_POLICY_COMPILE_CACHE_STATE_H

#include <vector>
#include <functional>
#include <unordered_map>
#include <chrono>
#include <queue>
#include <mutex>

#include "compile_cache_desc.h"
#include "compile_cache_hasher.h"

namespace ge {
class CacheInfo;
using CacheItemId = uint64_t;
constexpr CacheItemId KInvalidCacheItemId = std::numeric_limits<uint64_t>::max();

using DelCacheFunc = std::function<bool(CacheInfo &)>;
using CCStatType = std::unordered_map<uint64_t, std::vector<CacheInfo>>;

class CacheInfo {
friend class CompileCacheState;
public:
  CacheInfo(const time_t time_stamp, const CacheItemId item_id, const CompileCacheDesc &desc):
            time_stamp_(time_stamp), item_id_(item_id), desc_(desc) {}
  CacheInfo(const CacheInfo &other) :
            time_stamp_(other.time_stamp_),
            item_id_(other.item_id_), desc_(other.desc_) {}
  CacheInfo &operator=(const CacheInfo &other) {
    time_stamp_ = other.time_stamp_;
    item_id_ = other.item_id_;
    desc_ = other.desc_;
    return *this;
  }
  CacheInfo() = delete;
  ~CacheInfo() = default;

  void RefreshTimeStamp() {
    time_stamp_ = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  }

  const time_t &GetTimeStamp() const noexcept {
    return time_stamp_;
  }

  CacheItemId GetItemId() const noexcept {
    return item_id_;
  }

  const CompileCacheDesc &GetCompileCacheDesc() const noexcept {
    return desc_;
  }

private:
  time_t time_stamp_;
  CacheItemId item_id_;
  CompileCacheDesc desc_;
};

class CompileCacheState {
public:
  CompileCacheState() = default;
  ~CompileCacheState() = default;

  CacheItemId AddCache(const CompileCacheDesc &compile_cache_desc);

  std::vector<CacheItemId> DelCache(const DelCacheFunc &func);

  std::vector<CacheItemId> DelCache(const std::vector<CacheItemId> &delete_item);

  const CCStatType &GetState() const {
    return cc_state_;
  }

private:
  CacheItemId GetNextCacheItemId();
  void RecoveryCacheItemId(const std::vector<CacheItemId> &cache_items);

  std::mutex cc_state_mu_;
  std::mutex cache_item_mu_;

  int64_t cache_item_counter_ = 0L;
  std::queue<int64_t> cache_item_queue_;
  CCStatType cc_state_;
};
}  // namespace ge
#endif
