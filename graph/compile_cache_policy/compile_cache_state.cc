/**
 * Copyright (c) Huawei Technologies Co., Ltd
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

#include "graph/compile_cache_policy/compile_cache_state.h"
#include "framework/common/debug/ge_log.h"
namespace ge {
CacheItemId CompileCacheState::GetNextCacheItemId() {
  const std::lock_guard<std::mutex> lock(cache_item_mu_);
  if (cache_item_queue_.empty()) {
    return cache_item_counter_++;
  } else {
    const CacheItemId next_item_id = cache_item_queue_.front();
    cache_item_queue_.pop();
    return next_item_id;
  }
}

void CompileCacheState::RecoveryCacheItemId(const std::vector<CacheItemId> &cache_items) {
  const std::lock_guard<std::mutex> lock(cache_item_mu_);
  for (auto &item_id : cache_items) {
    cache_item_queue_.push(item_id);
  }
}

CacheItemId CompileCacheState::AddCache(const CompileCacheDesc &compile_cache_desc) {
  const CacheHashKey main_hash_key = CompileCacheHasher::GetCacheDescHashWithoutShape(compile_cache_desc);
  const std::lock_guard<std::mutex> lock(cc_state_mu_);
  const auto iter = cc_state_.find(main_hash_key);
  if (iter == cc_state_.end()) {
    const CacheItemId next_item_id = GetNextCacheItemId();
    const CacheInfo cache_info = CacheInfo(
        std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()), next_item_id, compile_cache_desc);
    const std::vector<CacheInfo> info = {cache_info};
    (void)cc_state_.insert({main_hash_key, std::move(info)});
    return next_item_id;
  }
  auto &cached_item = iter->second;
  for (size_t idx = 0UL; idx < cached_item.size(); idx++) {
    if (CompileCacheDesc::IsSameCompileDesc(cached_item[idx].desc_, compile_cache_desc)) {
      GELOGW("[AddCache] Same CompileCacheDesc has already been added, whose cache_item is %ld",
             cached_item[idx].item_id_);
      return cached_item[idx].item_id_;
    }
  }
  const CacheItemId next_item_id = GetNextCacheItemId();
  CacheInfo cache_info = CacheInfo(
      std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()), next_item_id, compile_cache_desc);
  cached_item.emplace_back(std::move(cache_info));
  return next_item_id;
}

std::vector<CacheItemId> CompileCacheState::DelCache(const DelCacheFunc &func) {
  std::vector<CacheItemId> delete_item;
  for (auto &item : cc_state_) {
    std::vector<CacheInfo> &cache_vec = item.second;
    for (auto iter = cache_vec.begin(); iter != cache_vec.end();) {
      if (func(*iter)) {
        delete_item.emplace_back((*iter).item_id_);
        const std::lock_guard<std::mutex> lock(cc_state_mu_);
        iter = cache_vec.erase(iter);
      } else {
         iter++;
      }
    }
  }
  RecoveryCacheItemId(delete_item);
  return delete_item;
}

std::vector<CacheItemId> CompileCacheState::DelCache(const std::vector<CacheItemId> &delete_item) {
  const DelCacheFunc lamb = [&, delete_item] (const CacheInfo &info) -> bool {
    const auto iter = std::find(delete_item.begin(), delete_item.end(), info.GetItemId());
    return iter != delete_item.end();
  };
  return DelCache(lamb);
}
}