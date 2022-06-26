/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "op_tiling/op_compile_info_manager.h"

namespace optiling {
CompileInfoManager::CompileInfoManager() {}
CompileInfoManager::~CompileInfoManager() {}

CompileInfoManager& CompileInfoManager::Instance() {
  static CompileInfoManager compile_info_manager_instance;
  return compile_info_manager_instance;
}

bool CompileInfoManager::HasCompileInfo(const std::string &key) {
  return this->compile_info_map_.find(key) != this->compile_info_map_.end();
}

CompileInfoPtr CompileInfoManager::GetCompileInfo(const std::string &key) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  const auto iter = this->compile_info_map_.find(key);
  if (iter == this->compile_info_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

void CompileInfoManager::SetCompileInfo(const std::string &key, CompileInfoPtr compile_info_ptr) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  (void)this->compile_info_map_.emplace(key, compile_info_ptr);
}

CompileInfoCache::CompileInfoCache() {}
CompileInfoCache::~CompileInfoCache() {}

CompileInfoCache& CompileInfoCache::Instance() {
  static CompileInfoCache compile_info_cache_instance;
  return compile_info_cache_instance;
}

bool CompileInfoCache::HasCompileInfo(const std::string &key) {
  return this->compile_info_map_.find(key) != this->compile_info_map_.end();
}

void* CompileInfoCache::GetCompileInfo(const std::string &key) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  const auto iter = this->compile_info_map_.find(key);
  if (iter == this->compile_info_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

void CompileInfoCache::SetCompileInfo(const std::string &key, void *value) {
  std::lock_guard<std::mutex> lock_guard(compile_info_mutex_);
  (void)this->compile_info_map_.emplace(key, value);
}
}  // namespace optiling
