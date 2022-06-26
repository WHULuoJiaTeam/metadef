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
#ifndef __OP_TILING_REGISTRY_IMPL_H__
#define __OP_TILING_REGISTRY_IMPL_H__

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "external/graph/tensor.h"
#include "register/op_tiling_registry.h"

namespace optiling {
namespace utils {
using ByteBuffer = std::stringstream;

class OpRunInfoImpl {
 public:
  OpRunInfoImpl() = default;
  ~OpRunInfoImpl() = default;

  OpRunInfoImpl(uint32_t block_dim, bool clear_atomic, uint32_t tiling_key)
      : block_dim(block_dim),
        clear_atomic(clear_atomic),
        tiling_key(tiling_key) {}

  void SetBlockDim(uint32_t block_dim);
  uint32_t GetBlockDim();

  void AddWorkspace(int64_t workspace);
  size_t GetWorkspaceNum();
  ge::graphStatus GetWorkspace(size_t idx, int64_t &workspace);
  ge::graphStatus GetAllWorkspaces(std::vector<int64_t> &workspace);

  void AddTilingData(const char *value, size_t size);
  ByteBuffer &GetAllTilingData();
  void SetAllTilingData(ByteBuffer &value);

  void SetClearAtomic(bool clear_atomic);
  bool GetClearAtomic() const;

  void SetTilingKey(uint32_t tiling_key);
  uint32_t GetTilingKey() const;

  uint32_t block_dim;
  bool clear_atomic;
  uint32_t tiling_key;
  ByteBuffer tiling_data;
  std::vector<int64_t> workspaces;
};

class OpCompileInfoImpl {
 public:
  OpCompileInfoImpl() = default;
  ~OpCompileInfoImpl() = default;
  OpCompileInfoImpl(const ge::AscendString &key, const ge::AscendString &value);

  void SetKey(const ge::AscendString &key);
  const ge::AscendString &GetKey() const;

  void SetValue(const ge::AscendString &value);
  const ge::AscendString &GetValue() const;

  ge::AscendString str;
  ge::AscendString key;
};
}  // namespace utils
}  // namespace optiling

#endif
