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
#include "op_tiling_registry_impl.h"

#include <securec.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "graph/debug/ge_log.h"

namespace optiling {
namespace utils {

OpCompileInfoImpl::OpCompileInfoImpl(const ge::AscendString &key,
                                     const ge::AscendString &value)
    : str(value), key(key) {}

void OpRunInfoImpl::SetBlockDim(uint32_t input_block_dim) {
  block_dim = input_block_dim;
}

void OpRunInfoImpl::AddWorkspace(int64_t workspace) {
  workspaces.push_back(workspace);
}

uint32_t OpRunInfoImpl::GetBlockDim() { return block_dim; }

size_t OpRunInfoImpl::GetWorkspaceNum() { return workspaces.size(); }

ge::graphStatus OpRunInfoImpl::GetWorkspace(size_t idx, int64_t &workspace) {
  if (!workspaces.empty() && idx < workspaces.size()) {
    workspace = workspaces[idx];
    return ge::GRAPH_SUCCESS;
  }
  return ge::GRAPH_FAILED;
}

ge::graphStatus OpRunInfoImpl::GetAllWorkspaces(
    std::vector<int64_t> &_workspaces) {
  _workspaces = workspaces;
  return ge::GRAPH_SUCCESS;
}

void OpRunInfoImpl::AddTilingData(const char *_value, size_t _size) {
  tiling_data.write(_value, _size);
  tiling_data.flush();
}

ByteBuffer &OpRunInfoImpl::GetAllTilingData() { return tiling_data; }

void OpRunInfoImpl::SetAllTilingData(ByteBuffer &value) {
  tiling_data.clear();
  std::string temp = value.str();
  tiling_data << temp;
}

void OpRunInfoImpl::SetClearAtomic(bool clear_atomic_input) {
  clear_atomic = clear_atomic_input;
}

bool OpRunInfoImpl::GetClearAtomic() const { return clear_atomic; }

void OpRunInfoImpl::SetTilingKey(uint32_t _tiling_key) {
  tiling_key = _tiling_key;
}

uint32_t OpRunInfoImpl::GetTilingKey() const { return tiling_key; }

void OpCompileInfoImpl::SetKey(const ge::AscendString &_key) { key = _key; }

void OpCompileInfoImpl::SetValue(const ge::AscendString &value) { str = value; }

const ge::AscendString &OpCompileInfoImpl::GetKey() const { return key; }

const ge::AscendString &OpCompileInfoImpl::GetValue() const { return str; }
}  // namespace utils
}  // namespace optiling
