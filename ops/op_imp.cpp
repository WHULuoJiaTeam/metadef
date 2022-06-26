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

#include <cstdint>
#include <functional>
#include <algorithm>
#include <vector>
#include "graph/debug/ge_log.h"

namespace ge {
namespace {
static graphStatus BroadCastRankAndDim(
    const std::vector<int64_t> &x1_shape, const std::vector<int64_t> &x2_shape, const int64_t len_diff,
    const std::function<void(const std::vector<int64_t> &out_shape)> &set_out_shape) {
  std::vector<int64_t> y_shape;
  y_shape.reserve(x1_shape.size());
  for (size_t i = 0UL; i < static_cast<size_t>(len_diff); i++) {
    y_shape.push_back(x1_shape[i]);
  }
  for (size_t i = 0UL; i < x2_shape.size(); i++) {
    const size_t idx_diff = i + static_cast<size_t>(len_diff);
    if ((x1_shape[idx_diff] != x2_shape[i]) && (std::min(x1_shape[idx_diff], x2_shape[i]) != 1)) {
      GE_LOGE("operands could not be broadcast together");
      return GRAPH_FAILED;
    }
    y_shape.push_back(std::max(x1_shape[idx_diff], x2_shape[i]));
  }
  set_out_shape(y_shape);
  return GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
BroadCastInfer(const std::function<std::vector<int64_t>()> &get_in1_shape,
               const std::function<std::vector<int64_t>()> &get_in2_shape,
               const std::function<void(const std::vector<int64_t> &out_shape)> &set_out_shape) {
  const auto x1_shape = get_in1_shape();
  const auto x2_shape = get_in2_shape();

  if ((x1_shape.size() >= static_cast<size_t>(std::numeric_limits<int64_t>::max())) ||
      (x2_shape.size() >= static_cast<size_t>(std::numeric_limits<int64_t>::max()))) {
    return GRAPH_FAILED;
  }

  if (x1_shape.empty()) {
    set_out_shape(x2_shape);
    return GRAPH_SUCCESS;
  }
  if (x2_shape.empty()) {
    set_out_shape(x1_shape);
    return GRAPH_SUCCESS;
  }

  const auto len_diff = static_cast<int64_t>(x1_shape.size()) - static_cast<int64_t>(x2_shape.size());
  if (len_diff >= 0) {
    return BroadCastRankAndDim(x1_shape, x2_shape, len_diff, set_out_shape);
  } else {
    return BroadCastRankAndDim(x2_shape, x1_shape, std::abs(len_diff), set_out_shape);
  }
}
}  // namespace ge
