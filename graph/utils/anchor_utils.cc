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

#include "graph/utils/anchor_utils.h"
#include <algorithm>
#include "graph/debug/ge_util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
// Get anchor status
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY AnchorStatus AnchorUtils::GetStatus(const DataAnchorPtr &data_anchor) {
  if (data_anchor == nullptr) {
    REPORT_INNER_ERROR("E18888", "param data_anchor is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The input data anchor is invalid.");
    return ANCHOR_RESERVED;
  }
  return data_anchor->status_;
}

// Set anchor status
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus AnchorUtils::SetStatus(const DataAnchorPtr &data_anchor,
                                                                                  const AnchorStatus anchor_status) {
  if ((data_anchor == nullptr) || (anchor_status == ANCHOR_RESERVED)) {
    REPORT_INNER_ERROR("E18888", "The input data anchor or input data format is invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The input data anchor or input data format is invalid.");
    return GRAPH_FAILED;
  }
  data_anchor->status_ = anchor_status;
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int32_t AnchorUtils::GetIdx(const AnchorPtr &anchor) {
  // Check if it can add edge between DataAnchor
  const auto data_anchor = Anchor::DynamicAnchorCast<DataAnchor>(anchor);
  if (data_anchor != nullptr) {
    return data_anchor->GetIdx();
  }
  // Check if it can add edge between ControlAnchor
  const auto ctrl_anchor = Anchor::DynamicAnchorCast<ControlAnchor>(anchor);
  if (ctrl_anchor != nullptr) {
    return ctrl_anchor->GetIdx();
  }
  return -1;
}
}  // namespace ge
