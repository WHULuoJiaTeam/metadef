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

#ifndef INC_GRAPH_AXIS_TYPE_INFO_H_
#define INC_GRAPH_AXIS_TYPE_INFO_H_

#include <vector>

#include "external/graph/ge_error_codes.h"

namespace ge {
using CutInfo = std::pair<int64_t, std::vector<int64_t>>;

enum class AxisType {
  UNSPLIT = 0,
  ELEMENTWISE,
  REDUCESUM,
  REDUCEMAX,
  REDUCEMIN,
  REDUCEMEAN,
  TRANSPOSE,
  SLIDINGWINDOW,
};

class AxisTypeInfo {
public:
  AxisTypeInfo() = default;
  ~AxisTypeInfo() = default;
  void SetAxisType(AxisType axis_type) { axis_type_ = axis_type; }
  AxisType GetAxisType() const { return axis_type_; }
  void AddInputCutInfo(CutInfo &input_cut_info);
  void AddOutputCutInfo(CutInfo &output_cut_info);
  graphStatus GetInputCutInfo(const size_t index, CutInfo &input_cut_info) const;
  graphStatus GetOutputCutInfo(const size_t index, CutInfo &output_cut_info) const;
  const std::vector<CutInfo> &GetRelateInputs() const { return relate_inputs_; }
  const std::vector<CutInfo> &GetRelateOutputs() const { return relate_outputs_; }

private:
  AxisType axis_type_;
  std::vector<CutInfo> relate_inputs_;
  std::vector<CutInfo> relate_outputs_;
};
}

#endif  // INC_GRAPH_AXIS_TYPE_INFO_H_
