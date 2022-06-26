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

#include "inc/graph/axis_type_info.h"

namespace ge {
void AxisTypeInfo::AddInputCutInfo(CutInfo &input_cut_info) {
  relate_inputs_.emplace_back(input_cut_info);
}

void AxisTypeInfo::AddOutputCutInfo(CutInfo &output_cut_info) {
  relate_outputs_.emplace_back(output_cut_info);
}

graphStatus AxisTypeInfo::GetInputCutInfo(const size_t index, CutInfo &input_cut_info) const {
  if (relate_inputs_.size() <= index) {
      return GRAPH_FAILED;
  }
  input_cut_info = relate_inputs_[index];
  return GRAPH_SUCCESS;
}

graphStatus AxisTypeInfo::GetOutputCutInfo(const size_t index, CutInfo &output_cut_info) const {
  if (relate_outputs_.size() <= index) {
      return GRAPH_FAILED;
  }
  output_cut_info = relate_outputs_[index];
  return GRAPH_SUCCESS;
}
}
