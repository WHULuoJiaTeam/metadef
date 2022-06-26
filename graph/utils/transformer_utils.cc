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

#include "transformer_utils.h"

#include "external/ge/ge_api_types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/attr_utils.h"
#include "inc/graph/debug/ge_attr_define.h"
#include "expand_dimension.h"
#include "transfer_shape_according_to_format.h"

namespace ge {
bool NodeShapeTransUtils::Init() {
  if (op_desc_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] input op_desc_ is nullptr!");
    return false;
  }
  in_num_ = op_desc_->MutableAllInputName().size();
  out_num_ = op_desc_->MutableAllOutputName().size();
  map_format_in_.resize(in_num_, FORMAT_RESERVED);
  map_ori_format_in_.resize(in_num_, FORMAT_RESERVED);
  map_dtype_in_.resize(in_num_, DT_UNDEFINED);
  map_format_out_.resize(out_num_, FORMAT_RESERVED);
  map_ori_format_out_.resize(out_num_, FORMAT_RESERVED);
  map_dtype_out_.resize(out_num_, DT_UNDEFINED);
  return true;
}
bool NodeShapeTransUtils::CatchFormatAndShape() {
  for (size_t i = 0; i < in_num_; i++) {
    auto tensor_desc_input = op_desc_->MutableInputDesc(i);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    auto format = tensor_desc_input->GetFormat();
    auto ori_format = tensor_desc_input->GetOriginFormat();
    if (format == ori_format) {
      GELOGD("Node is %s, input tensor idx is %zu. ori format: %s, format: %s is same! No need to catch format&shape!",
             op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(ori_format).c_str(),
             TypeUtils::FormatToSerialString(format).c_str());
      continue;
    }
    map_format_in_[i] = format;
    map_ori_format_in_[i] = ori_format;
    map_dtype_in_[i] = tensor_desc_input->GetDataType();
    tensor_desc_input->SetFormat(ori_format);
    tensor_desc_input->SetShape(tensor_desc_input->GetOriginShape());
  }

  for (size_t i = 0; i < out_num_; i++) {
    auto tensor_desc_output = op_desc_->MutableOutputDesc(i);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    auto format = tensor_desc_output->GetFormat();
    auto ori_format = tensor_desc_output->GetOriginFormat();
    if (format == ori_format) {
      GELOGD("Node is %s, output tensor idx is %zu. ori format: %s, format: %s is same! No need to catch format&shape!",
             op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(ori_format).c_str(),
             TypeUtils::FormatToSerialString(format).c_str());
      continue;
    }
    map_format_out_[i] = format;
    map_ori_format_out_[i] = ori_format;
    map_dtype_out_[i] = tensor_desc_output->GetDataType();

    if (format == ori_format) {
      continue;
    }
    tensor_desc_output->SetFormat(ori_format);
  }

  return true;
}

bool NodeShapeTransUtils::UpdateFormatAndShape() {
  transformer::ShapeTransferAccordingToFormat shape_transfer;
  for (size_t i = 0; i < in_num_; i++) {
    auto tensor_desc_input = op_desc_->MutableInputDesc(i);
    if (tensor_desc_input == nullptr) {
      continue;
    }
    // if can not find saved info, it says format and origin format is same when catched
    if (map_format_in_[i] == FORMAT_RESERVED) {
      GELOGD("Node is [%s], input tensor idx [%zu] is not been catched.Skip update action for it!",
             op_desc_->GetName().c_str(), i);
      tensor_desc_input->SetOriginFormat(tensor_desc_input->GetFormat());
      tensor_desc_input->SetOriginShape(tensor_desc_input->MutableShape());
      continue;
    }
    auto ori_format = tensor_desc_input->GetFormat();
    auto ori_shape = tensor_desc_input->MutableShape();
    auto curr_format = map_format_in_[i];
    if (ori_format == curr_format || curr_format == FORMAT_ND) {
      continue;
    }
    std::vector<int64_t> ori_shape_dims = ori_shape.GetDims();
    std::vector<int64_t> out_dims;
    ge::DataType dtype =  map_dtype_in_[i];

    // FE set and Ge get for PadDimention
    string infer_reshape_type;
    (void) AttrUtils::GetStr(*tensor_desc_input, ATTR_NAME_RESHAPE_INFER_TYPE, infer_reshape_type);
    bool is_success = transformer::ExpandDimension(op_desc_->GetType(), ori_format, curr_format, i,
                                                   infer_reshape_type, ori_shape_dims);
    if (!is_success) {
      REPORT_CALL_ERROR("E19999", "ExpandDimension failed, op type:%s", op_desc_->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Call][ExpandDimension] failed, op type:%s", op_desc_->GetType().c_str());
      return GRAPH_FAILED;
    }
    transformer::ShapeAndFormat shape_and_format_info {ori_shape_dims, out_dims, ori_format, curr_format, dtype,
                                                       transformer::EN_IMPL_CUSTOM_TBE};
    shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
    tensor_desc_input->SetFormat(curr_format);
    tensor_desc_input->SetShape(GeShape(out_dims));
  }

  for (size_t i = 0; i < out_num_; i++) {
    auto tensor_desc_output = op_desc_->MutableOutputDesc(i);
    if (tensor_desc_output == nullptr) {
      continue;
    }
    // if can not find saved info, it says format and origin format is same when catched
    if (map_ori_format_out_[i] == FORMAT_RESERVED) {
      GELOGD("Node is [%s], output tensor idx [%zu] is not been catched.Skip update action for it!",
             op_desc_->GetName().c_str(), i);
      tensor_desc_output->SetOriginFormat(tensor_desc_output->GetFormat());
      tensor_desc_output->SetOriginShape(tensor_desc_output->MutableShape());
      continue;
    }
    auto ori_shape = tensor_desc_output->MutableShape();
    auto curr_format = tensor_desc_output->GetFormat();
    if (curr_format != map_ori_format_out_[i]) {
      REPORT_INNER_ERROR("E19999", "Node is %s, out tensor idx is %zu. format: %s, "
                         "recorded origin format: %s is not same", op_desc_->GetName().c_str(), i,
                         TypeUtils::FormatToSerialString(curr_format).c_str(),
                         TypeUtils::FormatToSerialString(map_ori_format_out_[i]).c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Node is %s, out tensor idx is %zu. format: %s, "
             "recorded origin format: %s is not same", op_desc_->GetName().c_str(), i,
             TypeUtils::FormatToSerialString(curr_format).c_str(),
             TypeUtils::FormatToSerialString(map_ori_format_out_[i]).c_str());
      return GRAPH_FAILED;
    }
    tensor_desc_output->SetOriginShape(ori_shape);
    auto saved_format = map_format_out_[i];
    if (curr_format == saved_format || saved_format == FORMAT_ND) {
      GELOGD("Nodeis %s, out tensor idx is %zu. ori format: %s, recorded format: %s is same! No need to transfer",
             op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(curr_format).c_str(),
             TypeUtils::FormatToSerialString(saved_format).c_str());
      continue;
    }
    tensor_desc_output->SetFormat(saved_format);
    std::vector<int64_t> ori_shape_dims = ori_shape.GetDims();
    std::vector<int64_t> out_dims;
    ge::DataType dtype =  tensor_desc_output->GetDataType();

    // FE set and Ge get for PadDimention
    string infer_reshape_type;
    (void) AttrUtils::GetStr(*tensor_desc_output, ATTR_NAME_RESHAPE_INFER_TYPE, infer_reshape_type);
    bool is_success = transformer::ExpandDimension(op_desc_->GetType(), curr_format, saved_format, i,
                                                   infer_reshape_type, ori_shape_dims);
    if (!is_success) {
      REPORT_CALL_ERROR("E19999", "ExpandDimension failed, op type:%s.", op_desc_->GetType().c_str());
      GELOGE(GRAPH_FAILED, "[Call][ExpandDimension] failed, op type:%s.", op_desc_->GetType().c_str());
      return GRAPH_FAILED;
    }
    transformer::ShapeAndFormat shape_and_format_info {ori_shape_dims, out_dims, curr_format, saved_format, dtype,
                                                       transformer::EN_IMPL_CUSTOM_TBE};
    shape_transfer.GetShapeAccordingToFormat(shape_and_format_info);
    tensor_desc_output->SetShape(GeShape(out_dims));
    GELOGD("Node is %s, out tensor idx is %zu. Update format and shape success, ori format: %s, format: %s",
        op_desc_->GetName().c_str(), i, TypeUtils::FormatToSerialString(curr_format).c_str(),
        TypeUtils::FormatToSerialString(saved_format).c_str());
  }
  GELOGD("Node is %s. Update format and shape success", op_desc_->GetName().c_str());
  return true;
}
} // namespace ge
