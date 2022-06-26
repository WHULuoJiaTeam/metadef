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

#include "graph/op_desc.h"

#include "debug/ge_attr_define.h"
#include "debug/ge_util.h"
#include "external/graph/operator.h"
#include "framework/common/debug/ge_log.h"
#include "common/util/error_manager/error_manager.h"
#include "graph/common_error_codes.h"
#include "graph/ge_attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/operator_factory_impl.h"
#include "graph/op_desc_impl.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/transformer_utils.h"
#include "proto/ge_ir.pb.h"

using std::make_pair;
using std::shared_ptr;
using std::string;
using std::vector;

/*lint -save -e521 -e681 -e732 -e737*/
namespace ge {
const std::string ATTR_NAME_ID = "id";

const std::string ATTR_NAME_STREAM_ID = "stream_id";

const std::string ATTR_NAME_INPUT_NAME = "input_name";

const std::string ATTR_NAME_SRC_NAME = "src_name";

const std::string ATTR_NAME_SRC_INDEX = "src_index";

const std::string ATTR_NAME_INPUT = "input";

const std::string ATTR_NAME_OUTPUT = "output";

const std::string ATTR_NAME_INPUT_DESC = "input_desc";

const std::string ATTR_NAME_OUTPUT_DESC = "output_desc";

const std::string ATTR_NAME_DST_NAME = "dst_name";

const std::string ATTR_NAME_DST_INDEX = "dst_index";

const std::string ATTR_NAME_WORKSPACE = "workspace";

const std::string ATTR_NAME_WORKSPACE_BYTES = "workspace_bytes";

const std::string ATTR_NAME_IS_INPUT_CONST = "is_input_const";

const std::string ATTR_NAME_OP_INFER_DEPENDS = "_op_infer_depends";

const std::string ATTR_NAME_OP_KERNEL_LIB_NAME = "_ge_attr_op_kernel_lib_name";

OpDescImpl::OpDescImpl() {
  op_def_.InitDefault();
  if (op_def_.GetProtoMsg() != nullptr) {
    op_def_.GetProtoMsg()->set_has_out_attr(true);
  }
}

OpDescImpl::OpDescImpl(const std::string &name, const std::string &type) {
  op_def_.InitDefault();
  if (op_def_.GetProtoMsg() != nullptr) {
    op_def_.GetProtoMsg()->set_has_out_attr(true);
  }
  SetName(name);
  SetType(type);
}

OpDescImpl::OpDescImpl(const ProtoMsgOwner &proto_msg_owner,
                       ge::proto::OpDef *op_def)
    : op_def_(proto_msg_owner, op_def) {}

string OpDescImpl::GetName() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->name();
  }
  return "";
}

void OpDescImpl::SetName(const std::string &name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_name(name);
  }
}

string OpDescImpl::GetType() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->type();
  }
  return "";
}

void OpDescImpl::SetType(const string &type) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_type(type);
  }
}

graphStatus OpDescImpl::AddInputDesc(const ge::GeTensorDesc &input_desc) {
  int index = static_cast<int>(inputs_desc_.size());
  return AddInputDesc("__input" + std::to_string(index), input_desc);
}

graphStatus OpDescImpl::AddInputDesc(uint32_t index, const ge::GeTensorDesc &input_desc) {
  graphStatus ret = GRAPH_SUCCESS;
  if (index < inputs_desc_.size()) {
    //  InputsDesc[index] is exist, then update it
    ret = UpdateInputDesc(index, input_desc);
  } else {
    //  InputDesc[index] is not exist, then add it
    ret = AddInputDesc(input_desc);
  }
  return ret;
}

graphStatus OpDescImpl::AddInputDesc(const string &name, const ge::GeTensorDesc &input_desc) {
  if (input_name_idx_.find(name) != input_name_idx_.end()) {
    GELOGI("input %s is exist,  update it", name.c_str());
    graphStatus ret = UpdateInputDesc(name, input_desc);
    return ret;
  } else {
    int index = static_cast<int>(inputs_desc_.size());
    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(input_desc);
    if (in_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddInputDesc failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddInputDesc failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    inputs_desc_.push_back(in_desc);
    (void)input_name_idx_.insert(make_pair(name, index));
    if (find(register_input_name_.begin(), register_input_name_.end(), name) == register_input_name_.end()) {
      register_input_name_.push_back(name);
    }

    return GRAPH_SUCCESS;
  }
}

graphStatus OpDescImpl::AddInputDescMiddle(const string &name, const unsigned int num, size_t index) {
  for (unsigned int i = 0; i < num; i++) {
    string input_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC((input_name_idx_.find(input_name) == input_name_idx_.end()),
                     REPORT_INNER_ERROR("E19999", "Add input tensor_desc is existed. name[%s]", input_name.c_str());
                     GELOGE(ge::FAILED, "[Check][Param] Add input tensor_desc is existed. name[%s]",
                            input_name.c_str());
                     return GRAPH_FAILED);

    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddInputDescMiddle failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddInputDescMiddle failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    if (index > inputs_desc_.size()) {
      REPORT_INNER_ERROR("E19999", "AddInputDescMiddle failed, as param index(%zu) "
             "is bigger than inputs size(%zu).", index, inputs_desc_.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] AddInputDescMiddle failed, as param index(%zu) "
             "is bigger than inputs size(%zu).", index, inputs_desc_.size());
      return GRAPH_FAILED;
    }

    (void)inputs_desc_.insert(inputs_desc_.begin() + index + i, in_desc);

    // Update index in input_name_idx
    for (auto it = input_name_idx_.begin(); it != input_name_idx_.end(); ++it) {
      if (it->second >= (index + i)) {
        it->second += 1;
      }
    }

    (void)input_name_idx_.insert(make_pair(input_name, i + index));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddOutputDescMiddle(const string &name, const unsigned int num, size_t index) {
  for (unsigned int i = 0; i < num; i++) {
    string output_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC((output_name_idx_.find(output_name) == output_name_idx_.end()),
                     REPORT_INNER_ERROR("E19999", "Add output tensor_desc is existed. name[%s]", output_name.c_str());
                     return GRAPH_FAILED,
                    "[Check][Param] Add output tensor_desc is existed. name[%s]", output_name.c_str());

    std::shared_ptr<GeTensorDesc> out_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (out_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddOutputDescMiddle failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddOutputDescMiddle failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    if (index > outputs_desc_.size()) {
      REPORT_INNER_ERROR("E19999", "AddOutputDescMiddle failed, as param index(%zu) "
                         "is bigger than outputs size(%zu).", index, outputs_desc_.size());
      GELOGE(GRAPH_FAILED, "[Check][Param] AddOutputDescMiddle failed, as param index(%zu) "
             "is bigger than outputs size(%zu).", index, outputs_desc_.size());
      return GRAPH_FAILED;
    }

    (void)outputs_desc_.insert(outputs_desc_.begin() + index + i, out_desc);

    // Update index in input_name_idx
    for (auto it = output_name_idx_.begin(); it != output_name_idx_.end(); ++it) {
      if (it->second >= (index + i)) {
        it->second += 1;
      }
    }

    (void)output_name_idx_.insert(make_pair(output_name, i + index));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddInputDescForward(const string &name, const unsigned int num) {
  for (unsigned int i = 0; i < num; i++) {
    string input_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC((input_name_idx_.find(input_name) == input_name_idx_.end()),
                     REPORT_INNER_ERROR("E19999", "Add input tensor_desc is existed. name[%s]", input_name.c_str());
                     return GRAPH_FAILED,
                     "[Check][Param] Add input tensor_desc is existed. name[%s]", input_name.c_str());

    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddInputDescForward failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddInputDescForward failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }
    (void)inputs_desc_.insert(inputs_desc_.begin(), in_desc);

    // Update index in input_name_idx
    for (auto it = input_name_idx_.begin(); it != input_name_idx_.end(); ++it) {
      it->second += 1;
    }

    (void)input_name_idx_.insert(make_pair(input_name, 0));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddOutputDescForward(const string &name, const unsigned int num) {
  for (unsigned int i = 0; i < num; i++) {
    string output_name = name + std::to_string(i);
    GE_CHK_BOOL_EXEC((output_name_idx_.find(output_name) == output_name_idx_.end()),
                     REPORT_INNER_ERROR("E19999", "Add output tensor_desc is existed. name[%s]", output_name.c_str());
                     return GRAPH_FAILED,
                     "[Check][Param] Add output tensor_desc is existed. name[%s]", output_name.c_str());

    std::shared_ptr<GeTensorDesc> in_desc = ComGraphMakeShared<GeTensorDesc>(GeTensorDesc());
    if (in_desc == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddOutputDescForward failed, as malloc shared_ptr failed.");
      GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddOutputDescForward failed, as malloc shared_ptr failed.");
      return GRAPH_FAILED;
    }

    (void)outputs_desc_.insert(outputs_desc_.begin(), in_desc);

    // Update index in output_name_idx
    for (auto it = output_name_idx_.begin(); it != output_name_idx_.end(); ++it) {
      it->second += 1;
    }
    (void)output_name_idx_.insert(make_pair(output_name, 0));
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddOptionalInputDesc(const string &name,
                                             const ge::GeTensorDesc &input_desc) {
  if (OpDescImpl::AddInputDesc(name, input_desc) == GRAPH_FAILED) {
    return GRAPH_FAILED;
  }
  (void)optional_input_names_.insert(name);
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::UpdateInputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc) {
  if (index >= inputs_desc_.size()) {
    GELOGW("[UpdateInput][Check] Input index is invalid, index=%u, input_size=%zu", index, inputs_desc_.size());
    return GRAPH_FAILED;
  }

  inputs_desc_[index] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (inputs_desc_[index] == nullptr) {
    REPORT_CALL_ERROR("E19999", "UpdateInputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateInputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

bool OpDescImpl::OpDescMembersAreEqual(const OpDescImpl &r_op_desc) const {
  return (IsEqual(this->input_name_idx_, r_op_desc.input_name_idx_, "OpDesc.input_name_idx_") &&
          IsEqual(this->output_name_idx_, r_op_desc.output_name_idx_, "OpDesc.output_name_idx_") &&
          IsEqual(this->optional_input_names_, r_op_desc.optional_input_names_, "OpDesc.optional_input_names_") &&
          IsEqual(this->engine_name_, r_op_desc.engine_name_, "OpDesc.engine_name_") &&
          IsEqual(this->op_kernel_lib_name_, r_op_desc.op_kernel_lib_name_, "OpDesc.op_kernel_lib_name_"));
}

bool OpDescImpl::OpDescAttrsAreEqual(const OpDescImpl &r_op_desc) const {
  const auto &op_def = this->op_def_.GetProtoMsg();
  const auto &r_op_def = r_op_desc.op_def_.GetProtoMsg();
  if ((op_def != nullptr) && (r_op_def != nullptr)) {
    // Message OpDef in ge_ir.proto
    return (
        IsEqual(op_def->name(), r_op_def->name(), "OpDef_.name()") &&
        IsEqual(op_def->type(), r_op_def->type(), "OpDef_.type()") &&
        IsEqual(ToString(op_def->input()), ToString(r_op_def->input()), "OpDef_.input()") &&
        IsEqual(op_def->has_out_attr(), r_op_def->has_out_attr(), "OpDef_.has_out_attr()") &&
        IsEqual(op_def->stream_id(), r_op_def->stream_id(), "OpDef_.stream_id()") &&
        IsEqual(ToString(op_def->input_name()), ToString(r_op_def->input_name()), "OpDef_.input_name()") &&
        IsEqual(ToString(op_def->src_name()), ToString(r_op_def->src_name()), "OpDef_.src_name()") &&
        IsEqual(ToString(op_def->dst_name()), ToString(r_op_def->dst_name()), "OpDef_.dst_name()") &&
        IsEqual(ToString(op_def->src_index()), ToString(r_op_def->src_index()), "OpDef_.src_index()") &&
        IsEqual(ToString(op_def->dst_index()), ToString(r_op_def->dst_index()), "OpDef_.dst_index()") &&
        IsEqual(ToString(op_def->input_i()), ToString(r_op_def->input_i()), "OpDef_.input_i()") &&
        IsEqual(ToString(op_def->output_i()), ToString(r_op_def->output_i()), "OpDef_.output_i()") &&
        IsEqual(ToString(op_def->workspace()), ToString(r_op_def->workspace()), "OpDef_.workspace()") &&
        IsEqual(ToString(op_def->workspace_bytes()), ToString(r_op_def->workspace_bytes()),
                "OpDef_.workspace_bytes()") &&
        IsEqual(ToString(op_def->is_input_const()), ToString(r_op_def->is_input_const()), "OpDef_.is_input_const()"));
  } else {
    return ((op_def == nullptr) && (r_op_def == nullptr));
  }
}

bool OpDescImpl::OpDescGenTensorDescsAreEqual(const OpDescImpl &r_op_desc)
const {
  // 1.Verify inputs and outputs desc size
  const auto inputs_desc_size = this->inputs_desc_.size();
  const auto r_inputs_desc_size = r_op_desc.inputs_desc_.size();
  if (inputs_desc_size != r_inputs_desc_size) {
    REPORT_INNER_ERROR("E19999", "param r_op_desc inputs count(%zu) not equal to %s inputs count(%zu), "
                       "verify failed.", r_inputs_desc_size, this->GetName().c_str(), inputs_desc_size);
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of OpDesc's inputs desc verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  const auto outputs_desc_size = this->outputs_desc_.size();
  const auto r_outputs_desc_size = r_op_desc.outputs_desc_.size();
  if (outputs_desc_size != r_outputs_desc_size) {
    REPORT_INNER_ERROR("E19999", "param r_op_desc outputs count(%zu) not equal to %s outputs count(%zu), "
                       "verify failed.", r_inputs_desc_size, this->GetName().c_str(), inputs_desc_size);
    GELOGE(GRAPH_FAILED, "[Check][Param] Size of OpDesc's outputs desc verify failed, node name: %s.",
           this->GetName().c_str());
    return false;
  }
  // 2.Verify all inputs desc equal
  for (uint32_t i = 0; i < inputs_desc_size; i++) {
    const auto &in_ge_tensor_desc = this->GetInputDesc(i);
    const auto &r_in_ge_tensor_desc = r_op_desc.GetInputDesc(i);
    // Determine the connection relationship by GeTensorDesc
    if (!(in_ge_tensor_desc == r_in_ge_tensor_desc)) {
      REPORT_INNER_ERROR("E19999", "r_op_desc inputdesc(index:%u) not equal to %s inputdesc(index:%u), "
                         "verify failed.", i, this->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Link info of OpDesc's inputs desc verify failed, OpDesc name: %s.",
             this->GetName().c_str());
      return false;
    }
  }
  // 3.Verify all outputs desc equal
  for (uint32_t i = 0; i < outputs_desc_size; i++) {
    const auto &out_ge_tensor_desc = this->GetOutputDesc(i);
    const auto &r_out_ge_tensor_desc = r_op_desc.GetOutputDesc(i);
    if (!(out_ge_tensor_desc == r_out_ge_tensor_desc)) {
      REPORT_INNER_ERROR("E19999", "r_op_desc outputdesc(index:%u) not equal to %s outputdesc(index:%u), "
                         "verify failed.", i, this->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Link info of OpDesc's outputs desc verify failed, OpDesc name: %s.",
             this->GetName().c_str());
      return false;
    }
  }
  return true;
}

bool OpDescImpl::operator==(const OpDescImpl &r_op_desc) const {
  return (OpDescAttrsAreEqual(r_op_desc) && OpDescMembersAreEqual(r_op_desc) &&
          OpDescGenTensorDescsAreEqual(r_op_desc));
}

graphStatus OpDescImpl::UpdateInputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc) {
  auto it = input_name_idx_.find(name);
  if (it == input_name_idx_.end()) {
    GELOGW("[UpdateInput][Check] Can not find input desc named %s", name.c_str());
    return GRAPH_FAILED;
  }
  if (it->second >= inputs_desc_.size()) {
    REPORT_INNER_ERROR("E19999", "%u is out of range(0, %zu), check invalid", it->second, inputs_desc_.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] [%u] more than size:%zu of inputs_desc_", it->second, inputs_desc_.size());
    return GRAPH_FAILED;
  }

  inputs_desc_[it->second] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (inputs_desc_[it->second] == nullptr) {
    REPORT_CALL_ERROR("E19999", "UpdateInputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateInputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

bool OpDescImpl::InputIsSet(const string &name) const {
  auto it = input_name_idx_.find(name);
  if (it != input_name_idx_.end()) {
    GE_IF_BOOL_EXEC(it->second >= inputs_desc_.size(),
                    REPORT_INNER_ERROR("E19999", "input name(%s) id(%u) is out of range(0, %zu), check invalid",
                                       name.c_str(), it->second, inputs_desc_.size());
                    GELOGE(GRAPH_FAILED, "[Check][Param] it->second is invalid."); return false);
    auto tensor_desc = inputs_desc_[it->second];
    GE_IF_BOOL_EXEC(tensor_desc == nullptr,
                    REPORT_INNER_ERROR("E19999", "tensor_desc(index:%u) is null.", it->second);
                    GELOGE(GRAPH_FAILED, "[Check][Param] tensor_desc(index:%u) is null.", it->second); return false);
    auto dims = tensor_desc->GetShape().GetDims();
    if (dims.size() > 0) {
      return true;
    }
  }
  return false;
}

GeTensorDesc OpDescImpl::GetInputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG(index < inputs_desc_.size(), GeTensorDesc());
  return *(inputs_desc_[index].get());
}

GeTensorDesc OpDescImpl::GetInputDesc(const string &name) const {
  auto it = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), GeTensorDesc());
  GE_CHK_BOOL_RET_STATUS_NOLOG(it->second < inputs_desc_.size(), GeTensorDesc());
  return *(inputs_desc_[it->second].get());
}

GeTensorDescPtr OpDescImpl::MutableInputDesc(uint32_t index) const {
  GE_CHK_BOOL_EXEC(index < inputs_desc_.size(),
                   REPORT_INNER_ERROR("E19999", "index(%u) is out of range(0, %zu), check invalid",
                                      index, inputs_desc_.size());
                   GELOGE(ge::FAILED, "[Check][Param] Can't find the input desc %u", index);
                   return nullptr);
  if (inputs_desc_[index] == nullptr) {
    return nullptr;
  }
  if (inputs_desc_[index]->IsValid() != GRAPH_SUCCESS) {
    GELOGW("[Get][InputDesc] Input desc is invalid");
    return nullptr;
  }
  return inputs_desc_[index];
}

GeTensorDescPtr OpDescImpl::MutableInputDesc(const string &name) const {
  auto input_name_idx = GetAllInputName();
  auto it = input_name_idx.find(name);
  if (it == input_name_idx.end()) {
    GELOGW("[Get][InputDesc] Failed to get [%s] input desc", name.c_str());
    return nullptr;
  }
  return MutableInputDesc(it->second);
}

OpDesc::Vistor<string> OpDescImpl::GetAllInputNames(const ConstOpDescPtr &op_desc) const {
  vector<string> names;
  if (input_name_idx_.empty()) {
    return OpDesc::Vistor<string>(op_desc, names);
  }
  for (std::pair<string, uint32_t> input : input_name_idx_) {
    names.push_back(input.first);
  }
  return OpDesc::Vistor<string>(op_desc, names);
}

void OpDescImpl::SetOpKernelLibName(const std::string &name) {
  op_kernel_lib_name_ = name;
}

std::string OpDescImpl::GetOpKernelLibName() const {
  if (!op_kernel_lib_name_.empty()) {
    return op_kernel_lib_name_;
  }
  return "";
}

void OpDescImpl::SetOpEngineName(const std::string &name) {
  engine_name_ = name;
}

std::string OpDescImpl::GetOpEngineName() const { return engine_name_; }

OpDesc::Vistor<GeTensorDesc> OpDescImpl::GetAllInputsDesc(const ConstOpDescPtr &op_desc) const {
  vector<GeTensorDesc> temp{};
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      temp.push_back(*it);
    } else {
      GELOGW("[Get][InputDesc] This input_desc is invalid, it won't be return");
      continue;
    }
  }
  return OpDesc::Vistor<GeTensorDesc>(op_desc, temp);
}

OpDesc::Vistor<GeTensorDescPtr> OpDescImpl::GetAllInputsDescPtr(const ConstOpDescPtr &op_desc) const {
  vector<GeTensorDescPtr> temp{};
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      temp.push_back(it);
    } else {
      GELOGW("[Get][InputDesc] This input_desc is invalid, it won't be return");
      continue;
    }
  }
  return OpDesc::Vistor<GeTensorDescPtr>(op_desc, temp);
}

size_t OpDescImpl::GetInputsSize() const {
  //  Just return valid inputs size.InValid desc is set in default OPTION_INPUT register.
  size_t size = 0;
  for (const auto &it : inputs_desc_) {
    if (it->IsValid() == GRAPH_SUCCESS) {
      size++;
    }
  }
  return size;
}

size_t OpDescImpl::GetAllInputsSize() const { return inputs_desc_.size(); }

graphStatus OpDescImpl::AddOutputDesc(const ge::GeTensorDesc &output_desc) {
  int index = static_cast<int>(outputs_desc_.size());
  return AddOutputDesc("__output" + std::to_string(index), output_desc);
}

graphStatus OpDescImpl::AddOutputDesc(const string &name, const ge::GeTensorDesc &output_desc) {
  GE_CHK_BOOL_EXEC((output_name_idx_.find(name) == output_name_idx_.end()),
                   REPORT_INNER_ERROR("E19999", "Add output tensor_Desc is existed. name[%s]", name.c_str());
                   return GRAPH_FAILED,
                   "[Check][Param] Add output tensor_Desc is existed. name[%s]", name.c_str());
  int index = static_cast<int>(outputs_desc_.size());

  std::shared_ptr<GeTensorDesc> tensor = ComGraphMakeShared<GeTensorDesc>(output_desc);
  if (tensor == nullptr) {
    REPORT_CALL_ERROR("E19999", "AddOutputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] AddOutputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  outputs_desc_.push_back(tensor);
  (void)output_name_idx_.insert(make_pair(name, index));
  if (find(register_output_name_.begin(), register_output_name_.end(), name) == register_output_name_.end()) {
    register_output_name_.push_back(name);
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::UpdateOutputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc) {
  GE_CHK_BOOL_EXEC((index < outputs_desc_.size()),
                   REPORT_INNER_ERROR("E19999", "param index(%u) is out of range(0, %zu), check invalid",
                                      index, outputs_desc_.size());
                   return GRAPH_FAILED,
                   "[Check][Param] The index is invalid. index[%u]", index);
  outputs_desc_[index] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (outputs_desc_[index] == nullptr) {
    REPORT_CALL_ERROR("E19999", "UpdateOutputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateOutputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::UpdateOutputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc) {
  auto it = output_name_idx_.find(name);
  if (it == output_name_idx_.end()) {
    GELOGW("[Update][OutputDesc] Can not find the output desc named %s", name.c_str());
    return GRAPH_FAILED;
  }
  GE_IF_BOOL_EXEC(it->second >= outputs_desc_.size(),
                  REPORT_INNER_ERROR("E19999", "output name(%s) idx(%u) is out of range(0, %zu), check invalid",
                                     name.c_str(), it->second, outputs_desc_.size());
                  GELOGE(GRAPH_FAILED, "[Check][Param] it->second is invalid.");
                  return GRAPH_FAILED);
  outputs_desc_[it->second] = ComGraphMakeShared<GeTensorDesc>(tensor_Desc);
  if (outputs_desc_[it->second] == nullptr) {
    REPORT_CALL_ERROR("E19999", "UpdateOutputDesc failed, as malloc shared_ptr failed.");
    GELOGE(GRAPH_FAILED, "[Create][GeTensorDesc] UpdateOutputDesc failed, as malloc shared_ptr failed.");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GeTensorDesc OpDescImpl::GetOutputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG(index < outputs_desc_.size(), GeTensorDesc());
  return *(outputs_desc_[index].get());
}

GeTensorDesc OpDescImpl::GetOutputDesc(const string &name) const {
  auto it = output_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != output_name_idx_.end(), GeTensorDesc());
  GE_CHK_BOOL_RET_STATUS_NOLOG(it->second < outputs_desc_.size(), GeTensorDesc());
  return *(outputs_desc_[it->second].get());
}

GeTensorDescPtr OpDescImpl::MutableOutputDesc(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS(index < outputs_desc_.size(), nullptr, "Cann't find the output desc %u", index);
  return outputs_desc_[index];
}

GeTensorDescPtr OpDescImpl::MutableOutputDesc(const string &name) const {
  auto it = output_name_idx_.find(name);
  if (it == output_name_idx_.end()) {
    GELOGW("[Update][OutputDesc] Can not find the output desc named %s", name.c_str());
    return nullptr;
  }
  return MutableOutputDesc(it->second);
}

uint32_t OpDescImpl::GetAllOutputsDescSize() const {
  return static_cast<uint32_t>(outputs_desc_.size());
}

OpDesc::Vistor<GeTensorDesc> OpDescImpl::GetAllOutputsDesc(const ConstOpDescPtr &op_desc) const {
  vector<GeTensorDesc> temp{};
  for (const auto &it : outputs_desc_) {
    temp.push_back(*it);
  }
  return OpDesc::Vistor<GeTensorDesc>(op_desc, temp);
}

OpDesc::Vistor<GeTensorDescPtr> OpDescImpl::GetAllOutputsDescPtr(const ConstOpDescPtr &op_desc) const {
  return OpDesc::Vistor<GeTensorDescPtr>(op_desc, outputs_desc_);
}

size_t OpDescImpl::GetOutputsSize() const { return outputs_desc_.size(); }

ConstGeTensorDescPtr OpDescImpl::GetOutputDescPtr(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(outputs_desc_.size()), nullptr);
  return outputs_desc_[index];
}

ConstGeTensorDescPtr OpDescImpl::GetInputDescPtr(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < static_cast<uint32_t>(inputs_desc_.size()), nullptr);
  if (inputs_desc_[index] == nullptr) {
    return nullptr;
  }
  if (inputs_desc_[index]->IsValid() != GRAPH_SUCCESS) {
    GELOGW("[Get][InputDesc] Input desc %u is invalid", index);
    return nullptr;
  } else {
    return inputs_desc_[static_cast<size_t>(index)];
  }
}

ConstGeTensorDescPtr OpDescImpl::GetInputDescPtrDfault(uint32_t index) const {
  GE_CHK_BOOL_RET_STATUS_NOLOG((index) < (uint32_t)(inputs_desc_.size()), nullptr);
  return inputs_desc_[(int32_t)index];
}

ConstGeTensorDescPtr OpDescImpl::GetInputDescPtr(const string &name) const {
  auto it = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), shared_ptr<const GeTensorDesc>());
  return inputs_desc_[it->second];
}

graphStatus OpDescImpl::AddRegisterInputName(const std::string &name) {
  if (find(register_input_name_.begin(), register_input_name_.end(), name) == register_input_name_.end()) {
    register_input_name_.push_back(name);
  }

  return GRAPH_SUCCESS;
}

vector<string> OpDescImpl::GetRegisterInputName() const {
  return register_input_name_;
}

graphStatus OpDescImpl::AddDynamicInputDesc(const string &name, const unsigned int num, bool is_push_back) {
  if (is_push_back) {
    for (unsigned int i = 0; i < num; i++) {
      if (AddInputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS)
        return GRAPH_FAILED;
    }
  } else {
    if (AddInputDescForward(name, num) != GRAPH_SUCCESS)
      return GRAPH_FAILED;
  }
  if (AddRegisterInputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddDynamicInputDescByIndex(const string &name, const unsigned int num, size_t index) {
  if (AddInputDescMiddle(name, num, index) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::AddRegisterOutputName(const string &name) {
  if (find(register_output_name_.begin(), register_output_name_.end(), name) == register_output_name_.end()) {
    register_output_name_.push_back(name);
  }

  return GRAPH_SUCCESS;
}

vector<string> OpDescImpl::GetRegisterOutputName() const {
  return register_output_name_;
}

graphStatus OpDescImpl::AddDynamicOutputDesc(const string &name, const unsigned int num, bool is_push_back) {
  if (is_push_back) {
    for (unsigned int i = 0; i < num; i++) {
      if (AddOutputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS)
        return GRAPH_FAILED;
    }
  } else {
    if (AddOutputDescForward(name, num) != GRAPH_SUCCESS)
      return GRAPH_FAILED;
  }

  if (AddRegisterOutputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OpDescImpl::IsOptionalInput(const string &name) const {
  return optional_input_names_.find(name) != optional_input_names_.end();
}

bool OpDescImpl::IsOptionalInput(uint32_t index) const { return IsOptionalInput(GetInputNameByIndex(index)); }

std::map<string, uint32_t> OpDescImpl::GetAllInputName() const { return input_name_idx_; }

std::map<string, uint32_t> OpDescImpl::GetAllOutputName() { return output_name_idx_; }

std::map<string, uint32_t>& OpDescImpl::MutableAllInputName() { return input_name_idx_; }

std::map<string, uint32_t>& OpDescImpl::MutableAllOutputName() { return output_name_idx_; }

bool OpDescImpl::UpdateInputName(std::map<string, uint32_t> input_name_idx) {
  // Use inputDesc_.size() to contain the InValid OptionInput.GetInputsSize() will remove default OptionInput name.
  auto input_map_size = inputs_desc_.size();
  auto factory_map_size = input_name_idx.size();
  // It indicates that some inputs have no optional name.
  // The redundant optional name of factory needs to be deleted and then assigned
  if (input_map_size < factory_map_size) {
    GELOGI("org_input_name_num=%zu, factory_input_name_num=%zu", input_map_size, factory_map_size);
    for (auto it = input_name_idx.begin(); it != input_name_idx.end();) {
      if (it->second >= input_map_size) {
        it = input_name_idx.erase(it);
      } else {
        ++it;
      }
    }
    if (input_name_idx.size() == input_map_size) {
      GELOGI("UpdateInputName");
      input_name_idx_ = input_name_idx;
    } else {
      GELOGW("[Update][InputName] After update, org_input_name_num=%zu, factory_input_name_num=%zu", input_map_size,
             input_name_idx.size());
      return false;
    }
  } else if (input_map_size == factory_map_size) {
    input_name_idx_ = input_name_idx;
  } else {
    GELOGW("[Update][InputName] factory_input_name_num can not be less than org_input_name_num, exactly "
           "org_input_name_num=%zu, factory_input_name_num=%zu", input_map_size, factory_map_size);
    return false;
  }
  return true;
}

bool OpDescImpl::UpdateOutputName(std::map<string, uint32_t> output_name_idx) {
  size_t output_map_size = GetAllOutputsDescSize();
  size_t factory_map_size = output_name_idx.size();
  if (output_map_size < factory_map_size) {
    GELOGI("org_output_name_num=%zu, factory_output_name_num=%zu", output_map_size, factory_map_size);
    for (auto it = output_name_idx.begin(); it != output_name_idx.end();) {
      if (it->second >= output_map_size) {
        it = output_name_idx.erase(it);
      } else {
        ++it;
      }
    }
    if (output_name_idx.size() == output_map_size) {
      GELOGI("UpdateOutputName");
      output_name_idx_ = output_name_idx;
      return true;
    }
  } else if (output_map_size == factory_map_size) {
    output_name_idx_ = output_name_idx;
    return true;
  } else {
    GELOGW("[Update][OutputName] factory_output_name_num can not be less than org_output_name_num, exactly "
           "org_output_name_num=%zu, factory_output_name_num=%zu", output_map_size, output_name_idx.size());
    return false;
  }
  GELOGW("[Update][OutputName] After update, org_output_name_num=%zu, factory_output_name_num=%zu", output_map_size,
         factory_map_size);
  return false;
}

std::function<graphStatus(Operator &)> OpDescImpl::GetInferFunc() const { return infer_func_; }

std::function<graphStatus(Operator &)> OpDescImpl::GetVerifyFunc() const { return verifier_func_; }

void OpDescImpl::AddInferFunc(const std::function<graphStatus(Operator &)> &func) { infer_func_ = func; }

std::function<graphStatus(Operator &)> OpDescImpl::GetInferFormatFunc() const { return infer_format_func_; }

void OpDescImpl::AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func) { infer_format_func_ = func; }

void OpDescImpl::AddVerifierFunc(const std::function<graphStatus(Operator &)> &func) { verifier_func_ = func; }

graphStatus OpDescImpl::InferShapeAndType(const OpDescPtr &op_desc) {
  if (infer_func_ == nullptr) {
    infer_func_ = OperatorFactoryImpl::GetInferShapeFunc(GetType());
    if (infer_func_ == nullptr) {
      GELOGW("[InferShape][Check] %s does not have infer_func.", GetName().c_str());
      /// The infer_func has not been added for each operator in the current operator information library.
      /// No infer_func added operator skips the call
      /// and directly uses the shape information passed down by the upper framework
      return GRAPH_SUCCESS;
    }
  }
  Operator op_proxy = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  graphStatus ret = (graphStatus)infer_func_(op_proxy);
  op_proxy.BreakConnect();
  return ret;
}

graphStatus OpDescImpl::DefaultInferFormat(const ConstOpDescPtr &op_desc) {
  ge::Format first_none_nd_format = FORMAT_ND;
  auto input_descs = GetAllInputsDescPtr(op_desc);
  auto output_descs = GetAllOutputsDescPtr(op_desc);
  // Overall input and output,get the first non-nd format
  for (const auto &input_desc : input_descs) {
    Format origin_format = input_desc->GetOriginFormat();
    if (origin_format != FORMAT_ND) {
      first_none_nd_format = origin_format;
      break;
    }
  }
  for (const auto &output_desc : output_descs) {
    Format origin_format = output_desc->GetOriginFormat();
    if (origin_format != FORMAT_ND) {
      first_none_nd_format = origin_format;
      break;
    }
  }
  // Refresh all input output format
  GELOGD("Default infer format.node[%s], first none nod format is:%d", GetName().c_str(), first_none_nd_format);

  for (const auto &input_desc : input_descs) {
    Format origin_format = input_desc->GetOriginFormat();
    GELOGD("Default infer format[in].node[%s].origin format is:%d", GetName().c_str(), origin_format);
    if (origin_format == FORMAT_ND) {
      input_desc->SetOriginFormat(first_none_nd_format);
      input_desc->SetFormat(first_none_nd_format);
    }
  }
  for (const auto &output_desc : output_descs) {
    Format origin_format = output_desc->GetOriginFormat();
    GELOGD("Default infer format[out].node[%s].origin format is:%d", GetName().c_str(), origin_format);
    if (origin_format == FORMAT_ND) {
      output_desc->SetOriginFormat(first_none_nd_format);
      output_desc->SetFormat(first_none_nd_format);
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::OpVerify(const OpDescPtr &op_desc) {
  if (verifier_func_ == nullptr) {
    verifier_func_ = OperatorFactoryImpl::GetVerifyFunc(GetType());
  }
  if (verifier_func_ != nullptr) {
    Operator op_proxy = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    graphStatus ret = (graphStatus)verifier_func_(op_proxy);
    op_proxy.BreakConnect();
    return ret;
  }
  return GRAPH_SUCCESS;
}

string OpDescImpl::GetInputNameByIndex(uint32_t index) const {
  auto it = input_name_idx_.begin();
  for (; it != input_name_idx_.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != input_name_idx_.end(), "");
  return it->first;
}

int OpDescImpl::GetInputIndexByName(const string &name) const {
  auto it_find = input_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != input_name_idx_.end(), -1);
  return static_cast<int>(it_find->second);
}

int OpDescImpl::GetValidInputIndexByName(const string &name) const {
  map<string, uint32_t> valid_input_name_idx{};
  uint32_t j = 0;
  for (size_t i = 0; i < GetAllInputsSize(); i++) {
    if (MutableInputDesc(static_cast<uint32_t>(i)) != nullptr) {
      auto valid_name = GetInputNameByIndex(static_cast<uint32_t>(i));
      GE_CHK_BOOL_RET_STATUS_NOLOG(!valid_name.empty(), -1);
      valid_input_name_idx.insert({valid_name, j});
      j++;
    }
  }
  auto it_find = valid_input_name_idx.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != valid_input_name_idx.end(), -1);
  return static_cast<int>(it_find->second);
}

string OpDescImpl::GetValidInputNameByIndex(uint32_t index) const {
  map<string, uint32_t> valid_input_name_idx{};
  uint32_t j = 0;
  for (size_t i = 0; i < GetAllInputsSize(); i++) {
    if (MutableInputDesc(static_cast<uint32_t>(i)) != nullptr) {
      auto valid_name = GetInputNameByIndex(static_cast<uint32_t>(i));
      GE_CHK_BOOL_RET_STATUS_NOLOG(!valid_name.empty(), "");
      valid_input_name_idx.insert({valid_name, j});
      j++;
    }
  }
  auto it = valid_input_name_idx.begin();
  for (; it != valid_input_name_idx.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != valid_input_name_idx.end(), "");
  return it->first;
}

string OpDescImpl::GetOutputNameByIndex(uint32_t index) const {
  auto it = output_name_idx_.begin();
  for (; it != output_name_idx_.end(); ++it) {
    if (it->second == index) {
      break;
    }
  }
  GE_CHK_BOOL_RET_STATUS_NOLOG(it != output_name_idx_.end(), "");
  return it->first;
}

int OpDescImpl::GetOutputIndexByName(const string &name) const {
  auto it_find = output_name_idx_.find(name);
  GE_CHK_BOOL_RET_STATUS_NOLOG(it_find != output_name_idx_.end(), -1);
  return static_cast<int>(it_find->second);
}

ProtoAttrMapHelper OpDescImpl::MutableAttrMap() {
  if (op_def_.GetProtoMsg() == nullptr) {
    GELOGE(GRAPH_FAILED, "[Get][ProtoMsg] failed");
    return GeIrProtoHelper<ProtoAttrMap>();
  }
  return ProtoAttrMapHelper(op_def_.GetProtoOwner(), op_def_.GetProtoMsg()->mutable_attr());
}

ConstProtoAttrMapHelper OpDescImpl::GetAttrMap() const {
  return ConstProtoAttrMapHelper(op_def_.GetProtoOwner(), &op_def_.GetProtoMsg()->attr());
}

void OpDescImpl::SetId(int64_t id) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_id(id);
  }
}

int64_t OpDescImpl::GetId() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->id();
  }
  return 0;
}

void OpDescImpl::SetStreamId(int64_t stream_id) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->set_stream_id(stream_id);
  }
}

int64_t OpDescImpl::GetStreamId() const {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    return proto_msg->stream_id();
  }
  return 0;
}

void OpDescImpl::SetInputName(const vector<string> &input_name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_input_name();
    for (auto &item : input_name) {
      proto_msg->add_input_name(item);
    }
  }
}

vector<string> OpDescImpl::GetInputName() const {
  vector<string> input_name;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->input_name()) {
      input_name.push_back(item);
    }
  }
  return input_name;
}

void OpDescImpl::SetSrcName(const vector<string> &src_name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_src_name();
    for (auto &item : src_name) {
      proto_msg->add_src_name(item);
    }
  }
}

vector<string> OpDescImpl::GetSrcName() const {
  vector<string> src_name;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->src_name()) {
      src_name.push_back(item);
    }
  }
  return src_name;
}

void OpDescImpl::SetSrcIndex(const vector<int64_t> &src_index) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_src_index();
    for (auto &item : src_index) {
      proto_msg->add_src_index(item);
    }
  }
}

vector<int64_t> OpDescImpl::GetSrcIndex() const {
  vector<int64_t> src_index;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->src_index()) {
      src_index.push_back(item);
    }
  }
  return src_index;
}

void OpDescImpl::SetInputOffset(const vector<int64_t> &input) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_input_i();
    for (auto &item : input) {
      proto_msg->add_input_i(item);
    }
  }
}

vector<int64_t> OpDescImpl::GetInputOffset() const {
  vector<int64_t> input;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->input_i()) {
      input.push_back(item);
    }
  }
  return input;
}

void OpDescImpl::SetOutputOffset(const vector<int64_t> &output) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_output_i();
    for (auto &item : output) {
      proto_msg->add_output_i(item);
    }
  }
}

vector<int64_t> OpDescImpl::GetOutputOffset() const {
  vector<int64_t> output;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->output_i()) {
      output.push_back(item);
    }
  }
  return output;
}

void OpDescImpl::SetDstName(const vector<string> &dst_name) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_dst_name();
    for (auto &item : dst_name) {
      proto_msg->add_dst_name(item);
    }
  }
}

vector<string> OpDescImpl::GetDstName() const {
  vector<string> dst_name;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->dst_name()) {
      dst_name.push_back(item);
    }
  }
  return dst_name;
}

void OpDescImpl::SetDstIndex(const vector<int64_t> &dst_index) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_dst_index();
    for (auto &item : dst_index) {
      proto_msg->add_dst_index(item);
    }
  }
}

vector<int64_t> OpDescImpl::GetDstIndex() const {
  vector<int64_t> dst_index;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->dst_index()) {
      dst_index.push_back(item);
    }
  }
  return dst_index;
}

void OpDescImpl::SetWorkspace(const vector<int64_t> &workspace) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_workspace();
    for (auto &item : workspace) {
      proto_msg->add_workspace(item);
    }
  }
}

vector<int64_t> OpDescImpl::GetWorkspace() const {
  vector<int64_t> workspace;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->workspace()) {
      workspace.push_back(item);
    }
  }
  return workspace;
}

void OpDescImpl::SetWorkspaceBytes(const vector<int64_t> &workspace_bytes) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_workspace_bytes();
    for (auto &item : workspace_bytes) {
      proto_msg->add_workspace_bytes(item);
    }
  }
}

vector<int64_t> OpDescImpl::GetWorkspaceBytes() const {
  vector<int64_t> workspace_bytes;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto &item : proto_msg->workspace_bytes()) {
      workspace_bytes.push_back(item);
    }
  }
  return workspace_bytes;
}

void OpDescImpl::SetIsInputConst(const vector<bool> &is_input_const) {
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    proto_msg->clear_is_input_const();
    for (auto item : is_input_const) {
      proto_msg->add_is_input_const(item);
    }
  }
}

vector<bool> OpDescImpl::GetIsInputConst() const {
  vector<bool> is_input_const;
  auto proto_msg = op_def_.GetProtoMsg();
  if (proto_msg != nullptr) {
    for (auto item : proto_msg->is_input_const()) {
      is_input_const.push_back(item);
    }
  }
  return is_input_const;
}

graphStatus OpDescImpl::RestoreInputNameIdx(const string &name,
                                            const int &index) {
  if (input_name_idx_.find(name) != input_name_idx_.end()) {
    GELOGI("Restore input name index is existed. name[%s]", name.c_str());
  }
  (void)input_name_idx_.insert(make_pair(name, index));
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::RestoreOutputNameIdx(const string &name,
                                             const int &index) {
  if (output_name_idx_.find(name) != output_name_idx_.end()) {
    GELOGI("Restore output name index is existed. name[%s]", name.c_str());
  }
  (void)output_name_idx_.insert(make_pair(name, index));
  return GRAPH_SUCCESS;
}

graphStatus OpDescImpl::CallInferFunc(Operator &op, const OpDescPtr &op_desc) {
  if (infer_func_ == nullptr) {
    infer_func_ = OperatorFactoryImpl::GetInferShapeFunc(GetType());
    if (infer_func_ == nullptr) {
      GELOGW("[InferShape][Check] %s does not have infer_func.", GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
  }
  std::unique_ptr<NodeShapeTransUtils> transformer(new(std::nothrow) NodeShapeTransUtils(op_desc));
  if (transformer == nullptr) {
    REPORT_CALL_ERROR("E19999", "Alloc Memory failed.");
    GELOGE(GRAPH_FAILED, "[Alloc][Memory] failed");
    return GRAPH_FAILED;
  }
  auto is_init_success = transformer->Init();
  if (!is_init_success) {
    GELOGE(GRAPH_FAILED, "[Call][Init] for transformer failed");
    return GRAPH_FAILED;
  }
  if (!transformer->CatchFormatAndShape()) {
    GELOGE(GRAPH_FAILED, "[Call][CatchFormatAndShape] for transformer failed!");
    return GRAPH_FAILED;
  }
  graphStatus graph_status = (graphStatus)infer_func_(op);
  if (graph_status != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Call][InferFunc] for %s failed. ret:%u", GetName().c_str(), graph_status);
    return GRAPH_FAILED;
  }
  if (!transformer->UpdateFormatAndShape()) {
    GELOGE(GRAPH_FAILED, "[Call][UpdateFormatAndShape] for transformer failed!");
    return GRAPH_FAILED;
  }
  return graph_status;
}

graphStatus OpDescImpl::CallInferFormatFunc(Operator &op, const ConstOpDescPtr &op_desc) {
  if (infer_format_func_ == nullptr) {
    infer_format_func_ = OperatorFactoryImpl::GetInferFormatFunc(GetType());
    if (infer_format_func_ == nullptr) {
      return DefaultInferFormat(op_desc);
    }
  }
  return (graphStatus)infer_format_func_(op);
}

graphStatus OpDescImpl::CallInferValueRangeFunc(Operator &op, const ConstOpDescPtr &op_desc) {
  if (infer_value_range_func_ == nullptr) {
    auto infer_value_range_param = OperatorFactoryImpl::GetInferValueRangePara(GetType());
    if (!infer_value_range_param.is_initialized) {
      REPORT_CALL_ERROR("E19999", "Node %s does not register func to infer value range.", GetName().c_str());
      GELOGE(GRAPH_PARAM_INVALID, "Node %s does not register func to infer value range.", GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }

    infer_value_range_func_ = infer_value_range_param.infer_value_func;
    if (infer_value_range_func_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "Value range infer func of node %s has been registered, but infer func is nullptr.",
                        GetName().c_str());
      GELOGE(GRAPH_PARAM_INVALID, "Value range infer func of node %s has been registered, but infer func is nullptr.",
             GetName().c_str());
      return GRAPH_PARAM_INVALID;
    }
  }
  return (graphStatus) infer_value_range_func_(op);
}

std::string OpDescImpl::GetSubgraphInstanceName(uint32_t index) const {
  if (static_cast<size_t>(index) >= subgraph_instance_names_.size()) {
    return "";
  }
  return subgraph_instance_names_.at(index);
}

const std::vector<std::string> &OpDescImpl::GetSubgraphInstanceNames() const {
  return subgraph_instance_names_;
}

void OpDescImpl::RemoveSubgraphInstanceName(const std::string &name) {
  for (auto iter = subgraph_instance_names_.begin(); iter != subgraph_instance_names_.end(); ++iter) {
    if (*iter == name) {
      *iter = "";
      return;
    }
  }
}

graphStatus OpDescImpl::AddSubgraphName(const std::string &name) {
  GELOGI("Add subgraph name is %s", name.c_str());
  auto iter = subgraph_names_to_index_.find(name);
  if (iter != subgraph_names_to_index_.end()) {
    GELOGW("[Add][Subgraph] Subgraph name %s exists, index %u", name.c_str(), iter->second);
    return GRAPH_FAILED;
  }
  auto size = subgraph_names_to_index_.size();
  subgraph_names_to_index_[name] = size;
  subgraph_instance_names_.resize(size + 1);
  return GRAPH_SUCCESS;
}

const std::map<std::string, uint32_t> & OpDescImpl::GetSubgraphNameIndexes() const {
  return subgraph_names_to_index_;
}

graphStatus OpDescImpl::SetSubgraphInstanceName(uint32_t index, const std::string &name) {
  GELOGI("Add sub graph instance name is %s, index is %u", name.c_str(), index);
  if (index >= subgraph_instance_names_.size()) {
    REPORT_INNER_ERROR("E19999", "Index %u exceeds the max instance count %zu", index, subgraph_instance_names_.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Index %u exceeds the max instance count %zu", index,
           subgraph_instance_names_.size());
    return GRAPH_PARAM_INVALID;
  }
  subgraph_instance_names_[index] = name;
  return GRAPH_SUCCESS;
}

void OpDescImpl::RegisterSubgraphIrName(const string &name, SubgraphType type) {
  subgraph_ir_names_to_type_[name] = type;
}

const std::map<std::string, SubgraphType> &OpDescImpl::GetSubgraphIrNames() const {
  return subgraph_ir_names_to_type_;
}

SubgraphType OpDescImpl::GetSubgraphTypeByIrName(const std::string &name) const {
  auto iter = subgraph_ir_names_to_type_.find(name);
  if (iter == subgraph_ir_names_to_type_.end()) {
    return kSubgraphTypeEnd;
  }
  return iter->second;
}


graphStatus OpDescImpl::GetSubgraphNameByInstanceName(const std::string &instance_name,
                                                      std::string &subgraph_name) const {
  for (size_t idx = 0; idx < subgraph_instance_names_.size(); ++idx) {
    if (subgraph_instance_names_[idx] != instance_name) {  // find subgraph index.
      continue;
    }

    for (auto name_to_index : subgraph_names_to_index_) {
      if (name_to_index.second != idx) {   // find subgraph name.
        continue;
      }

      subgraph_name = name_to_index.first;
      return GRAPH_SUCCESS;
    }
  }

  return GRAPH_PARAM_INVALID;
}

graphStatus OpDescImpl::InferDataSlice(const OpDescPtr &op_desc) {
  if (infer_data_slice_func_ == nullptr) {
    infer_data_slice_func_ = OperatorFactoryImpl::GetInferDataSliceFunc(GetType());
    if (infer_data_slice_func_ == nullptr) {
      GELOGW("[InferDataSlice][Check] %s does not have infer data slice func.", GetName().c_str());
      return NO_DEPENDENCE_FUNC;
    }
  }
  Operator op_proxy = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  graphStatus ret = (graphStatus)infer_data_slice_func_(op_proxy);
  op_proxy.BreakConnect();
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc()
    : impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl())) {
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::~OpDesc() {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const std::string &name, const std::string &type)
    : impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl(name, type))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const OpDesc &op_desc)
    : AttrHolder(op_desc),
      impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl(*(op_desc.impl_)))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(OpDesc &&op_desc)
    : AttrHolder(std::move(op_desc)),
      impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl(std::move(*(op_desc.impl_))))) {}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::OpDesc(const ProtoMsgOwner &proto_msg_owner,
                                                              ge::proto::OpDef *op_def)
    : impl_(std::shared_ptr<OpDescImpl>(new OpDescImpl(proto_msg_owner, op_def))) {
  if (op_def != nullptr && !op_def->has_out_attr()) {
    op_def->set_has_out_attr(true);

    int64_t id = 0;
    (void)AttrUtils::GetInt(this, ATTR_NAME_ID, id);
    op_def->set_id(id);

    int64_t stream_id = 0;
    (void)AttrUtils::GetInt(this, ATTR_NAME_STREAM_ID, stream_id);
    op_def->set_stream_id(stream_id);

    vector<string> input_name;
    (void)AttrUtils::GetListStr(this, ATTR_NAME_INPUT_NAME, input_name);
    for (auto &item : input_name) {
      op_def->add_input_name(item);
    }
    vector<string> src_name;
    (void)AttrUtils::GetListStr(this, ATTR_NAME_SRC_NAME, src_name);
    for (auto &item : src_name) {
      op_def->add_src_name(item);
    }
    vector<int64_t> src_index;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_SRC_INDEX, src_index);
    for (auto &item : src_index) {
      op_def->add_src_index(item);
    }
    vector<int64_t> input;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_INPUT, input);
    for (auto &item : input) {
      op_def->add_input_i(item);
    }
    vector<int64_t> output;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_OUTPUT, output);
    for (auto &item : output) {
      op_def->add_output_i(item);
    }
    vector<string> dst_name;
    (void)AttrUtils::GetListStr(this, ATTR_NAME_DST_NAME, dst_name);
    for (auto &item : dst_name) {
      op_def->add_dst_name(item);
    }
    vector<int64_t> dst_index;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_DST_INDEX, dst_index);
    for (auto &item : dst_index) {
      op_def->add_dst_index(item);
    }
    vector<int64_t> workspace;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_WORKSPACE, workspace);
    for (auto &item : workspace) {
      op_def->add_workspace(item);
    }
    vector<int64_t> workspace_bytes;
    (void)AttrUtils::GetListInt(this, ATTR_NAME_WORKSPACE_BYTES, workspace_bytes);
    for (auto &item : workspace_bytes) {
      op_def->add_workspace_bytes(item);
    }
    vector<bool> is_input_const;
    (void)AttrUtils::GetListBool(this, ATTR_NAME_IS_INPUT_CONST, is_input_const);
    for (auto item : is_input_const) {
      op_def->add_is_input_const(item);
    }
    auto input_desc_mutable_list = (*op_def->mutable_attr())[ATTR_NAME_INPUT_DESC].mutable_list();
    if (input_desc_mutable_list != nullptr) {
      *op_def->mutable_input_desc() = *(input_desc_mutable_list->mutable_td());
    }
    auto output_desc_mutable_list = (*op_def->mutable_attr())[ATTR_NAME_OUTPUT_DESC].mutable_list();
    if (output_desc_mutable_list != nullptr) {
      *op_def->mutable_output_desc() = *(output_desc_mutable_list->mutable_td());
    }
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetName() const {
  return impl_->GetName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetName(const std::string &name) {
  return impl_->SetName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetType() const {
  return impl_->GetType();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetType(const string &type) {
  return impl_->SetType(type);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddInputDesc(const ge::GeTensorDesc &input_desc) {
  return impl_->AddInputDesc(input_desc);
}

graphStatus OpDesc::AddInputDesc(uint32_t index, const ge::GeTensorDesc &input_desc) {
  return impl_->AddInputDesc(index, input_desc);
}

graphStatus OpDesc::AddInputDesc(const string &name, const ge::GeTensorDesc &input_desc) {
  return impl_->AddInputDesc(name, input_desc);
}

graphStatus OpDesc::AddInputDescMiddle(const string &name, const unsigned int num, size_t index) {
  return impl_->AddInputDescMiddle(name, num, index);
}

graphStatus OpDesc::AddOutputDescMiddle(const string &name, const unsigned int num, size_t index) {
  return impl_->AddOutputDescMiddle(name, num, index);
}

graphStatus OpDesc::AddInputDescForward(const string &name, const unsigned int num) {
  return impl_->AddInputDescForward(name, num);
}

graphStatus OpDesc::AddOutputDescForward(const string &name, const unsigned int num) {
  return impl_->AddOutputDescForward(name, num);
}

graphStatus OpDesc::AddOptionalInputDesc(const string &name, const ge::GeTensorDesc &input_desc) {
  return impl_->AddOptionalInputDesc(name, input_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::UpdateInputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc) {
  return impl_->UpdateInputDesc(index, tensor_Desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescMembersAreEqual(const OpDesc &r_op_desc) const {
  return impl_->OpDescMembersAreEqual(*(r_op_desc.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescAttrsAreEqual(const OpDesc &r_op_desc) const {
  return impl_->OpDescAttrsAreEqual(*(r_op_desc.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::OpDescGenTensorDescsAreEqual(const OpDesc &r_op_desc)
    const {
  return impl_->OpDescGenTensorDescsAreEqual(*(r_op_desc.impl_));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool OpDesc::operator==(const OpDesc &r_op_desc) const {
  return (OpDescAttrsAreEqual(r_op_desc) && OpDescMembersAreEqual(r_op_desc) &&
          OpDescGenTensorDescsAreEqual(r_op_desc));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc& OpDesc::operator=(OpDesc op_desc) {
  if (&op_desc == this) {
    return *this;
  }
  AttrHolder::Swap(op_desc);
  *impl_ = *(op_desc.impl_);
  return *this;
}

graphStatus OpDesc::UpdateInputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc) {
  return impl_->UpdateInputDesc(name, tensor_Desc);
}

bool OpDesc::InputIsSet(const string &name) const {
  return impl_->InputIsSet(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc OpDesc::GetInputDesc(uint32_t index) const {
  return impl_->GetInputDesc(index);
}

GeTensorDesc OpDesc::GetInputDesc(const string &name) const {
  return impl_->GetInputDesc(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableInputDesc(uint32_t index) const {
  return impl_->MutableInputDesc(index);
}

GeTensorDescPtr OpDesc::MutableInputDesc(const string &name) const {
  return impl_->MutableInputDesc(name);
}

GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<string> OpDesc::GetAllInputNames() const {
  return impl_->GetAllInputNames(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpKernelLibName(const std::string &name) {
  impl_->SetOpKernelLibName(name);
  auto ret = AttrUtils::SetStr(this, ATTR_NAME_OP_KERNEL_LIB_NAME, name);
  if (!ret) {
    REPORT_CALL_ERROR("E19999", "set %s to op failed.", ATTR_NAME_OP_KERNEL_LIB_NAME.c_str());
    GELOGE(GRAPH_FAILED, "[Set][Str] %s to op failed.", ATTR_NAME_OP_KERNEL_LIB_NAME.c_str());
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOpKernelLibName() const {
  string op_kernel_lib_name = impl_->GetOpKernelLibName();
  if (op_kernel_lib_name.empty()) {
    (void)AttrUtils::GetStr(this, ATTR_NAME_OP_KERNEL_LIB_NAME,
                            op_kernel_lib_name);
  }
  return op_kernel_lib_name;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpEngineName(const std::string &name) {
  impl_->SetOpEngineName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetOpEngineName() const {
  return impl_->GetOpEngineName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllInputsDesc() const {
  return impl_->GetAllInputsDesc(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDescPtr> OpDesc::GetAllInputsDescPtr() const {
  return impl_->GetAllInputsDescPtr(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetInputsSize() const {
  //  Just return valid inputs size.InValid desc is set in default OPTION_INPUT register.
  return impl_->GetInputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetAllInputsSize() const {
  return impl_->GetAllInputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddOutputDesc(const ge::GeTensorDesc &output_desc) {
  return impl_->AddOutputDesc(output_desc);
}

graphStatus OpDesc::AddOutputDesc(const string &name, const ge::GeTensorDesc &output_desc) {
  return impl_->AddOutputDesc(name, output_desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDesc::UpdateOutputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc) {
  return impl_->UpdateOutputDesc(index, tensor_Desc);
}

graphStatus OpDesc::UpdateOutputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc) {
  return impl_->UpdateOutputDesc(name, tensor_Desc);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDesc OpDesc::GetOutputDesc(uint32_t index) const {
  return impl_->GetOutputDesc(index);
}

GeTensorDesc OpDesc::GetOutputDesc(const string &name) const {
  return impl_->GetOutputDesc(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableOutputDesc(uint32_t index) const {
  return impl_->MutableOutputDesc(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY GeTensorDescPtr OpDesc::MutableOutputDesc(const string &name) const {
  return impl_->MutableOutputDesc(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY uint32_t OpDesc::GetAllOutputsDescSize() const {
  return impl_->GetAllOutputsDescSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDesc> OpDesc::GetAllOutputsDesc() const {
  return impl_->GetAllOutputsDesc(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDesc::Vistor<GeTensorDescPtr> OpDesc::GetAllOutputsDescPtr() const {
  return impl_->GetAllOutputsDescPtr(shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY size_t OpDesc::GetOutputsSize() const {
  return impl_->GetOutputsSize();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr OpDesc::GetOutputDescPtr(uint32_t index) const {
  return impl_->GetOutputDescPtr(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr OpDesc::GetInputDescPtr(uint32_t index) const {
  return impl_->GetInputDescPtr(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr
OpDesc::GetInputDescPtrDfault(uint32_t index) const {
  return impl_->GetInputDescPtrDfault(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY ConstGeTensorDescPtr OpDesc::GetInputDescPtr(const string &name) const {
  return impl_->GetInputDescPtr(name);
}

graphStatus OpDesc::AddRegisterInputName(const std::string &name) {
  return impl_->AddRegisterInputName(name);
}

vector<string> OpDesc::GetRegisterInputName() const {
  return impl_->GetRegisterInputName();
}

graphStatus OpDesc::AddDynamicInputDesc(const string &name, const unsigned int num, bool is_push_back) {
  return impl_->AddDynamicInputDesc(name, num, is_push_back);
}

graphStatus OpDesc::AddDynamicInputDescByIndex(const string &name, const unsigned int num, size_t index) {
  if (AddInputDescMiddle(name, num, index) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDesc::AddRegisterOutputName(const string &name) {
  return impl_->AddRegisterOutputName(name);
}

vector<string> OpDesc::GetRegisterOutputName() const {
  return impl_->GetRegisterOutputName();
}

graphStatus OpDesc::AddDynamicOutputDesc(const string &name, const unsigned int num, bool is_push_back) {
  if (is_push_back) {
    for (unsigned int i = 0; i < num; i++) {
      if (AddOutputDesc(name + std::to_string(i), GeTensorDesc()) != GRAPH_SUCCESS)
        return GRAPH_FAILED;
    }
  } else {
    if (AddOutputDescForward(name, num) != GRAPH_SUCCESS)
      return GRAPH_FAILED;
  }

  if (AddRegisterOutputName(name) != GRAPH_SUCCESS) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OpDesc::IsOptionalInput(const string &name) const {
  return impl_->IsOptionalInput(name);
}

bool OpDesc::IsOptionalInput(uint32_t index) const { return IsOptionalInput(GetInputNameByIndex(index)); }

std::map<string, uint32_t> OpDesc::GetAllInputName() const {
  return impl_->GetAllInputName();
}

std::map<string, uint32_t> OpDesc::GetAllOutputName() {
  return impl_->GetAllOutputName();
}

std::map<string, uint32_t>& OpDesc::MutableAllInputName() {
  return impl_->MutableAllInputName();
}

std::map<string, uint32_t>& OpDesc::MutableAllOutputName() {
  return impl_->MutableAllOutputName();
}

bool OpDesc::UpdateInputName(std::map<string, uint32_t> input_name_idx) {
  return impl_->UpdateInputName(input_name_idx);
}

bool OpDesc::UpdateOutputName(std::map<string, uint32_t> output_name_idx) {
  return impl_->UpdateOutputName(output_name_idx);
}

std::function<graphStatus(Operator &)> OpDesc::GetInferFunc() const {
  return impl_->GetInferFunc();
}

std::function<graphStatus(Operator &)> OpDesc::GetVerifyFunc() const {
  return impl_->GetVerifyFunc();
}

void OpDesc::AddInferFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddInferFunc(func);
}

std::function<graphStatus(Operator &)> OpDesc::GetInferFormatFunc() const {
  return impl_->GetInferFormatFunc();
}

void OpDesc::AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddInferFormatFunc(func);
}

void OpDesc::AddVerifierFunc(const std::function<graphStatus(Operator &)> &func) {
  impl_->AddVerifierFunc(func);
}

graphStatus OpDesc::InferShapeAndType() {
  return impl_->InferShapeAndType(shared_from_this());
}

graphStatus OpDesc::DefaultInferFormat() {
  return impl_->DefaultInferFormat(shared_from_this());
}

graphStatus OpDesc::OpVerify() {
  return impl_->OpVerify(shared_from_this());

}

graphStatus OpDesc::CommonVerify() const {
  for (const string &iname : GetAllInputNames()) {
    // Checking shape of all inputs
    vector<int64_t> ishape = GetInputDescPtr(iname)->GetShape().GetDims();
    if (ishape == DUMMY_SHAPE) {
      continue;
    }
    for (int64_t dim : ishape) {
      GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(dim < -2,
          ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
              {GetName(), "input " + iname + " shape", "contains negative or zero dimension"});
          return GRAPH_FAILED,
          "Op[%s]'s input %s shape contains negative or zero dimension.", GetName().c_str(), iname.c_str());
    }
  }
  // Check all attributes defined
  const auto &all_attributes = GetAllAttrs();
  for (const auto &name : GetAllAttrNames()) {
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(all_attributes.find(name) == all_attributes.end(),
        ErrorManager::GetInstance().ATCReportErrMessage("E19014", {"opname", "value", "reason"},
            {GetName(), "attribute " + name, "is empty"});
            return GRAPH_FAILED,
            "operator attribute %s is empty.", name.c_str());
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetInputNameByIndex(uint32_t index) const {
  return impl_->GetInputNameByIndex(index);
}

int OpDesc::GetInputIndexByName(const string &name) const {
  return impl_->GetInputIndexByName(name);
}

int OpDesc::GetValidInputIndexByName(const string &name) const {
  return impl_->GetValidInputIndexByName(name);
}

string OpDesc::GetValidInputNameByIndex(uint32_t index) const {
  return impl_->GetValidInputNameByIndex(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY string OpDesc::GetOutputNameByIndex(uint32_t index) const {
  return impl_->GetOutputNameByIndex(index);
}

int OpDesc::GetOutputIndexByName(const string &name) const {
  return impl_->GetOutputIndexByName(name);
}

ProtoAttrMapHelper OpDesc::MutableAttrMap() {
  return impl_->MutableAttrMap();
}

ConstProtoAttrMapHelper OpDesc::GetAttrMap() const {
  return impl_->GetAttrMap();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetId(int64_t id) {
  impl_->SetId(id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetId() const {
  return impl_->GetId();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetStreamId(int64_t stream_id) {
  impl_->SetStreamId(stream_id);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY int64_t OpDesc::GetStreamId() const {
  return impl_->GetStreamId();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetInputName(const vector<string> &input_name) {
  impl_->SetInputName(input_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetInputName() const {
  return impl_->GetInputName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetSrcName(const vector<string> &src_name) {
  impl_->SetSrcName(src_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetSrcName() const {
  return impl_->GetSrcName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetSrcIndex(const vector<int64_t> &src_index) {
  impl_->SetSrcIndex(src_index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetSrcIndex() const {
  return impl_->GetSrcIndex();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetInputOffset(const vector<int64_t> &input) {
  impl_->SetInputOffset(input);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetInputOffset() const {
  return impl_->GetInputOffset();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOutputOffset(const vector<int64_t> &output) {
  impl_->SetOutputOffset(output);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetOutputOffset() const {
  return impl_->GetOutputOffset();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetDstName(const vector<string> &dst_name) {
  impl_->SetDstName(dst_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetDstName() const {
  return impl_->GetDstName();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetOpInferDepends(const vector<string> &depend_names) {
  auto ret = AttrUtils::SetListStr(this, ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  if (!ret) {
    GELOGE(GRAPH_FAILED, "[Set][Attr] op_infer_depends fail.");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<string> OpDesc::GetOpInferDepends() const {
  vector<string> depend_names;
  (void)AttrUtils::GetListStr(this, ATTR_NAME_OP_INFER_DEPENDS, depend_names);
  return depend_names;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetDstIndex(const vector<int64_t> &dst_index) {
  impl_->SetDstIndex(dst_index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetDstIndex() const {
  return impl_->GetDstIndex();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetWorkspace(const vector<int64_t> &workspace) {
  impl_->SetWorkspace(workspace);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetWorkspace() const {
  return impl_->GetWorkspace();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetWorkspaceBytes(const vector<int64_t> &workspace_bytes) {
  impl_->SetWorkspaceBytes(workspace_bytes);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<int64_t> OpDesc::GetWorkspaceBytes() const {
  return impl_->GetWorkspaceBytes();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::SetIsInputConst(const vector<bool> &is_input_const) {
  impl_->SetIsInputConst(is_input_const);
  // If comes from ME,which is_input_const exist as attrs, outside no need to check GE_TRAIN flag
  auto ret = AttrUtils::SetListBool(this, ATTR_NAME_IS_INPUT_CONST, is_input_const);
  if (ret != true) {
    GELOGE(GRAPH_FAILED, "[Set][Attr] is_input_const fail.");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY vector<bool> OpDesc::GetIsInputConst() const {
  return impl_->GetIsInputConst();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::RestoreInputNameIdx(const string &name,
                                                                                       const int &index) {
  return impl_->RestoreInputNameIdx(name, index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::RestoreOutputNameIdx(const string &name,
                                                                                        const int &index) {
  return impl_->RestoreOutputNameIdx(name, index);
}

graphStatus OpDesc::CallInferFunc(Operator &op) {
  return impl_->CallInferFunc(op, shared_from_this());
}
graphStatus OpDesc::CallInferFormatFunc(Operator &op) {
  return impl_->CallInferFormatFunc(op, shared_from_this());
}
graphStatus OpDesc::CallInferValueRangeFunc(Operator &op) {
  return impl_->CallInferValueRangeFunc(op, shared_from_this());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY std::string OpDesc::GetSubgraphInstanceName(uint32_t index) const {
  return impl_->GetSubgraphInstanceName(index);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::vector<std::string> &OpDesc::GetSubgraphInstanceNames()
    const {
  return impl_->GetSubgraphInstanceNames();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void OpDesc::RemoveSubgraphInstanceName(const std::string &name) {
  impl_->RemoveSubgraphInstanceName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::AddSubgraphName(const std::string &name) {
  return impl_->AddSubgraphName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY const std::map<std::string, uint32_t> & OpDesc::GetSubgraphNameIndexes()
    const {
  return impl_->GetSubgraphNameIndexes();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus OpDesc::SetSubgraphInstanceName(uint32_t index, const std::string &name) {
  return impl_->SetSubgraphInstanceName(index, name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
void OpDesc::RegisterSubgraphIrName(const string &name, SubgraphType type) {
  impl_->RegisterSubgraphIrName(name, type);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
const std::map<std::string, SubgraphType> &OpDesc::GetSubgraphIrNames() const {
  return impl_->GetSubgraphIrNames();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
SubgraphType OpDesc::GetSubgraphTypeByIrName(const std::string &name) const {
  return impl_->GetSubgraphTypeByIrName(name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus OpDesc::GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const {
  return impl_->GetSubgraphNameByInstanceName(instance_name, subgraph_name);
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus OpDesc::InferDataSlice() {
  return impl_->InferDataSlice(shared_from_this());
}
}  // namespace ge
