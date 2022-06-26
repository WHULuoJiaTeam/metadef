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

#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include <cstdint>
#include <algorithm>
#include <mutex>
#include <queue>
#include <set>
#include "debug/ge_log.h"
#include "debug/ge_op_types.h"
#include "debug/ge_util.h"
#include "external/graph/attr_value.h"
#include "graph/compute_graph.h"
#include "graph/ge_context.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/node_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "utils/graph_utils.h"
#include "utils/op_desc_utils.h"
#include "utils/tensor_adapter.h"
#include "utils/tensor_utils.h"

using std::enable_shared_from_this;
using std::make_pair;
using std::shared_ptr;
using std::string;
using std::to_string;
using std::vector;

/*lint -save -e529 -e728*/
namespace ge {
/*lint -e446 -e732*/
/*lint -e665*/
class OpIO {
 public:
  OpIO(const string &name, int index, const OperatorImplPtr &owner) : name_(name), index_(index), owner_(owner) {}

  ~OpIO() = default;

  string GetName() const { return name_; }

  int GetIndex() const { return index_; }

  OperatorImplPtr GetOwner() const { return owner_; }

  bool operator==(const OpIO &r_value) const {
    return (this->name_ == r_value.GetName()) && (this->index_ == r_value.GetIndex()) &&
           (this->GetOwner() == r_value.GetOwner());
  }

 private:
  string name_;
  int index_;
  std::shared_ptr<OperatorImpl> owner_;
};

class TensorTypeImpl {
 public:
  TensorTypeImpl() = default;
  ~TensorTypeImpl() = default;

  std::vector<DataType> dt_vec_;
};

TensorType::TensorType(DataType dt) {
  tensor_type_impl_ = ComGraphMakeShared<TensorTypeImpl>();
  if (tensor_type_impl_ != nullptr) {
    tensor_type_impl_->dt_vec_.push_back(dt);
  }
}

TensorType::TensorType(const std::initializer_list<DataType> &types) {
  tensor_type_impl_ = ComGraphMakeShared<TensorTypeImpl>();
  if (tensor_type_impl_ != nullptr) {
    tensor_type_impl_->dt_vec_ = types;
  }
}

class OperatorImpl : public std::enable_shared_from_this<OperatorImpl> {
  friend class GraphBuilderImpl;
  friend class OpDescUtils;

 public:
  explicit OperatorImpl(const string &name, const string &type) : op_desc_(ComGraphMakeShared<OpDesc>(name, type)) {
    if (op_desc_ == nullptr) {
      GELOGW("[Check][Param] Make op_desc failed");
    }
  }
  explicit OperatorImpl(const OpDescPtr &op_desc) : op_desc_(op_desc) {}
  explicit OperatorImpl(ge::ConstNodePtr node) : node_(std::move(node)) {
    if (node_ != nullptr && node_->GetOpDesc() != nullptr) {
      op_desc_ = node_->GetOpDesc();
    }
  }
  ~OperatorImpl() {}
  void SetInputImpl(const string &dst_name, const ge::Operator &src_oprt) {
    GE_CHK_BOOL_EXEC(!dst_name.empty(), REPORT_INNER_ERROR("E19999", "param dst_name is empty, check invalid.");
                     return, "[Check][Param] dst name is empty");
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr.");
                     return, "[Check][Param] op_desc_ is nullptr.");
    GE_CHK_BOOL_EXEC(src_oprt.operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr.");
                     return, "[Check][Param] operator_impl_ is nullptr.");
    GE_CHK_BOOL_EXEC(src_oprt.operator_impl_->op_desc_ != nullptr,
                     REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr.");
                     return, "[Check][Param] op_desc_ is nullptr.");

    auto src_op_impl = src_oprt.GetOperatorImplPtr();
    GE_CHK_BOOL_EXEC(src_op_impl != nullptr, REPORT_INNER_ERROR("E19999", "Src impl is null.");
                     return, "[Get][OperatorImplPtr] Src impl is null.");
    GE_CHK_BOOL_EXEC(src_op_impl->op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "Src impl's opdesc is null.");
                     return, "[Check][Param] Src impl's opdesc is null.");
    GE_CHK_BOOL_EXEC(src_oprt.operator_impl_->op_desc_->GetOutputsSize() == 1,
                     REPORT_INNER_ERROR("E19999", "The source operator[%s] must be single output operator",
                                        src_oprt.operator_impl_->op_desc_->GetName().c_str());
                     return,
                     "[Check][Param] The source operator[%s] must be single output operator",
                     src_oprt.operator_impl_->op_desc_->GetName().c_str());

    uint32_t src_index = 0;
    string src_name = src_op_impl->op_desc_->GetOutputNameByIndex(src_index);
    GE_CHK_BOOL_EXEC(!src_name.empty(),
                     REPORT_INNER_ERROR("E19999", "Src output's name is empty, index:%u.", src_index);
                     return, "[Get][OutputName] Src output's name is empty, index:%u.", src_index);

    OpIO out_handler(src_name, src_index, src_op_impl);
    input_link_.insert(std::make_pair(dst_name, out_handler));

    int dst_index = op_desc_->GetInputIndexByName(dst_name);
    GE_CHK_BOOL_EXEC(dst_index >= 0,
                     REPORT_INNER_ERROR("E19999", "Find input index by name failed. name[%s], op name:%s",
                                        dst_name.c_str(), op_desc_->GetName().c_str());
                     return, "[Get][InputIndex] Find input index by name failed. name[%s], op name:%s",
                     dst_name.c_str(), op_desc_->GetName().c_str());

    bool is_const = false;
    if (src_oprt.GetOpType() == CONSTANT) {
      is_const = true;
    }
    auto is_input_const = op_desc_->GetIsInputConst();
    for (int i = static_cast<int>(is_input_const.size()); i <= dst_index; ++i) {
      is_input_const.push_back(false);
    }

    is_input_const[dst_index] = is_const;
    op_desc_->SetIsInputConst(is_input_const);

    OpIO op_dst(dst_name, dst_index, shared_from_this());
    src_op_impl->UpdateLinkMapImpl(src_name, op_dst);
    auto output_desc = src_op_impl->GetOutputDesc(src_name);
    auto input_desc = op_desc_->GetInputDesc(dst_name);
    if (input_desc.GetFormat() == FORMAT_RESERVED) {
      output_desc.SetFormat(FORMAT_ND);
    } else {
      output_desc.SetFormat(input_desc.GetFormat());
    }
    // Fix for linking opdesc
    if (op_desc_->UpdateInputDesc(dst_name, output_desc) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "UpdateInputDesc failed, dst name is %s, src name is %s",
                        dst_name.c_str(), src_name.c_str());
      GELOGE(GRAPH_FAILED, "[Update][InputDesc] failed, dst name is %s, src name is %s", dst_name.c_str(),
             src_name.c_str());
      return;
    }
  }

  void SetInputImpl(const string &dst_name, const ge::OutHandler &out_handler) {
    GE_CHK_BOOL_EXEC(!dst_name.empty(), REPORT_INNER_ERROR("E19999", "param dst_name is empty, check invalid.");
                     return, "[Check][Param] dst name is empty");
    GE_CHK_BOOL_EXEC(out_handler != nullptr,
                     REPORT_INNER_ERROR("E19999", "param out_handler is nullptr, check invalid.");
                     return, "[Check][Param] SetInputImpl faild, as out_handler is nullptr.");
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr.");
                     return, "[Check][Param] op_desc_ is nullptr.");
    input_link_.insert(std::make_pair(dst_name, *out_handler));

    string src_name = out_handler->GetName();
    int dst_index = op_desc_->GetInputIndexByName(dst_name);
    GE_CHK_BOOL_EXEC(dst_index >= 0,
                     REPORT_INNER_ERROR("E19999", "Find input index by name failed. name[%s], op name:%s",
                                        dst_name.c_str(), op_desc_->GetName().c_str());
                     return, "[Get][InputIndex] Find input index by name failed. name[%s], op name:%s",
                     dst_name.c_str(), op_desc_->GetName().c_str());
    auto out_op_impl = out_handler->GetOwner();
    GE_CHK_BOOL_EXEC(out_op_impl != nullptr && out_op_impl->GetOpDescImpl() != nullptr,
                     REPORT_INNER_ERROR("E19999", "out_handler invalid. name[%s]", dst_name.c_str());
                     return, "[Get][Impl] out_handler invalid. name[%s]", dst_name.c_str());
    bool is_const = false;
    if (out_op_impl->GetOpDescImpl()->GetType() == CONSTANT) {
      is_const = true;
    }
    auto is_input_const = op_desc_->GetIsInputConst();
    for (int i = static_cast<int>(is_input_const.size()); i <= dst_index; ++i) {
      is_input_const.push_back(false);
    }
    is_input_const[dst_index] = is_const;
    op_desc_->SetIsInputConst(is_input_const);

    OpIO in_handler(dst_name, dst_index, shared_from_this());
    GE_CHK_BOOL_EXEC(out_op_impl != nullptr,
                     REPORT_INNER_ERROR("E19999", "out_handler invalid. name[%s]", dst_name.c_str());
                     return, "[Get][Impl] of out_handler failed.");

    out_op_impl->UpdateLinkMapImpl(src_name, in_handler);
    auto src_output_desc = out_op_impl->GetOutputDesc(src_name);
    auto dst_input_desc = op_desc_->GetInputDesc(dst_name);
    if (dst_input_desc.GetFormat() == FORMAT_RESERVED) {
      src_output_desc.SetFormat(FORMAT_ND);
      src_output_desc.SetOriginFormat(FORMAT_ND);
    } else {
      src_output_desc.SetFormat(dst_input_desc.GetFormat());
      src_output_desc.SetOriginFormat(dst_input_desc.GetOriginFormat());
    }
    GE_CHK_BOOL_EXEC(op_desc_->UpdateInputDesc(dst_name, src_output_desc) == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "UpdateInputDesc failed, dst name is %s, src name is %s",
                                       dst_name.c_str(), src_name.c_str());
                     return, "[Update][InputDesc] failed, dst name is %s, src name is %s",
                     dst_name.c_str(), src_name.c_str()); // fix for linking opdesc
  }

  void AddControlInputImp(const ge::Operator &src_oprt) {
    if (src_oprt.operator_impl_ == nullptr) {
      REPORT_INNER_ERROR("E19999", "Src operator impl is nullptr, check invalid");
      GELOGE(FAILED, "[Check][Param] Src operator impl is nullptr");
      return;
    }
    for (auto &input : control_input_link_) {
      if (input.lock() == src_oprt.operator_impl_) {
        return;
      }
    }
    control_input_link_.push_back(src_oprt.operator_impl_);
    src_oprt.operator_impl_->control_output_link_.push_back(shared_from_this());
  }

  graphStatus GetInputImpl(const string &dst_name, ge::OpIO &out_handler) {
    auto out = input_link_.find(dst_name);
    if (out == input_link_.end()) {
      return GRAPH_FAILED;
    }
    out_handler = out->second;
    return GRAPH_SUCCESS;
  }

  graphStatus GetInputConstData(const string &dst_name, Tensor &data) {
    auto node_ptr = GetNode();
    GE_IF_BOOL_EXEC(node_ptr == nullptr, return GetInputConstDataOut(dst_name, data);)

    // For inner compute graph
    auto op_desc = node_ptr->GetOpDesc();
    GE_CHECK_NOTNULL(op_desc);
    auto index = op_desc->GetInputIndexByName(dst_name);
    auto in_data_anchor = node_ptr->GetInDataAnchor(index);
    GE_CHECK_NOTNULL(in_data_anchor);
    auto out_data_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(out_data_anchor);
    auto peer_node = out_data_anchor->GetOwnerNode();

    // if tensor has host mem, init data by ATTR_NAME_VALUE first
    auto tensor = op_desc->MutableInputDesc(index);
    GeTensorPtr tensor_value = nullptr;
    if (AttrUtils::MutableTensor(tensor, ATTR_NAME_VALUE, tensor_value)) {
      GELOGD("Get ATTR_NAME_VALUE from %d input of %s, Tensor addr is %p, tensor value data type is %d.", index,
             op_desc->GetName().c_str(), tensor.get(), tensor_value->GetTensorDesc().GetDataType());
      data = TensorAdapter::GeTensor2Tensor(tensor_value);
      return GRAPH_SUCCESS;
    }
    // Try get from runtime inference context
    auto context_id = std::to_string(GetContext().ContextId());
    RuntimeInferenceContext *runtime_infer_ctx = nullptr;
    if (RuntimeInferenceContext::GetContext(context_id, &runtime_infer_ctx) == GRAPH_SUCCESS) {
      GELOGD("To get constant from runtime inference context. context_id = %s", context_id.c_str());
      auto ret = runtime_infer_ctx->GetTensor(peer_node->GetOpDesc()->GetId(), out_data_anchor->GetIdx(), data);
      if (ret == GRAPH_SUCCESS) {
        return GRAPH_SUCCESS;
      }
    }
    if (peer_node->GetType() == ENTER || peer_node->GetType() == REFENTER) {
      auto enter_in_data_anchor = peer_node->GetInDataAnchor(0);
      GE_CHECK_NOTNULL(enter_in_data_anchor);
      auto enter_peer_out_data_anchor = enter_in_data_anchor->GetPeerOutAnchor();
      GE_CHECK_NOTNULL(enter_peer_out_data_anchor);
      peer_node = enter_peer_out_data_anchor->GetOwnerNode();
    }
    auto peer_op_desc = peer_node->GetOpDesc();
    GE_CHECK_NOTNULL(peer_op_desc);
    auto peer_op_type = peer_op_desc->GetType();
    if (peer_op_type == CONSTANTOP || peer_op_type == CONSTANT) {
      auto const_op_impl = ComGraphMakeShared<OperatorImpl>(peer_node);
      GE_CHECK_NOTNULL(const_op_impl);
      Operator const_op(std::move(const_op_impl));
      return const_op.GetAttr(ATTR_NAME_WEIGHTS, data);
    } else if (peer_op_type == DATA) {
      auto parent_node = NodeUtils::GetParentInput(peer_node);
      while ((parent_node != nullptr) && (parent_node->GetType() == DATA)) {
        parent_node = NodeUtils::GetParentInput(parent_node);
      }
      if ((parent_node != nullptr)
          && ((parent_node->GetType() == CONSTANT) || (parent_node->GetType() == CONSTANTOP))) {
        auto const_op_impl = ComGraphMakeShared<OperatorImpl>(parent_node);
        GE_CHECK_NOTNULL(const_op_impl);
        Operator const_op(std::move(const_op_impl));
        return const_op.GetAttr(ATTR_NAME_WEIGHTS, data);
      }
    }

    auto op_name = GetName();
    GELOGW("[Get][ConstInput] Node[%s]'s input[%s]'s peer node is not const", op_name.c_str(), dst_name.c_str());
    return GRAPH_FAILED;
  }

  graphStatus GetInputConstDataOut(const string &dst_name, Tensor &data) {
    ge::OpIO out_handle("", 0, nullptr);
    if (GetInputImpl(dst_name, out_handle) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "%s get input impl failed", dst_name.c_str());
      GELOGE(FAILED, "[Get][InputImpl] failed, dst_name:%s", dst_name.c_str());
      return GRAPH_FAILED;
    }
    if (out_handle.GetOwner() != nullptr && out_handle.GetOwner()->GetOpDescImpl() != nullptr) {
      Operator const_op(out_handle.GetOwner());
      const auto &op_desc_impl_type = out_handle.GetOwner()->GetOpDescImpl()->GetType();
      if (op_desc_impl_type == CONSTANTOP) {
        return const_op.GetAttr(ATTR_NAME_WEIGHTS, data);
      } else if (op_desc_impl_type == CONSTANT) {
        return const_op.GetAttr(ATTR_NAME_WEIGHTS, data);
      }
    }
    return GRAPH_FAILED;
  }

  bool InputIsSet(const string &name) {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return false, "[Check][Param] op_desc_ is nullptr.");
    return op_desc_->InputIsSet(name);
  }

  string GetName() const {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return string(), "[Check][Param] op_desc_ is nullptr.");
    return op_desc_->GetName();
  }

  GeTensorDesc GetInputDesc(const string &name) const {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");
    return op_desc_->GetInputDesc(name);
  }

  GeTensorDesc GetInputDesc(uint32_t index) const {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");
    return op_desc_->GetInputDesc(index);
  }

  graphStatus UpdateInputDesc(const string &name, const GeTensorDesc &tensor_desc) {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return GRAPH_FAILED, "[Check][Param] op_desc_ is nullptr.");

    return op_desc_->UpdateInputDesc(name, tensor_desc);
  }

  OutHandler GetOutput(const string &name) {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return nullptr, "[Check][Param] op_desc_ is nullptr.");

    int src_index = op_desc_->GetOutputIndexByName(name);
    GE_CHK_BOOL_EXEC(src_index >= 0,
                     REPORT_INNER_ERROR("E19999", "Find src index by name failed. name[%s]", name.c_str());
                     return nullptr, "[Get][OutputIndex] Find src index by name failed. name[%s]", name.c_str());
    shared_ptr<OpIO> output_ptr = ComGraphMakeShared<OpIO>(name, src_index, shared_from_this());
    if (output_ptr == nullptr) {
      REPORT_CALL_ERROR("E19999", "OpIO make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OpIO make shared failed");
      return nullptr;
    }
    return output_ptr;
  }

  OutHandler GetOutput(uint32_t index) {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return nullptr, "[Check][Param] op_desc_ is nullptr.");
    string name = op_desc_->GetOutputNameByIndex(index);
    if (name.empty()) {
      REPORT_INNER_ERROR("E19999", "Find src name by index failed. index[%u]", index);
      GELOGE(GRAPH_FAILED, "[Get][OutputName] Find src name by index failed. index[%u]", index);
      return nullptr;
    }
    shared_ptr<OpIO> output_ptr = ComGraphMakeShared<OpIO>(name, index, shared_from_this());
    if (output_ptr == nullptr) {
      REPORT_CALL_ERROR("E19999", "OpIO make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OpIO make shared failed");
      return nullptr;
    }
    return output_ptr;
  }

  GeTensorDesc GetOutputDesc(const string &name) const {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");

    return op_desc_->GetOutputDesc(name);
  }

  GeTensorDesc GetOutputDesc(uint32_t index) const {
    GE_CHK_BOOL_EXEC(op_desc_ != nullptr, REPORT_INNER_ERROR("E19999", "op_desc_ is nullptr, check invalid.");
                     return GeTensorDesc(), "[Check][Param] op_desc_ is nullptr.");

    return op_desc_->GetOutputDesc(index);
  }

  graphStatus UpdateOutputDesc(const string &name, const GeTensorDesc &tensor_desc) {
    GE_CHK_BOOL_RET_STATUS(op_desc_ != nullptr, GRAPH_FAILED, "[Check][Param] op_desc is nullptr.");

    auto res = op_desc_->UpdateOutputDesc(name, tensor_desc);
    if (res == GRAPH_SUCCESS) {
      for (auto ol : output_links_[name]) {
        if (ol.GetOwner() == nullptr) {
          GELOGW("[Update][Check] %s get owner is nullptr", ol.GetName().c_str());
          continue;
        }
        GE_CHK_BOOL_RET_STATUS(ol.GetOwner()->UpdateInputDesc(ol.GetName(), tensor_desc) == GRAPH_SUCCESS, GRAPH_FAILED,
                               "[Update][InputDesc] Could not update next operator's input %s.", ol.GetName().c_str());
      }
    }
    return res;
  }

  size_t GetInputsSize() const {
    GE_IF_BOOL_EXEC(op_desc_ == nullptr, return 0);
    return op_desc_->GetInputsSize();
  }

  size_t GetOutputsSize() const {
    GE_IF_BOOL_EXEC(op_desc_ == nullptr, return 0);
    return op_desc_->GetOutputsSize();
  }

  graphStatus SetAttr(const string &name, GeAttrValue &&attr_value) {
    GE_CHK_BOOL_RET_STATUS(op_desc_ != nullptr, GRAPH_FAILED, "[Check][Param] op_desc is nullptr.");
    return op_desc_->SetAttr(name, std::move(attr_value));
  }

  graphStatus GetAttr(const string &name, GeAttrValue &attr_value) const {
    GE_CHK_BOOL_RET_STATUS(op_desc_ != nullptr, GRAPH_FAILED, "[Check][Param] op_desc is nullptr.");
    return op_desc_->GetAttr(name, attr_value);
  }

  OpDescPtr GetOpDescImpl() const { return op_desc_; }

  void UpdateLinkMapImpl(const string &src_name, OpIO &op_dst) {
    auto it_find = output_links_.find(src_name);
    if (it_find == output_links_.end()) {
      std::vector<OpIO> dsts{op_dst};
      output_links_.insert(std::make_pair(src_name, dsts));
    } else {
      it_find->second.push_back(op_dst);
    }
  }

  Operator ToOperator() { return Operator(shared_from_this()); }

  static OpDescPtr GetOpDesc(const Operator &oprt) {
    GE_IF_BOOL_EXEC(oprt.operator_impl_ == nullptr, return nullptr);
    return oprt.operator_impl_->op_desc_;
  }

  void ClearOutputLinks() noexcept { output_links_.clear(); }

  void ClearInputLinks() noexcept { input_link_.clear(); }

  ge::ConstNodePtr GetNode() { return node_; }

  void SetInferenceContext(const InferenceContextPtr &inference_context) { inference_context_ = inference_context; }

  InferenceContextPtr GetInferenceContext() const { return inference_context_; }

  void SubgraphRegister(const std::string &ir_name, bool dynamic) {
    op_desc_->RegisterSubgraphIrName(ir_name, dynamic ? kDynamic : kStatic);
  }

  void SubgraphCountRegister(const std::string &ir_name, uint32_t count) {
    if (op_desc_->GetSubgraphTypeByIrName(ir_name) == kStatic) {
      op_desc_->AddSubgraphName(ir_name);
      subgraph_names_to_builders_[ir_name] = nullptr;
    } else {
      for (uint32_t i = 0; i < count; ++i) {
        string key_name = ir_name + std::to_string(i);
        op_desc_->AddSubgraphName(key_name);
        subgraph_names_to_builders_[key_name] = nullptr;
      }
    }
  }

  void SetSubgraphBuilder(const std::string &ir_name, uint32_t index, const SubgraphBuilder &builder) {
    string key_name = ir_name;
    if (op_desc_->GetSubgraphTypeByIrName(ir_name) == kDynamic) {
      key_name += std::to_string(index);
    }

    auto it = subgraph_names_to_builders_.find(key_name);
    if (it == subgraph_names_to_builders_.end()) {
      REPORT_INNER_ERROR("E19999", "Failed to set subgraph builder for name %s index %u.",
                         ir_name.c_str(), index);
      GELOGE(PARAM_INVALID, "[Check][Param] Failed to set subgraph builder for name %s index %u.",
             ir_name.c_str(), index);
      return;
    }
    it->second = builder;
  }

  SubgraphBuilder GetSubgraphBuilder(const std::string &ir_name, uint32_t index) const {
    string key_name = ir_name;
    if (op_desc_->GetSubgraphTypeByIrName(ir_name) == kDynamic) {
      key_name += std::to_string(index);
    }

    return GetSubgraphBuilder(key_name);
  }

  SubgraphBuilder GetSubgraphBuilder(const std::string &name) const {
    auto iter = subgraph_names_to_builders_.find(name);
    if (iter == subgraph_names_to_builders_.end()) {
      REPORT_INNER_ERROR("E19999", "Failed to get subgraph builder for name %s", name.c_str());
      GELOGE(PARAM_INVALID, "[Check][Param] Failed to get subgraph builder for name %s", name.c_str());
      return nullptr;
    }

    return iter->second;
  }

  std::vector<std::string> GetSubgraphNames() const {
    std::vector<std::string> names;
    for (const auto &subgraph_name_to_type : op_desc_->GetSubgraphIrNames()) {
      names.emplace_back(subgraph_name_to_type.first);
    }
    return names;
  }

  size_t GetSubgraphNamesCount() const {
    return op_desc_->GetSubgraphIrNames().size();
  }

  OpDescPtr op_desc_ = nullptr;

 private:
  ge::ConstNodePtr node_{nullptr};
  ge::InferenceContextPtr inference_context_;
  std::map<string, std::vector<OpIO>> output_links_{};
  std::map<string, OpIO> input_link_{};
  std::vector<std::weak_ptr<OperatorImpl>> control_input_link_{};
  std::vector<std::weak_ptr<OperatorImpl>> control_output_link_{};
  std::map<std::string, SubgraphBuilder> subgraph_names_to_builders_;
};

// Used to manage OperatorImpl instances created by ge api.
class OperatorKeeper {
 private:
  OperatorKeeper() = default;
  ~OperatorKeeper() {
    for (const auto &iter : operators_) {
      if (iter) {
        iter->ClearInputLinks();
        iter->ClearOutputLinks();
      }
    }
  }
  std::set<OperatorImplPtr> operators_;
  std::mutex mutex_;

 public:
  static OperatorKeeper &GetInstance() {
    static OperatorKeeper instance;
    return instance;
  }
  void CheckInOperator(const OperatorImplPtr &op_impl) {
    if (op_impl) {
      std::lock_guard<std::mutex> lock(mutex_);
      operators_.insert(op_impl);
    }
  }
  void CheckOutOperator(const OperatorImplPtr &op_impl) {
    if (op_impl) {
      std::lock_guard<std::mutex> lock(mutex_);
      operators_.erase(op_impl);
    }
  }
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromNode(ge::ConstNodePtr node_ptr) {
  ge::OperatorImplPtr operator_impl_ptr = ComGraphMakeShared<OperatorImpl>(node_ptr);
  if (operator_impl_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
    return Operator("default");
  }
  return operator_impl_ptr->ToOperator();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::CopyOperators(ComputeGraphPtr &dst_compute_graph,
                           const std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                           const std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                           const std::map<string, ge::Operator> &src_op_list,
                           std::map<string, ge::Operator> &dst_op_list) {
  GE_CHECK_NOTNULL(dst_compute_graph);

  std::map<OperatorImplPtr, NodePtr> all_node_info;

  for (const auto &itr : src_op_list) {
    auto name = itr.first;
    const ge::Operator &src_op = itr.second;
    GE_CHECK_NOTNULL(src_op.operator_impl_);
    OperatorImplPtr scr_op_impl_ptr = src_op.operator_impl_;
    GE_CHECK_NOTNULL(scr_op_impl_ptr->op_desc_);
    ge::Operator dst_op;
    OpDescPtr dst_op_desc = nullptr;
    if (scr_op_impl_ptr->node_ == nullptr) {
      // cannot find op_desc in compute graph, need creat new op_desc
      // otherwise use existing op_desc
      auto it = op_desc_old_2_new.find(scr_op_impl_ptr->op_desc_);
      if (it != op_desc_old_2_new.end()) {
        dst_op_desc = it->second;
      } else {
        dst_op_desc = AttrUtils::CopyOpDesc(scr_op_impl_ptr->op_desc_);
        if (dst_op_desc == nullptr) {
          REPORT_CALL_ERROR("E19999", "CopyOpDesc from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Copy][OpDesc] from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          return GRAPH_FAILED;
        }
        dst_op_desc->CopyAttrsFrom(*scr_op_impl_ptr->op_desc_);
        dst_op_desc->SetName(scr_op_impl_ptr->op_desc_->GetName());
      }
      dst_op = CreateOperatorFromOpDesc(dst_op_desc);
    } else {
      auto original_op_desc = scr_op_impl_ptr->node_->GetOpDesc();
      if (scr_op_impl_ptr->op_desc_ != original_op_desc) {
        REPORT_INNER_ERROR("E19999", "node and op_desc of operator are not equal.");
        GELOGE(GRAPH_FAILED, "[Check][Param] node and op_desc of operator are not equal.");
        return GRAPH_FAILED;
      }
      NodePtr dst_node = nullptr;
      // cannot find node in compute graph, need creat new node
      // otherwise use existing node and op_desc
      auto it = node_old_2_new.find(scr_op_impl_ptr->node_);
      if (it != node_old_2_new.end()) {
        dst_node = it->second;
      } else {
        dst_op_desc = AttrUtils::CopyOpDesc(scr_op_impl_ptr->op_desc_);
        if (dst_op_desc == nullptr) {
          REPORT_CALL_ERROR("E19999", "CopyOpDesc from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Copy][OpDesc] from %s failed", scr_op_impl_ptr->op_desc_->GetName().c_str());
          return GRAPH_FAILED;
        }
        dst_op_desc->CopyAttrsFrom(*scr_op_impl_ptr->op_desc_);
        dst_op_desc->SetName(scr_op_impl_ptr->op_desc_->GetName());
        dst_node = NodeUtils::CreatNodeWithoutGraph(dst_op_desc);
        GE_CHECK_NOTNULL(dst_node);
        // to do link egdes
      }
      dst_op = CreateOperatorFromNode(dst_node);
      all_node_info.emplace(dst_op.GetOperatorImplPtr(), dst_node);
    }
    dst_op.operator_impl_->subgraph_names_to_builders_ = src_op.operator_impl_->subgraph_names_to_builders_;
    dst_op_list.emplace(name, dst_op);
  }

  dst_compute_graph->SetAllNodesInfo(all_node_info);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
OpDescUtils::CopyOperatorLinks(const std::map<string, ge::Operator> &src_op_list,
                               std::map<string, ge::Operator> &dst_op_list) {
  for (const auto &it : src_op_list) {
    auto &src_op = it.second;
    auto op_name = it.first;
    auto &dst_op = dst_op_list[op_name];
    OperatorImplPtr src_impl_ptr = src_op.GetOperatorImplPtr();
    OperatorImplPtr dst_impl_ptr = dst_op.GetOperatorImplPtr();
    for (const auto &itr : src_impl_ptr->input_link_) {
      std::string dst_name = itr.first;
      const OpIO &op_io = itr.second;
      OperatorImplPtr input_impl_ptr = op_io.GetOwner();
      GE_CHECK_NOTNULL(input_impl_ptr);
      auto iter = dst_op_list.find(input_impl_ptr->GetName());
      if (iter == dst_op_list.end()) {
        REPORT_INNER_ERROR("E19999", "Find dst operator:%s failed", input_impl_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Find dst operator:%s failed", input_impl_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto &input_op = iter->second;
      dst_op.SetInput(dst_name, input_op);
    }

    for (const auto &itr : src_impl_ptr->control_input_link_) {
      OperatorImplPtr input_ctrl_impl_ptr = itr.lock();
      GE_CHECK_NOTNULL(input_ctrl_impl_ptr);
      auto iter = dst_op_list.find(input_ctrl_impl_ptr->GetName());
      if (iter == dst_op_list.end()) {
        REPORT_INNER_ERROR("E19999", "Find dst ctrl operator:%s failed", input_ctrl_impl_ptr->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] Find dst ctrl operator:%s failed", input_ctrl_impl_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }
      auto &ctrl_input_op = iter->second;
      dst_op.AddControlInput(ctrl_input_op);
    }
  }
  return GRAPH_SUCCESS;
}

Operator::Operator(const std::string &type) {
  static uint32_t index = 0;
  string name = type + "_" + std::to_string(index++);
  operator_impl_ = ComGraphMakeShared<OperatorImpl>(name, type);
  if (operator_impl_ == nullptr) {
    GELOGW("[Check][Param] Make OperatorImpl failed");
  }
  OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
}

Operator::Operator(const char *type) {
  if (type != nullptr) {
    std::string op_type = type;
    static uint32_t index = 0;
    string name = op_type + "_" + std::to_string(index++);
    operator_impl_ = ComGraphMakeShared<OperatorImpl>(name, op_type);
    if (operator_impl_ == nullptr) {
      GELOGW("[Check][Param] Make OperatorImpl failed");
    }
    OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
  } else {
    GELOGW("[Check][Param] Operator type is nullptr");
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY Operator OpDescUtils::CreateOperatorFromOpDesc(OpDescPtr op_desc) {
  shared_ptr<OperatorImpl> operator_impl_ptr;
  operator_impl_ptr = ComGraphMakeShared<OperatorImpl>(op_desc);
  if (operator_impl_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
    return Operator("default");
  }
  OperatorKeeper::GetInstance().CheckInOperator(operator_impl_ptr);
  return operator_impl_ptr->ToOperator();
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OpDescPtr OpDescUtils::GetOpDescFromOperator(const Operator &oprt) {
  return OperatorImpl::GetOpDesc(oprt);
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const string &name, const string &type) {
  operator_impl_ = ComGraphMakeShared<OperatorImpl>(name, type);
  if (operator_impl_ == nullptr) {
    REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
    GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
    return;
  }
  OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const AscendString &name, const AscendString &type) {
  if ((name.GetString() != nullptr) && (type.GetString() != nullptr)) {
    string op_name = name.GetString();
    string op_type = type.GetString();
    operator_impl_ = ComGraphMakeShared<OperatorImpl>(op_name, op_type);
    if (operator_impl_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
      return;
    }
    OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
  } else {
    GELOGW("[Check][Param] Operator input parameter is nullptr");
  }
}

GE_FUNC_HOST_VISIBILITY Operator::Operator(const char *name, const char *type) {
  if ((name != nullptr) && (type != nullptr)) {
    string op_name = name;
    string op_type = type;
    operator_impl_ = ComGraphMakeShared<OperatorImpl>(op_name, op_type);
    if (operator_impl_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "OperatorImpl make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] OperatorImpl make shared failed");
      return;
    }
    OperatorKeeper::GetInstance().CheckInOperator(operator_impl_);
  } else {
    GELOGW("[Check][Param] Operator input parameter is nullptr");
  }
}

Operator::Operator(ge::OperatorImplPtr &&op_impl) { operator_impl_ = std::move(op_impl); }

bool Operator::IsEmpty() const {
  if (operator_impl_ == nullptr) {
    return true;
  }
  return false;
}

string Operator::GetName() const {
  if (operator_impl_ != nullptr) {
    return operator_impl_->GetName();
  }
  return "";
}

graphStatus Operator::GetName(AscendString &name) const {
  if (operator_impl_ != nullptr) {
    string op_name = operator_impl_->GetName();
    name = op_name.c_str();
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_HOST_VISIBILITY Operator &Operator::SetInput(const string &dst_name, const ge::Operator &src_oprt) {
  // Describe the connection relationship between operators, no create action
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator impl is nullptr, check invalid.");
                   return *this, "[Check][Param] operator impl is nullptr.");
  operator_impl_->SetInputImpl(dst_name, src_oprt);
  return *this;
}

GE_FUNC_HOST_VISIBILITY Operator &Operator::SetInput(const char *dst_name, const ge::Operator &src_oprt) {
  GE_CHK_BOOL_EXEC(dst_name != nullptr, REPORT_INNER_ERROR("E19999", "param dst name is nullptr, check invalid");
                   return *this, "[Check][Param] Operator dst name is nullptr.");
  // Describe the connection relationship between operators, no create action
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator impl is nullptr.");
  std::string dst_op_name = dst_name;
  operator_impl_->SetInputImpl(dst_op_name, src_oprt);
  return *this;
}

Operator &Operator::SetInput(const string &dst_name, const ge::OutHandler &out_handler) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] operator impl is nullptr.");
  operator_impl_->SetInputImpl(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt, const std::string &name) {
  auto out_handler = src_oprt.GetOutput(name);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by name:%s failed, out_handler is nullptr.", name.c_str());
                   return *this, "[Get][Output] by name:%s failed, out_handler is nullptr.", name.c_str());
  (void)SetInput(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const char *dst_name, const ge::Operator &src_oprt, const char *name) {
  GE_CHK_BOOL_EXEC(dst_name != nullptr, REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid.");
                   return *this, "[Check][Param] Dst name is nullptr.");
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
                   return *this, "[Check][Param] Name is nullptr.");
  std::string op_name = name;
  std::string dst_op_name = dst_name;
  auto out_handler = src_oprt.GetOutput(op_name);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by name:%s failed, out_handler is nullptr.",
                                      op_name.c_str());
                   return *this, "[Get][Output] by name:%s failed, out_handler is nullptr.", op_name.c_str());
  (void)SetInput(dst_op_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const std::string &dst_name, const ge::Operator &src_oprt, uint32_t index) {
  auto out_handler = src_oprt.GetOutput(index);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by index:%u failed, out_handler is nullptr.", index);
                   return *this, "[Get][Output] by index:%u failed, out_handler is nullptr.", index);
  (void)SetInput(dst_name, out_handler);
  return *this;
}

Operator &Operator::SetInput(const char *dst_name, const ge::Operator &src_oprt, uint32_t index) {
  GE_CHK_BOOL_EXEC(dst_name != nullptr, REPORT_INNER_ERROR("E19999", "param dst_name is nullptr, check invalid");
                   return *this, "[Check][Param] Dst name is nullptr.");
  auto out_handler = src_oprt.GetOutput(index);
  GE_CHK_BOOL_EXEC(out_handler != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOutput by index:%u failed, out_handler is nullptr.", index);
                   return *this, "[Get][Output] by index:%u failed, out_handler is nullptr.", index);
  std::string op_dst_name = dst_name;
  (void)SetInput(dst_name, out_handler);
  return *this;
}

Operator &Operator::AddControlInput(const Operator &src_oprt) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator impl is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
    return *this;
  }
  operator_impl_->AddControlInputImp(src_oprt);
  return *this;
}

graphStatus Operator::GetInputConstData(const string &dst_name, Tensor &data) const {
  GE_CHECK_NOTNULL(operator_impl_);
  graphStatus ret = operator_impl_->GetInputConstData(dst_name, data);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("[Get][ConstInput] %s get input const data failed", dst_name.c_str());
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstData(const char *dst_name, Tensor &data) const {
  GE_CHECK_NOTNULL(dst_name);
  GE_CHECK_NOTNULL(operator_impl_);
  std::string op_dst_name = dst_name;
  graphStatus ret = operator_impl_->GetInputConstData(op_dst_name, data);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("[Get][ConstInput] %s get input const data failed", op_dst_name.c_str());
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetInputConstDataOut(const string &dst_name, Tensor &data) const {
  GE_CHECK_NOTNULL(operator_impl_);
  if (operator_impl_->GetInputConstDataOut(dst_name, data) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "%s get input const data out failed", dst_name.c_str());
    GELOGE(GRAPH_FAILED, "[Get][Tensor] %s get input const data out failed", dst_name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

std::shared_ptr<const Node>  Operator::GetNode() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetNode();
}

TensorDesc Operator::GetInputDesc(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(name));
}

TensorDesc Operator::GetInputDescByName(const char *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(op_name));
}

void Operator::SetInferenceContext(const InferenceContextPtr &inference_context) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  operator_impl_->SetInferenceContext(inference_context);
}

InferenceContextPtr Operator::GetInferenceContext() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetInferenceContext();
}

TensorDesc Operator::GetInputDesc(uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(index));
}

graphStatus Operator::TryGetInputDesc(const string &name, TensorDesc &tensor_desc) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  auto check = operator_impl_->InputIsSet(name);
  if (check)
    tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(name));
  return check ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Operator::TryGetInputDesc(const char *name, TensorDesc &tensor_desc) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  auto check = operator_impl_->InputIsSet(op_name);
  if (check)
    tensor_desc = TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(op_name));
  return check ? GRAPH_SUCCESS : GRAPH_FAILED;
}

graphStatus Operator::UpdateInputDesc(const std::string &name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateInputDesc(name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateInputDesc(const char *name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return operator_impl_->UpdateInputDesc(op_name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

OutHandler Operator::GetOutput(const string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetOutput(name);
}

OutHandler Operator::GetOutput(uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return nullptr, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetOutput(index);
}

TensorDesc Operator::GetOutputDesc(const std::string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(name));
}

TensorDesc Operator::GetOutputDescByName(const char *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(op_name));
}

TensorDesc Operator::GetOutputDesc(uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(index));
}

graphStatus Operator::UpdateOutputDesc(const std::string &name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateOutputDesc(name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateOutputDesc(const char *name, const ge::TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return operator_impl_->UpdateOutputDesc(op_name, TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

TensorDesc Operator::GetDynamicInputDesc(const string &name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(name + std::to_string(index)));
}

TensorDesc Operator::GetDynamicInputDesc(const char *name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetInputDesc(op_name + std::to_string(index)));
}

graphStatus Operator::UpdateDynamicInputDesc(const string &name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateInputDesc(name + std::to_string(index),
                                         TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateDynamicInputDesc(const char *name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  std::string op_name = name;
  return operator_impl_->UpdateInputDesc(op_name + std::to_string(index),
                                         TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

TensorDesc Operator::GetDynamicOutputDesc(const string &name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] operator impl is nullptr.");
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(name + std::to_string(index)));
}

TensorDesc Operator::GetDynamicOutputDesc(const char *name, uint32_t index) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return TensorDesc(), "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return TensorAdapter::GeTensorDesc2TensorDesc(operator_impl_->GetOutputDesc(op_name + std::to_string(index)));
}

graphStatus Operator::UpdateDynamicOutputDesc(const string &name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->UpdateOutputDesc(name + std::to_string(index),
                                          TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::UpdateDynamicOutputDesc(const char *name, uint32_t index, const TensorDesc &tensor_desc) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return operator_impl_->UpdateOutputDesc(op_name + std::to_string(index),
                                          TensorAdapter::TensorDesc2GeTensorDesc(tensor_desc));
}

graphStatus Operator::InferShapeAndType() {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");

  return operator_impl_->GetOpDescImpl()->CallInferFunc(*this);
}

graphStatus Operator::VerifyAllAttr(bool disable_common_verifier) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");

  if (!disable_common_verifier && (graphStatus)Operator::VerifyAll() == GRAPH_FAILED) {
    return GRAPH_FAILED;
  } else {
    return (graphStatus)operator_impl_->GetOpDescImpl()->OpVerify();
  }
}

GE_FUNC_HOST_VISIBILITY size_t Operator::GetInputsSize() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] OperatorImpl_ is nullptr");
  return operator_impl_->GetInputsSize();
}

GE_FUNC_HOST_VISIBILITY size_t Operator::GetOutputsSize() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] OperatorImpl_ is nullptr");
  return operator_impl_->GetOutputsSize();
}

// According to op get the attrs name and type
namespace {
const std::map<GeAttrValue::ValueType, std::string> kAttrTypesMap = {
    {GeAttrValue::VT_NONE, "VT_STRING"},
    {GeAttrValue::VT_STRING, "VT_STRING"},
    {GeAttrValue::VT_FLOAT, "VT_FLOAT"},
    {GeAttrValue::VT_BOOL, "VT_BOOL"},
    {GeAttrValue::VT_INT, "VT_INT"},
    {GeAttrValue::VT_TENSOR_DESC, "VT_TENSOR_DESC"},
    {GeAttrValue::VT_TENSOR, "VT_TENSOR"},
    {GeAttrValue::VT_BYTES, "VT_BYTES"},
    {GeAttrValue::VT_GRAPH, "VT_GRAPH"},
    {GeAttrValue::VT_NAMED_ATTRS, "VT_NAMED_ATTRS"},
    {GeAttrValue::VT_LIST_LIST_INT, "VT_LIST_LIST_INT"},
    {GeAttrValue::VT_DATA_TYPE, "VT_DATA_TYPE"},
    {GeAttrValue::VT_LIST_BASE, "VT_LIST_BASE"},
    {GeAttrValue::VT_LIST_STRING, "VT_LIST_STRING"},
    {GeAttrValue::VT_LIST_FLOAT, "VT_LIST_FLOAT"},
    {GeAttrValue::VT_LIST_BOOL, "VT_LIST_BOOL"},
    {GeAttrValue::VT_LIST_INT, "VT_LIST_INT"},
    {GeAttrValue::VT_LIST_TENSOR_DESC, "VT_LIST_TENSOR_DESC"},
    {GeAttrValue::VT_LIST_TENSOR, "VT_LIST_TENSOR"},
    {GeAttrValue::VT_LIST_BYTES, "VT_LIST_BYTES"},
    {GeAttrValue::VT_GRAPH, "VT_GRAPH"},
    {GeAttrValue::VT_LIST_NAMED_ATTRS, "VT_LIST_NAMED_ATTRS"},
    {GeAttrValue::VT_LIST_DATA_TYPE, "VT_LIST_DATA_TYPE"},
};
} // namespace
const std::map<std::string, std::string> Operator::GetAllAttrNamesAndTypes() const {
  std::map<std::string, std::string> attr_types;

  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return attr_types, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return attr_types, "[Get][OpDescImpl] is nullptr.");
  std::map<string, GeAttrValue> attr_map = operator_impl_->GetOpDescImpl()->GetAllAttrs();

  map<string, GeAttrValue>::iterator iter;
  for (iter = attr_map.begin(); iter != attr_map.end(); ++iter) {
    string name = iter->first;
    GeAttrValue attr_value = iter->second;

    GeAttrValue::ValueType type = attr_value.GetValueType();

    auto iter2 = kAttrTypesMap.find(type);
    if (iter2 != kAttrTypesMap.end()) {
      attr_types[name] = iter2->second;
    }
  }

  return attr_types;
}

graphStatus Operator::GetAllAttrNamesAndTypes(std::map<AscendString, AscendString> &attr_name_types) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");
  std::map<string, GeAttrValue> attr_map = operator_impl_->GetOpDescImpl()->GetAllAttrs();

  map<string, GeAttrValue>::iterator iter;
  for (iter = attr_map.begin(); iter != attr_map.end(); ++iter) {
    string name = iter->first;
    GeAttrValue attr_value = iter->second;

    GeAttrValue::ValueType type = attr_value.GetValueType();

    auto iter2 = kAttrTypesMap.find(type);
    if (iter2 != kAttrTypesMap.end()) {
      AscendString temp(name.c_str());
      attr_name_types[temp] = AscendString(iter2->second.c_str());
    }
  }

  return GRAPH_SUCCESS;
}

void Operator::InputRegister(const string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  (void)operator_impl_->GetOpDescImpl()->AddInputDesc(name, GeTensorDesc());
}

void Operator::OptionalInputRegister(const string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddOptionalInputDesc(name,
                                                              GeTensorDesc(GeShape(), FORMAT_RESERVED, DT_UNDEFINED));
}

void Operator::InferFuncRegister(const std::function<graphStatus(Operator &)> &func) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddInferFunc(func);
}

void Operator::InferFormatFuncRegister(const std::function<graphStatus(Operator &)> &func) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddInferFormatFunc(func);
}

void Operator::VerifierFuncRegister(const std::function<graphStatus(Operator &)> &func) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddVerifierFunc(func);
}

void Operator::OutputRegister(const string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  // [No need to verify return value]
  (void)operator_impl_->GetOpDescImpl()->AddOutputDesc(name, GeTensorDesc());
}

void Operator::DynamicInputRegister(const string &name, const unsigned int num, bool is_push_back) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_INPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "set attr %s to op:%s failed.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return, "[Set][Int] %s to op:%s failed", name.c_str(),
                   operator_impl_->GetOpDescImpl()->GetName().c_str());
  (void)operator_impl_->GetOpDescImpl()->AddDynamicInputDesc(name, num, is_push_back);
}

void Operator::DynamicInputRegisterByIndex(const string &name, const unsigned int num, size_t index) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  operator_impl_->GetOpDescImpl()->AddDynamicInputDescByIndex(name, num, index);
}

int Operator::GetDynamicInputNum(const string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  int num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_INPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Init] %s failed", name.c_str());
  return num;
}

int Operator::GetDynamicInputNum(const char *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return 0, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  string op_name = name;
  int num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_INPUT_TD_NUM(op_name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", op_name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Int] %s failed", op_name.c_str());
  return num;
}

void Operator::DynamicOutputRegister(const string &name, const unsigned int num, bool is_push_back) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  GE_CHK_BOOL_EXEC(AttrUtils::SetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_OUTPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "set attr %s to op:%s failed.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return, "[Set][Int] %s to op:%s failed", name.c_str(),
                   operator_impl_->GetOpDescImpl()->GetName().c_str());
  (void)operator_impl_->GetOpDescImpl()->AddDynamicOutputDesc(name, num, is_push_back);
}

int Operator::GetDynamicOutputNum(const string &name) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  int num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_OUTPUT_TD_NUM(name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Init] %s failed", name.c_str());
  return num;
}

int Operator::GetDynamicOutputNum(const char *name) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
                   return 0, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return 0, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return 0, "[Get][OpDescImpl] is nullptr.");
  std::string op_name = name;
  int num = 0;
  GE_CHK_BOOL_EXEC(AttrUtils::GetInt(operator_impl_->GetOpDescImpl(), DYNAMIC_OUTPUT_TD_NUM(op_name), num),
                   REPORT_INNER_ERROR("E19999", "get attr %s failed, op:%s.", op_name.c_str(),
                                      operator_impl_->GetOpDescImpl()->GetName().c_str());
                   return num, "[Get][Init] %s failed", op_name.c_str());
  return num;
}

void Operator::RequiredAttrRegister(const string &name) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return, "[Get][OpDescImpl] is nullptr.");
  operator_impl_->GetOpDescImpl()->AddRequiredAttr(name);
}

graphStatus Operator::VerifyAll() {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_->GetOpDescImpl() != nullptr,
                   REPORT_INNER_ERROR("E19999", "GetOpDescImpl failed, as return nullptr.");
                   return GRAPH_FAILED, "[Get][OpDescImpl] is nullptr.");

  // Check all inputs defined
  for (const string &iname : operator_impl_->GetOpDescImpl()->GetAllInputNames()) {
    GE_CHK_BOOL_RET_STATUS(operator_impl_->GetOpDescImpl()->IsOptionalInput(iname) || operator_impl_->InputIsSet(iname),
                           GRAPH_FAILED, "[Check][Param] operator input %s is not linked.", iname.c_str());
    vector<int64_t> ishape = operator_impl_->GetOpDescImpl()->GetInputDesc(iname).GetShape().GetDims();
    for (int64_t dim : ishape) {
      GE_CHK_BOOL_RET_STATUS(dim > 0, GRAPH_FAILED,
                             "[Check][Param] operator input %s shape contains negative or zero dimension, "
                             "node:%s, index:%d.",
                             iname.c_str(), operator_impl_->GetOpDescImpl()->GetName().c_str(),
                             operator_impl_->GetOpDescImpl()->GetInputIndexByName(iname));
    }
  }
  // Check all attributes defined
  const auto all_attributes = operator_impl_->GetOpDescImpl()->GetAllAttrs();
  for (const auto &name : operator_impl_->GetOpDescImpl()->GetAllAttrNames()) {
    GE_CHK_BOOL_RET_STATUS(all_attributes.find(name) != all_attributes.end(), GRAPH_FAILED,
                           "[Check][Param] operator attribute %s is empty.", name.c_str());
  }

  return GRAPH_SUCCESS;
}

string Operator::GetOpType() const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return "Data", "[Check][Param] operator impl is nullptr.");
  return OperatorImpl::GetOpDesc(*this)->GetType();
}

graphStatus Operator::GetOpType(AscendString &type) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_type = OperatorImpl::GetOpDesc(*this)->GetType();
  type = op_type.c_str();
  return GRAPH_SUCCESS;
}

Operator &Operator::SetInput(const std::string &dst_name, uint32_t dst_index, const ge::Operator &src_oprt) {
  string dynamic_dst_name = DYNAMIN_INPUT_NAME(dst_name, dst_index);
  return SetInput(dynamic_dst_name, src_oprt);
}

Operator &Operator::SetInput(const std::string &dst_name, uint32_t dst_index, const ge::Operator &src_oprt,
                             const std::string &name) {
  string dynamic_dst_name = DYNAMIN_INPUT_NAME(dst_name, dst_index);
  return SetInput(dynamic_dst_name, src_oprt, name);
}

OperatorImplPtr Operator::GetOperatorImplPtr() const { return operator_impl_; }

#define OP_ATTR_SET_IMP(ArgType, AttrUtilsFun)                                                                         \
  Operator &Operator::SetAttr(const string &name, ArgType attr_value) {                                                \
    if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {                                     \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());                         \
      return *this;                                                                                                    \
    }                                                                                                                  \
    if (!AttrUtils::Set##AttrUtilsFun(operator_impl_->GetOpDescImpl(), name, attr_value)) {                            \
      GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());                                                     \
    }                                                                                                                  \
    return *this;                                                                                                      \
  }                                                                                                                    \
  Operator &Operator::SetAttr(const char *name, ArgType attr_value) {                                                  \
    if (name == nullptr) {                                                                                             \
      REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");                                           \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator attr name is nullptr.");                                           \
      return *this;                                                                                                    \
    }                                                                                                                  \
    std::string op_name = name;                                                                                        \
    if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {                                     \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");                  \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", op_name.c_str());                      \
      return *this;                                                                                                    \
    }                                                                                                                  \
    if (!AttrUtils::Set##AttrUtilsFun(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {                         \
      GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());                                                  \
    }                                                                                                                  \
    return *this;                                                                                                      \
  }

#define OP_ATTR_GET_IMP(ArgType, AttrUtilsFun)                                                                         \
  graphStatus Operator::GetAttr(const string &name, ArgType attr_value) const {                                        \
    if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {                                     \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());                         \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    if (!AttrUtils::Get##AttrUtilsFun(operator_impl_->GetOpDescImpl(), name, attr_value)) {                            \
      GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());                                                     \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    return GRAPH_SUCCESS;                                                                                              \
  }                                                                                                                    \
  graphStatus Operator::GetAttr(const char *name, ArgType attr_value) const {                                          \
    if (name == nullptr) {                                                                                             \
      REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");                                            \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator attr name is nullptr.");                                           \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {                                     \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name);                                 \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    std::string op_name = name;                                                                                        \
    if (!AttrUtils::Get##AttrUtilsFun(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {                         \
      GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());                                                  \
      return GRAPH_FAILED;                                                                                             \
    }                                                                                                                  \
    return GRAPH_SUCCESS;                                                                                              \
  }

void Operator::BreakConnect() const {
  if (operator_impl_ == nullptr) {
    GELOGW("[Check][Param] operator impl is nullptr");
    return;
  }
  operator_impl_->ClearInputLinks();
  operator_impl_->ClearOutputLinks();
  OperatorKeeper::GetInstance().CheckOutOperator(operator_impl_);
}

#define OP_ATTR_REG_IMP(ArgType, AttrUtilsFun)                                                                         \
  void Operator::AttrRegister(const string &name, ArgType attr_value) {                                                \
    if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {                                     \
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");                   \
      GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());                         \
      return;                                                                                                          \
    }                                                                                                                  \
    if (!AttrUtils::Set##AttrUtilsFun(operator_impl_->GetOpDescImpl(), name, attr_value)) {                            \
      GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());                                                \
    }                                                                                                                  \
  } // lint !e665

OP_ATTR_SET_IMP(int64_t, Int)
OP_ATTR_SET_IMP(int32_t, Int)
OP_ATTR_SET_IMP(uint32_t, Int)
OP_ATTR_GET_IMP(int64_t &, Int)
OP_ATTR_GET_IMP(int32_t &, Int)
OP_ATTR_GET_IMP(uint32_t &, Int)
OP_ATTR_SET_IMP(const vector<int64_t> &, ListInt)
OP_ATTR_SET_IMP(const vector<int32_t> &, ListInt)
OP_ATTR_SET_IMP(const vector<uint32_t> &, ListInt)
OP_ATTR_SET_IMP(std::initializer_list<int64_t> &&, ListInt)
OP_ATTR_GET_IMP(vector<int64_t> &, ListInt)
OP_ATTR_GET_IMP(vector<int32_t> &, ListInt)
OP_ATTR_GET_IMP(vector<uint32_t> &, ListInt)
OP_ATTR_GET_IMP(vector<vector<int64_t>> &, ListListInt)
OP_ATTR_SET_IMP(const vector<vector<int64_t>> &, ListListInt)

OP_ATTR_SET_IMP(float, Float)
OP_ATTR_GET_IMP(float &, Float)
OP_ATTR_SET_IMP(const vector<float> &, ListFloat)
OP_ATTR_GET_IMP(vector<float> &, ListFloat) // lint !e665

OP_ATTR_SET_IMP(bool, Bool)
OP_ATTR_GET_IMP(bool &, Bool)
OP_ATTR_SET_IMP(const vector<bool> &, ListBool)
OP_ATTR_GET_IMP(vector<bool> &, ListBool) // lint !e665

OP_ATTR_SET_IMP(const GeAttrValue::NAMED_ATTRS &, NamedAttrs)
OP_ATTR_GET_IMP(GeAttrValue::NAMED_ATTRS &, NamedAttrs)
OP_ATTR_SET_IMP(const vector<GeAttrValue::NAMED_ATTRS> &, ListNamedAttrs)
OP_ATTR_GET_IMP(vector<GeAttrValue::NAMED_ATTRS> &, ListNamedAttrs)  // lint !e665

OP_ATTR_REG_IMP(int64_t, Int)
OP_ATTR_REG_IMP(const vector<int64_t> &, ListInt)
OP_ATTR_REG_IMP(float, Float)
OP_ATTR_REG_IMP(const vector<float> &, ListFloat)
OP_ATTR_REG_IMP(const string &, Str)
OP_ATTR_REG_IMP(const vector<string> &, ListStr)
OP_ATTR_REG_IMP(bool, Bool)
OP_ATTR_REG_IMP(const vector<bool> &, ListBool)
OP_ATTR_REG_IMP(const vector<vector<int64_t>> &, ListListInt)
OP_ATTR_REG_IMP(const GeAttrValue::NAMED_ATTRS &, NamedAttrs)
OP_ATTR_REG_IMP(const vector<GeAttrValue::NAMED_ATTRS> &, ListNamedAttrs)

#undef OP_ATTR_SET_IMP
#undef OP_ATTR_GET_IMP
#undef OP_ATTR_REG_IMP

void Operator::AttrRegister(const string &name, const AscendString &attr_value) {
  if (attr_value.GetString() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Attr %s register param is invalid.", name.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Attr %s register param is invalid.", name.c_str());
    return;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  std::string str_attr_value = attr_value.GetString();
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), name, str_attr_value)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const string &name, const std::vector<AscendString> &attr_value) {
  std::vector<std::string> str_attr_values;
  for (auto &val : attr_value) {
    if (val.GetString() == nullptr) {
      REPORT_INNER_ERROR("E19999", "Attr %s register value is invalid.", name.c_str());
      GELOGE(GRAPH_FAILED, "Attr %s register value is invalid.", name.c_str());
      return;
    }
    str_attr_values.emplace_back(val.GetString());
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetListStr(operator_impl_->GetOpDescImpl(), name, str_attr_values)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

Operator &Operator::SetAttr(const string &name, const string &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
      GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
      return *this;
    }
    if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
      GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
    }
    return *this;
}

graphStatus Operator::GetAttr(const string &name, string &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
      REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
      GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
      return GRAPH_FAILED;
    }
    if (!AttrUtils::GetStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
      GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
      return GRAPH_FAILED;
    }
    return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const string &name, const std::vector<string> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetListStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const string &name, std::vector<string> &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetListStr(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char *name, const char *attr_value) {
  if (name == nullptr || attr_value == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr or attr_value is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return *this;
  }

  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  std::string op_attr_value = attr_value;
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char *name, const AscendString &attr_value) {
  if (name == nullptr || attr_value.GetString() == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  std::string op_attr_value = attr_value.GetString();
  if (!AttrUtils::SetStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const char *name, AscendString &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator input parameters is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  std::string op_attr_value;
  if (!AttrUtils::GetStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  attr_value = AscendString(op_attr_value.c_str());
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const char *name, const std::vector<AscendString> &attr_values) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return  *this;
  }
  std::vector<std::string> op_attr_values;
  for (auto &attr_value : attr_values) {
    if (attr_value.GetString() == nullptr) {
      REPORT_INNER_ERROR("E19999", "Operator ascend string name is nullptr, check invalid");
      GELOGE(GRAPH_FAILED, "[Check][Param] Operator ascend string name is nullptr.");
      return  *this;
    }
    op_attr_values.emplace_back(attr_value.GetString());
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  if (!AttrUtils::SetListStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_values)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const char *name, std::vector<AscendString> &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator input parameters name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  std::vector<std::string> op_attr_values;
  if (!AttrUtils::GetListStr(operator_impl_->GetOpDescImpl(), op_name, op_attr_values)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  for (auto &op_attr_value : op_attr_values) {
    attr_value.emplace_back(AscendString(op_attr_value.c_str()));
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const string &name, const Tensor &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  GeTensor tensor = TensorAdapter::AsGeTensor(attr_value);
  if (!AttrUtils::SetTensor(operator_impl_->GetOpDescImpl(), name, tensor)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char *name, const Tensor &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  GeTensor tensor = TensorAdapter::AsGeTensor(attr_value);
  if (!AttrUtils::SetTensor(operator_impl_->GetOpDescImpl(), op_name, tensor)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const string &name, const vector<Tensor> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  vector<GeTensor> val_list;
  for (const auto &item : attr_value) {
    auto tensor = TensorAdapter::AsGeTensor(item);
    val_list.push_back(tensor);
  }
  if (!AttrUtils::SetListTensor(operator_impl_->GetOpDescImpl(), name, val_list)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char *name, const vector<Tensor> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  vector<GeTensor> val_list;
  for (const auto &item : attr_value) {
    auto tensor = TensorAdapter::AsGeTensor(item);
    val_list.push_back(tensor);
  }
  if (!AttrUtils::SetListTensor(operator_impl_->GetOpDescImpl(), op_name, val_list)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const string &name, Tensor &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  ConstGeTensorPtr tensor;
  if (!AttrUtils::GetTensor(operator_impl_->GetOpDescImpl(), name, tensor)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  attr_value = TensorAdapter::GeTensor2Tensor(tensor);
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char *name, Tensor &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  ConstGeTensorPtr tensor;
  if (!AttrUtils::GetTensor(operator_impl_->GetOpDescImpl(), op_name, tensor)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  attr_value = TensorAdapter::GeTensor2Tensor(tensor);
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const string &name, vector<Tensor> &attr_value) const {
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  vector<ConstGeTensorPtr> val_list;
  if (!AttrUtils::GetListTensor(operator_impl_->GetOpDescImpl(), name, val_list)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  for (auto &tensor : val_list) {
    attr_value.push_back(TensorAdapter::GeTensor2Tensor(tensor));
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char *name, vector<Tensor> &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  vector<ConstGeTensorPtr> val_list;
  if (!AttrUtils::GetListTensor(operator_impl_->GetOpDescImpl(), op_name, val_list)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  for (auto &tensor : val_list) {
    attr_value.push_back(TensorAdapter::GeTensor2Tensor(tensor));
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const string &name, const OpBytes &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetZeroCopyBytes(operator_impl_->GetOpDescImpl(), name,
                                   Buffer::CopyFrom(attr_value.data(), attr_value.size()))) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char *name, const OpBytes &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  if (!AttrUtils::SetZeroCopyBytes(operator_impl_->GetOpDescImpl(), op_name,
                                   Buffer::CopyFrom(attr_value.data(), attr_value.size()))) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const string &name, OpBytes &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  Buffer buffer;
  if (!AttrUtils::GetZeroCopyBytes(operator_impl_->GetOpDescImpl(), name, buffer)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (buffer.data() == nullptr) {
    REPORT_CALL_ERROR("E19999", "buffer data is null, op:%s", operator_impl_->GetOpDescImpl()->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] buffer data is null.");
    return GRAPH_FAILED;
  }
  attr_value.assign(buffer.data(), buffer.data() + buffer.size());
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char *name, OpBytes &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  Buffer buffer;
  if (!AttrUtils::GetZeroCopyBytes(operator_impl_->GetOpDescImpl(), op_name, buffer)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (buffer.data() == nullptr) {
    REPORT_CALL_ERROR("E19999", "buffer data is null, op:%s", operator_impl_->GetOpDescImpl()->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] buffer data is null.");
    return GRAPH_FAILED;
  }
  attr_value.assign(buffer.data(), buffer.data() + buffer.size());
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const string &name, ge::AttrValue &&attrValue) {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator impl is nullptr.");
  (void)operator_impl_->SetAttr(name, std::move(attrValue.impl->geAttrValue_));
  return *this;
}

Operator &Operator::SetAttr(const char *name, ge::AttrValue &&attrValue) {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return *this, "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  (void)operator_impl_->SetAttr(op_name, std::move(attrValue.impl->geAttrValue_));
  return *this;
}

graphStatus Operator::GetAttr(const string &name, ge::AttrValue &attrValue) const {
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
  return operator_impl_->GetAttr(name, attrValue.impl->geAttrValue_);
}

graphStatus Operator::GetAttr(const char *name, ge::AttrValue &attrValue) const {
  GE_CHK_BOOL_EXEC(name != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
  GE_CHK_BOOL_EXEC(operator_impl_ != nullptr, REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
                   return GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
  std::string op_name = name;
  return operator_impl_->GetAttr(op_name, attrValue.impl->geAttrValue_);
}

Operator &Operator::SetAttr(const string &name, const std::vector<ge::DataType> &attr_value) {
  if (operator_impl_ == nullptr || !operator_impl_->GetOpDescImpl()) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetListDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char *name, const std::vector<ge::DataType> &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || !operator_impl_->GetOpDescImpl()) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  if (!AttrUtils::SetListDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const string &name, std::vector<ge::DataType> &attr_value) const {
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetListDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char *name, std::vector<ge::DataType> &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  attr_value.clear();
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  if (!AttrUtils::GetListDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

Operator &Operator::SetAttr(const string &name, const ge::DataType &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return *this;
  }
  if (!AttrUtils::SetDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
  return *this;
}

Operator &Operator::SetAttr(const char *name, const ge::DataType &attr_value) {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return *this;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return *this;
  }
  std::string op_name = name;
  if (!AttrUtils::SetDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", op_name.c_str());
  }
  return *this;
}

graphStatus Operator::GetAttr(const string &name, ge::DataType &attr_value) const {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return GRAPH_FAILED;
  }
  if (!AttrUtils::GetDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

graphStatus Operator::GetAttr(const char *name, ge::DataType &attr_value) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return GRAPH_FAILED;
  }
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr, name %s.", name);
    return GRAPH_FAILED;
  }
  std::string op_name = name;
  if (!AttrUtils::GetDataType(operator_impl_->GetOpDescImpl(), op_name, attr_value)) {
    GELOGW("[Get][Attr] Get attr name %s failed", op_name.c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

void Operator::AttrRegister(const string &name, const std::vector<ge::DataType> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetListDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const string &name, const ge::DataType &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetDataType(operator_impl_->GetOpDescImpl(), name, attr_value)) {
    GELOGW("[Set][Attr] Set attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const string &name, const Tensor &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  auto tensor = TensorAdapter::AsGeTensor(attr_value);
  if (!AttrUtils::SetTensor(operator_impl_->GetOpDescImpl(), name, tensor)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const string &name, const vector<Tensor> &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  vector<GeTensor> val_list;
  for (const auto &item : attr_value) {
    val_list.push_back(TensorAdapter::AsGeTensor(item));
  }
  if (!AttrUtils::SetListTensor(operator_impl_->GetOpDescImpl(), name, val_list)) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::AttrRegister(const string &name, const OpBytes &attr_value) {
  if (operator_impl_ == nullptr || operator_impl_->GetOpDescImpl() == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr or opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  if (!AttrUtils::SetZeroCopyBytes(operator_impl_->GetOpDescImpl(), name,
                                   Buffer::CopyFrom(attr_value.data(), attr_value.size()))) {
    GELOGW("[Register][Attr] Reg attr name %s failed", name.c_str());
  }
}

void Operator::SubgraphRegister(const std::string &name, bool dynamic) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  operator_impl_->SubgraphRegister(name, dynamic ? kDynamic : kStatic);
}

void Operator::SubgraphCountRegister(const std::string &name, uint32_t count) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", name.c_str());
    return;
  }
  operator_impl_->SubgraphCountRegister(name, count);
}

void Operator::SetSubgraphBuilder(const std::string &ir_name, uint32_t index, const SubgraphBuilder &builder) {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr, name %s.", ir_name.c_str());
    return;
  }
  operator_impl_->SetSubgraphBuilder(ir_name, index, builder);
}

std::vector<std::string> Operator::GetSubgraphNames() const {
  return operator_impl_->GetSubgraphNames();
}

graphStatus Operator::GetSubgraphNames(std::vector<AscendString> &names) const {
  std::vector<std::string> subgraph_names = operator_impl_->GetSubgraphNames();
  for (auto &subgraph_name : subgraph_names) {
    names.emplace_back(subgraph_name.c_str());
  }
  return GRAPH_SUCCESS;
}

SubgraphBuilder Operator::GetDynamicSubgraphBuilder(const string &ir_name, uint32_t index) const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] operator impl is nullptr.");
    return nullptr;
  }
  return operator_impl_->GetSubgraphBuilder(ir_name, index);
}

SubgraphBuilder Operator::GetDynamicSubgraphBuilder(const char *ir_name, uint32_t index) const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator impl is nullptr.");
    return nullptr;
  }
  if (ir_name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param ir_name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return nullptr;
  }
  std::string op_ir_name = ir_name;
  return operator_impl_->GetSubgraphBuilder(op_ir_name, index);
}

SubgraphBuilder Operator::GetSubgraphBuilder(const string &ir_name) const {
  return GetDynamicSubgraphBuilder(ir_name, 0);
}

SubgraphBuilder Operator::GetSubgraphBuilder(const char *ir_name) const {
  std::string graph_ir_name;
  if (ir_name != nullptr) {
    graph_ir_name = ir_name;
  }
  return GetDynamicSubgraphBuilder(graph_ir_name, 0);
}

Graph Operator::GetSubgraphImpl(const string &name) const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GE_LOGE("[Check][Param] Failed to get subgraph %s, the operator impl is null", name.c_str());
    return Graph("");
  }
  auto op_desc = OpDescUtils::GetOpDescFromOperator(*this);
  if (op_desc == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s, because the op_desc is nullptr.", name.c_str());
    GE_LOGE("[Get][OpDesc] Failed to get subgraph %s, the op_desc is null", name.c_str());
    return Graph("");
  }
  const auto &subgraph_names_to_index = op_desc->GetSubgraphNameIndexes();
  auto iter = subgraph_names_to_index.find(name);
  if (iter == subgraph_names_to_index.end()) {
    REPORT_INNER_ERROR("E19999", "Failed to get subgraph %s, the name may be invalid", name.c_str());
    GE_LOGE("[Check][Param] Failed to get subgraph %s, the name may be invalid", name.c_str());
    return Graph("");
  }
  auto subgraph_instance_name = op_desc->GetSubgraphInstanceName(iter->second);
  if (subgraph_instance_name.empty()) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s index %u, the subgraph may not be added",
                      name.c_str(), iter->second);
    GE_LOGE("[Get][Subgraph] %s index %u failed, because the subgraph may not be added",
            name.c_str(), iter->second);
    return Graph("");
  }

  auto node = operator_impl_->GetNode();
  if (node == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s, because the node is null", name.c_str());
    GE_LOGE("[Get][Node] Failed to get subgraph %s, because the node is null", name.c_str());
    return Graph("");
  }
  auto root_graph = GraphUtils::FindRootGraph(node->GetOwnerComputeGraph());
  if (root_graph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s, because can not find the root graph,node:%s",
                      name.c_str(), node->GetName().c_str());
    GE_LOGE("[Get][Subgraph] subgraph %s failed, because can not find the root graph", name.c_str());
    return Graph("");
  }
  auto subgraph = root_graph->GetSubgraph(subgraph_instance_name);
  if (subgraph == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to get subgraph %s index %u, because can not find the instance %s "
                      "from the root graph", name.c_str(), iter->second, subgraph_instance_name.c_str());
    GE_LOGE("[Get][Subgraph] %s index %u failed, because can not find the instance %s from the root graph",
            name.c_str(), iter->second, subgraph_instance_name.c_str());
    return Graph("");
  }
  return GraphUtils::CreateGraphFromComputeGraph(subgraph);
}

Graph Operator::GetSubgraph(const string &name) const {
  return GetSubgraphImpl(name);
}

Graph Operator::GetSubgraph(const char *name) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Get subgraph failed, name is nullptr.");
    return Graph("");
  }
  std::string op_name = name;
  return GetSubgraphImpl(op_name);
}

Graph Operator::GetDynamicSubgraph(const string &name, uint32_t index) const {
  return GetSubgraph(name + std::to_string(index));
}

Graph Operator::GetDynamicSubgraph(const char *name, uint32_t index) const {
  if (name == nullptr) {
    REPORT_INNER_ERROR("E19999", "param name is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator name is nullptr.");
    return Graph("");
  }
  std::string op_name = name;
  return GetSubgraph(op_name + std::to_string(index));
}

size_t Operator::GetSubgraphNamesCount() const {
  if (operator_impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "operator_impl_ is nullptr, check invalid.");
    GE_LOGE("[Check][Param] Failed to get subgraph names count, the operator impl is null");
    return 0;
  }
  return operator_impl_->GetSubgraphNamesCount();
}

class GraphBuilderImpl {
 public:
  explicit GraphBuilderImpl(const string &name) : graph_(ComGraphMakeShared<ComputeGraph>(name)) {
    if (graph_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "ComputeGraph make shared failed");
      GELOGE(GRAPH_FAILED, "[Call][ComGraphMakeShared] ComputeGraph make shared failed");
      return;
    }
  }

  ~GraphBuilderImpl() {}

  ComputeGraphPtr BuildGraph(const std::vector<Operator> &inputs) {
    std::vector<OperatorImplPtr> vec_inputs;
    for (auto &it : inputs) {
      auto src_op_impl = it.operator_impl_;
      GE_CHK_BOOL_EXEC(src_op_impl != nullptr, REPORT_INNER_ERROR("E19999", "src_op_impl is nullptr, check invalid.");
                       return nullptr, "[Check][Param] Operator Impl is null.");
      GE_CHK_BOOL_EXEC(src_op_impl->op_desc_ != nullptr,
                       REPORT_INNER_ERROR("E19999", "impl's opdesc is nullptr, check invalid.");
                       return nullptr, "[Check][Param] Operator impl's opdesc is null.");

      string type = src_op_impl->op_desc_->GetType();
      auto node_op = ge::OperatorFactory::CreateOperator("node_op", type);
      auto tensor_desc = ge::OpDescUtils::GetOpDescFromOperator(node_op);
      node_op.BreakConnect();

      GE_CHK_BOOL_EXEC(tensor_desc != nullptr, continue, "[Get][Opdesc] tensor_desc is null.");
      if ((tensor_desc->GetInputsSize() == 0 && tensor_desc->GetOutputsSize() > 0) || type == DATA ||
          type == VARIABLE || type == INITDATA || type == GETNEXT) {
        vec_inputs.push_back(it.operator_impl_);
      } else {
        GELOGW("[BuildGraph][CheckInput] Input operator should be Data, Variable operator or operator that has output "
               "but no input.");
      }
    }
    GE_CHK_BOOL_EXEC(!vec_inputs.empty(),
                     REPORT_INNER_ERROR("E19999", "User Input do not include operator such as "
                                        "Data, Variable operator or operator that has output but no input.");
                     return nullptr, "[Check][Param] User Input do not include operator such as "
                     "Data, Variable operator or operator that has output but no input.");
    auto ret = WalkAllOperators(vec_inputs);
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS, return nullptr, "[Call][WalkAllOperators] failed, ret:%d.", ret);

    ret = AddEdge();
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS, return nullptr, "[Add][Edge] failed, ret:%d.", ret);

    return graph_;
  }

  const std::map<OperatorImplPtr, NodePtr> &GetAllNodesInfo() const { return all_nodes_info_; }

 private:
  graphStatus WalkAllOperators(const std::vector<OperatorImplPtr> &vec_ops) {
    GE_CHK_BOOL_EXEC(graph_ != nullptr,
                     REPORT_INNER_ERROR("E19999", "graph_ is nullptr, check invalid.");
                     return GRAPH_FAILED, "[Check][Param] graph_ is null.");
    std::queue<std::vector<OperatorImplPtr>> que;
    que.push(vec_ops);
    while (!que.empty()) {
      auto vec_tem = que.front();
      que.pop();
      for (const auto &op_impl : vec_tem) {
        GE_CHK_BOOL_EXEC(op_impl != nullptr,
                         REPORT_INNER_ERROR("E19999", "op_impl is nullptr, check invalid.");
                         return GRAPH_FAILED, "[Check][Param] Operator Impl is null.");
        GE_CHK_BOOL_EXEC_INFO(all_nodes_info_.find(op_impl) == all_nodes_info_.end(), continue,
                              "This node %s has created.", op_impl->GetName().c_str())
        auto node_ptr = graph_->AddNode(op_impl->op_desc_);
        GE_CHK_BOOL_EXEC(node_ptr != nullptr,
                         REPORT_CALL_ERROR("E19999", "add node failed.");
                         return GRAPH_FAILED, "[Add][Node] failed.");
        all_nodes_info_.insert(std::make_pair(op_impl, node_ptr));

        auto &out_links = op_impl->output_links_;
        std::vector<OperatorImplPtr> vec_op_forward{};
        for (const auto &out_link : out_links) {
          for (const auto &op_forward : out_link.second) {
            vec_op_forward.push_back(op_forward.GetOwner());
          }
        }

        auto &out_control_links = op_impl->control_output_link_;
        for (const auto &out_link : out_control_links) {
          vec_op_forward.push_back(out_link.lock());
        }
        que.push(vec_op_forward);

        auto &in_links = op_impl->input_link_;
        std::vector<OperatorImplPtr> vec_op_back_forward{};
        for (const auto &in_link : in_links) {
          vec_op_back_forward.push_back(in_link.second.GetOwner());
        }

        auto &in_control_links = op_impl->control_input_link_;
        for (const auto &in_link : in_control_links) {
          vec_op_back_forward.push_back(in_link.lock());
        }
        que.push(vec_op_back_forward);

        if (WalkAllSubgraphs(node_ptr, op_impl) != GRAPH_SUCCESS) {
          return GRAPH_FAILED;
        }
      }
    }
    return MoveSubgraphToRoot(graph_);
  }

  graphStatus WalkAllSubgraphs(const NodePtr &node, const OperatorImplPtr &op_impl) {
    const string name = node->GetName();
    for (auto &name_idx : op_impl->op_desc_->GetSubgraphNameIndexes()) {
      const SubgraphBuilder &builder = op_impl->GetSubgraphBuilder(name_idx.first);
      if (builder == nullptr) {
        GELOGW("[Check][Param] Node %s has no builder", name.c_str());
        continue;
      }

      Graph graph = builder();  // Build subgraph from user define builder.
      const ComputeGraphPtr &subgraph = GraphUtils::GetComputeGraph(graph);
      GE_CHK_BOOL_EXEC(subgraph != nullptr,
                       REPORT_CALL_ERROR("E19999", "Node: %s, Build graph failed.", name.c_str());
                       return GRAPH_FAILED, "[Get][Graph] Node: %s, Build graph failed.", name.c_str());

      subgraph->SetParentNode(node);
      subgraph->SetParentGraph(graph_);
      if (graph_->AddSubgraph(subgraph->GetName(), subgraph) != GRAPH_SUCCESS) {
        return GRAPH_FAILED;
      }

      if (op_impl->op_desc_->SetSubgraphInstanceName(name_idx.second, subgraph->GetName()) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "Failed to set subgraph %s index %u", subgraph->GetName().c_str(), name_idx.second);
        GELOGE(GRAPH_FAILED, "[Set][SubGraph] %s index %u failed", subgraph->GetName().c_str(), name_idx.second);
        return GRAPH_FAILED;
      }
    }

    return GRAPH_SUCCESS;
  }

  graphStatus MoveSubgraphToRoot(const ComputeGraphPtr &graph) {
    const ComputeGraphPtr &root_graph = GraphUtils::FindRootGraph(graph);
    if (root_graph == nullptr) {
      REPORT_CALL_ERROR("E19999", "failed to find root graph of %s", graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Find][RootGraph] failed for graph:%s.", graph->GetName().c_str());
      return GRAPH_FAILED;
    }

    if (root_graph == graph) {
      auto subgraphs = graph->GetAllSubgraphs();
      for (auto &subgraph : subgraphs) {
        if (MoveSubgraphToRoot(subgraph) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
      }
    } else {
      auto subgraphs = graph->GetAllSubgraphs();
      for (auto &subgraph : subgraphs) {
        if (root_graph->AddSubgraph(subgraph->GetName(), subgraph) != GRAPH_SUCCESS) {
          return GRAPH_FAILED;
        }
        graph->RemoveSubgraph(subgraph->GetName());
        if (MoveSubgraphToRoot(subgraph) != GRAPH_SUCCESS) {
            return GRAPH_FAILED;
        }
      }
    }

    return GRAPH_SUCCESS;
  }

  graphStatus AddEdge() {
    for (const auto &node_info : all_nodes_info_) {
      auto src_op_impl_ptr = node_info.first;
      auto src_node_ptr = node_info.second;

      GE_IF_BOOL_EXEC(src_op_impl_ptr == nullptr || src_node_ptr == nullptr, continue);
      auto out_links = src_op_impl_ptr->output_links_;
      GE_CHK_BOOL_EXEC(src_op_impl_ptr->op_desc_ != nullptr,
                       REPORT_INNER_ERROR("E19999", "Src operator impl's op_desc is nullptr, check invalid.");
                       return GRAPH_FAILED, "[Check][Param] Src operator impl's op_desc is null.");
      auto &op_desc = src_op_impl_ptr->op_desc_;
      GE_IF_BOOL_EXEC(op_desc == nullptr, continue);
      for (const auto &out : out_links) {
        auto src_idx = op_desc->GetOutputIndexByName(out.first);
        GE_CHK_BOOL_EXEC(src_idx >= 0,
                         REPORT_INNER_ERROR("E19999", "Find output index by name:%s in op:%s failed",
                                            out.first.c_str(), op_desc->GetName().c_str());
                         return GRAPH_FAILED, "[Get][Index] Find output index by name:%s failed", out.first.c_str());

        auto src_anchor = src_node_ptr->GetOutDataAnchor(src_idx);
        GE_CHK_BOOL_EXEC(src_anchor != nullptr,
                         REPORT_INNER_ERROR("E19999", "GetOutDataAnchor failed, index:%d, op:%s.",
                                            src_idx, op_desc->GetName().c_str());
                         return GRAPH_FAILED, "[Get][OutDataAnchor] failed, index:%d.", src_idx);

        for (const auto &dst_opio : out.second) {
          auto dst_node_info = all_nodes_info_.find(dst_opio.GetOwner());
          GE_CHK_BOOL_EXEC(dst_node_info != all_nodes_info_.end(),
                           REPORT_INNER_ERROR("E19999", "Find Dst node failed, op:%s.", op_desc->GetName().c_str());
                           return GRAPH_FAILED, "[Check][Param] Find Dst node failed.");

          GE_IF_BOOL_EXEC(dst_node_info->second == nullptr, continue);

          auto dst_anchor = dst_node_info->second->GetInDataAnchor(dst_opio.GetIndex());
          GE_CHK_BOOL_EXEC(dst_anchor != nullptr,
                           REPORT_INNER_ERROR("E19999", "GetInDataAnchor failed, index:%d, op:%s",
                                              dst_opio.GetIndex(), op_desc->GetName().c_str());
                           return GRAPH_FAILED, "GetInDataAnchor failed, index:%d", dst_opio.GetIndex());

          auto ret = GraphUtils::AddEdge(src_anchor, dst_anchor);
          GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                           REPORT_CALL_ERROR("E19999", "add edge from node[%s][%d] to node[%s][%d] failed.",
                                             src_node_ptr->GetName().c_str(), src_anchor->GetIdx(),
                                             dst_node_info->second->GetName().c_str(), dst_anchor->GetIdx());
                           return GRAPH_FAILED, "[Add][Edge] from node[%s][%d] to node[%s][%d] failed.",
                           src_node_ptr->GetName().c_str(), src_anchor->GetIdx(),
                           dst_node_info->second->GetName().c_str(), dst_anchor->GetIdx());
        }
      }
      auto out_control_anchor = src_node_ptr->GetOutControlAnchor();
      for (const auto &control_out : src_op_impl_ptr->control_output_link_) {
        auto dst_node_info = all_nodes_info_.find(control_out.lock());
        if (dst_node_info == all_nodes_info_.end()) {
          REPORT_INNER_ERROR("E19999", "Find Dst node failed.");
          GELOGE(GRAPH_FAILED, "[Check][Param] Find Dst node failed.");
          return GRAPH_FAILED;
        }
        GE_IF_BOOL_EXEC(dst_node_info->second == nullptr, continue);
        auto in_control_anchor = dst_node_info->second->GetInControlAnchor();
        auto ret = GraphUtils::AddEdge(out_control_anchor, in_control_anchor);
        if (ret != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "add edge failed. srcNode %s:%s, dstNode %s:%s", op_desc->GetName().c_str(),
                            op_desc->GetType().c_str(), dst_node_info->second->GetName().c_str(),
                            dst_node_info->second->GetType().c_str());
          GELOGE(ret, "[Add][Edge] failed. srcNode %s:%s, dstNode %s:%s", op_desc->GetName().c_str(),
                 op_desc->GetType().c_str(), dst_node_info->second->GetName().c_str(),
                 dst_node_info->second->GetType().c_str());
          return ret;
        }
      }
    }
    return GRAPH_SUCCESS;
  }

  ComputeGraphPtr graph_ = nullptr;
  std::map<OperatorImplPtr, NodePtr> all_nodes_info_{};
};

inline bool HasSameNameNode(const ComputeGraphPtr &compute_graph) {
  for (const auto &graph : compute_graph->GetAllSubgraphs()) {
    std::set<string> node_names;
    for (auto const &node : graph->GetDirectNode()) {
      auto result = node_names.insert(node->GetName());
      if (!result.second) {
        REPORT_INNER_ERROR("E19999", "[Check][Param] graph %s has same name node%s",
                           graph->GetName().c_str(), node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] graph %s has same name node%s",
               graph->GetName().c_str(), node->GetName().c_str());
        return true;
      }
    }
  }

  std::set<string> node_names;
  for (auto const &node : compute_graph->GetDirectNode()) {
    auto result = node_names.insert(node->GetName());
    if (!result.second) {
      REPORT_INNER_ERROR("E19999", "[Check][Param] graph %s has same name node%s",
                         compute_graph->GetName().c_str(), node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] graph %s has same name node%s",
             compute_graph->GetName().c_str(), node->GetName().c_str());
      return true;
    }
  }
  return false;
}

ComputeGraphPtr GraphUtils::CreateGraphFromOperator(const string &name, const vector<ge::Operator> &inputs) {
  auto graph_builder_impl = GraphBuilderImpl(name);
  ComputeGraphPtr compute_graph = graph_builder_impl.BuildGraph(inputs);
  GE_CHK_BOOL_EXEC(compute_graph != nullptr,
                   REPORT_INNER_ERROR("E19999", "BuildGraph failed, as return nullptr.");
                   return compute_graph, "[Build][Graph] Computer graph is nullptr");
  compute_graph->SetAllNodesInfo(graph_builder_impl.GetAllNodesInfo());
  if (HasSameNameNode(compute_graph)) {
    GELOGW("[CreateGraph][Check] Nodes with same name exist in one compute graph is not allowed, graph_name: %s",
           name.c_str());
    compute_graph = nullptr;
  }

  return compute_graph;
}

void GraphUtils::BreakConnect(const std::map<OperatorImplPtr, NodePtr> &all_nodes_infos) {
  for (const auto &it : all_nodes_infos) {
    OperatorImplPtr op_impl = it.first;
    if (op_impl == nullptr) {
      GELOGW("[BreakConnect][Check] Operator impl is null");
      continue;
    }
    op_impl->ClearOutputLinks();
    op_impl->ClearInputLinks();
    OperatorKeeper::GetInstance().CheckOutOperator(op_impl);
  }
}
/*lint +e446 +e732*/
/*lint +e665*/
} // namespace ge
