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

#ifndef GRAPH_OP_DESC_IMPL_H_
#define GRAPH_OP_DESC_IMPL_H_

#include <string>
#include <vector>
#include "graph/op_desc.h"

namespace ge {
class OpDescImpl {
 public:
  OpDescImpl();
  OpDescImpl(const std::string &name, const std::string &type);
  OpDescImpl(const ProtoMsgOwner &proto_msg_owner, ge::proto::OpDef *op_def);

  ~OpDescImpl() = default;

  string GetName() const;
  void SetName(const std::string &name);
  string GetType() const;
  void SetType(const string &type);

  graphStatus AddInputDesc(const ge::GeTensorDesc &input_desc);
  graphStatus AddInputDesc(uint32_t index, const ge::GeTensorDesc &input_desc);
  graphStatus AddInputDesc(const string &name, const ge::GeTensorDesc &input_desc);
  graphStatus AddInputDescMiddle(const string &name, const unsigned int num, size_t index);
  graphStatus AddOutputDescMiddle(const string &name, const unsigned int num, size_t index);
  graphStatus AddInputDescForward(const string &name, const unsigned int num);
  graphStatus AddOutputDescForward(const string &name, const unsigned int num);
  graphStatus AddOptionalInputDesc(const string &name, const ge::GeTensorDesc &input_desc);

  graphStatus UpdateInputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc);
  graphStatus UpdateInputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc);

  bool OpDescMembersAreEqual(const OpDescImpl &r_op_desc) const;
  bool OpDescAttrsAreEqual(const OpDescImpl &r_op_desc) const;
  bool OpDescGenTensorDescsAreEqual(const OpDescImpl &r_op_desc) const;

  bool operator==(const OpDescImpl &r_op_desc) const;

  bool InputIsSet(const string &name) const;

  GeTensorDesc GetInputDesc(uint32_t index) const;
  GeTensorDesc GetInputDesc(const string &name) const;
  GeTensorDescPtr MutableInputDesc(uint32_t index) const;
  GeTensorDescPtr MutableInputDesc(const string &name) const;
  OpDesc::Vistor<string> GetAllInputNames(const ConstOpDescPtr &op_desc) const;

  void SetOpKernelLibName(const std::string &name);
  std::string GetOpKernelLibName() const;
  void SetOpEngineName(const std::string &name);
  std::string GetOpEngineName() const;

  OpDesc::Vistor<GeTensorDesc> GetAllInputsDesc(const ConstOpDescPtr &op_desc) const;
  OpDesc::Vistor<GeTensorDescPtr> GetAllInputsDescPtr(const ConstOpDescPtr &op_desc) const;

  size_t GetInputsSize() const;
  size_t GetAllInputsSize() const;

  graphStatus AddOutputDesc(const ge::GeTensorDesc &output_desc);
  graphStatus AddOutputDesc(const string &name, const ge::GeTensorDesc &output_desc);
  graphStatus UpdateOutputDesc(uint32_t index, const ge::GeTensorDesc &tensor_Desc);
  graphStatus UpdateOutputDesc(const string &name, const ge::GeTensorDesc &tensor_Desc);
  GeTensorDesc GetOutputDesc(uint32_t index) const;
  GeTensorDesc GetOutputDesc(const string &name) const;
  GeTensorDescPtr MutableOutputDesc(uint32_t index) const;
  GeTensorDescPtr MutableOutputDesc(const string &name) const;

  uint32_t GetAllOutputsDescSize() const;
  OpDesc::Vistor<GeTensorDesc> GetAllOutputsDesc(const ConstOpDescPtr &op_desc) const;
  OpDesc::Vistor<GeTensorDescPtr> GetAllOutputsDescPtr(const ConstOpDescPtr &op_desc) const;
  ConstGeTensorDescPtr GetOutputDescPtr(uint32_t index) const;
  size_t GetOutputsSize() const;

  ConstGeTensorDescPtr GetInputDescPtr(uint32_t index) const;
  ConstGeTensorDescPtr GetInputDescPtrDfault(uint32_t index) const;
  ConstGeTensorDescPtr GetInputDescPtr(const string &name) const;

  graphStatus AddRegisterInputName(const std::string &name);
  vector<string> GetRegisterInputName() const;

  graphStatus AddDynamicInputDesc(const string &name, const unsigned int num, bool is_push_back);
  graphStatus AddDynamicInputDescByIndex(const string &name, const unsigned int num, size_t index);

  graphStatus AddRegisterOutputName(const string &name);
  vector<string> GetRegisterOutputName() const;

  graphStatus AddDynamicOutputDesc(const string &name, const unsigned int num, bool is_push_back);
  bool IsOptionalInput(const string &name) const;
  bool IsOptionalInput(uint32_t index) const;
  std::map<string, uint32_t> GetAllInputName() const;
  std::map<string, uint32_t> GetAllOutputName();
  std::map<string, uint32_t>& MutableAllInputName();
  std::map<string, uint32_t>& MutableAllOutputName();
  bool UpdateInputName(std::map<string, uint32_t> input_name_idx);
  bool UpdateOutputName(std::map<string, uint32_t> output_name_idx);

  std::function<graphStatus(Operator &)> GetInferFunc() const;
  std::function<graphStatus(Operator &)> GetVerifyFunc() const;
  void AddInferFunc(const std::function<graphStatus(Operator &)> &func);
  std::function<graphStatus(Operator &)> GetInferFormatFunc() const;
  void AddInferFormatFunc(const std::function<graphStatus(Operator &)> &func);
  void AddVerifierFunc(const std::function<graphStatus(Operator &)> &func);

  graphStatus InferShapeAndType(const OpDescPtr &op_desc);
  graphStatus DefaultInferFormat(const ConstOpDescPtr &op_desc);
  graphStatus OpVerify(const OpDescPtr &op_desc);

  string GetInputNameByIndex(uint32_t index) const;
  int GetInputIndexByName(const string &name) const;
  int GetValidInputIndexByName(const string &name) const;
  string GetValidInputNameByIndex(uint32_t index) const;

  string GetOutputNameByIndex(uint32_t index) const;
  int GetOutputIndexByName(const string &name) const;

  ProtoAttrMapHelper MutableAttrMap();
  ConstProtoAttrMapHelper GetAttrMap() const;

  void SetId(int64_t id);
  int64_t GetId() const;

  void SetStreamId(int64_t stream_id);
  int64_t GetStreamId() const;

  void SetInputName(const vector<string> &input_name);
  vector<string> GetInputName() const;

  void SetSrcName(const vector<string> &src_name);
  vector<string> GetSrcName() const;

  void SetSrcIndex(const vector<int64_t> &src_index);
  vector<int64_t> GetSrcIndex() const;

  void SetInputOffset(const vector<int64_t> &input);
  vector<int64_t> GetInputOffset() const;

  void SetOutputOffset(const vector<int64_t> &output);
  vector<int64_t> GetOutputOffset() const;

  void SetDstName(const vector<string> &dst_name);
  vector<string> GetDstName() const;

  void SetDstIndex(const vector<int64_t> &dst_index);
  vector<int64_t> GetDstIndex() const;

  void SetWorkspace(const vector<int64_t> &workspace);
  vector<int64_t> GetWorkspace() const;

  void SetWorkspaceBytes(const vector<int64_t> &workspace_bytes);
  vector<int64_t> GetWorkspaceBytes() const;

  void SetIsInputConst(const vector<bool> &is_input_const);
  vector<bool> GetIsInputConst() const;

  graphStatus RestoreInputNameIdx(const string &name, const int &index);
  graphStatus RestoreOutputNameIdx(const string &name, const int &index);

  graphStatus CallInferFunc(Operator &op, const OpDescPtr &op_desc);
  graphStatus CallInferFormatFunc(Operator &op, const ConstOpDescPtr &op_desc);
  graphStatus CallInferValueRangeFunc(Operator &op, const ConstOpDescPtr &op_desc);

  std::string GetSubgraphInstanceName(uint32_t index) const;
  const std::vector<std::string> &GetSubgraphInstanceNames() const;
  void RemoveSubgraphInstanceName(const std::string &name);
  graphStatus AddSubgraphName(const std::string &name);
  const std::map<std::string, uint32_t> &GetSubgraphNameIndexes() const;
  graphStatus SetSubgraphInstanceName(uint32_t index, const std::string &name);

  void RegisterSubgraphIrName(const string &name, SubgraphType type);
  const std::map<std::string, SubgraphType> &GetSubgraphIrNames() const;
  SubgraphType GetSubgraphTypeByIrName(const std::string &name) const;
  graphStatus GetSubgraphNameByInstanceName(const std::string &instance_name, std::string &subgraph_name) const;
  graphStatus InferDataSlice(const OpDescPtr &op_desc);

 private:
  friend class AttrUtils;
  friend class OpDescUtils;
  friend class ModelSerializeImp;
  friend class OnnxUtils;
  friend class GraphUtils;
  GeIrProtoHelper<ge::proto::OpDef> op_def_;
  std::vector<std::string> subgraph_instance_names_;

  // subgraph names to index, for a `if` operator:
  // then_branch: 0
  // else_branch: 1
  // or for a `case` node:
  // branches0: 0
  // branches1: 1
  // branches2: 2
  std::map<std::string, uint32_t> subgraph_names_to_index_;

  // subgraph ir names to type, for a `if` operator:
  // then_branch: static
  // else_branch: static
  // or for a `case` op:
  // branches: dynamic
  std::map<std::string, SubgraphType> subgraph_ir_names_to_type_;

  vector<GeTensorDescPtr> inputs_desc_{};
  map<string, uint32_t> input_name_idx_{};
  vector<string> register_input_name_{};
  std::set<string> optional_input_names_{};
  vector<GeTensorDescPtr> outputs_desc_{};
  map<string, uint32_t> output_name_idx_{};
  vector<string> register_output_name_{};
  std::function<graphStatus(Operator &)> infer_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_format_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_value_range_func_ = nullptr;
  std::function<graphStatus(Operator &)> verifier_func_ = nullptr;
  std::function<graphStatus(Operator &)> infer_data_slice_func_ = nullptr;
  string op_kernel_lib_name_;
  string engine_name_;
};
}  // namespace ge
#endif  // GRAPH_OP_DESC_IMPL_H_
