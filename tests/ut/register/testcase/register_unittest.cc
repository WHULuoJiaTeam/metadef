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
#define private public
#define protected public
#include "external/register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#include "external/graph/operator.h"
#include "graph/utils/node_utils.h"
#include "graph/node_impl.h"
#include "graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/operator_factory_impl.h"
#include "graph/compute_graph_impl.h"
#include "external/register/register.h"
#undef private
#undef protected

#include <gtest/gtest.h>
#include <iostream>
#include <nlohmann/json.hpp>

#include "framework/common/debug/ge_log.h"
#include "register/op_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling_utils.h"
#include "op_tiling/op_tiling_constants.h"
#include "register/op_compile_info_base.h"
#include "op_tiling.h"

#include "graph/graph.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/attr_value.h"

#include "graph/debug/ge_util.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_attr_define.h"

#include "proto/tensorflow/attr_value.pb.h"
#include "proto/tensorflow/node_def.pb.h"


using namespace domi;
using namespace ge;
using namespace optiling;

class CompileInfoJson : public CompileInfoBase {
public:
  CompileInfoJson(const std::string &json) : json_str_(json) {}
  ~CompileInfoJson() {}
private:
  std::string json_str_;
};

class UtestRegister : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse);

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *compile_info_hash,
                                      const char *inputs, const char *outputs, const char *attrs, char *run_info_json,
                                      size_t run_info_len, uint64_t *elapse);

bool op_tiling_stub_v2(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

bool op_tiling_stub_v3(const Operator &op, const void* value, OpRunInfoV2 &run_info) {
  return true;
}

void* op_parse_stub_v3(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  static int x = 1024;
  void *p = &x;
  return p;
}

bool op_tiling_stub_v4(const Operator &op, const CompileInfoPtr value, OpRunInfoV2 &run_info) {
  return true;
}

CompileInfoPtr op_parse_stub_v4(const Operator &op, const ge::AscendString &compile_info_json) {
//  static void *p = new int(3);
  CompileInfoPtr info = std::make_shared<CompileInfoJson>("qwer");
  return info;
}

REGISTER_OP_TILING_V2(ReluV2, op_tiling_stub_v2);
REGISTER_OP_TILING_V3(ReluV3, op_tiling_stub_v3, op_parse_stub_v3);
REGISTER_OP_TILING_V4(ReluV4, op_tiling_stub_v4, op_parse_stub_v4);

TEST_F(UtestRegister, test_register_dynamic_outputs_op_only_has_partial_output) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node_src = builder.AddNode("ParseSingleExample", "ParseSingleExample",
                    {"serialized", "dense_defaults_0", "dense_defaults_1", "dense_defaults_2"},
                    {"dense_values_0", "dense_values_1", "dense_values_2"});
  // build op_src attrs
  vector<string> dense_keys = {"image/class/lable","image/encode", "image/format"};
  vector<DataType> t_dense = {DT_INT64, DT_STRING, DT_STRING};
  AttrUtils::SetListStr(node_src->GetOpDesc(), "dense_keys", dense_keys);
  AttrUtils::SetListStr(node_src->GetOpDesc(), "dense_shapes", {});
  AttrUtils::SetInt(node_src->GetOpDesc(), "num_sparse", 0);
  AttrUtils::SetListStr(node_src->GetOpDesc(), "sparse_keys", {});
  AttrUtils::SetListStr(node_src->GetOpDesc(), "sparse_types", {});
  AttrUtils::SetListDataType(node_src->GetOpDesc(), "Tdense", t_dense);
  auto graph = builder.GetGraph();

  // get op_src
  ge::Operator op_src = OpDescUtils::CreateOperatorFromNode(node_src);
  ge::Operator op_dst = ge::Operator("ParseSingleExample");
  std::shared_ptr<ge::OpDesc> op_desc_dst = ge::OpDescUtils::GetOpDescFromOperator(op_dst);
  op_desc_dst->AddRegisterInputName("dense_defaults");
  op_desc_dst->AddRegisterOutputName("sparse_indices");
  op_desc_dst->AddRegisterOutputName("sparse_values");
  op_desc_dst->AddRegisterOutputName("sparse_shapes");
  op_desc_dst->AddRegisterOutputName("dense_values");

  // simulate parse_single_example plugin
  std::vector<DynamicInputOutputInfo> value;
  DynamicInputOutputInfo input(kInput, "dense_defaults", 14, "Tdense", 6);
  value.push_back(input);
  DynamicInputOutputInfo output(kOutput, "sparse_indices", 14, "num_sparse", 10);
  value.push_back(output);
  DynamicInputOutputInfo output1(kOutput, "sparse_values", 13, "sparse_types", 12);
  value.push_back(output1);
  DynamicInputOutputInfo output2(kOutput, "sparse_shapes", 13, "num_sparse", 10);
  value.push_back(output2);
  DynamicInputOutputInfo output3(kOutput, "dense_values", 12, "Tdense", 6);
  value.push_back(output3);
  DynamicInputOutputInfo invalidput(kInvalid, "Invalid", 7, "Invalid", 7);
  value.push_back(invalidput);

  // pre_check
  EXPECT_EQ(op_dst.GetOutputsSize(), 0);
  auto ret = AutoMappingByOpFnDynamic(op_src, op_dst, value);

  // check add 3 output to op_dst
  EXPECT_EQ(ret, domi::SUCCESS);
  EXPECT_EQ(op_dst.GetOutputsSize(), 3);

  // for AutoMappingByOpFnDynamic failed test
  ge::Operator op_src_fail(nullptr);
  ret = AutoMappingByOpFnDynamic(op_src_fail, op_dst, value);
  EXPECT_EQ(ret, domi::FAILED);

  std::vector<DynamicInputOutputInfo> value_fail;
  ret = AutoMappingByOpFnDynamic(op_src, op_dst, value_fail);
  DynamicInputOutputInfo input_fail(kInput, "", 0, "", 0);
  value_fail.push_back(input_fail);
  ret = AutoMappingByOpFnDynamic(op_src, op_dst, value_fail);
  EXPECT_EQ(ret, domi::FAILED);
}

void GraphInit(domi::tensorflow::GraphDef &graph_def) {
  // add node, set info
  domi::tensorflow::NodeDef *placeholder0 = graph_def.add_node();
  placeholder0->set_name("placeholder0");
  placeholder0->set_op("PlaceHolder");

  // add node, set info, add edges
  domi::tensorflow::NodeDef *add0 = graph_def.add_node();
  add0->set_name("add0");
  add0->set_op("Add");
  add0->add_input("placeholder0");
  add0->add_input("placeholder1");

  // 1. add node
  auto placeholder1 = graph_def.add_node();
  auto add1 = graph_def.add_node();
  auto mul0 = graph_def.add_node();
  auto mul1 = graph_def.add_node();
  auto add2 = graph_def.add_node();
  auto retval0 = graph_def.add_node();
  auto retval1 = graph_def.add_node();

  // 2. set info
  placeholder1->set_name("placeholder1");
  placeholder1->set_op("PlaceHolder");
  add1->set_name("add1");
  add1->set_op("Add");
  add2->set_name("add2");
  add2->set_op("Add");
  mul0->set_name("mul0");
  mul0->set_op("Mul");
  mul1->set_name("mul1");
  mul1->set_op("Mul");
  retval0->set_name("retval0");
  retval0->set_op("_RetVal");
  retval1->set_name("retval1");
  retval1->set_op("_RetVal");

  // 3. add edges
  mul0->add_input("placeholder0");
  mul0->add_input("placeholder1");
  mul1->add_input("placeholder0");
  mul1->add_input("add0");
  mul1->add_input("^mul0");
  add1->add_input("mul0");
  add1->add_input("placeholder1");
  add2->add_input("mul1");
  add2->add_input("mul0");
  retval0->add_input("add2:0");
  retval1->add_input("add1:0");
}


int32_t AutoMappingSubgraphIndexInput(int32_t data_index) {
  return 0;
}
int32_t AutoMappingSubgraphIndexOutput(int32_t netoutput_index) {
  return 0;
}
Status AutoMappingSubgraphIndexInput2(int32_t data_index, int32_t &parent_input_index) {
  return domi::SUCCESS;
}
Status AutoMappingSubgraphIndexOutput2(int32_t netoutput_index, int32_t &parent_output_index) {
  parent_output_index++;
  return domi::SUCCESS;
}
Status AutoMappingSubgraphIndexOutput2Failed(int32_t netoutput_index, int32_t &parent_output_index) {
  return domi::FAILED;
}

TEST_F(UtestRegister, AutoMappingSubgraphIndex) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto input = builder.AddNode("data", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "Data";
  auto func_node = builder.AddNode("func_node", FRAMEWORKOP, 1, 1);
  func_node->impl_->op_->impl_->meta_data_.type_ = "FrameworkOp";
  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput, AutoMappingSubgraphIndexOutput);
  EXPECT_EQ(stat, domi::FAILED);
}

TEST_F(UtestRegister, AutoMappingSubgraphIndexByDataNode) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto func_node = builder.AddNode("func_node", PARTITIONEDCALL, 1, 1);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2);
  EXPECT_EQ(stat, domi::SUCCESS);

  auto input = builder.AddNode("Retval", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "_Retval";
  AttrUtils::SetInt(input->GetOpDesc(), "retval_index", 0);
  builder.AddDataEdge(input, 0, func_node, 0);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingSubgraphIndexByDataNode2) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto input = builder.AddNode("index", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "Data";
  AttrUtils::SetInt(input->GetOpDesc(), "index", 0);
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto func_node = builder.AddNode("func_node", PARTITIONEDCALL, 1, 1);
  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);
  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2);
  EXPECT_EQ(stat, domi::SUCCESS);
}


TEST_F(UtestRegister, AutoMappingSubgraphOutputFail) {
  Status stat;
  auto builder = ut::GraphBuilder("root");
  auto output = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto input = builder.AddNode("data", DATA, 1, 1);
  input->impl_->op_->impl_->meta_data_.type_ = "Data";
  auto func_node = builder.AddNode("func_node", FRAMEWORKOP, 1, 1);
  func_node->impl_->op_->impl_->meta_data_.type_ = "FrameworkOp";
  builder.AddDataEdge(input, 0, func_node, 0);
  builder.AddDataEdge(func_node, 0, output, 0);

  auto computeGraph = builder.GetGraph();
  Graph graph = GraphUtils::CreateGraphFromComputeGraph(computeGraph);

  stat = AutoMappingSubgraphIndex(graph, AutoMappingSubgraphIndexInput2, AutoMappingSubgraphIndexOutput2Failed);
  EXPECT_EQ(stat, domi::FAILED);
}

TEST_F(UtestRegister, AutoMappingFnDynamicInputTest) {
  Status stat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);
  map<std::string, std::pair<std::string, std::string>> name_attr_value;
  name_attr_value.insert(make_pair(std::string("in"), make_pair(std::string("dynamicName1"), std::string("int"))));
  name_attr_value.insert(make_pair(std::string("out"), make_pair(std::string("dynamicName2"), std::string("float"))));

  ge::Operator op_dst = ge::Operator("Add", "int");
  const domi::tensorflow::NodeDef *node;

  int32_t node_size = graph_def.node_size();
  for(int i=0; i<node_size; i++) {
    node = graph_def.mutable_node(i);
    stat = AutoMappingFnDynamic(node, op_dst, name_attr_value, 1, 1);
    EXPECT_EQ(stat, domi::SUCCESS);
  }
}

TEST_F(UtestRegister, AutoMappingFnDynamicInput) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);

  // test for add attrs
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue inValue;
  inValue.set_s(std::string("stringValue"));
  inValue.set_i(66);
  node->mutable_attr()->insert({"inVal", inValue});
  name_attrs.insert(make_pair(std::string("in"), make_pair(std::string("inName1"), std::string("inVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingFnDynamicOutput) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);

  // test for add attrs
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue outValue;
  outValue.set_b(true);
  outValue.set_i(88);
  node->mutable_attr()->insert({"outVal", outValue});
  name_attrs.insert(make_pair(std::string("out"), make_pair(std::string("outName1"), std::string("outVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::SUCCESS);
}

TEST_F(UtestRegister, AutoMappingFunctionkFunc) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  op_dst.SubgraphRegister("subVal", true);
  op_dst.SubgraphCountRegister("subVal", 6);

  // test for add attrs
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue attrValue;
  attrValue.set_i(88);
  domi::tensorflow::NameAttrList *nameAttrList = new domi::tensorflow::NameAttrList();
  nameAttrList->set_name("nameAttrList");
  attrValue.unsafe_arena_set_allocated_func(nameAttrList);
  node->mutable_attr()->insert({"subVal", attrValue});
  name_attrs.insert(make_pair(std::string("out"), make_pair(std::string("outName1"), std::string("subVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::FAILED);
}

TEST_F(UtestRegister, AutoMappingFunctionkList) {
  Status retStat;
  domi::tensorflow::GraphDef graph_def;
  GraphInit(graph_def);

  ge::Operator op_dst = ge::Operator("Add", "int");
  op_dst.SubgraphRegister("subVal", true);

  // test for add attrs
  domi::tensorflow::NodeDef *node = graph_def.mutable_node(0);
  map<std::string, std::pair<std::string, std::string>> name_attrs;
  domi::tensorflow::AttrValue attrValue;
  attrValue.set_i(88);
  domi::tensorflow::AttrValue_ListValue *attrValListVal = new domi::tensorflow::AttrValue_ListValue();
  attrValListVal->add_s("list0");
  attrValListVal->add_s("list1");
  attrValue.unsafe_arena_set_allocated_list(attrValListVal);
  // list.func
  domi::tensorflow::NameAttrList *nameAttrList = new domi::tensorflow::NameAttrList();
  nameAttrList->set_name("nameAttrList");
  attrValListVal->add_func();

  node->mutable_attr()->insert({"subVal", attrValue});
  name_attrs.insert(make_pair(std::string("out"), make_pair(std::string("outName1"), std::string("subVal"))));
  retStat = AutoMappingFnDynamic(node, op_dst, name_attrs, 1, 1);
  EXPECT_EQ(retStat, domi::SUCCESS);
}


domi::Status inputFunc(int32_t data_index, int32_t &parent_input_index) {
  parent_input_index++;
  return (parent_input_index<0) ? domi::FAILED : domi::SUCCESS;
}

domi::Status outputFunc(int32_t netoutput_index, int32_t &parent_output_index) {
  parent_output_index++;
  return (parent_output_index<2) ? domi::FAILED : domi::SUCCESS;
}

domi::Status AutoMappingSubgraphIOIndexFuncCB(const ge::Graph &graph,
  const std::function<Status(int32_t data_index, int32_t &parent_input_index)> &input,
  const std::function<Status(int32_t netoutput_index, int32_t &parent_output_index)> &output) {
  static int test_idx = -2;

  switch(test_idx)
  {
    case -2:
      return input(0, test_idx);
    case -1:
      return input(0, test_idx);
    case 0:
      return output(0, test_idx);
    case 1:
      return output(0, test_idx);
  }
}

TEST_F(UtestRegister, FrameworkRegistryTest) {
  REGISTER_AUTOMAPPING_SUBGRAPH_IO_INDEX_FUNC(TENSORFLOW, AutoMappingSubgraphIOIndexFuncCB);

  FrameworkRegistry &cur = FrameworkRegistry::Instance();
  cur.AddAutoMappingSubgraphIOIndexFunc(domi::CAFFE, AutoMappingSubgraphIOIndexFuncCB);

  const ge::Graph graph("graph_test");
  AutoMappingSubgraphIOIndexFunc func = cur.GetAutoMappingSubgraphIOIndexFunc(domi::CAFFE);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::FAILED);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::SUCCESS);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::FAILED);
  EXPECT_EQ(func(graph, inputFunc, outputFunc), domi::SUCCESS);
}

TEST_F(UtestRegister, OpRegistrationDataWithNoImpl) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  opRegData.impl_.reset();

  EXPECT_EQ(opRegData.GetOmOptype()=="", true);
  EXPECT_EQ(opRegData.GetFrameworkType(), domi::FRAMEWORK_RESERVED);
  EXPECT_EQ(opRegData.GetOriginOpTypeSet().empty(), true);
  EXPECT_EQ(opRegData.GetParseParamFn(), nullptr);
  EXPECT_EQ(opRegData.GetParseParamByOperatorFn(), nullptr);
  EXPECT_EQ(opRegData.GetFusionParseParamFn(), nullptr);
  EXPECT_EQ(opRegData.GetFusionParseParamByOpFn(), nullptr);
  EXPECT_EQ(opRegData.GetImplyType(), domi::ImplyType::BUILDIN);
  EXPECT_EQ(opRegData.GetParseSubgraphPostFn(), nullptr);
  EXPECT_EQ(opRegData.GetParseOpToGraphFn(), nullptr);
  ParseSubgraphFuncV2 func;
  EXPECT_EQ(opRegData.GetParseSubgraphPostFn(func), domi::FAILED);
}

TEST_F(UtestRegister, OmOptypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  OpReceiver oprcver(opRegData);
  opRegData.GetOmOptype();

  AscendString OmOptype;
  Status stat = opRegData.GetOmOptype(OmOptype);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, FrameworkTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  opRegData.FrameworkType(domi::LUOJIANET);
  EXPECT_EQ(opRegData.GetFrameworkType(), domi::LUOJIANET);
}

TEST_F(UtestRegister, OriOpTypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  OpRegistrationData opRegData2("OmOptype2");

  std::initializer_list<std::string> OptypeList1{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList1);
  std::vector<AscendString> OptypeList2 = {AscendString("Div"), AscendString("Mul")};
  opRegData.OriginOpType(OptypeList2);

  opRegData2.OriginOpType(std::string("Add"));
  opRegData2.OriginOpType("Sub");

  opRegData.GetOriginOpTypeSet();
  std::set<ge::AscendString> opTypeSet;
  Status stat = opRegData.GetOriginOpTypeSet(opTypeSet);
  EXPECT_EQ(stat, domi::SUCCESS);
}

TEST_F(UtestRegister, OpRegistryImplyTypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);
  std::vector<AscendString> OptypeList2 = {AscendString("Div"), AscendString("Mul")};
  opRegData.OriginOpType(OptypeList2);

  // set ImplyType
  opRegData.ImplyType(domi::ImplyType::CUSTOM);
  EXPECT_EQ(opRegData.GetImplyType(), domi::ImplyType::CUSTOM);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  domi::ImplyType implType = opReg->GetImplyTypeByOriOpType(std::string("Add"));
  EXPECT_EQ(implType, domi::ImplyType::CUSTOM);

  implType = opReg->GetImplyType(std::string("OmOptype"));
  EXPECT_EQ(implType, domi::ImplyType::CUSTOM);
  implType = opReg->GetImplyType(std::string("strOmOptype"));
  EXPECT_EQ(implType, domi::ImplyType::BUILDIN);

  vector<std::string> vecOpType;
  vecOpType.clear();
  opReg->GetOpTypeByImplyType(vecOpType, domi::ImplyType::CUSTOM);
  EXPECT_EQ(vecOpType.empty(), false);
  vecOpType.clear();
  opReg->GetOpTypeByImplyType(vecOpType, domi::ImplyType::AI_CPU);
  EXPECT_EQ(vecOpType.empty(), true);
}

TEST_F(UtestRegister, DelInputWithTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));
  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseParamsFn(domi::AutoMappingFn);
  EXPECT_NE(opRegData.GetParseParamFn(), nullptr);

  // insert input into vector
  const vector<int> input_order{0, 1, 3, 2};
  opRegData.InputReorderVector(input_order);

  opRegData.DelInputWithCond(1, std::string("attrName_1"), true);
  opRegData.DelInputWithCond(2, "attrName_2", false);

  opRegData.DelInputWithOriginalType(3, std::string("Add"));
  opRegData.DelInputWithOriginalType(4, "Sub");

  OpRegistry *opReg = OpRegistry::Instance();
  ASSERT_NE(opReg, nullptr);
  bool retBool = opReg->Register(opRegData);
  ASSERT_EQ(retBool, true);

  std::vector<RemoveInputConfigure> rmConfigVec;
  rmConfigVec = opReg->GetRemoveInputConfigure(std::string("Add"));
  EXPECT_EQ(rmConfigVec.empty(), true);
  rmConfigVec = opReg->GetRemoveInputConfigure(std::string("Mul"));
  EXPECT_EQ(rmConfigVec.empty(), true);
  rmConfigVec = opReg->GetRemoveInputConfigure(std::string("Mul666"));
  EXPECT_EQ(rmConfigVec.empty(), true);
}

TEST_F(UtestRegister, GetOmTypeByOriOpTypeTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);
  std::string om_type;
  EXPECT_EQ(opReg->GetOmTypeByOriOpType(std::string("Sub"), om_type), true);
  EXPECT_EQ(opReg->GetOmTypeByOriOpType(std::string("Sub1"), om_type), false);
}

domi::Status FusionParseParamsFnCB(const std::vector<const google::protobuf::Message *> Msg, ge::Operator &Op) {
  return domi::SUCCESS;
}
domi::Status FusionParseParamsFnCB2(const std::vector<ge::Operator> &VecOp, ge::Operator &Op) {
  return domi::FAILED;
}
domi::Status ParseSubgraphPostFnCB(const std::string &subgraph_name, const ge::Graph &graph) {
  return domi::SUCCESS;
}
domi::Status ParseSubgraphPostFnCB2(const ge::AscendString &subgraph_name, const ge::Graph &graph) {
  return domi::SUCCESS;
}
domi::Status ParseOpToGraphFnCB(const ge::Operator &Op, ge::Graph &Graph) {
  return domi::SUCCESS;
}

TEST_F(UtestRegister, ParseParamFuncTest) {
  const std::string strOmOptype = "OmOptype";
  OpRegistrationData opRegData(strOmOptype);

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);
  std::vector<AscendString> OptypeListAStr = {AscendString("Div"), AscendString("Mul")};
  opRegData.OriginOpType(OptypeListAStr);

  opRegData.ParseParamsFn(domi::AutoMappingFn);
  EXPECT_NE(opRegData.GetParseParamFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetParseParamFunc(std::string("OmOptype1"), std::string("Sub")), nullptr);
  EXPECT_EQ(opReg->GetParseParamFunc(std::string("OmOptype"), std::string("Sub")), nullptr);
}

TEST_F(UtestRegister, FusionParseParamFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.FusionParseParamsFn(FusionParseParamsFnCB);
  EXPECT_NE(opRegData.GetFusionParseParamFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetFusionParseParamFunc(std::string("OmOptype"), std::string("Sub")), nullptr);
  EXPECT_EQ(opReg->GetFusionParseParamFunc(std::string("OmOptype1"), std::string("Sub")), nullptr);
}

TEST_F(UtestRegister, GetParseOpToGraphFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseOpToGraphFn(ParseOpToGraphFnCB);
  EXPECT_NE(opRegData.GetParseOpToGraphFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);
  std::string om_type;

  EXPECT_EQ(opReg->GetParseOpToGraphFunc(std::string("OmOptype"), std::string("Add")), nullptr);
  EXPECT_EQ(opReg->GetParseOpToGraphFunc(std::string("OmOptype"), std::string("Mul")), nullptr);
}

TEST_F(UtestRegister, ParseParamByOperatorFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseParamsByOperatorFn(domi::AutoMappingByOpFn);
  EXPECT_NE(opRegData.GetParseParamByOperatorFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetParseParamByOperatorFunc(std::string("int")), nullptr);
  EXPECT_EQ(opReg->GetParseParamByOperatorFunc(std::string("Add")), nullptr);
}

TEST_F(UtestRegister, FusionParseParamByOpFuncTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.FusionParseParamsFn(FusionParseParamsFnCB);
  EXPECT_NE(opRegData.GetFusionParseParamFn(), nullptr);

  opRegData.FusionParseParamsFn(FusionParseParamsFnCB2);
  EXPECT_NE(opRegData.GetFusionParseParamByOpFn(), nullptr);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetFusionParseParamByOpFunc(std::string("strOmOptype"), std::string("Add")), nullptr);
  EXPECT_EQ(opReg->GetFusionParseParamByOpFunc(std::string("OmOptype"), std::string("Add")), nullptr);
}

TEST_F(UtestRegister, ParseSubgraphPostFnTest) {
  OpRegistrationData opRegData(std::string("OmOptype"));

  std::initializer_list<std::string> OptypeList{std::string("Add"), std::string("Sub")};
  opRegData.OriginOpType(OptypeList);

  opRegData.ParseSubgraphPostFn(ParseSubgraphPostFnCB);
  EXPECT_NE(opRegData.GetParseSubgraphPostFn(), nullptr);

  opRegData.ParseSubgraphPostFn(ParseSubgraphPostFnCB2);
  EXPECT_NE(opRegData.GetParseSubgraphPostFn(), nullptr);

  ParseSubgraphFuncV2 Getfunc;
  opRegData.GetParseSubgraphPostFn(Getfunc);

  OpRegistry *opReg = OpRegistry::Instance();
  opReg->Register(opRegData);

  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("strOmOptype")), nullptr);
  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("OmOptype")), nullptr);

  domi::ParseSubgraphFuncV2 parse_subgraph_func;
  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("OmOptype"), parse_subgraph_func), domi::SUCCESS);
  EXPECT_EQ(opReg->GetParseSubgraphPostFunc(std::string("strOmOptype"), parse_subgraph_func), domi::FAILED);
}

TEST_F(UtestRegister, optiling_py_interface) {
  const nlohmann::json j = R"([
      {
          "name": "test_0",
          "dtype": "int8",
          "value": 1,
          "const_value": [
            1,
            1,
            1,
            1
          ],
          "shape": [
            4,
            4,
            4,
            4
          ],
          "format": "ND"
      },
      {
          "name": "test_1",
          "dtype": "list_int",
          "value": [
            1,
            1,
            1,
            1
          ]
      },
      {
          "name": "test_2"
      },
      {
          "name": "test_2",
          "dtype": "list_list_int",
          "value": [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]
          ]
      },
      {
          "name": "test_0",
          "dtype": "list_list_int64",
          "value": [
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2]
          ]
      },
      {
          "name": "test_3",
          "dtype": "test",
          "value": "1"
      }
      ])"_json;

  std::string json_str = j.dump();
  ge::Operator op("NULL");
  const char* optype = "ReluV2";
  const char* optype_v3 = "ReluV3";
  const char* optype_v4 = "ReluV4";
  const char* cmp_info = "";
  const char* inputs = "";
  const char* outputs = "";
  char* runinfo = "";
  size_t size = 3;
  const char* cmp_info_hash = "";
  uint64_t *elapse = nullptr;
  const char *attrs = json_str.c_str();
  TbeOpTilingPyInterface(optype, cmp_info, cmp_info_hash, attrs, attrs, attrs, runinfo, size, elapse);
  TbeOpTilingPyInterface(optype_v3, cmp_info, cmp_info_hash, attrs, attrs, attrs, runinfo, size, elapse);
  TbeOpTilingPyInterface(optype_v4, cmp_info, cmp_info_hash, attrs, attrs, attrs, runinfo, size, elapse);
  TbeOpTilingPyInterfaceEx2(optype, cmp_info, attrs, attrs, runinfo, size, cmp_info_hash, elapse);
  TbeOpTilingPyInterfaceEx2(optype_v3, cmp_info, attrs, attrs, runinfo, size, cmp_info_hash, elapse);
  TbeOpTilingPyInterfaceEx2(optype_v4, cmp_info, attrs, attrs, runinfo, size, cmp_info_hash, elapse);
}
