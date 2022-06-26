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

#include <gtest/gtest.h>

#define protected public
#define private public

#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/utils/constant_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/op_desc_impl.h"
#include "graph/node_impl.h"
#include "graph/runtime_inference_context.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/anchor_utils.h"

#undef private
#undef protected

namespace ge {
class UtestOpDescUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};


namespace {
///     Data    const1
///        \  /
///        addn
///
ComputeGraphPtr BuildGraph1() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto addn = builder.AddNode("addn", "AddN", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor0});

  builder.AddDataEdge(data, 0, addn, 0);
  builder.AddDataEdge(const1, 0, addn, 1);
  return builder.GetGraph();
}
///   (p_const)addn    const1
///          /     \   /
///        cast     mul
///
ComputeGraphPtr BuildGraph2() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto addn = builder.AddNode("addn", "AddN", 0, 2);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto cast = builder.AddNode("cast", "Cast", 1, 1);
  auto mul = builder.AddNode("mul", "Mul", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  AttrUtils::SetBool(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  AttrUtils::SetListInt(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0,1});
  AttrUtils::SetListTensor(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor0, tensor0});
  OpDescUtils::SetWeights(const1, {tensor0});

  builder.AddDataEdge(addn, 0, cast, 0);
  builder.AddDataEdge(addn, 1, mul, 0);
  builder.AddDataEdge(const1, 0, mul, 1);
  return builder.GetGraph();
}
///   (p_const)addn    const1
///          /     \   /
///        enter     mul
///         |
///       cast
ComputeGraphPtr BuildGraph3() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto addn = builder.AddNode("addn", "AddN", 0, 2);
  auto const1 = builder.AddNode("const1", "Const", 0, 1);
  auto enter = builder.AddNode("enter", "Enter", 1, 1);
  auto cast = builder.AddNode("cast", "Cast", 1, 1);
  auto mul = builder.AddNode("mul", "Mul", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  AttrUtils::SetBool(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_CONST, true);
  AttrUtils::SetListInt(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT_INDICES, {0,1});
  AttrUtils::SetListTensor(addn->GetOpDesc(), ATTR_NAME_POTENTIAL_WEIGHT, {tensor0, tensor0});
  OpDescUtils::SetWeights(const1, {tensor0});

  AttrUtils::SetBool(enter->GetOpDesc(), ENTER_ATTR_CONSTANT_FLAG, true);

  builder.AddDataEdge(addn, 0, enter, 0);
  builder.AddDataEdge(addn, 1, mul, 0);
  builder.AddDataEdge(const1, 0, mul, 1);
  builder.AddDataEdge(enter, 0, cast, 0);
  return builder.GetGraph();
}
}
TEST_F(UtestOpDescUtils, SetWeight) {
  auto graph = BuildGraph1();

  auto addn_node = graph->FindNode("addn");
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  map<int, ge::GeTensorPtr> weight0;
  weight0[-1] = tensor;
  auto ret = ge::OpDescUtils::SetWeights(*addn_node, weight0);
  EXPECT_NE(ret, 0);

  map<int, ge::GeTensorPtr> weight1;
  weight1[1] = tensor;
  ret = ge::OpDescUtils::SetWeights(*addn_node, weight1);
  EXPECT_EQ(ret, 0);
  auto const_node = graph->FindNode("const1");
  auto const_tensor = OpDescUtils::MutableWeights(const_node);
  EXPECT_EQ(const_tensor[0]->MutableData().size(), 3);
  auto in_nodes = addn_node->GetInAllNodes();
  EXPECT_EQ(in_nodes.size(), 2);

  map<int, ge::GeTensorPtr> weight2;
  weight2[2] = tensor;
  ret = ge::OpDescUtils::SetWeights(*addn_node, weight2);
  EXPECT_EQ(ret, 0);
  auto in_nodes1 = addn_node->GetInAllNodes();
  EXPECT_EQ(in_nodes1.size(), 3);
}

TEST_F(UtestOpDescUtils, GetRealConstInputNodeAndAnchor) {
  auto graph = BuildGraph1();
  auto add_node = graph->FindNode("addn");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*add_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 1);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "const1");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 0);
}
TEST_F(UtestOpDescUtils, GetMixConstInputNodeAndAnchor) {
  auto graph = BuildGraph2();
  auto mul_node = graph->FindNode("mul");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*mul_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 2);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 1);
  EXPECT_EQ(nodes_2_out_anchor[1].first->GetName(), "const1");
  EXPECT_EQ(nodes_2_out_anchor[1].second->GetIdx(), 0);
}
TEST_F(UtestOpDescUtils, GetInputDataByIndexForMixInputConst) {
  auto graph = BuildGraph2();
  auto mul_node = graph->FindNode("mul");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*mul_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 2);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 1);
  EXPECT_EQ(nodes_2_out_anchor[1].first->GetName(), "const1");
  EXPECT_EQ(nodes_2_out_anchor[1].second->GetIdx(), 0);

  auto weights = OpDescUtils::GetWeightsFromNodes(nodes_2_out_anchor);
  EXPECT_EQ(weights.size(), 2);
  EXPECT_EQ(weights[0]->GetTensorDesc().GetDataType(), DT_INT32);
  EXPECT_EQ(weights[1]->GetTensorDesc().GetDataType(), DT_INT32);
}
TEST_F(UtestOpDescUtils, GetPotentailWeightByIndexAccrossEnter) {
  auto graph = BuildGraph3();
  auto cast_node = graph->FindNode("cast");
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*cast_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 1);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 0);

  auto weights = OpDescUtils::GetWeightsFromNodes(nodes_2_out_anchor);
  EXPECT_EQ(weights.size(), 1);
  EXPECT_EQ(weights[0]->GetTensorDesc().GetDataType(), DT_INT32);
}

TEST_F(UtestOpDescUtils, GetInputConstDataByIndex_01) {
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 23;
  data_buf[10] = 32;
  auto ge_tensor = std::make_shared<GeTensor>();
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const_node = builder.AddNode("Const", "Const", 0, 1);
  AttrUtils::SetTensor(const_node->GetOpDesc(), "value", ge_tensor);
  auto case_node = builder.AddNode("Case", "Case", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(const_node, 0, case_node, 0);
  builder.AddDataEdge(case_node, 0, netoutput, 0);
  auto parent_graph = builder.GetGraph();

  ut::GraphBuilder sub_builder = ut::GraphBuilder("subgraph_graph");
  auto sub_data = sub_builder.AddNode("sub_data", "Data", 0, 1);
  auto sub_const = sub_builder.AddNode("sub_const", "Const", 0, 1);
  AttrUtils::SetTensor(sub_const->GetOpDesc(), "value", ge_tensor);
  auto add = sub_builder.AddNode("Add", "Add", 2, 1);
  auto sub_netoutput = sub_builder.AddNode("sub_netoutput", "NetOutput", 1, 0);
  sub_builder.AddDataEdge(sub_data, 0, add, 0);
  sub_builder.AddDataEdge(sub_const, 0, add, 1);
  sub_builder.AddDataEdge(add, 0, sub_netoutput, 0);

  auto subgraph = sub_builder.GetGraph();
  subgraph->SetParentNode(case_node);
  subgraph->SetParentGraph(parent_graph);
  parent_graph->AddSubgraph(subgraph->GetName(), subgraph);
  AttrUtils::SetInt(sub_data->GetOpDesc(), "_parent_node_index", 0);

  auto op_desc = add->GetOpDesc();
  op_desc->impl_->input_name_idx_["sub_data"] = 0;
  op_desc->impl_->input_name_idx_["sub_const"] = 1;
  auto op = OpDescUtils::CreateOperatorFromNode(add);
  RuntimeInferenceContext runtime_ctx;
  OpDescUtils::SetRuntimeContextToOperator(op, &runtime_ctx);
  GeTensorDesc desc;
  GeTensorPtr tensor = std::make_shared<GeTensor>(desc);
  tensor->SetData(data_buf, 4096);

  int64_t node_id = 1;
  int output_id = 0;
  runtime_ctx.SetTensor(node_id, output_id, std::move(tensor));
  ConstGeTensorBarePtr ge_tensor_res = nullptr;
  ge_tensor_res = OpDescUtils::GetInputConstData(op, 1);

  ASSERT_TRUE(ge_tensor_res != nullptr);
  const TensorData tmp(ge_tensor_res->GetData());
  const uint8_t* res_buf = tmp.GetData();
  ASSERT_EQ(res_buf[0], 23);
  ASSERT_EQ(res_buf[10], 32);
}

TEST_F(UtestOpDescUtils, GetInputConstDataByIndex_02) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto data2 = builder.AddNode("Data2", "Data", 0, 1);
  auto enter = builder.AddNode("Enter", "Enter", 1, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 2, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data2, 0, enter, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(enter, 0, transdata, 1);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();

  auto ge_tensor = std::make_shared<GeTensor>();
  uint8_t data_buf[4096] = {0};
  data_buf[0] = 23;
  data_buf[10] = 32;
  ge_tensor->SetData(data_buf, 4096);

  auto op_desc = transdata->GetOpDesc();
  op_desc->impl_->input_name_idx_["Data"] = 0;
  op_desc->impl_->input_name_idx_["Enter"] = 1;
  auto tensor_desc = op_desc->MutableInputDesc(0);
  AttrUtils::SetTensor(tensor_desc, "_value", ge_tensor);

  auto op = OpDescUtils::CreateOperatorFromNode(transdata);
  ConstGeTensorBarePtr ge_tensor_res = nullptr;
  ConstGeTensorBarePtr ge_tensor_res2 = nullptr;
  ge_tensor_res = OpDescUtils::GetInputConstData(op, 0);
  ge_tensor_res2 = OpDescUtils::GetInputConstData(op, 1);
  ASSERT_TRUE(ge_tensor_res != nullptr);
  ASSERT_TRUE(ge_tensor_res2 == nullptr);
  const TensorData tmp(ge_tensor_res->GetData());
  const uint8_t* res_buf = tmp.GetData();
  ASSERT_EQ(res_buf[0], 23);
  ASSERT_EQ(res_buf[10], 32);
}


TEST_F(UtestOpDescUtils, DefaultInferFormat) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape());
  tensor_desc->SetFormat(FORMAT_ND);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  EXPECT_EQ(op_desc->DefaultInferFormat(), 0);
  auto input_desc = op_desc->MutableInputDesc(0);
  EXPECT_EQ(input_desc->GetFormat(), FORMAT_ND);
  auto output_desc = op_desc->MutableOutputDesc(0);
  EXPECT_EQ(output_desc->GetFormat(), FORMAT_ND);
}


TEST_F(UtestOpDescUtils, OpDescBuilder) {
  OpDescBuilder builder("name", "type");
  builder.AddDynamicInput("AddDy", 1);
  EXPECT_NE(&builder, nullptr);
  const GeTensorDesc ten = GeTensorDesc(GeShape());
  builder.AddDynamicInput(std::string("AddDy2"), 2, ten);
  EXPECT_NE(&builder, nullptr);
  builder.AddDynamicOutput("AddDyOut", 3);
  EXPECT_NE(&builder, nullptr);
  builder.AddDynamicOutput(std::string("AddDyOut2"), 4, ten);
  EXPECT_NE(&builder, nullptr);
}

TEST_F(UtestOpDescUtils, OpDescUtils) {
  OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
  EXPECT_EQ(OpDescUtils::SetSubgraphInstanceName("subgraph_name", "subgraph_instance_name", odp), GRAPH_PARAM_INVALID);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  GeTensorPtr tp = std::make_shared<GeTensor>();
  OpDescPtr odp1 = std::make_shared<OpDesc>("name1", "type1");
  auto optr = odp1.get();
  optr = nullptr;
  EXPECT_EQ(OpDescUtils::MutableWeights(odp1), nullptr);
  EXPECT_EQ(OpDescUtils::ClearWeights(data_node), GRAPH_SUCCESS);
  NodePtr np = std::make_shared<Node>();
  EXPECT_EQ(OpDescUtils::ClearWeights(np), GRAPH_PARAM_INVALID);
  EXPECT_EQ(OpDescUtils::ClearInputDesc(data_node), true);
  odp->AddInputDesc(GeTensorDesc());
  EXPECT_EQ(OpDescUtils::GetWeights(data_node).size(), 0);
  EXPECT_EQ(OpDescUtils::GetWeights(nullptr).size(), 0);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*data_node).size(), 0);
  EXPECT_EQ(OpDescUtils::SetWeights(*odp, nullptr), GRAPH_FAILED);
  EXPECT_EQ(OpDescUtils::ClearInputDesc(odp, 0), true);
  EXPECT_EQ(OpDescUtils::ClearInputDesc(odp, 1), false);
  EXPECT_EQ(odp->impl_->inputs_desc_.size(), 0);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(odp), false);
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(data_node), true);
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(odp, 0), false);
  EXPECT_EQ(OpDescUtils::HasQuantizeFactorParams(*odp), false);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(*data_node, 1), false);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(data_node, 1), false);
}

TEST_F(UtestOpDescUtils, OpDescUtilsSupply) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  auto one_node = builder.AddNode("One", "One", 3, 3);
  InDataAnchorPtr in_anch = std::make_shared<InDataAnchor>(data_node, 111);
  OutDataAnchorPtr out_anch = std::make_shared<OutDataAnchor>(data_node, 222);
  auto node3 = builder.AddNode("Data3", "Data3", 3, 3);
  InControlAnchorPtr inc_anch = std::make_shared<InControlAnchor>(node3, 33);
  EXPECT_EQ(attr_node->AddLinkFrom(data_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*attr_node).size(), 0);
  std::vector<ge::NodePtr> node_v;
  node_v.push_back(data_node);
  node_v.push_back(attr_node);
  EXPECT_EQ(OpDescUtils::GetInputData(node_v).size(), 0);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(*attr_node), 1);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(attr_node), 1);
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(*attr_node, 1), GeTensorDesc());
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(attr_node, 1), GeTensorDesc());
  size_t st = 0;
  EXPECT_EQ(OpDescUtils::GetNonConstInputIndex(attr_node, 1, st), false);
  EXPECT_EQ(OpDescUtils::GetConstInputs(nullptr).size(), 0);
  EXPECT_EQ(OpDescUtils::GetNonConstTensorDesc(attr_node).size(), 1);
  Operator op("name", "type");
  op.operator_impl_ = nullptr;
  EXPECT_EQ(OpDescUtils::GetInputConstData(op, 0), nullptr);
}

TEST_F(UtestOpDescUtils, ClearInputDesc_Nullptr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 1);
  data_node->impl_->op_->impl_ = nullptr;
  EXPECT_EQ(OpDescUtils::ClearInputDesc(data_node), false);
}

TEST_F(UtestOpDescUtils, ClearOutputDesc_Nullptr) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data_node = builder.AddNode("Data", "Data", 1, 1);
  EXPECT_EQ(data_node->GetAllInDataAnchors().size(), 1);
  data_node->impl_->op_->impl_ = nullptr;
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(data_node), false);
}

TEST_F(UtestOpDescUtils, ClearOutputDesc_Normal) {
  OpDescPtr odp = std::make_shared<OpDesc>("name", "type");
  EXPECT_NE(odp, nullptr);
  EXPECT_NE(odp->impl_, nullptr);
  EXPECT_EQ(odp->impl_->outputs_desc_.size(), 0);
  odp->impl_->outputs_desc_.push_back(std::make_shared<GeTensorDesc>());
  EXPECT_EQ(OpDescUtils::ClearOutputDesc(odp, 0), true);
}

TEST_F(UtestOpDescUtils, GetWeightsFromNodes) {
  auto graph = BuildGraph3();
  auto cast_node = graph->FindNode("cast");
  auto enter_node = graph->FindNode("enter");
  auto in_nodes_and_anchors = cast_node->GetInDataNodesAndAnchors();
  EXPECT_EQ(in_nodes_and_anchors.size(), 1);
  EXPECT_EQ(in_nodes_and_anchors.begin()->first->GetName(), "enter");
  EXPECT_EQ(in_nodes_and_anchors.begin()->second->GetIdx(), 0);
  auto opdsc1 = in_nodes_and_anchors.begin()->first->GetOpDesc();
  bool is_potential_const1 = false;
  auto has_attr1 = AttrUtils::GetBool(opdsc1, ATTR_NAME_POTENTIAL_CONST, is_potential_const1);
  EXPECT_EQ(has_attr1, false);

  EXPECT_EQ(in_nodes_and_anchors.size(), 1);
  auto nodes_2_out_anchor = OpDescUtils::GetConstInputNodeAndAnchor(*cast_node);
  EXPECT_EQ(nodes_2_out_anchor.size(), 1);
  EXPECT_EQ(nodes_2_out_anchor[0].first->GetName(), "addn");
  EXPECT_EQ(nodes_2_out_anchor[0].second->GetIdx(), 0);

  auto opdsc = nodes_2_out_anchor[0].first->GetOpDesc();
  bool is_potential_const = false;
  auto has_attr = AttrUtils::GetBool(opdsc, ATTR_NAME_POTENTIAL_CONST, is_potential_const);
  EXPECT_EQ(has_attr, true);
  auto weights = OpDescUtils::GetWeightsFromNodes(nodes_2_out_anchor);
}

TEST_F(UtestOpDescUtils, GetConstInputNode_Const_Enter_Other) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*const1).size(), 1);

  auto enter1 = builder.AddNode("enter1", "Enter", 1, 1);
  auto enter2 = builder.AddNode("enter2", "Enter", 1, 1);
  EXPECT_EQ(enter1->AddLinkFrom(enter2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*enter1).size(), 0);

  auto other1 = builder.AddNode("other1", "Enter", 1, 1);
  auto other2 = builder.AddNode("other2", "other", 1, 1);
  EXPECT_EQ(other1->AddLinkFrom(other2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetConstInputNode(*other1).size(), 0);
}

TEST_F(UtestOpDescUtils, GetInputData_Weight) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor0});

  std::vector<ge::NodePtr> vec;
  vec.push_back(const1);
  EXPECT_EQ(OpDescUtils::GetInputData(vec).size(), 1);
}

TEST_F(UtestOpDescUtils, GetNonConstInputsSize) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);

  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(nullptr), 0);
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*const1), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(*const1), 0);
  EXPECT_EQ(const1->GetAllInDataAnchors().size(), 1);
  auto in_anch = const1->GetAllInDataAnchors().at(0);
  EXPECT_EQ(AnchorUtils::SetStatus(in_anch, ANCHOR_DATA), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputsSize(*const1), 1);
}

TEST_F(UtestOpDescUtils, AddConstOpToAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  EXPECT_EQ(const1->GetAllInDataAnchors().size(), 1);
  auto in_anch = const1->GetAllInDataAnchors().at(0);
  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));

  EXPECT_EQ(OpDescUtils::AddConstOpToAnchor(in_anch, tensor0), GRAPH_SUCCESS);
}

TEST_F(UtestOpDescUtils, GetNonConstInputIndex) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);

  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*attr_node), GRAPH_SUCCESS);
  size_t st = 0;
  EXPECT_EQ(OpDescUtils::GetNonConstInputIndex(attr_node, 1, st), false);
}

TEST_F(UtestOpDescUtils, GetNonConstInputTensorDesc) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(), 2);
  auto in_anch = attr_node->GetAllInDataAnchors().at(0);
  EXPECT_NE(in_anch, nullptr);

  EXPECT_EQ(AnchorUtils::SetStatus(in_anch, ANCHOR_DATA), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(attr_node, 1), GeTensorDesc());
}

TEST_F(UtestOpDescUtils, GetNonConstInputTensorDesc_SetStatus) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(attr_node->GetAllInDataAnchors().size(), 2);
  auto in_anch = attr_node->GetAllInDataAnchors().at(0);
  EXPECT_NE(in_anch, nullptr);

  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(attr_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstInputTensorDesc(attr_node, 1), GeTensorDesc());
}

TEST_F(UtestOpDescUtils, IsNonConstInput) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(attr_node, 1), false);


  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*attr_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(attr_node, 1), false);

  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(const1, 0), false);
  EXPECT_EQ(OpDescUtils::IsNonConstInput(const2, 0), false);
}

TEST_F(UtestOpDescUtils, GetNonConstTensorDesc) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto attr_node = builder.AddNode("Attr", "Attr", 2, 2);
  EXPECT_EQ(OpDescUtils::GetNonConstTensorDesc(nullptr).size(), 0);
  EXPECT_EQ(NodeUtils::SetAllAnchorStatus(*attr_node), GRAPH_SUCCESS);
  EXPECT_EQ(OpDescUtils::GetNonConstTensorDesc(attr_node).size(), 0);
}

TEST_F(UtestOpDescUtils, GetConstInputs_Const) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto const2 = builder.AddNode("const2", "Const", 1, 1);
  EXPECT_EQ(const1->AddLinkFrom(const2), GRAPH_SUCCESS);
  EXPECT_EQ(const1->GetType(), "Const");
  EXPECT_EQ(OpDescUtils::GetConstInputs(*const1).size(), 1);
}

TEST_F(UtestOpDescUtils, GetConstInputs_Switch) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto sw1 = builder.AddNode("sw1", "Switch", 1, 1);
  auto mm1 = builder.AddNode("mm1", "MatMul", 1, 1);
  EXPECT_EQ(sw1->AddLinkFrom(mm1), GRAPH_SUCCESS);
  EXPECT_EQ(sw1->GetType(), "Switch");
  EXPECT_EQ(mm1->GetType(), "MatMul");
  EXPECT_EQ(OpDescUtils::GetConstInputs(*sw1).size(), 0);
}

TEST_F(UtestOpDescUtils, MutableWeights) {
  auto node = std::make_shared<Node>();
  node = nullptr;
  EXPECT_EQ(OpDescUtils::MutableWeights(node).size(), 0);
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto ph = builder.AddNode("ph", "PlaceHolder", 1, 1);
  EXPECT_EQ(OpDescUtils::MutableWeights(*ph).size(), 0);
}

TEST_F(UtestOpDescUtils, MutableWeights_Nullptr) {
  OpDescPtr odp = std::make_shared<OpDesc>();
  odp = nullptr;
  EXPECT_EQ(OpDescUtils::MutableWeights(odp), nullptr);
}

TEST_F(UtestOpDescUtils, SetWeights) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  std::map<int, ge::GeTensorPtr> weights_map;
  weights_map[1] = std::make_shared<GeTensor>();
  EXPECT_EQ(OpDescUtils::SetWeights(*const1, weights_map), GRAPH_SUCCESS);

  auto non1 = builder.AddNode("nonconst1", "NonConst", 1, 1);
  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  EXPECT_EQ(OpDescUtils::SetWeights(non1, {tensor0}), GRAPH_SUCCESS);

  weights_map[2] = tensor0;
  EXPECT_EQ(OpDescUtils::SetWeights(*const1, weights_map), GRAPH_PARAM_INVALID);
}

}
