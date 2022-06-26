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
#include <iostream>
#define private public
#include "graph/graph.h"
#include "graph/operator.h"
#include "graph/compute_graph.h"
#include "graph/compute_graph_impl.h"
#include "graph/op_desc.h"
#include "graph/node.h"
#include "graph/utils/graph_utils.h"
#include "external/graph/graph.h"
#include "graph/compute_graph_impl.h"
#include "inc/external/graph/operator_reg.h"
#include "inc/external/graph/operator.h"
#include "inc/external/graph/operator_factory.h"
#include "inc/external/graph/graph.h"
#include "inc/graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "graph/ge_attr_value.h"
#undef private

using namespace ge;

class UtestGraph : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static ComputeGraphPtr BuildSubComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("subgraph");
  auto data = builder.AddNode("sub_Data", "sub_Data", 0, 1);
  auto netoutput = builder.AddNode("sub_Netoutput", "sub_NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  return graph;
}

// construct graph which contains subgraph
static ComputeGraphPtr BuildComputeGraph() {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  transdata->GetOpDesc()->AddSubgraphName("subgraph");
  transdata->GetOpDesc()->SetSubgraphInstanceName(0, "subgraph");
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  builder.AddDataEdge(data, 0, transdata, 0);
  builder.AddDataEdge(transdata, 0, netoutput, 0);
  auto graph = builder.GetGraph();
  // add subgraph
  transdata->SetOwnerComputeGraph(graph);
  ComputeGraphPtr subgraph = BuildSubComputeGraph();
  subgraph->SetParentGraph(graph);
  subgraph->SetParentNode(transdata);
  graph->AddSubgraph("subgraph", subgraph);
  return graph;
}

TEST_F(UtestGraph, copy_graph_01) {
  ge::OpDescPtr add_op(new ge::OpDesc("add1", "Add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  ge::Graph copy_graph("copy_graph");
  ASSERT_EQ(copy_graph.CopyFrom(graph), ge::GRAPH_SUCCESS);
  Graph graph2("graph2");
  ASSERT_EQ(copy_graph.CopyFrom(graph2), GRAPH_FAILED);

  auto cp_compute_graph = ge::GraphUtils::GetComputeGraph(copy_graph);
  ASSERT_NE(cp_compute_graph, nullptr);
  ASSERT_NE(cp_compute_graph, compute_graph);
  ASSERT_EQ(cp_compute_graph->GetDirectNodesSize(), 1);
  auto cp_add_node = cp_compute_graph->FindNode("add1");
  ASSERT_NE(cp_add_node, nullptr);
  ASSERT_NE(cp_add_node, add_node);
}

TEST_F(UtestGraph, copy_graph_02) {
  ge::OpDescPtr if_op(new ge::OpDesc("if", "If"));
  if_op->AddDynamicInputDesc("input", 1);
  if_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto if_node = compute_graph->AddNode(if_op);
  auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  ge::Graph copy_graph("copy_graph");

  if_op->AddSubgraphName("then_branch");
  if_op->AddSubgraphName("else_branch");
  if_op->SetSubgraphInstanceName(0, "then");
  if_op->SetSubgraphInstanceName(1, "else");

  ge::OpDescPtr add_op1(new ge::OpDesc("add1", "Add"));
  add_op1->AddDynamicInputDesc("input", 2);
  add_op1->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> then_compute_graph(new ge::ComputeGraph("then"));
  auto add_node1 = then_compute_graph->AddNode(add_op1);
  then_compute_graph->SetParentNode(if_node);
  then_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(then_compute_graph);

  ge::OpDescPtr add_op2(new ge::OpDesc("add2", "Add"));
  add_op2->AddDynamicInputDesc("input", 2);
  add_op2->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> else_compute_graph(new ge::ComputeGraph("else"));
  auto add_node2 = else_compute_graph->AddNode(add_op2);
  else_compute_graph->SetParentNode(if_node);
  else_compute_graph->SetParentGraph(compute_graph);
  compute_graph->AddSubgraph(else_compute_graph);

  ASSERT_EQ(copy_graph.CopyFrom(graph), ge::GRAPH_SUCCESS);

  auto cp_compute_graph = ge::GraphUtils::GetComputeGraph(copy_graph);
  ASSERT_NE(cp_compute_graph, nullptr);
  ASSERT_NE(cp_compute_graph, compute_graph);
  ASSERT_EQ(cp_compute_graph->GetDirectNodesSize(), 1);
  auto cp_if_node = cp_compute_graph->FindNode("if");
  ASSERT_NE(cp_if_node, nullptr);
  ASSERT_NE(cp_if_node, if_node);

  auto cp_then_compute_graph = cp_compute_graph->GetSubgraph("then");
  ASSERT_NE(cp_then_compute_graph, nullptr);
  ASSERT_NE(cp_then_compute_graph, then_compute_graph);
  ASSERT_EQ(cp_then_compute_graph->GetDirectNodesSize(), 1);
  auto cp_add_node1 = cp_then_compute_graph->FindNode("add1");
  ASSERT_NE(cp_add_node1, nullptr);
  ASSERT_NE(cp_add_node1, add_node1);

  auto cp_else_compute_graph = cp_compute_graph->GetSubgraph("else");
  ASSERT_NE(cp_else_compute_graph, nullptr);
  ASSERT_NE(cp_else_compute_graph, else_compute_graph);
  ASSERT_EQ(cp_else_compute_graph->GetDirectNodesSize(), 1);
  auto cp_add_node2 = cp_else_compute_graph->FindNode("add2");
  ASSERT_NE(cp_add_node2, nullptr);
  ASSERT_NE(cp_add_node2, add_node2);
}

REG_OP(Mul)
    .OP_END_FACTORY_REG(Mul)
IMPL_INFER_VALUE_RANGE_FUNC(Mul, func){
  std::cout << "test" << std::endl;
  return GRAPH_SUCCESS;
}

REG_OP(Test2)
    .OP_END_FACTORY_REG(Test2)
IMPL_INFER_VALUE_RANGE_FUNC(Test2, func2){
  std::cout << "test" << std::endl;
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraph, test_infer_value_range_register_succ) {
  string op_type = "Add";
  INFER_VALUE_RANGE_DEFAULT_REG(Add);
  INFER_VALUE_RANGE_DEFAULT_REG(Test1);
  auto para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, true);
  ASSERT_EQ(para.infer_value_func, nullptr);

  op_type = "Mul";
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Mul, INPUT_HAS_VALUE_RANGE, func);
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Test2, INPUT_IS_DYNAMIC, func2);
  para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, true);
  ASSERT_NE(para.infer_value_func, nullptr);

  op_type = "Sub";
  para = OperatorFactoryImpl::GetInferValueRangePara(op_type);
  ASSERT_EQ(para.is_initialized, false);
}

REG_OP(Shape)
    .OP_END_FACTORY_REG(Shape)
IMPL_INFER_VALUE_RANGE_FUNC(Shape, ShapeValueInfer){
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  auto output_tensor_desc = op_desc->MutableOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> in_shape_range;
  op_desc->MutableInputDesc(0)->GetShapeRange(in_shape_range);
  if (!in_shape_range.empty()) {
    output_tensor_desc->SetValueRange(in_shape_range);
  }
  return GRAPH_SUCCESS;
}

TEST_F(UtestGraph, test_value_range_infer_and_set_get) {
  using std::make_pair;
  string op_type = "Shape";
  INFER_VALUE_RANGE_CUSTOM_FUNC_REG(Shape, INPUT_IS_DYNAMIC, ShapeValueInfer);
  auto graph = std::make_shared<ComputeGraph>("test_graph");
  auto shape_op_desc = std::make_shared<OpDesc>("node_name", op_type);
  GeTensorDesc tensor_desc(GeShape({-1, -1, 4, 192}), ge::FORMAT_NCHW, DT_INT32);
  std::vector<std::pair<int64_t, int64_t>> shape_range = {make_pair(1, 100), make_pair(1, 240),
                                                          make_pair(4, 4),   make_pair(192, 192)};
  tensor_desc.SetShapeRange(shape_range);
  shape_op_desc->AddInputDesc(tensor_desc);
  GeTensorDesc out_tensor_desc(GeShape({4}), ge::FORMAT_NCHW, DT_INT32);
  shape_op_desc->AddOutputDesc(out_tensor_desc);
  auto shape_node = graph->AddNode(shape_op_desc);
  Operator op = OpDescUtils::CreateOperatorFromNode(shape_node);
  auto ret = shape_node->GetOpDesc()->CallInferValueRangeFunc(op);
  ASSERT_EQ(ret, GRAPH_SUCCESS);

  auto output_0_desc = shape_node->GetOpDesc()->GetOutputDesc(0);
  std::vector<std::pair<int64_t, int64_t>> value_range;
  output_0_desc.GetValueRange(value_range);
  EXPECT_EQ(value_range.size(), 4);

  std::vector<int64_t> target_value_range = {1, 100, 1, 240, 4, 4, 192, 192};
  std::vector<int64_t> output_value_range;
  for (auto pair : value_range) {
    output_value_range.push_back(pair.first);
    output_value_range.push_back(pair.second);
  }
  EXPECT_EQ(target_value_range, output_value_range);
}

TEST_F(UtestGraph, get_all_graph_nodes) {
  ComputeGraphPtr graph = BuildComputeGraph();
  auto nodes = graph->GetAllNodes();
  EXPECT_EQ(nodes.size(), 5);
}

TEST_F(UtestGraph, SetOutputs_ops) {
  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<ge::Operator> outputs = {op1, op2, op3};

  Graph graph;
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 0);
  // EXPECT_TRUE(graph.impl_->output_name_.empty()); // impl缺少头文件，找不到声明
}

TEST_F(UtestGraph, SetOutputs_string) {
  using std::make_pair;
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::string op_n1 = std::string("add");
  std::string op_n2 = std::string("op2");
  std::string op_n3 = std::string("op3");

  std::vector<std::pair<Operator, std::string>> outputs = {make_pair(op1, op_n1), make_pair(op2, op_n2),
                                                          make_pair(op3, op_n3)};
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, SetOutputs_AscendString) {
  using std::make_pair;
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  AscendString op_n1 = AscendString("add");
  AscendString op_n2 = AscendString("op2");
  AscendString op_n3 = AscendString("op3");

  std::vector<std::pair<Operator, AscendString>> outputs = {make_pair(op1, op_n1), make_pair(op2, op_n2),
                                                          make_pair(op3, op_n3)};
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, SetOutputs_Index) {
  using std::make_pair;
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  Graph graph2;

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<size_t> vec_index1 = {0,1,2};
  std::vector<size_t> vec_index2 = {0};
  std::vector<size_t> vec_index3 = {0};

  std::vector<std::pair<Operator, std::vector<size_t>>> outputs = {make_pair(op1, vec_index1),
    make_pair(op2, vec_index2),  make_pair(op3, vec_index3)};
  graph2.SetOutputs(outputs);
  graph.SetOutputs(outputs);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, SetTargets) {
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  add_op->AddDynamicInputDesc("input", 2);
  add_op->AddDynamicOutputDesc("output", 1);
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  Graph graph2;

  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<size_t> vec_index1 = {0,1,2};
  std::vector<size_t> vec_index2 = {0};
  std::vector<size_t> vec_index3 = {0};

  std::vector<ge::Operator> targets = {op1, op2, op3};

  graph2.SetTargets(targets);
  graph.SetTargets(targets);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, SetNeedIteration) {
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
  Graph graph2;

  graph2.SetNeedIteration(true);
  graph.SetNeedIteration(false);
  EXPECT_EQ(graph.GetAllNodes().size(), 1);
}

TEST_F(UtestGraph, GetDirectNode) {
  ge::OpDescPtr add_op(new ge::OpDesc("add_0", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph(new ge::ComputeGraph("test_graph"));
  auto add_node = compute_graph->AddNode(add_op);
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);

  ge::OpDescPtr add_op2(new ge::OpDesc("add_1", "add"));
  std::shared_ptr<ge::ComputeGraph> compute_graph2 = nullptr;
  Graph graph2 = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph2);

  Graph graph3;

  std::vector<GNode> gnodes, gnodes2, gnodes3;

  gnodes = graph.GetDirectNode();
  gnodes2 = graph2.GetDirectNode();
  gnodes3 = graph3.GetDirectNode();
  EXPECT_EQ(gnodes.size(), 1);
}

TEST_F(UtestGraph, RemoveNode) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  auto v_nodes = cgp->GetAllNodes();
  EXPECT_EQ(v_nodes.size(), 5);

  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(cgp);

  auto nodes = graph.GetAllNodes();
  graph.RemoveNode(nodes[4]);
  EXPECT_EQ(graph.GetAllNodes().size(), 4);

  graph.RemoveNode(nodes[0], true);
  EXPECT_EQ(graph.GetAllNodes().size(), 3);
}

TEST_F(UtestGraph, AddRemoveEdge1) {
  Operator op1 = Operator("add");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");

  Graph graph("a_graph");
  Graph graph2;

  GNode node1 = graph.AddNodeByOp(op1);
  GNode node2 = graph.AddNodeByOp(op2);
  GNode node3 = graph.AddNodeByOp(op3);

  auto ret =graph.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.AddControlEdge(node2, node3);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);

  graph2.AddNodeByOp(op1);
  ret =graph2.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph2.AddControlEdge(node2, node3);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph2.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, AddRemoveEdge2) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  ComputeGraphPtr cgp = builder.GetGraph();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(cgp);

  auto nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 1);

  GNode node1 = nodes[0];
  GNode node2;

  auto ret =graph.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_FAILED);
  ret = graph.AddControlEdge(node1, node2);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, AddRemoveEdge3) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 0, 1);
  auto transdata = builder.AddNode("Transdata", "Transdata", 1, 1);
  auto netoutput = builder.AddNode("Netoutput", "NetOutput", 1, 0);
  ComputeGraphPtr cgp = builder.GetGraph();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(cgp);

  auto nodes = graph.GetAllNodes();
  EXPECT_EQ(nodes.size(), 3);

  GNode node1 = nodes[0];
  GNode node2 = nodes[1];
  GNode node3 = nodes[2];

  auto ret = graph.AddDataEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph.AddControlEdge(node2, node3);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  ret = graph.RemoveEdge(node1, 0, node2, 0);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, ConstructFromInputs1) {
  Graph graph;
  Operator op1 = Operator("op1");
  Operator op2 = Operator("op2");
  Operator op3 = Operator("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  auto ret = graph.ConstructFromInputs({}, name);
  EXPECT_EQ(ret, nullptr);

  ret = graph.ConstructFromInputs(inputs, AscendString(nullptr));
  EXPECT_EQ(ret, nullptr);

  ret = graph.ConstructFromInputs(inputs, name);
  EXPECT_EQ(ret, nullptr);
}

REG_OP(Phony0)
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(value, Tensor, Tensor())
    .OP_END_FACTORY_REG(Phony0);

REG_OP(Phony1)
    .DYNAMIC_INPUT(x, TensorType::NumberType())
    .OUTPUT(y, TensorType::NumberType())
    .REQUIRED_ATTR(N, Int)
    .OP_END_FACTORY_REG(Phony1);

REG_OP(Phony2)
    .INPUT(x,
           TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                       DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .INPUT(shape, TensorType({DT_INT32, DT_INT64}))
    .OUTPUT(y,
            TensorType({DT_FLOAT, DT_FLOAT16, DT_INT8, DT_INT16, DT_UINT16, DT_UINT8, DT_INT32, DT_INT64, DT_UINT32,
                        DT_UINT64, DT_BOOL, DT_DOUBLE}))
    .ATTR(axis, Int, 0)
    .ATTR(num_axes, Int, -1)
    .OP_END_FACTORY_REG(Phony2);

TEST_F(UtestGraph, ConstructFromInputs2) {
  Graph graph;
  Operator op1 = op::Phony0("op1");
  Operator op2 = op::Phony1("op2");
  Operator op3 = op::Phony2("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  auto ret = graph.ConstructFromInputs(inputs, name);
  EXPECT_NE(ret, nullptr);
}

TEST_F(UtestGraph, SaveLoadFile) {
  system("rm -rf ./ut_graph1.txt");
  system("rm -rf ./ut_graph2.txt");

  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(cgp);

  auto ret = graph.SaveToFile(nullptr);
  EXPECT_EQ(ret, GRAPH_FAILED);

  ret = graph.SaveToFile("./ut_graph1.txt");
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  ret = graph.SaveToFile(std::string("./ut_graph2.txt"));
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  Graph graph2;
  ret = graph2.LoadFromFile(nullptr);
  EXPECT_EQ(ret, GRAPH_FAILED);

  Graph graph3;
  ret = graph3.LoadFromFile("./ut_graph1.txt");
  EXPECT_EQ(ret, GRAPH_FAILED);

  Graph graph4;
  ret = graph4.LoadFromFile(std::string("./ut_graph2.txt"));
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, GetName) {
  Graph graph;
  AscendString name;
  auto ret = graph.GetName(name);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, RecoverGraphOperators) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(cgp);
  auto ret = GraphUtils::RecoverGraphOperators(graph);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, GetOpName) {
  ComputeGraphPtr cgp = BuildComputeGraph();
  Graph graph = ge::GraphUtils::CreateGraphFromComputeGraph(cgp);

  Operator op1("add");
  auto ret = graph.AddOp(op1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<std::string> op_names1;
  ret = graph.GetAllOpName(op_names1);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<AscendString> op_names2;
  ret = graph.GetAllOpName(op_names2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}

TEST_F(UtestGraph, FindOpByName) {
  Graph graph;
  Operator op1 = op::Phony0("op1");
  Operator op2 = op::Phony1("op2");
  Operator op3 = op::Phony2("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  GraphPtr gptr = Graph::ConstructFromInputs(inputs, name);

  EXPECT_EQ(gptr->GetAllNodes().size(), 2);

  Operator op1_2;
  auto ret = gptr->FindOpByName(nullptr, op1_2);
  ret = gptr->FindOpByName("op1", op1_2);
  EXPECT_EQ(ret, GRAPH_FAILED);

  Operator op2_2;
  ret = gptr->FindOpByName(std::string("op2"), op2_2);
  EXPECT_EQ(ret, GRAPH_FAILED);
}

TEST_F(UtestGraph, FindOpByType) {
  Graph graph;
  Operator op1 = op::Phony0("op1");
  Operator op2 = op::Phony1("op2");
  Operator op3 = op::Phony2("op3");
  std::vector<Operator> inputs = {op1, op2, op3};
  AscendString name = "graph_name";

  GraphPtr gptr = Graph::ConstructFromInputs(inputs, name);

  std::vector<ge::Operator> op1_2;
  auto ret = gptr->FindOpByType(nullptr, op1_2);
  ret = gptr->FindOpByType("const", op1_2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);

  std::vector<ge::Operator> op2_2;
  ret = gptr->FindOpByType(std::string("data"), op2_2);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
}


