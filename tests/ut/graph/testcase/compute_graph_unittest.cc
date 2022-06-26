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
#include "graph/compute_graph.h"
#include "graph/compute_graph_impl.h"
#include "graph/op_desc.h"
#include "graph/op_desc_impl.h"
#include "graph/ge_tensor.h"
#include "graph/utils/ge_ir_utils.h"
#include "graph_builder_utils.h"
#undef private
#undef protected
#include "debug/ge_op_types.h"
#include "inc/graph/debug/ge_attr_define.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils.h"
namespace ge
{
  class UtestComputeGraph : public testing::Test {
    protected:
    void SetUp() {}
    void TearDown() {}
  };

TEST_F(UtestComputeGraph, GetAllNodes_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc(tensor_desc->Clone());
  graph->AddNode(op_desc);
  graph->AddNode(op_desc);

  auto node_filter = [](const Node &node){ return true;};
  auto graph_filter = [](const Node &node, const char *str, const ComputeGraphPtr &graph){ return true;};
  auto out_nodes = graph->GetAllNodes(node_filter, graph_filter);
  EXPECT_EQ(out_nodes.size(), 2);
}

TEST_F(UtestComputeGraph, GetNodes_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>();
  op_desc->AddInputDesc(tensor_desc->Clone());
  graph->AddNode(op_desc);
  graph->AddNode(op_desc);
  auto node_filter = [](const Node &node){ return true;};
  auto graph_filter = [](const Node &node, const char *str, const ComputeGraphPtr &graph){ return true;};
  auto out_nodes = graph->GetNodes(true, node_filter, graph_filter);
  EXPECT_EQ(out_nodes.size(), 2);
}

TEST_F(UtestComputeGraph, AddNodeFront_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  auto node = graph->AddNode(op_desc);

  auto op_desc1 = std::make_shared<OpDesc>("add_front", "add_front");
  op_desc1->AddInputDesc(tensor_desc->Clone());
  auto nodeptr = graph->AddNodeFront(node);
  EXPECT_EQ(node, nodeptr);
}

TEST_F(UtestComputeGraph, RemoveNode_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);
  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  auto node = graph->AddNode(op_desc);

  EXPECT_EQ(graph->RemoveNode(node), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, GraphMembersAreEqual_success) {
  auto graph1 = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  graph1->AddNode(op_desc);
  graph1->AddNode(op_desc);

  auto graph2 = std::make_shared<ComputeGraph>("graph");
  graph2->AddNode(op_desc);
  EXPECT_EQ(graph1->GraphMembersAreEqual(*(graph2)), false);
  graph2->AddNode(op_desc);
  EXPECT_EQ(graph1->GraphMembersAreEqual(*(graph2)), true);
}

TEST_F(UtestComputeGraph, GraphAttrsAreEqual_success) {
  auto graph1 = std::make_shared<ComputeGraph>("graph1");

  int64_t val = 0;
  AnyValue anyvalue;
  anyvalue.SetValue(val);
  graph1->SetAttr("test", anyvalue);

  auto graph2 = std::make_shared<ComputeGraph>("graph2");
  EXPECT_EQ(graph1->GraphAttrsAreEqual(*(graph2)), false);

  graph2->SetAttr("test", anyvalue);
  EXPECT_EQ(graph1->GraphAttrsAreEqual(*(graph2)), true);
}

TEST_F(UtestComputeGraph, VectorInputNodePtrIsEqual_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", "node1");
  op_desc->AddInputDesc(tensor_desc->Clone());
  auto node = graph->AddNode(op_desc);

  std::vector<NodePtr> leftnodes{node};
  std::vector<NodePtr> rightnodes{node};
  EXPECT_EQ(graph->VectorInputNodePtrIsEqual(leftnodes, rightnodes), true);
  rightnodes.push_back(node);
  EXPECT_EQ(graph->VectorInputNodePtrIsEqual(leftnodes, rightnodes), false);
}

TEST_F(UtestComputeGraph, RemoveConstInput_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetDataType(DT_FLOAT);
  tensor_desc->SetFormat(FORMAT_CHWN);

  auto op_desc = std::make_shared<OpDesc>("node1", CONSTANT);
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  auto node1 = graph->AddNode(op_desc);
  auto node2 = graph->AddNode(op_desc);
  GraphUtils::AddEdge(node1->GetOutControlAnchor(), node2->GetInControlAnchor());
  EXPECT_EQ(graph->RemoveConstInput(node1), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, RemoveSubGraph_success) {
  auto rootgraph = std::make_shared<ComputeGraph>("rootgraph");
  auto subgraph = std::make_shared<ComputeGraph>("subgraph");
  rootgraph->AddSubGraph(subgraph);
  EXPECT_EQ(rootgraph->RemoveSubGraph(subgraph), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, Set_GetShareParamLayer_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  std::map<std::vector<std::string>, std::vector<std::string>> params_share_map{{{"test"},{"test"}}};
  graph->SetShareParamLayer(params_share_map);
  EXPECT_EQ(graph->GetShareParamLayer().size(), 1);
}

TEST_F(UtestComputeGraph, Set_GetGraphOutNodes_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  std::map<std::string, std::vector<int32_t>> out_nodes_map{{"test",{1}}};
  auto opdesc = std::make_shared<OpDesc>();
  graph->SetGraphOutNodes(out_nodes_map);
  EXPECT_EQ(graph->GetGraphOutNodes().size(), 1);
  std::map<std::string, std::vector<int32_t>> append_out_nodes_map{{"test2",{2}}};
  graph->AppendGraphOutNodes(append_out_nodes_map);
  EXPECT_EQ(graph->GetGraphOutNodes().size(), 2);
}

TEST_F(UtestComputeGraph, Set_GetOrigGraph_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto origin_graph = std::make_shared<ComputeGraph>("origin_graph");
  graph->SetOrigGraph(origin_graph);
  EXPECT_NE(graph->GetOrigGraph(), nullptr);
}

TEST_F(UtestComputeGraph, GetOutputSize_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto nodes = std::make_shared<Node>();
  graph->AddOutputNode(nodes);
  EXPECT_EQ(graph->GetOutputSize(), 1);
}

TEST_F(UtestComputeGraph, GetInputSize_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto nodes = std::make_shared<Node>();
  graph->AddInputNode(nodes);
  EXPECT_EQ(graph->GetInputSize(), 1);
}

TEST_F(UtestComputeGraph, Set_GetNeedIteration_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  graph->SetNeedIteration(true);
  EXPECT_EQ(graph->GetNeedIteration(), true);
}

TEST_F(UtestComputeGraph, UpdateInputMapping_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto opdesc = std::make_shared<OpDesc>(ATTR_NAME_PARENT_NODE_INDEX, DATA);
  opdesc->AddInputDesc("name1", tensor_desc->Clone());
  opdesc->AddOutputDesc("name2", tensor_desc->Clone());
  auto node = graph->AddNode(opdesc);
  ge::AttrUtils::SetInt(opdesc, ATTR_NAME_PARENT_NODE_INDEX, 1);

  graph->AddInputNode(node);
  std::map<uint32_t, uint32_t> input_mapping{{0,1}};
  EXPECT_EQ(graph->UpdateInputMapping(input_mapping), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, UpdateOutputMapping_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({1}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto opdesc = std::make_shared<OpDesc>(ATTR_NAME_PARENT_NODE_INDEX, NETOUTPUT);
  opdesc->AddInputDesc("name1", tensor_desc->Clone());
  opdesc->AddOutputDesc("name2", tensor_desc->Clone());
  auto node = graph->AddNode(opdesc);
  ge::AttrUtils::SetInt(opdesc, ATTR_NAME_PARENT_NODE_INDEX, 1);
  graph->AddOutputNode(node);
  std::map<uint32_t, uint32_t> output_mapping{{0,1}};
  EXPECT_EQ(graph->UpdateOutputMapping(output_mapping), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, ReorderEventNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode(ATTR_NAME_PARENT_NODE_INDEX, SEND, 1, 1);
  const auto &node2 = builder.AddNode(ATTR_NAME_PARENT_NODE_INDEX, RECV, 1, 1);
  const auto &node3 = builder.AddNode(ATTR_NAME_PARENT_NODE_INDEX, RECV, 1, 1);
  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);
  builder.AddControlEdge(node2, node3);
  auto graph = builder.GetGraph();

  EXPECT_EQ(graph->ReorderEventNodes(), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, DFSTopologicalSorting_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  std::vector<NodePtr> vec_nodes{node1, node2, node3};

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);

  std::vector<NodePtr> stack{};
  auto graph = builder.GetGraph();
  std::map<NodePtr, uint32_t> map_in_edge_num{};
  EXPECT_EQ(graph->DFSTopologicalSorting(vec_nodes, map_in_edge_num, stack, false),
    GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, BFSTopologicalSorting_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  std::vector<NodePtr> vec_nodes{node1, node2, node3};

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);

  std::deque<NodePtr> stack{};
  auto graph = builder.GetGraph();
  std::map<NodePtr, uint32_t> map_in_edge_num{};
  EXPECT_EQ(graph->BFSTopologicalSorting(vec_nodes, map_in_edge_num, stack), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, CollectBreadthOutNode_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node2, 0, node1, 0);
  builder.AddControlEdge(node2, node1);
  builder.AddControlEdge(node1, node3);
  std::map<NodePtr, uint32_t> map_in_edge_num{};
  std::map<std::string, NodePtr> breadth_node_map{};
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->CollectBreadthOutNode(node1, map_in_edge_num, breadth_node_map), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, TopologicalSorting_success) {
  const auto func = [](const NodePtr &node1, const NodePtr &node2) -> bool { return true; };
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  auto graph = builder.GetGraph();
  graph->TopologicalSorting(func);
  EXPECT_EQ(node1->GetOpDesc()->GetId(), 1);
  EXPECT_EQ(node2->GetOpDesc()->GetId(), 0);
}

TEST_F(UtestComputeGraph, SortNodes_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 1);
  const auto &node4 = builder.AddNode("node4", "node4", 1, 0);

  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);
  builder.AddControlEdge(node2, node4);
  auto graph = builder.GetGraph();
  std::map<NodePtr, uint32_t> map_in_edge_num{{node1, 2},{node2, 2},{node3, 2}};
  std::vector<NodePtr> stack{};
  EXPECT_EQ(graph->SortNodes(stack, map_in_edge_num), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, GetInEdgeSize_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 0, 1);
  builder.AddDataEdge(node2, 0, node1, 0);
  builder.AddDataEdge(node3, 0, node1, 1);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->GetInEdgeSize(node1), 2);
}

TEST_F(UtestComputeGraph, GetOutEdgeSize_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 1, 0);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 0);
  builder.AddDataEdge(node1, 0, node2, 0);
  builder.AddDataEdge(node1, 1, node3, 0);
  auto graph = builder.GetGraph();
  graph->Dump();
  EXPECT_EQ(graph->GetOutEdgeSize(node1), 2);
}

TEST_F(UtestComputeGraph, IsValid_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  EXPECT_EQ(graph->IsValid(), false);
}

TEST_F(UtestComputeGraph, InValid_success) {
  const auto func = [](const NodePtr &node1, const NodePtr &node2) -> bool { return true; };
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  auto graph = builder.GetGraph();
  graph->TopologicalSorting(func);
  EXPECT_EQ(graph->IsValid(), true);
  graph->InValid();
  EXPECT_EQ(graph->IsValid(), false);
}

TEST_F(UtestComputeGraph, Swap_success) {
  auto builder1 = ut::GraphBuilder("graph1");
  const auto &node1 = builder1.AddNode("node1", "node1", 0, 0);
  auto graph1 = builder1.GetGraph();
  auto builder2 = ut::GraphBuilder("graph2");
  const auto &node2 = builder2.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder2.AddNode("node3", "node3", 0, 0);
  auto graph2 = builder2.GetGraph();

  graph1->Swap(*(graph2));
  EXPECT_EQ(graph1->GetNodes(false).size(), 2);
  EXPECT_EQ(graph2->GetNodes(false).size(), 1);
  EXPECT_EQ(graph1->GetName(), "graph2");
  EXPECT_EQ(graph2->GetName(), "graph1");
}

TEST_F(UtestComputeGraph, InsertToNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node1", 0, 0);
  auto graph = builder.GetGraph();
  graph->InsertToNodeList(graph->impl_->nodes_.begin(), node3);
  EXPECT_EQ(*(graph->impl_->nodes_.begin()), node3);
}

TEST_F(UtestComputeGraph, PushBackToNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node3", 0, 0);
  auto graph = builder.GetGraph();
  graph->PushBackToNodeList(node1);
  auto node_list = graph->GetDirectNode();
  EXPECT_EQ(*(node_list.end() - 1), node1);
}

TEST_F(UtestComputeGraph, EmplaceBackToNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node1", 0, 0);
  auto graph = builder.GetGraph();
  graph->EmplaceBackToNodeList(node1);
  auto node_list = graph->GetDirectNode();
  EXPECT_EQ(*(node_list.end() - 1) , node1);
}

TEST_F(UtestComputeGraph, ClearNodeList_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 0);
  const auto &node3 = builder.AddNode("node3", "node1", 0, 0);
  auto graph = builder.GetGraph();
  graph->ClearNodeList();
  EXPECT_EQ(graph->GetDirectNode().size(), 0);
}

TEST_F(UtestComputeGraph, IsolateNode_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 2, 2);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 0);
  const auto &node4 = builder.AddNode("node4", "node4", 0, 1);
  const auto &node5 = builder.AddNode("node5", "node5", 1, 0);
  builder.AddDataEdge(node2, 0, node1, 0);
  builder.AddDataEdge(node1, 0, node3, 0);
  builder.AddControlEdge(node1, node4);
  builder.AddControlEdge(node5, node1);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->IsolateNode(node1), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, RemoveExtraOutEdge_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 1);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  const auto &node3 = builder.AddNode("node3", "node3", 1, 0);
  builder.AddControlEdge(node1, node2);
  builder.AddControlEdge(node3, node1);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->RemoveExtraOutEdge(node1), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, Verify_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 0, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->Verify(), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, InferOriginFormat_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->InferOriginFormat(), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, InferShapeInNeed_success) {
  auto builder = ut::GraphBuilder("graph");
  const auto &node1 = builder.AddNode("node1", "node1", 1, 0);
  const auto &node2 = builder.AddNode("node2", "node2", 0, 1);
  builder.AddDataEdge(node1, 0, node2, 0);
  auto graph = builder.GetGraph();
  EXPECT_EQ(graph->InferShapeInNeed(), GRAPH_SUCCESS);
}

TEST_F(UtestComputeGraph, SetSessionID_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto session_id = graph->GetSessionID() + 1;
  graph->SetSessionID(session_id);
  EXPECT_EQ(graph->GetSessionID(), session_id);
}

TEST_F(UtestComputeGraph, SetGraphID_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto graph_id = graph->GetGraphID() + 1;
  graph->SetGraphID(graph_id);
  EXPECT_EQ(graph->GetGraphID(), graph_id);
}

TEST_F(UtestComputeGraph, SetSummaryGraph_success) {
  auto graph = std::make_shared<ComputeGraph>("graph");
  auto summary_flag = !graph->IsSummaryGraph();
  graph->SetSummaryFlag(summary_flag);
  EXPECT_EQ(graph->IsSummaryGraph(), summary_flag);
}
}
