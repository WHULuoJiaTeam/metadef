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

#include "graph/tuning_utils.h"

#include "../debug/ge_util.h"
#include "../debug/ge_op_types.h"
#include "graph/utils/scope_guard.h"
#include "graph/node_impl.h"

namespace ge {
namespace {
const int kControlIndex = -1;
const std::string peer_node_name_attr = "_peerNodeName";
const std::string parent_node_name_attr = "_parentNodeName";
const std::string alias_name_attr = "_aliasName";
const std::string alias_indexes_attr = "_aliasIndexes";
const std::string parent_node_attr = "parentNode";
const std::string parent_node_anchor_index_attr = "_parentNodeAnchorIndex";
const std::string tuning_subgraph_prefix = "/aicore_subgraph_";
const std::string non_tuning_subgraph_prefix = "/subgraph_";
const std::set<std::string> kPartitionOpTypes = {PLACEHOLDER, END};
const std::set<std::string> kExeTypes = {DATA, CONSTANT, NETOUTPUT};
}
NodeNametoNodeNameMap TuningUtils::data_2_end_;
NodetoNodeNameMap TuningUtils::data_node_2_end_node_ ;
NodetoNodeMap TuningUtils::data_node_2_netoutput_node_;
NodeVec TuningUtils::netoutput_nodes_;
NodeVec TuningUtils::merged_graph_nodes_;
SubgraphCreateOutNode TuningUtils::create_output_;
std::mutex TuningUtils::mutex_;

std::string TuningUtils::PrintCheckLog() {
  std::stringstream ss;
  ss << "d2e:{";
  for (const auto &pair : data_2_end_) {
    ss << "data:" << pair.first << "-" << "end:" << pair.second;
    ss << " | ";
  }
  ss << "}";
  ss << "netoutputs:{";
  for (const auto &node : netoutput_nodes_) {
    ss << "netoutput:" << node->GetName();
    ss << " | ";
  }
  ss << "}";
  return ss.str();
}

std::string TuningUtils::GetNodeNameByAnchor(const Anchor *anchor) {
  if (anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "Anchor is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Anchor is nullptr");
    return "Null";
  }
  auto node = anchor->GetOwnerNode();
  return node == nullptr ? "Null" : node->GetName();
}

// part 1
graphStatus TuningUtils::ConvertGraphToFile(std::vector<ComputeGraphPtr> tuning_subgraphs,
                                            std::vector<ComputeGraphPtr> non_tuning_subgraphs,
                                            bool exe_flag, const std::string &path, const std::string &user_path) {
  int64_t i = 0;
  int64_t j = 0;
  std::lock_guard<std::mutex> lock(mutex_);
  for (auto &subgraph : tuning_subgraphs) {
    create_output_.emplace(subgraph, nullptr);
    auto help_info = HelpInfo{i, exe_flag, true, path, user_path};
    if (MakeExeGraph(subgraph, help_info) != SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Invoke][MakeExeGraph] TUU:subgraph %zu generate exe graph failed", i);
      return GRAPH_FAILED;
    }
    i++;
  }

  for (auto &subgraph : non_tuning_subgraphs) {
    create_output_.emplace(subgraph, nullptr);
    auto help_info = HelpInfo{j, true, false, path, user_path};
    if (MakeExeGraph(subgraph, help_info) != SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Invoke][MakeExeGraph] TUU:non tuning_subgraph %zu generate exe graph failed", j);
      return GRAPH_FAILED;
    }
    j++;
  }
  create_output_.clear();
  return SUCCESS;
}

// +---------------+
// | pld     pld   |
// |  \      /     |
// | relu relu     |
// |   \   /       |
// |   add         |
// |    |          |
// |   end         |
// +---------------+
//        |
//        |
//        V
// +---------------+
// | data   data   |
// |  \      /     |
// | relu relu     |
// |   \   /       |
// |   add         |
// |    |          |
// |  netoutput    |
// +---------------+
graphStatus TuningUtils::MakeExeGraph(ComputeGraphPtr &exe_graph,
                                      const HelpInfo& help_info) {
  GE_CHECK_NOTNULL(exe_graph);
  graphStatus ret = exe_graph->TopologicalSortingGraph(true);
  if (ret != SUCCESS) {
    GraphUtils::DumpGEGraphToOnnx(*exe_graph, "black_box");
    REPORT_CALL_ERROR("E19999", "TopologicalSortingGraph [%s] failed, saved to file black_box ret:%d.",
                      exe_graph->GetName().c_str(), ret);
    GELOGE(ret, "[Sort][Graph] Graph[%s] topological sort failed, saved to file black_box ret:%d.",
           exe_graph->GetName().c_str(), ret);
    return ret;
  }
  // clear graph id
  if (!AttrUtils::SetStr(*exe_graph, ATTR_NAME_SESSION_GRAPH_ID, "")) {
    REPORT_CALL_ERROR("E19999", "TUU:clear graph %s session_graph_id failed", exe_graph->GetName().c_str());
    GELOGE(FAILED, "[Invoke][SetStr] TUU:clear graph %s session_graph_id failed", exe_graph->GetName().c_str());
    return FAILED;
  }
  GELOGI("TUU:clear [%s] session_graph_id success", exe_graph->GetName().c_str());
  // if not make exe, just dump and return
  if (!help_info.exe_flag) {
    DumpGraphToPath(exe_graph, help_info.index, help_info.is_tuning_graph, help_info.path);
    GELOGI("TUU:just return, dump original sub_graph[%s]index[%ld]", exe_graph->GetName().c_str(), help_info.index);
    return SUCCESS;
  }
  // modify sub graph
  for (NodePtr &node : exe_graph->GetDirectNode()) {
    // 1.handle pld
    if (node->GetType() == PLACEHOLDER) {
      if (HandlePld(node) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "TUU:Failed to handle node %s from graph %s", node->GetName().c_str(),
                          exe_graph->GetName().c_str());
        GELOGE(FAILED, "[Invoke][HandlePld] TUU:Failed to handle node %s from graph %s", node->GetName().c_str(),
               exe_graph->GetName().c_str());
        return FAILED;
      }
    }
    // 2.handle end
    if (node->GetType() == END) {
      if (HandleEnd(node) != SUCCESS) {
        REPORT_CALL_ERROR("E19999", "TUU:Failed to handle node %s from graph %s", node->GetName().c_str(),
                          exe_graph->GetName().c_str());
        GELOGE(FAILED, "[Invoke][HandlePld] TUU:Failed to handle node %s from graph %s", node->GetName().c_str(),
               exe_graph->GetName().c_str());
        return FAILED;
      }
    }
  }
  ret = exe_graph->TopologicalSortingGraph(true);
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Graph[%s] topological sort failed, ret:%d.", exe_graph->GetName().c_str(), ret);
    GELOGE(ret, "[Sort][Graph] Graph [%s] topological sort failed, ret:%d.", exe_graph->GetName().c_str(), ret);
    return ret;
  }
  // dump subgraphs which modified by us
  if (help_info.user_path.empty()) {
    DumpGraphToPath(exe_graph, help_info.index, help_info.is_tuning_graph, help_info.path);
  } else {
    GraphUtils::DumpGEGraph(exe_graph, "", true, help_info.user_path);
  }
  return SUCCESS;
}

void TuningUtils::DumpGraphToPath(ComputeGraphPtr &exe_graph, int64_t index,
                                  bool is_tuning_graph, std::string path) {
  if (!path.empty()) {
    if (is_tuning_graph) {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + tuning_subgraph_prefix + std::to_string(index) + ".txt");
    } else {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + non_tuning_subgraph_prefix + std::to_string(index) + ".txt");
    }
  } else {
    path = "./";
    if (is_tuning_graph) {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + tuning_subgraph_prefix + std::to_string(index) + ".txt");
    } else {
      GraphUtils::DumpGEGraph(exe_graph, "", true, path + non_tuning_subgraph_prefix + std::to_string(index) + ".txt");
    }
  }
}

graphStatus TuningUtils::CreateDataNode(NodePtr &node, NodePtr &data_node) {
  auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  OpDescPtr data_op_desc;
  const vector<ge::GeTensorPtr> weight = OpDescUtils::MutableWeights(node);
  if (!weight.empty()) {
    data_op_desc = ComGraphMakeShared<OpDesc>(node->GetName(), CONSTANT);
  } else {
    data_op_desc = ComGraphMakeShared<OpDesc>(node->GetName(), DATA);
  }
  GE_CHECK_NOTNULL(data_op_desc);
  auto pld_op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(pld_op_desc);
  auto output_desc = pld_op_desc->GetOutputDesc(0); // only one output for pld and data
  // data inputdesc & outputdesc set as same
  if (data_op_desc->AddInputDesc(output_desc) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddInputDesc failed, TUU:data node %s", data_op_desc->GetName().c_str());
    GELOGE(FAILED, "[Add][InputDesc] failed, TUU:data node %s", data_op_desc->GetName().c_str());
    return FAILED;
  }
  if (data_op_desc->AddOutputDesc(output_desc) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddOutputDesc failed, TUU:data node %s", data_op_desc->GetName().c_str());
    GELOGE(FAILED, "[Add][OutputDesc] failed, TUU:data node %s", data_op_desc->GetName().c_str());
    return FAILED;
  }
  data_node = graph->AddNode(data_op_desc);
  GE_CHECK_NOTNULL(data_node);
  if (data_node->GetType() == CONSTANT) {
    if (OpDescUtils::SetWeights(data_node, weight) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "TUU:const node %s add weight failed", data_op_desc->GetName().c_str());
      GELOGE(FAILED, "[Set][Weights] TUU:const node %s add weight failed", data_op_desc->GetName().c_str());
      return FAILED;
    }
  }
  if (data_node->SetOwnerComputeGraph(graph) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SetOwnerComputeGraph failed, node:%s", node->GetName().c_str());
    GELOGE(FAILED, "[Set][OwnerComputeGraph] failed, node:%s", node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

graphStatus TuningUtils::AddAttrToDataNodeForMergeGraph(const NodePtr &pld, NodePtr &data_node) {
  auto op_desc = data_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);

  auto pld_desc = pld->GetOpDesc();
  GE_CHECK_NOTNULL(pld_desc);
  // inherit
  // a.  set `end's input node type` as attr
  std::string parent_op_type;
  if (!AttrUtils::GetStr(pld_desc, "parentOpType", parent_op_type)) {
    REPORT_CALL_ERROR("E19999", "TUU:pld %s get parentOpType failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] TUU:pld %s get parentOpType failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void) AttrUtils::SetStr(op_desc, "parentOpType", parent_op_type);
  // b. set `end's input node name` as attr
  std::string parent_op_name;
  if (!AttrUtils::GetStr(pld_desc, parent_node_name_attr, parent_op_name)) {
    REPORT_CALL_ERROR("E19999", "TUU:pld %s get _parentNodeName failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] TUU:pld %s get _parentNodeName failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void) AttrUtils::SetStr(op_desc, parent_node_name_attr, parent_op_name);
  // c. set `end's input node's out anchor index` as attr
  int parent_node_anchor_index;
  if (!AttrUtils::GetInt(pld_desc, "anchorIndex", parent_node_anchor_index)) {
    REPORT_CALL_ERROR("E19999", "TUU:pld %s get anchorIndex failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] TUU:pld %s get anchorIndex failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void) AttrUtils::SetInt(op_desc, parent_node_anchor_index_attr, parent_node_anchor_index);
  GELOGD("TUU:from node %s(%s) to add attr to node %s(%s) success",
         pld->GetName().c_str(), pld->GetType().c_str(), data_node->GetName().c_str(), data_node->GetType().c_str());
  // d. set `end node name` as attr
  std::string peer_end_name;
  if (!AttrUtils::GetStr(pld_desc, peer_node_name_attr, peer_end_name)) {
    REPORT_CALL_ERROR("E19999", "TUU:pld %s get _peerNodeName failed", pld_desc->GetName().c_str());
    GELOGE(FAILED, "[Invoke][GetStr] TUU:pld %s get _peerNodeName failed", pld_desc->GetName().c_str());
    return FAILED;
  }
  (void) AttrUtils::SetStr(op_desc, peer_node_name_attr, peer_end_name);
  GELOGD("TUU:from node %s(%s) to add attr to node %s(%s) success",
         pld->GetName().c_str(), pld->GetType().c_str(), data_node->GetName().c_str(), data_node->GetType().c_str());
  return SUCCESS;
}

graphStatus TuningUtils::ChangePld2Data(NodePtr &node, NodePtr &data_node) {
  auto type_pld = node->GetType();
  auto type_data = data_node->GetType();
  if (type_pld != PLACEHOLDER || kExeTypes.count(type_data) == 0) {
    REPORT_INNER_ERROR("E19999", "TUU:Failed to change node %s from type %s to type %s",
                       node->GetName().c_str(), type_pld.c_str(), type_data.c_str());
    GELOGE(FAILED, "[Check][Param] TUU:Failed to change node %s from type %s to type %s",
           node->GetName().c_str(), type_pld.c_str(), type_data.c_str());
    return FAILED;
  }
  auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  std::vector<int> output_map(node->GetAllOutDataAnchorsSize());
  for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    output_map[i] = static_cast<int>(i);
  }

  auto ret = GraphUtils::ReplaceNodeAnchors(data_node, node, {}, output_map);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "TUU:Failed to replace node %s by node %s, ret:%u",
                      node->GetName().c_str(), data_node->GetName().c_str(), ret);
    GELOGE(FAILED, "[Replace][Node] %s by node %s failed, ret:%u",
           node->GetName().c_str(), data_node->GetName().c_str(), ret);
    return FAILED;
  }

  NodeUtils::UnlinkAll(*node);

  ret = GraphUtils::RemoveNodeWithoutRelink(graph, node);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "TUU:Failed to remove node %s from graph:%s",
                      node->GetName().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Remove][Node] %s from graph:%s failed.", node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }

  GELOGD("TUU:Remove node %s(%s) by the ChangePld2Data process, replace it with node %s(%s)",
         node->GetName().c_str(), node->GetType().c_str(), data_node->GetName().c_str(), data_node->GetType().c_str());
  return ret;
}

graphStatus TuningUtils::HandlePld(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  if (HandleContinuousInputNodeNextData(node) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Handle][Node] TUU:Failed to handle continuous node next to data node:%s",
           node->GetName().c_str());
    return GRAPH_FAILED;
  }

  NodePtr data_node = nullptr;
  // 1. create data node
  if (CreateDataNode(node, data_node) != SUCCESS) {
    GELOGE(FAILED, "[Create][DataNode] TUU:Failed to handle node %s from graph %s",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  // 2. add necessary info to data_node for recovery whole graph
  if (AddAttrToDataNodeForMergeGraph(node, data_node) != SUCCESS) {
    GELOGE(FAILED, "[Add][Attr] TUU:Failed to handle node %s from graph %s",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  // 3. replace pld node by data node created before
  if (ChangePld2Data(node, data_node) != SUCCESS) {
    GELOGE(FAILED, "[Change][Pld2Data] TUU:Failed to handle node %s from graph %s",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  GELOGD("TUU:pld[%s] handle success", node->GetName().c_str());
  return SUCCESS;
}

graphStatus TuningUtils::CreateNetOutput(NodePtr &node, NodePtr &out_node) {
  GE_CHECK_NOTNULL(node);
  auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  auto search = create_output_.find(graph);
  if (search == create_output_.end()) {
    REPORT_INNER_ERROR("E19999", "TUU:node %s's owner sub graph %s not exist in create_output map",
                       node->GetName().c_str(), graph->GetName().c_str());
    GELOGE(FAILED, "[Check][Param] TUU:node %s's owner sub graph %s not exist in create_output map",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  if (search->second != nullptr) {
    out_node = search->second;
    GELOGD("TUU:sub graph %s has created output node, just return", graph->GetName().c_str());
    return SUCCESS;
  }
  auto out_op_desc = ComGraphMakeShared<OpDesc>(node->GetName(), NETOUTPUT);
  GE_CHECK_NOTNULL(out_op_desc);
  out_node = graph->AddNode(out_op_desc);
  GE_CHECK_NOTNULL(out_node);
  if (out_node->SetOwnerComputeGraph(graph) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "TUU:SetOwnerComputeGraph failed, graph:%s", graph->GetName().c_str());
    GELOGE(FAILED, "[Set][Graph] TUU:SetOwnerComputeGraph failed, graph:%s", graph->GetName().c_str());
    return FAILED;
  }
  create_output_[graph] = out_node;
  return SUCCESS;
}

graphStatus TuningUtils::AddAttrToNetOutputForMergeGraph(const NodePtr &end, NodePtr &out_node, int64_t index) {
  GE_CHECK_NOTNULL(end);
  GE_CHECK_NOTNULL(out_node);
  auto op_desc = out_node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  std::vector<std::string> alias_names = {};
  (void) AttrUtils::GetListStr(op_desc, alias_name_attr, alias_names);
  alias_names.push_back(end->GetName());
  (void) AttrUtils::SetListStr(op_desc, alias_name_attr, alias_names);

  std::vector<std::int64_t> indexes = {};
  (void) AttrUtils::GetListInt(op_desc, alias_indexes_attr, indexes);
  indexes.push_back(index);
  (void) AttrUtils::SetListInt(op_desc, alias_indexes_attr, indexes);

  return SUCCESS;
}

graphStatus TuningUtils::LinkEnd2NetOutput(NodePtr &end_node, NodePtr &out_node) {
  GE_CHECK_NOTNULL(end_node);
  GE_CHECK_NOTNULL(out_node);
  // get end in node is control node or normal node
  AnchorPtr end_in_anchor = (end_node->GetInDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                            ? Anchor::DynamicAnchorCast<Anchor>(end_node->GetInControlAnchor())
                            : Anchor::DynamicAnchorCast<Anchor>(end_node->GetInDataAnchor(0));
  GE_CHECK_NOTNULL(end_in_anchor);
  auto src_anchor = end_in_anchor->GetFirstPeerAnchor();  // src_anchor should be only 1
  GE_CHECK_NOTNULL(src_anchor);
  if (GraphUtils::RemoveEdge(src_anchor, end_in_anchor) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "TUU:remove end input edge from from %s(%d) to %s(%d) failed. "
                      "node_name:%s, graph_name:%s",
                      GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
                      GetNodeNameByAnchor(end_in_anchor.get()).c_str(), end_in_anchor->GetIdx(),
                      end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
    GELOGE(FAILED, "[Remove][Edge] TUU:remove end input edge from from %s(%d) to %s(%d) failed. "
           "node_name:%s, graph_name:%s", GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
           GetNodeNameByAnchor(end_in_anchor.get()).c_str(), end_in_anchor->GetIdx(),
           end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
    return FAILED;
  }
  // add edge between `end in node` and `out_node`
  if (src_anchor->IsTypeOf<OutDataAnchor>()) {
    std::shared_ptr<InDataAnchor>
        anchor = ComGraphMakeShared<InDataAnchor>(out_node, out_node->GetAllInDataAnchors().size());
    GE_CHECK_NOTNULL(anchor);
    GE_CHECK_NOTNULL(out_node->impl_);
    out_node->impl_->in_data_anchors_.push_back(anchor);
    if (GraphUtils::AddEdge(src_anchor, anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "TUU:add edge from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
                        GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
                        GetNodeNameByAnchor(anchor.get()).c_str(), anchor->GetIdx(),
                        end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
      GELOGE(FAILED, "[Add][Edge] from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
             GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
             GetNodeNameByAnchor(anchor.get()).c_str(), anchor->GetIdx(),
             end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
    auto end_op_desc = end_node->GetOpDesc();
    GE_CHECK_NOTNULL(end_op_desc);
    auto out_node_op_desc = out_node->GetOpDesc();
    GE_CHECK_NOTNULL(out_node_op_desc);
    // end node always has one input
    if (out_node_op_desc->AddInputDesc(end_op_desc->GetInputDesc(0)) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "TUU:node %s add input desc failed.", out_node_op_desc->GetName().c_str());
      GELOGE(FAILED, "[Add][InputDesc] failed, TUU:node %s .", out_node_op_desc->GetName().c_str());
      return FAILED;
    }
    // add necessary info to out_node for recovery whole graph
    if (AddAttrToNetOutputForMergeGraph(end_node, out_node, anchor->GetIdx()) != SUCCESS) {
      GELOGE(FAILED, "[Add][Attr] TUU:Failed to handle node %s from graph %s",
             end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
  } else if (src_anchor->IsTypeOf<OutControlAnchor>()) {
    OpDescPtr noop = nullptr;
    noop = ComGraphMakeShared<OpDesc>(end_node->GetName() + NOOP, NOOP);
    GE_CHECK_NOTNULL(noop);
    auto noop_node = end_node->GetOwnerComputeGraph()->AddNode(noop);
    GE_CHECK_NOTNULL(noop_node);
    auto out_in_anchor = out_node->GetInControlAnchor();
    if (GraphUtils::AddEdge(src_anchor, noop_node->GetInControlAnchor()) != GRAPH_SUCCESS ||
        GraphUtils::AddEdge(noop_node->GetOutControlAnchor(), out_in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "TUU:add edge from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
                        GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
                        GetNodeNameByAnchor(noop_node->GetInControlAnchor().get()).c_str(),
                        noop_node->GetInControlAnchor()->GetIdx(), end_node->GetName().c_str(),
                        end_node->GetOwnerComputeGraph()->GetName().c_str());
      GELOGE(FAILED, "[Add][Edge] from %s(%d) to %s(%d) failed. node_name:%s, graph_name:%s",
             GetNodeNameByAnchor(src_anchor.get()).c_str(), src_anchor->GetIdx(),
             GetNodeNameByAnchor(noop_node->GetInControlAnchor().get()).c_str(),
             noop_node->GetInControlAnchor()->GetIdx(), end_node->GetName().c_str(),
             end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
    // add necessary info to out_node for recovery whole graph
    if (AddAttrToNetOutputForMergeGraph(end_node, out_node, kControlIndex) != SUCCESS) {
      GELOGE(FAILED, "[Add][Attr] TUU:Failed to handle node %s from graph %s", end_node->GetName().c_str(),
             end_node->GetOwnerComputeGraph()->GetName().c_str());
      return FAILED;
    }
  } else {
    REPORT_INNER_ERROR("E19999", "TUU: node_name:%s, graph_name:%s handled failed",
                       end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
    GELOGE(FAILED, "[Handle][Node] TUU: node_name:%s, graph_name:%s handled failed",
           end_node->GetName().c_str(), end_node->GetOwnerComputeGraph()->GetName().c_str());
    return FAILED;
  }

  return SUCCESS;
}

graphStatus TuningUtils::ChangeEnd2NetOutput(NodePtr &end_node, NodePtr &out_node) {
  GE_CHECK_NOTNULL(end_node);
  GE_CHECK_NOTNULL(out_node);
  auto type_end = end_node->GetType();
  auto type_out = out_node->GetType();
  if (type_end != END || type_out != NETOUTPUT) {
    REPORT_INNER_ERROR("E19999", "TUU:Failed to change end_node %s from type %s to type %s",
                       end_node->GetName().c_str(), type_end.c_str(), type_out.c_str());
    GELOGE(FAILED, "[Check][Param] TUU:Failed to change end_node %s from type %s to type %s",
           end_node->GetName().c_str(), type_end.c_str(), type_out.c_str());
    return FAILED;
  }
  // link all `end nodes's in node` to this out_node
  if (LinkEnd2NetOutput(end_node, out_node) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][LinkEnd2NetOutput] failed, TUU:end_node [%s].", end_node->GetName().c_str());
    return FAILED;
  }
  // remove `end node`
  NodeUtils::UnlinkAll(*end_node);
  auto graph = end_node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  if (GraphUtils::RemoveNodeWithoutRelink(graph, end_node) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "TUU:end node [%s] RemoveNodeWithoutRelink failed.", end_node->GetName().c_str());
    GELOGE(FAILED, "[Remove][Node]TUU:end node [%s] RemoveNodeWithoutRelink failed.", end_node->GetName().c_str());
    return FAILED;
  }
  return SUCCESS;
}

graphStatus TuningUtils::HandleEnd(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  auto graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(graph);
  NodePtr out_node = nullptr;

  // 1. create net_output node , add only one NetOutput node to one subgraph
  if (CreateNetOutput(node, out_node) != SUCCESS) {
    GELOGE(FAILED, "[Create][NetOutput] TUU:Failed to handle node %s from graph %s",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  // 2. replace all end nodes by one output node created before
  if (ChangeEnd2NetOutput(node, out_node) != SUCCESS) {
    GELOGE(FAILED, "[Invoke][ChangeEnd2NetOutput] TUU:Failed to handle node %s from graph %s",
           node->GetName().c_str(), graph->GetName().c_str());
    return FAILED;
  }
  GELOGD("TUU:end[%s] handle success", node->GetName().c_str());
  return SUCCESS;
}

// part 2
graphStatus TuningUtils::ConvertFileToGraph(const map<int64_t, string> &options, ge::Graph &graph) {
  std::function<void()> callback = [&]() {
    data_2_end_.clear();
    data_node_2_end_node_.clear();
    data_node_2_netoutput_node_.clear();
    netoutput_nodes_.clear();
    merged_graph_nodes_.clear();
  };
  GE_MAKE_GUARD(release, callback);
  // 1. get all subgraph object
  std::vector<ComputeGraphPtr> graphs;
  // options format like {index:"subgraph_path"}
  for (const auto &pair : options) {
    ComputeGraphPtr compute_graph = ComGraphMakeShared<ComputeGraph>(std::to_string(pair.first));
    if (!ge::GraphUtils::LoadGEGraph(pair.second.c_str(), *compute_graph)) {
      REPORT_CALL_ERROR("E19999", "LoadGEGraph from file:%s failed", pair.second.c_str());
      GELOGE(FAILED, "[Load][Graph] from file:%s failed", pair.second.c_str());
    }
    graphs.push_back(compute_graph);
  }
  // 2. merge graph
  ComputeGraphPtr merged_graph = ComGraphMakeShared<ComputeGraph>("whole_graph_after_tune");
  GE_CHECK_NOTNULL(merged_graph);
  if (MergeAllSubGraph(graphs, merged_graph) != SUCCESS) {
    GELOGE(FAILED, "[Merge][Graph] failed");
    return FAILED;
  }
  // 3. set parent graph
  for (const auto &node : merged_graph->GetDirectNode()) {
    GE_CHECK_NOTNULL(node);
    if (node->SetOwnerComputeGraph(merged_graph) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "TUU:node %s set owner graph failed", node->GetName().c_str());
      GELOGE(FAILED, "[Set][Graph] TUU:node %s set owner graph failed", node->GetName().c_str());
      return FAILED;
    }
  }
  graph = GraphUtils::CreateGraphFromComputeGraph(merged_graph);
  return SUCCESS;
}

// +----------------------------------+
// | const const                      |
// |  \     /                         |
// | netoutput(end,end)               |
// +----------------------------------+
//         +
// +----------------------------------+
// | data(pld)   data(pld)            |
// |  \         /                     |
// | relu     relu                    |
// |   \      /                       |
// |    \   /                         |
// |    add                           |
// |     |                            |
// |  netoutput(end)                  |
// +----------------------------------+
//         +
// +----------------------------------+
// |  data(pld)                       |
// |      /                           |
// |  netoutput                       |
// +----------------------------------+
//        |
//        |
//        V
// +----------------------------------+
// | const     const                  |
// |  \         /                     |
// | relu     relu                    |
// |   \      /                       |
// |    \   /                         |
// |    add                           |
// |     |                            |
// |  netoutput                       |
// +----------------------------------+
graphStatus TuningUtils::MergeAllSubGraph(std::vector<ComputeGraphPtr> &subgraphs,
                                          ComputeGraphPtr &output_merged_compute_graph) {
  GE_CHECK_NOTNULL(output_merged_compute_graph);
  // 1. handle all subgraphs
  for (auto &subgraph : subgraphs) {
    Status ret_status = MergeSubGraph(subgraph);
    if (ret_status != SUCCESS) {
      GELOGE(ret_status, "[Invoke][MergeSubGraph] TUU:subgraph %s merge failed", subgraph->GetName().c_str());
      return ret_status;
    }
  }

  for (const auto &node: merged_graph_nodes_) {
    (void) output_merged_compute_graph->AddNode(node);
    GELOGD("TUU:graph %s add node %s success", output_merged_compute_graph->GetName().c_str(), node->GetName().c_str());

    vector<string> recover_attr_name;
    (void) ge::AttrUtils::GetListStr(node->GetOpDesc(), ATTR_NAME_NEED_RECOVER_ATTR, recover_attr_name);
    if (!recover_attr_name.empty()) {
      for (const auto &attr_name : recover_attr_name) {
        if (!ge::AttrUtils::SetBool(node->GetOpDesc(), attr_name, true)) {
          REPORT_CALL_ERROR("E19999", "Recover attr %s for node:%s failed.",
                            attr_name.c_str(), node->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Set][Bool]Recover attr %s for node:%s failed.",
                 attr_name.c_str(), node->GetName().c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }

  // 2. remove data and output node added by us
  if (RemoveDataNetoutputEdge(output_merged_compute_graph) != SUCCESS) {
    GELOGE(FAILED, "[Remove][Edge] TUU:Failed to merge graph %s", output_merged_compute_graph->GetName().c_str());
    return FAILED;
  }
  graphStatus ret = output_merged_compute_graph->TopologicalSorting();
  if (ret != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Graph[%s] topological sort failed, ret:%d.",
                      output_merged_compute_graph->GetName().c_str(), ret);
    GELOGE(ret, "[Sort][Graph] Graph[%s] topological sort failed, ret:%d.",
           output_merged_compute_graph->GetName().c_str(), ret);
    return ret;
  }
  GELOGD("TUU:Print-%s", PrintCheckLog().c_str());
  GELOGI("TUU:output_merged_compute_graph %s success", output_merged_compute_graph->GetName().c_str());
  return SUCCESS;
}

graphStatus TuningUtils::MergeSubGraph(ComputeGraphPtr &subgraph) {
  for (auto &node : subgraph->GetDirectNode()) {
    if (kPartitionOpTypes.count(node->GetType()) > 0) {
      REPORT_INNER_ERROR("E19999", "TUU:subgraph passed in should not contain nodes of end or pld type");
      GELOGE(FAILED, "[Check][Param] TUU:subgraph passed in should not contain nodes of end or pld type");
      return FAILED;
    }
    // handle data converted from pld node
    if (node->GetType() == DATA || node->GetType() == CONSTANT) {
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::string peer_out_name;
      bool has_valid_str =
          (AttrUtils::GetStr(op_desc, peer_node_name_attr, peer_out_name)) && (!peer_out_name.empty());
      if (has_valid_str) {
        std::lock_guard<std::mutex> lock(mutex_);
        data_2_end_.emplace(op_desc->GetName(), peer_out_name);
        data_node_2_end_node_.emplace(node, peer_out_name);
        continue;
      }
    }
    // handle netoutput converted from end node
    if (node->GetType() == NETOUTPUT) {
      auto op_desc = node->GetOpDesc();
      GE_CHECK_NOTNULL(op_desc);
      std::vector<string> out_alias_name;
      bool has_valid_str =
          (AttrUtils::GetListStr(op_desc, alias_name_attr, out_alias_name)) && (!out_alias_name.empty());
      if (has_valid_str) {
        std::lock_guard<std::mutex> lock(mutex_);
        netoutput_nodes_.emplace_back(node);
      }
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      merged_graph_nodes_.emplace_back(node);
    }
    GELOGD("TUU:subgraph %s add node %s success", subgraph->GetName().c_str(), node->GetName().c_str());
  }
  GELOGI("TUU:merge subgraph %s success", subgraph->GetName().c_str());
  return SUCCESS;
}

NodePtr TuningUtils::FindNode(const std::string &name, int64_t &in_index) {
  for (const auto &node : netoutput_nodes_) {
    if (node == nullptr) {
      continue;
    }
    std::vector<string> out_alias_name;
    std::vector<int64_t> alias_indexes;
    if (AttrUtils::GetListStr(node->GetOpDesc(), alias_name_attr, out_alias_name) &&
        AttrUtils::GetListInt(node->GetOpDesc(), alias_indexes_attr, alias_indexes) &&
        (out_alias_name.size() == alias_indexes.size())) {
      for (size_t i = 0; i < out_alias_name.size(); i++) {
        if (out_alias_name[i] == name) {
          in_index = alias_indexes[i];
          return node;
        }
      }
    }
  }
  return nullptr;
}

graphStatus TuningUtils::RemoveDataNetoutputEdge(ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph);
  // 1. traverse
  for (auto &pair : data_node_2_end_node_) {
    auto data_node = pair.first;
    GE_CHECK_NOTNULL(data_node);
    auto end_name = pair.second;
    int64_t index = 0;
    auto netoutput_node = FindNode(end_name, index);
    GELOGD("TUU:start to find info[%s][%s][%ld] ", data_node->GetName().c_str(), end_name.c_str(), index);
    GE_CHECK_NOTNULL(netoutput_node);
    data_node_2_netoutput_node_.emplace(data_node, netoutput_node);
    // 2. get `data out anchor` and `net output in anchor` and `net output in node's out anchor`
    AnchorPtr data_out_anchor = (data_node->GetOutDataAnchor(0)->GetFirstPeerAnchor() == nullptr)
                                ? Anchor::DynamicAnchorCast<Anchor>(data_node->GetOutControlAnchor())
                                : Anchor::DynamicAnchorCast<Anchor>(data_node->GetOutDataAnchor(0));
    AnchorPtr net_output_in_anchor = nullptr;
    AnchorPtr src_out_anchor = nullptr;
    if (index != kControlIndex) {
      net_output_in_anchor = netoutput_node->GetInDataAnchor(index);
      src_out_anchor = net_output_in_anchor->GetFirstPeerAnchor();
    } else {
      net_output_in_anchor = netoutput_node->GetInControlAnchor();
      for (const auto &out_ctrl : net_output_in_anchor->GetPeerAnchors()) {
        auto noop_node = out_ctrl->GetOwnerNode();
        GE_CHECK_NOTNULL(noop_node);
        if (noop_node->GetType() == NOOP && noop_node->GetName() == end_name + NOOP) {
          src_out_anchor = noop_node->GetInControlAnchor()->GetFirstPeerAnchor();
          // remove noop node
          NodeUtils::UnlinkAll(*noop_node);
          if (GraphUtils::RemoveJustNode(graph, noop_node) != SUCCESS) {
            REPORT_CALL_ERROR("E19999", "TUU:noop node [%s] RemoveNodeWithoutRelink failed.",
                              noop_node->GetName().c_str());
            GELOGE(FAILED, "[Remove][Node]TUU:noop node [%s] RemoveNodeWithoutRelink failed.",
                   noop_node->GetName().c_str());
            return FAILED;
          }
          break;
        }
      }
    }
    GELOGD("TUU:get out node:%s 's in anchor(%d) peer_src_node:%s 's out anchor(%d)  match info[%s][%s][%ld]",
           netoutput_node->GetName().c_str(), net_output_in_anchor->GetIdx(),
           src_out_anchor->GetOwnerNode()->GetName().c_str(), src_out_anchor->GetIdx(), data_node->GetName().c_str(),
           end_name.c_str(), index);

    // 3. relink
    // unlink netoutput_node with it's input in stage 4
    GE_CHECK_NOTNULL(data_out_anchor);
    for (const auto &peer_in_anchor : data_out_anchor->GetPeerAnchors()) {
      if (GraphUtils::RemoveEdge(data_out_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "[Remove][Edge] from %s(%d) to %s(%d) failed. "
                          "node_name:(data:%s;netoutput:%s), graph_name:%s",
                          GetNodeNameByAnchor(data_out_anchor.get()).c_str(), data_out_anchor->GetIdx(),
                          GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
                          data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        GELOGE(FAILED, "[Remove][Edge] from %s(%d) to %s(%d) failed. node_name:(data:%s;netoutput:%s), graph_name:%s",
               GetNodeNameByAnchor(data_out_anchor.get()).c_str(), data_out_anchor->GetIdx(),
               GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
               data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
      if (GraphUtils::AddEdge(src_out_anchor, peer_in_anchor) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "TUU:add edge from %s(%d) to %s(%d) failed. "
                          "node_name:(data:%s;netoutput:%s), graph_name:%s",
                          GetNodeNameByAnchor(src_out_anchor.get()).c_str(), src_out_anchor->GetIdx(),
                          GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
                          data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        GELOGE(FAILED, "[Add][Edge] from %s(%d) to %s(%d) failed. node_name:(data:%s;netoutput:%s), graph_name:%s",
               GetNodeNameByAnchor(src_out_anchor.get()).c_str(), src_out_anchor->GetIdx(),
               GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx(),
               data_node->GetName().c_str(), netoutput_node->GetName().c_str(), graph->GetName().c_str());
        return FAILED;
      }
    }
  }
  // 4. remove out nodes added by us
  for (auto &node: netoutput_nodes_) {
    NodeUtils::UnlinkAll(*node);
    if (GraphUtils::RemoveNodeWithoutRelink(graph, node) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "TUU:Failed to remove node %s from graph", node->GetName().c_str());
      GELOGE(FAILED, "[Remove][Node] %s from graph failed.", node->GetName().c_str());
      return FAILED;
    }
    GELOGD("TUU:Remove node %s by the RemoveDataNetoutputEdge process success", node->GetName().c_str());
  }
  return SUCCESS;
}

graphStatus TuningUtils::HandleContinuousInputNodeNextData(NodePtr &node) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_anchor : node->GetAllOutAnchors()) {
    for (const auto &peer_in_anchor : out_anchor->GetPeerAnchors()) {
      auto next_node = peer_in_anchor->GetOwnerNode();
      vector<string> remove_attr_names;
      bool is_no_padding_continuous_input = false;
      bool is_continuous_input = false;
      bool is_no_task = false;
      (void) ge::AttrUtils::GetBool(next_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, is_continuous_input);
      (void) ge::AttrUtils::GetBool(next_node->GetOpDesc(),
                                    ATTR_NAME_NOPADDING_CONTINUOUS_INPUT,
                                    is_no_padding_continuous_input);
      (void) ge::AttrUtils::GetBool(next_node->GetOpDesc(), ATTR_NAME_NOTASK, is_no_task);
      if (is_continuous_input) {
        if (!ge::AttrUtils::SetBool(next_node->GetOpDesc(), ATTR_NAME_CONTINUOUS_INPUT, false)) {
          REPORT_CALL_ERROR("E19999", "Remove attr ATTR_NAME_CONTINUOUS_INPUT for node:%s failed.",
                            next_node->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Remove][Attr] ATTR_NAME_CONTINUOUS_INPUT for node:%s failed.",
                 next_node->GetName().c_str());
          return GRAPH_FAILED;
        }
        remove_attr_names.emplace_back(ATTR_NAME_CONTINUOUS_INPUT);
      }
      if (is_no_padding_continuous_input) {
        if (!ge::AttrUtils::SetBool(next_node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, false)) {
          REPORT_CALL_ERROR("E19999", "Remove attr ATTR_NAME_NOPADDING_CONTINUOUS_INPUT for node:%s failed.",
                            next_node->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Remove][Attr] ATTR_NAME_NOPADDING_CONTINUOUS_INPUT for node:%s failed.",
                 next_node->GetName().c_str());
          return GRAPH_FAILED;
        }
        remove_attr_names.emplace_back(ATTR_NAME_NOPADDING_CONTINUOUS_INPUT);
      }
      if ((is_continuous_input || is_no_padding_continuous_input) && is_no_task) {
        if (!ge::AttrUtils::SetBool(next_node->GetOpDesc(), ATTR_NAME_NOTASK, false)) {
          REPORT_CALL_ERROR("E19999", "Remove attr ATTR_NAME_NOTASK for node:%s failed.",
                            next_node->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Remove][Attr] ATTR_NAME_NOTASK for node:%s failed.", next_node->GetName().c_str());
          return GRAPH_FAILED;
        }
        remove_attr_names.emplace_back(ATTR_NAME_NOTASK);
      }
      if (!remove_attr_names.empty()) {
        if (!ge::AttrUtils::SetListStr(next_node->GetOpDesc(),
                                       ATTR_NAME_NEED_RECOVER_ATTR,
                                       remove_attr_names)) {
          REPORT_CALL_ERROR("E19999", "Set attr ATTR_NAME_NEED_RECOVER_ATTR for node:%s failed.",
                            next_node->GetName().c_str());
          GELOGE(GRAPH_FAILED, "[Set][Attr] ATTR_NAME_NEED_RECOVER_ATTR for node:%s failed.",
                 next_node->GetName().c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}
}
