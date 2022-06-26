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

#include "utils/graph_utils.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <queue>
#include <atomic>
#include <mutex>

#include "./ge_context.h"
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "proto/ge_ir.pb.h"
#include "utils/attr_utils.h"
#include "utils/ge_ir_utils.h"
#include "utils/node_utils.h"
#include "utils/file_utils.h"
#include "graph/utils/dumper/ge_graph_dumper.h"
#include "debug/ge_op_types.h"
#include "external/ge/ge_api_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/compute_graph_impl.h"
#include "graph/op_desc_impl.h"
#include "mmpa/mmpa_api.h"

using google::protobuf::io::FileOutputStream;

namespace ge {
enum DumpGraphLevel {
  kDumpLevel1 = 1,
  kDumpLevel2 = 2,
  kDumpLevel3 = 3,
  kDumpLevelOther,
};

namespace{
const int32_t kBaseOfIntegerValue = 10;
#ifdef FMK_SUPPORT_DUMP
const char *const kDumpGeGraph = "DUMP_GE_GRAPH";
const int kDumpGraphIndexWidth = 8;
#endif

const char *const kNpuCollectPath = "NPU_COLLECT_PATH";
const char *const kDumpGraphPath = "DUMP_GRAPH_PATH";
const char *const kDumpGraphLevel = "DUMP_GRAPH_LEVEL";
const char *const kDumpStrBuild = "Build";
const char *const kDumpStrPartition = "partition";
const char *const kDumpStrOptimizeSubgraph = "OptimizeSubGraph";
const char *const kDumpStrSubgraphFunc = "sub_graph";
const char *const kDumpStrAicpu = "Aicpu";
const int32_t kNameMax = 255;
const int32_t kMaxRecursionDepth = 10;
};

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "addedge failed because param src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const AnchorPtr &src,
                                                                               const AnchorPtr &dst) {
  OutDataAnchorPtr src_data = Anchor::DynamicAnchorCast<OutDataAnchor>(src);
  InDataAnchorPtr dst_data = Anchor::DynamicAnchorCast<InDataAnchor>(dst);
  OutControlAnchorPtr src_control = Anchor::DynamicAnchorCast<OutControlAnchor>(src);
  InControlAnchorPtr dst_control = Anchor::DynamicAnchorCast<InControlAnchor>(dst);
  if ((src_data != nullptr) && (dst_data != nullptr) && (src_data->LinkTo(dst_data) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_data != nullptr) && (dst_control != nullptr) && (src_data->LinkTo(dst_control) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_control != nullptr) && (src_control->LinkTo(dst_control) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  if ((src_control != nullptr) && (dst_data != nullptr) && (src_control->LinkTo(dst_data) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src or dst is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const Format &src_format,
                                                                               const InDataAnchorPtr &dst,
                                                                               const Format &dst_format) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    auto ret = AnchorUtils::SetFormat(src, src_format);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "SetFormat failed, ret:%d, format:%d", ret, static_cast<int>(src_format));
      GELOGE(GRAPH_FAILED, "[Set][Format] failed, ret:%d, format is %d", ret, static_cast<int>(src_format));
      return ret;
    }
    ret = AnchorUtils::SetFormat(dst, dst_format);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "SetFormat failed, ret:%d, format is %d", ret, static_cast<int>(dst_format));
      GELOGE(GRAPH_FAILED, "[Set][Format] failed, ret:%d, format is %d", ret, static_cast<int>(dst_format));
      return ret;
    }
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutControlAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AddEdge(const OutDataAnchorPtr &src,
                                                                               const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->LinkTo(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "AddEdge failed because src is nullptr or run LinkTo failed.");
  GELOGE(GRAPH_FAILED, "[Add][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InDataAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const AnchorPtr &src,
                                                                                  const AnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutControlAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  REPORT_CALL_ERROR("E19999", "RemoveEdge failed because src is nullptr or run Unlink failed.");
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveEdge(const OutDataAnchorPtr &src,
                                                                                  const InControlAnchorPtr &dst) {
  if ((src != nullptr) && (src->Unlink(dst) == GRAPH_SUCCESS)) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Remove][Edge] Failed.");
  return GRAPH_FAILED;
}

graphStatus GraphUtils::ReplaceEdgeDst(const OutDataAnchorPtr &src, const InDataAnchorPtr &dst,
                                       const InDataAnchorPtr &new_dst) {
  if (RemoveEdge(src, dst) == GRAPH_SUCCESS && AddEdge(src, new_dst) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Replace][EdgeDst] Failed.");
  return GRAPH_FAILED;
}

graphStatus GraphUtils::ReplaceEdgeDst(const OutControlAnchorPtr &src, const InControlAnchorPtr &dst,
                                       const InControlAnchorPtr &new_dst) {
  if (RemoveEdge(src, dst) == GRAPH_SUCCESS && AddEdge(src, new_dst) == GRAPH_SUCCESS) {
    return GRAPH_SUCCESS;
  }
  GELOGE(GRAPH_FAILED, "[Replace][EdgeDst] Failed.");
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertNodeBetweenDataAnchors(
    const OutDataAnchorPtr &src, const InDataAnchorPtr &dst, const NodePtr &new_node) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(new_node);

  InDataAnchorPtr node_in_anchor = new_node->GetInDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_in_anchor != nullptr, GRAPH_FAILED,
                         "[Invoke][GetInDataAnchor] this node has not inDataAnchor");
  OutDataAnchorPtr node_out_anchor = new_node->GetOutDataAnchor(0);
  GE_CHK_BOOL_RET_STATUS(node_out_anchor != nullptr, GRAPH_FAILED,
                         "[Invoke][GetOutDataAnchor] this node has not outDataAnchor");
  GE_CHK_STATUS_RET(src->ReplacePeer(dst, node_in_anchor, node_out_anchor), "[Replace][Peer] Failed");
  return GRAPH_SUCCESS;
}


GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveSubgraphRecursively(const ComputeGraphPtr &compute_graph,
                                      const NodePtr &remove_node) {
  GE_CHECK_NOTNULL(compute_graph);
  if (remove_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param remove node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // Check if this node is belong to this compute graph, maybe a little slow
  const auto &all_nodes_in_graph = compute_graph->GetDirectNode();
  if (std::find(all_nodes_in_graph.begin(), all_nodes_in_graph.end(), remove_node) == all_nodes_in_graph.end()) {
    REPORT_INNER_ERROR("E19999", "Can not find node %s in graph %s.",
                       remove_node->GetName().c_str(), compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Can not find node %s in graph %s.",
           remove_node->GetName().c_str(), compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  // Find all subgraph of this node
  const auto &root_graph = GraphUtils::FindRootGraph(compute_graph);
  std::vector<ComputeGraphPtr> subgraphs;
  std::vector<NodePtr> all_nodes;
  std::deque<NodePtr> candidates;
  NodePtr remove_node_new = remove_node;
  candidates.emplace_back(remove_node_new);
  while (!candidates.empty()) {
    const NodePtr node = candidates.front();
    all_nodes.emplace_back(node);
    candidates.pop_front();

    OpDescPtr op_desc = node->GetOpDesc();
    if (op_desc == nullptr) {
      continue;
    }

    const auto &subgraph_names = op_desc->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
      auto subgraph = root_graph->GetSubgraph(*name_iter);
      if (subgraph != nullptr && subgraph->impl_ != nullptr) {
        subgraphs.emplace_back(subgraph);
        candidates.insert(candidates.begin(), subgraph->impl_->nodes_.begin(), subgraph->impl_->nodes_.end());
      }
    }
  }
  // Remove all subgraph
  for (const auto &remove_graph : subgraphs) {
    if (root_graph->RemoveSubGraph(remove_graph) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "RemoveSubGraph failed, sub graph name is %s, compute graph is %s.",
                        remove_node->GetName().c_str(), compute_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Remove][SubGraph] failed, sub graph name is %s, compute graph is %s.",
             remove_node->GetName().c_str(), compute_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::RemoveNodeWithoutRelink(const ComputeGraphPtr &compute_graph, const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(compute_graph->impl_);
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should not be null.");
    return GRAPH_FAILED;
  }

  // If the node save as input node, delete it
  (void)compute_graph->RemoveInputNode(node);

  // If the node save as output node, delete it
  (void)compute_graph->RemoveOutputNode(node);

  // If the node has sub-graphs, delete them
  auto ret = RemoveSubgraphRecursively(compute_graph, node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Remove][SubGraph] recursively failed.");
    return GRAPH_FAILED;
  }

  auto iter = find(compute_graph->impl_->nodes_.begin(), compute_graph->impl_->nodes_.end(), node);
  if (iter != compute_graph->impl_->nodes_.end()) {
    compute_graph->EraseFromNodeList(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

/// Add two edges to the new node, respectively connecting the SRC and DST
/// associated with the original edge
/// A ---> B transferred to  A ---> N ---> B
graphStatus InsertTransNode(ComputeGraph &compute_graph, const InDataAnchorPtr &in_data_anchor,
                            const std::vector<OpDescPtr> &vec_op_desc) {
  GE_CHECK_NOTNULL(in_data_anchor);
  for (const auto &op_desc : vec_op_desc) {
    GE_CHECK_NOTNULL(op_desc);

    auto ret = op_desc->AddInputDesc(GeTensorDesc());
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "AddInputDesc failed, op:%s", op_desc->GetName().c_str());
                     return GRAPH_FAILED, "[Add][InputDesc] failed, op:%s", op_desc->GetName().c_str());
    ret = op_desc->AddOutputDesc(GeTensorDesc());
    GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                     REPORT_CALL_ERROR("E19999", "AddOutputDesc failed, op:%s", op_desc->GetName().c_str());
                     return GRAPH_FAILED, "[Add][OnputDesc] failed, op:%s", op_desc->GetName().c_str());
    auto node_to_insert = compute_graph.AddNode(op_desc);

    GE_CHECK_NOTNULL(node_to_insert);
    GE_CHECK_NOTNULL(in_data_anchor->GetPeerOutAnchor());

    auto src = in_data_anchor->GetPeerOutAnchor()->GetOwnerNode();
    if (!src) {
      REPORT_INNER_ERROR("E19999", "peeroutanchor has no ownernode.");
      GELOGE(GRAPH_FAILED, "[Get][Node] src nullptr error.");
      return GRAPH_FAILED;
    }

    auto src_out_index = in_data_anchor->GetPeerOutAnchor()->GetIdx();

    auto dst = in_data_anchor->GetOwnerNode();
    if (!dst) {
      REPORT_INNER_ERROR("E19999", "in_data_anchor has no ownernode.");
      GELOGE(GRAPH_FAILED, "[Get][Node] dst nullptr error.");
      return GRAPH_FAILED;
    }

    auto dst_in_index = in_data_anchor->GetIdx();

    auto in_data_anchor_src_format = AnchorUtils::GetFormat(in_data_anchor->GetPeerOutAnchor());
    auto in_data_anchor_dst_format = AnchorUtils::GetFormat(in_data_anchor);

    GE_CHECK_NOTNULL(src->GetOutDataAnchor(src_out_index));
    GE_CHECK_NOTNULL(dst->GetInDataAnchor(dst_in_index));

    ret = GraphUtils::RemoveEdge(src->GetOutDataAnchor(src_out_index), dst->GetInDataAnchor(dst_in_index));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "RemoveEdge from %s to %s failed, ret:%d.",
                        src->GetName().c_str(), dst->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Remove][Edge] from %s to %s failed, ret:%d",
             src->GetName().c_str(), dst->GetName().c_str(), ret);
      return GRAPH_FAILED;
    }

    GE_CHECK_NOTNULL(node_to_insert->GetInDataAnchor(0));
    GE_CHECK_NOTNULL(node_to_insert->GetOutDataAnchor(0));

    ret = GraphUtils::AddEdge(src->GetOutDataAnchor(src_out_index), node_to_insert->GetInDataAnchor(0));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddEdge from %s to %s failed, ret:%d",
                        src->GetName().c_str(), node_to_insert->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             src->GetName().c_str(), node_to_insert->GetName().c_str(), ret);
      return ret;
    }
    ret = GraphUtils::AddEdge(node_to_insert->GetOutDataAnchor(0), dst->GetInDataAnchor(dst_in_index));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddEdge from %s to %s failed, ret:%d",
                        node_to_insert->GetName().c_str(), dst->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             node_to_insert->GetName().c_str(), dst->GetName().c_str(), ret);
      return ret;
    }

    if (op_desc->HasAttr("input_format")) {
      int64_t input_format = 0;
      int64_t output_format = 0;
      if (!AttrUtils::GetInt(op_desc, "input_format", input_format)) {
        GELOGW("[Get][Attr] Get attr input_format failed");
        continue;
      }
      if (!AttrUtils::GetInt(op_desc, "output_format", output_format)) {
        GELOGW("[Get][Attr] Get attr output_format failed");
        continue;
      }

      GE_CHECK_NOTNULL(node_to_insert->GetInDataAnchor(0)->GetPeerOutAnchor());
      GE_CHK_BOOL_RET_STATUS(node_to_insert->GetOutDataAnchor(0)->GetPeerInDataAnchors().empty(), GRAPH_FAILED,
                             "[Get][PeerInDataAnchors] of node:%s out data anchor(0) is empty, check invalid.",
                             node_to_insert->GetName().c_str());
      GE_CHECK_NOTNULL(node_to_insert->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0));

      auto status =
          AnchorUtils::SetFormat(node_to_insert->GetInDataAnchor(0)->GetPeerOutAnchor(), in_data_anchor_src_format);
      if (status != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "SetFormat:%d to node:%s in data anchor(0)'s peer anchor failed",
                          static_cast<int>(in_data_anchor_src_format), node_to_insert->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Format] %d to node:%s in data anchor(0)'s peer anchor failed",
               static_cast<int>(in_data_anchor_src_format), node_to_insert->GetName().c_str());
        return status;
      }
      status = AnchorUtils::SetFormat(node_to_insert->GetInDataAnchor(0), static_cast<Format>(input_format));
      if (status != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "SetFormat:%ld to node:%s in data anchor(0) failed",
                          input_format, node_to_insert->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Format] %ld to node:%s in data anchor(0) failed",
               input_format, node_to_insert->GetName().c_str());
        return status;
      }
      status = AnchorUtils::SetFormat(node_to_insert->GetOutDataAnchor(0), static_cast<Format>(output_format));
      if (status != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "SetFormat:%ld to node:%s out data anchor(0) failed",
                          output_format, node_to_insert->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Format] %ld to node:%s out data anchor(0) failed",
               output_format, node_to_insert->GetName().c_str());
        return status;
      }
      status = AnchorUtils::SetFormat(node_to_insert->GetOutDataAnchor(0)->GetPeerInDataAnchors().at(0),
                                      in_data_anchor_dst_format);
      if (status != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "SetFormat:%d to node:%s out data anchor(0)'s peer anchor failed",
                          static_cast<int>(in_data_anchor_dst_format), node_to_insert->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Set][Format] %d to node:%s out data anchor(0)'s peer anchor failed",
               static_cast<int>(in_data_anchor_dst_format), node_to_insert->GetName().c_str());
        return status;
      }
    }
    std::vector<ge::NodePtr> original_nodes;
    GraphUtils::RecordOriginalNames(original_nodes, node_to_insert);
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertTransNode(
    ComputeGraphPtr compute_graph, const InDataAnchorPtr &in_data_anchor, const std::vector<OpDescPtr> &vec_op_desc) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(in_data_anchor);
  graphStatus ret =
      ge::InsertTransNode(*compute_graph, in_data_anchor, vec_op_desc) == GRAPH_SUCCESS ? GRAPH_SUCCESS : GRAPH_FAILED;
  return ret;
}

///
/// @brief Insert node: src->insert_node:input_index, insert_node:output_index->dst
/// @param [in] src
/// @param [in] dsts
/// @param [in] insert_node
/// @param [in] input_index
/// @param [in] output_index
/// @return graphStatus
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::InsertNodeAfter(const OutDataAnchorPtr &src,
    const std::vector<InDataAnchorPtr> &dsts, const NodePtr &insert_node, uint32_t input_index, uint32_t output_index) {
  GE_CHECK_NOTNULL(src);
  GE_CHECK_NOTNULL(insert_node);

  NodePtr src_node = src->GetOwnerNode();
  if (src_node->GetOwnerComputeGraph() != insert_node->GetOwnerComputeGraph()) {
    REPORT_INNER_ERROR("E19999" "src:%s and insert_node:%s not exist in the same graph.",
                       src_node->GetName().c_str(), insert_node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] src:%s and insert_node:%s not exist in the same graph.",
           src_node->GetName().c_str(), insert_node->GetName().c_str());
    return GRAPH_FAILED;
  }

  if (AddEdge(src, insert_node->GetInDataAnchor(input_index)) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "AddEdge %s->%s failed.", src_node->GetName().c_str(), insert_node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][Edge] %s->%s failed.", src_node->GetName().c_str(), insert_node->GetName().c_str());
    return GRAPH_FAILED;
  }

  OutControlAnchorPtr src_out_ctrl_anchor = src_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(src_out_ctrl_anchor);

  bool ctrl_edge_flag = true;
  std::string type = NodeUtils::GetNodeType(src->GetOwnerNode());
  if ((type == SWITCH) || (type == REFSWITCH) || (type == SWITCHN)) {
    ctrl_edge_flag = false;
  }

  for (auto &dst : dsts) {
    GE_CHECK_NOTNULL(dst);
    NodePtr dst_node = dst->GetOwnerNode();
    GELOGI("Insert node %s between %s->%s.",
           insert_node->GetName().c_str(), src_node->GetName().c_str(), dst_node->GetName().c_str());
    if (src_node->GetOwnerComputeGraph() != dst_node->GetOwnerComputeGraph()) {
      REPORT_INNER_ERROR("E19999", "src:%s and dst:%s not exist in the same graph.",
                         src_node->GetName().c_str(), dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] src:%s and dst:%s not exist in the same graph.",
             src_node->GetName().c_str(), dst_node->GetName().c_str());
      return GRAPH_FAILED;
    }

    (void)RemoveEdge(src, dst);
    if (AddEdge(insert_node->GetOutDataAnchor(output_index), dst) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "ReplaceEdge from %s->%s to %s->%s failed.", src_node->GetName().c_str(),
                        dst_node->GetName().c_str(), insert_node->GetName().c_str(), dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Replace][Edge] from %s->%s to %s->%s failed.", src_node->GetName().c_str(),
             dst_node->GetName().c_str(), insert_node->GetName().c_str(), dst_node->GetName().c_str());
      return GRAPH_FAILED;
    }

    if (!ctrl_edge_flag) { continue; }
    for (const InControlAnchorPtr& peer_in_ctrl_anchor : src_out_ctrl_anchor->GetPeerInControlAnchors()) {
      if ((RemoveEdge(src_out_ctrl_anchor, peer_in_ctrl_anchor) != GRAPH_SUCCESS) ||
          (AddEdge(insert_node->GetOutControlAnchor(), peer_in_ctrl_anchor) != GRAPH_SUCCESS)) {
        REPORT_CALL_ERROR("E19999", "ReplaceEdge from %s->%s to %s->%s failed.",
                          src_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
                          insert_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Replace][Edge] from %s->%s to %s->%s failed.",
               src_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str(),
               insert_node->GetName().c_str(), peer_in_ctrl_anchor->GetOwnerNode()->GetName().c_str());
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus GraphUtils::InsertNodeBefore(const InDataAnchorPtr &dst,
                                         const NodePtr &insert_node,
                                         uint32_t input_index,
                                         uint32_t output_index) {
  GE_CHECK_NOTNULL(dst);
  GE_CHECK_NOTNULL(insert_node);
  auto dst_node = dst->GetOwnerNode();
  GE_CHECK_NOTNULL(dst_node);
  if (dst_node->GetOwnerComputeGraph() != insert_node->GetOwnerComputeGraph()) {
    GELOGE(GRAPH_FAILED, "[INSERT][NODE] dst:%s and insert_node:%s not exist in the same graph.",
           dst_node->GetName().c_str(), insert_node->GetName().c_str());
    return GRAPH_FAILED;
  }

  auto src_node_out_anchor = dst->GetPeerOutAnchor();
  GE_CHECK_NOTNULL(src_node_out_anchor);
  auto src_node = src_node_out_anchor->GetOwnerNode();
  GE_CHECK_NOTNULL(src_node);

  // insert node
  if ((RemoveEdge(src_node_out_anchor, dst) != GRAPH_SUCCESS) ||
      (AddEdge(src_node_out_anchor, insert_node->GetInDataAnchor(input_index)) != GRAPH_SUCCESS) ||
      (AddEdge(insert_node->GetOutDataAnchor(output_index), dst) != GRAPH_SUCCESS)) {
    GELOGE(GRAPH_FAILED, "[INSERT][NODE] %s between %s->%s failed",
           insert_node->GetName().c_str(),
           src_node->GetName().c_str(),
           dst_node->GetName().c_str());
    return GRAPH_FAILED;
  }
  GELOGI("[INSERT][NODE] %s between %s->%s",
         insert_node->GetName().c_str(),
         src_node->GetName().c_str(),
         dst_node->GetName().c_str());

  // update control edges
  auto in_ctrl_anchor = dst_node->GetInControlAnchor();
  GE_CHECK_NOTNULL(in_ctrl_anchor);
  auto insert_node_in_ctrl_anchor = insert_node->GetInControlAnchor();
  for (const auto &peer_out_ctrl_anchor : in_ctrl_anchor->GetPeerOutControlAnchors()) {
    auto peer_node = peer_out_ctrl_anchor->GetOwnerNode();
    auto node_type = NodeUtils::GetNodeType(peer_node);
    if (node_type == ATOMICADDRCLEAN) {
      continue;
    }
    if ((RemoveEdge(peer_out_ctrl_anchor, in_ctrl_anchor) != GRAPH_SUCCESS) ||
        (AddEdge(peer_out_ctrl_anchor, insert_node_in_ctrl_anchor) != GRAPH_SUCCESS)) {
      GELOGE(GRAPH_FAILED, "[INSERT][NODE] replace control edge from %s->%s to %s->%s failed.",
             peer_node != nullptr ? peer_node->GetName().c_str() : "NULL",
             dst_node->GetName().c_str(),
             peer_node != nullptr ? peer_node->GetName().c_str() : "NULL",
             insert_node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraph &compute_graph,
                                                                                      const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] The node ptr should be not null.");
    return GRAPH_FAILED;
  }
  if (compute_graph.impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "The compute graph impl should be not null, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] The compute graph impl should be not null.");
    return GRAPH_FAILED;
  }
  auto iter = find(compute_graph.impl_->nodes_.begin(), compute_graph.impl_->nodes_.end(), node);
  if (iter != compute_graph.impl_->nodes_.end()) {
    compute_graph.EraseFromNodeList(iter);
    return GRAPH_SUCCESS;
  }
  return GRAPH_FAILED;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::RemoveJustNode(ComputeGraphPtr compute_graph,
                                                                                      const NodePtr &node) {
  GE_CHECK_NOTNULL(compute_graph);
  GE_CHECK_NOTNULL(node);
  graphStatus ret = (RemoveJustNode(*compute_graph, node) == GRAPH_SUCCESS ? GRAPH_SUCCESS : GRAPH_FAILED);
  return ret;
}

void GraphUtils::RecordOriginalNames(std::vector<ge::NodePtr> original_nodes, const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
                   return, "[Check][Param] node is null.");
  std::vector<std::string> original_names;
  for (const auto &node_tmp : original_nodes) {
    std::vector<std::string> names_tmp;
    ge::OpDescPtr opdesc_tmp = node_tmp->GetOpDesc();
    if (opdesc_tmp == nullptr) {
      GELOGE(GRAPH_FAILED, "[Check][Param] Node %s get opdesc is nullptr", node_tmp->GetName().c_str());
      continue;
    }
    auto ret = ge::AttrUtils::GetListStr(opdesc_tmp, ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp);
    if (!ret) {
      GELOGW("[Get][Attr] Get attr _datadump_original_op_names failed");
      continue;
    }
    if (names_tmp.size() != 0) {
      original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
    } else {
      original_names.push_back(opdesc_tmp->GetName());
    }
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names),
                   REPORT_INNER_ERROR("E19999", "Set original_op_names to node:%s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_names to node:%s fail.", node->GetName().c_str());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::RecordOriginalNames(std::vector<std::string> names_tmp,
                                                                                    const ge::NodePtr &node) {
  GE_CHK_BOOL_EXEC(node != nullptr, REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
                   return, "[Check][Param] node is null.");
  std::vector<std::string> original_names;
  if (names_tmp.size() != 0) {
    original_names.insert(original_names.end(), names_tmp.begin(), names_tmp.end());
  } else {
    std::string tmp;
    original_names.push_back(tmp);
  }
  GE_CHK_BOOL_EXEC(ge::AttrUtils::SetListStr(node->GetOpDesc(), ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names),
                   REPORT_INNER_ERROR("E19999", "Set original_op_names to node %s fail.", node->GetName().c_str());
                   return, "[Invoke][SetListStr] Set original_op_names to node %s fail.", node->GetName().c_str());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::MatchDumpStr(const std::string &suffix) {
  char dump_level[MMPA_MAX_PATH ] = { 0x00 };
  INT32 res = mmGetEnv(kDumpGraphLevel, dump_level, MMPA_MAX_PATH);
  int64_t dump_graph_level = (res == EN_OK) ? std::strtol(dump_level, nullptr, kBaseOfIntegerValue) : kDumpLevel2;

  if (dump_graph_level == kDumpLevel1) {
    return false;
  }

  if (dump_graph_level == kDumpLevel2 && ((suffix.find(kDumpStrPartition) != std::string::npos) ||
                                          (suffix.find(kDumpStrOptimizeSubgraph) != std::string::npos) ||
                                          (suffix.find(kDumpStrAicpu) != std::string::npos) ||
                                          (suffix.find(kDumpStrSubgraphFunc) != std::string::npos))) {
    return true;
  }

  if (dump_graph_level == kDumpLevel3 && suffix.compare(kDumpStrBuild) != 0) {
    return true;
  }

  return false;
}

namespace {
void GetDumpGraphPrefix(std::stringstream& stream_file_name) {
  static std::string path_prefix;
  if (path_prefix.empty()) {
    char *npu_collect_path = std::getenv(kNpuCollectPath);
    char *dump_graph_path = std::getenv(kDumpGraphPath);
    if (npu_collect_path != nullptr) {
      std::string base_path_str(npu_collect_path);
      stream_file_name << base_path_str << "/extra-info/graph/" << mmGetPid() << "_" << GetContext().DeviceId() << "/";
    } else if (dump_graph_path != nullptr) {
      std::string dump_graph_path_str(dump_graph_path);
      stream_file_name << (dump_graph_path_str.empty() ? "" : dump_graph_path_str + "/");
    } else {
      stream_file_name << "./";
    }
    path_prefix = stream_file_name.str();
  } else {
    stream_file_name << path_prefix;
  }
}
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraph(const ge::ComputeGraphPtr &graph,
                                                                            const std::string &suffix,
                                                                            bool is_always_dump,
                                                                            const std::string &user_graph_name) {
#ifdef FMK_SUPPORT_DUMP
  GraphDumperRegistry::GetDumper().Dump(graph, suffix);
  static std::mutex mutex;
  char dump_ge_graph[MMPA_MAX_PATH] = { 0x00 };
  INT32 res = mmGetEnv(kDumpGeGraph, dump_ge_graph, MMPA_MAX_PATH);
  GE_IF_BOOL_EXEC(res != EN_OK && !is_always_dump, return;);

  // dump the graph according to different graph level
  if (GraphUtils::MatchDumpStr(suffix) && (!is_always_dump)) {
    return;
  }

  // file name
  static std::atomic_long atomic_file_index(0);
  auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump om txt: %ld", file_index);

  thread_local long max_dump_file_num = 0;
  if (max_dump_file_num == 0) {
    string opt = "0";
    (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_NUM, opt);
    max_dump_file_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if (max_dump_file_num != 0 && file_index > max_dump_file_num) {
    GELOGW("[DumpGraph][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
           file_index, max_dump_file_num);
    return;
  }

  std::stringstream stream_file_name;
  {
    std::lock_guard<std::mutex> lock(mutex);
    GetDumpGraphPrefix(stream_file_name);
    if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
      int32_t ret = CreateDirectory(stream_file_name.str());
      if (ret != 0) {
        GELOGW("[DumpGraph][CreateDirectory] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
        stream_file_name.str("");
        stream_file_name << "./";
      }
    }
  }

  stream_file_name << "ge_proto_" << std::setw(kDumpGraphIndexWidth) << std::setfill('0') << file_index;
  stream_file_name << "_" << suffix << ".txt";
  std::string proto_file = user_graph_name.empty() ? stream_file_name.str() : user_graph_name;

  // Create buffer
  ge::Model model("", "");
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(graph)));
  Buffer buffer;
  const int64_t kDumpLevel =
      (dump_ge_graph != nullptr) ? std::strtol(dump_ge_graph, nullptr, kBaseOfIntegerValue) : ge::OnnxUtils::NO_DUMP;
  model.Save(buffer, kDumpLevel != ge::OnnxUtils::DUMP_ALL && !is_always_dump);

  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    std::string str(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      GELOGE(GRAPH_FAILED, "[Invoke][Parse] parse from string failed.");
      return;
    }
    char real_path[MMPA_MAX_PATH] = {0x00};
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(proto_file.c_str()) >= MMPA_MAX_PATH,
                                   REPORT_INNER_ERROR("E19999", "file path is too longer! file:%s", proto_file.c_str());
                                   return, "[Check][Param] file path is too longer!");
    GE_IF_BOOL_EXEC(mmRealPath(proto_file.c_str(), real_path, MMPA_MAX_PATH) != EN_OK,
                    GELOGI("file %s does not exist, it will be created.", proto_file.c_str()));

    GraphUtils::WriteProtoToTextFile(ge_proto, real_path);
  }
#else
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGrph(const ge::ComputeGraphPtr &graph,
                                                                           const std::string &path,
                                                                           const std::string &suffix) {
  // file name
  static std::atomic_long atomic_file_index(0);
  auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump om txt: %ld", file_index);

  thread_local long max_dump_file_num = 0;
  if (max_dump_file_num == 0) {
    string opt = "0";
    (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_NUM, opt);
    max_dump_file_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if (max_dump_file_num != 0 && file_index > max_dump_file_num) {
    GELOGW("[DumpGraph][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
           file_index, max_dump_file_num);
    return;
  }

  std::stringstream stream_file_name;
  stream_file_name << path.c_str() << "/ge_proto_" << std::setw(5) << std::setfill('0')
                   << file_index;
  stream_file_name << "_" << suffix << ".txt";
  std::string proto_file = stream_file_name.str();

  // Create buffer
  ge::Model model("", "");
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(graph)));
  Buffer buffer;
  const int64_t kDumpLevel = ge::OnnxUtils::NO_DUMP;
  model.Save(buffer, kDumpLevel != ge::OnnxUtils::DUMP_ALL);

  // Write file
  ge::proto::ModelDef ge_proto;
  if (buffer.GetData() != nullptr) {
    std::string str(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize());
    if (!ge_proto.ParseFromString(str)) {
      GELOGE(GRAPH_FAILED, "[Invoke][Parse] parse from string failed.");
      return;
    }
    char real_path[MMPA_MAX_PATH] = {0x00};
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(proto_file.c_str()) >= MMPA_MAX_PATH,
                                   REPORT_INNER_ERROR("E19999", "file path is too longer! file:%s", proto_file.c_str());
                                   return, "[Check][Param] file path is too longer! file:%s", proto_file.c_str());
    GE_IF_BOOL_EXEC(mmRealPath(proto_file.c_str(), real_path, MMPA_MAX_PATH) != EN_OK,
                    GELOGI("file %s does not exist, it will be created.", proto_file.c_str()));

    GraphUtils::WriteProtoToTextFile(ge_proto, real_path);
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char *file,
                                                                            ge::ComputeGraph &compute_graph) {
  ge::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  if (!ReadProtoFromTextFile(file, &model_def)) {
    GELOGE(GRAPH_FAILED, "[Get][ModelDef] failed from file:%s", file);
    return false;
  }
  ge::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  if (model.Load(model_def) == GRAPH_SUCCESS) {
    GE_CHK_BOOL_EXEC(GraphUtils::GetComputeGraph(model.GetGraph()) != nullptr,
                     REPORT_INNER_ERROR("E19999", "Get computer graph is nullptr, model file:%s.", file);
                     return false, "[Get][ComputerGraph] is nullptr");
    compute_graph = *(GraphUtils::GetComputeGraph(model.GetGraph()));
    return true;
  } else {
    REPORT_CALL_ERROR("E19999", "Get Model failed from ModelDef:%s.", file);
    GELOGE(GRAPH_FAILED, "[Get][Model] failed from ModelDef:%s", file);
    return false;
  }
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraph(const char *file,
                                                                            ge::ComputeGraphPtr &compute_graph) {
  ge::proto::ModelDef model_def;
  // Get ModelDef object from file generated by DumpGEGraph()
  if (!ReadProtoFromTextFile(file, &model_def)) {
    GELOGE(GRAPH_FAILED, "[Get][ModelDef] failed from file:%s", file);
    return false;
  }
  ge::Model model;
  // Get Model object from ModelDef by deserialize ModelDef
  if (model.Load(model_def) == GRAPH_SUCCESS) {
    GE_CHK_BOOL_EXEC(GraphUtils::GetComputeGraph(model.GetGraph()) != nullptr,
                     REPORT_INNER_ERROR("E19999", "Get computer graph is nullptr, model file:%s.", file);
                     return false, "[Get][ComputerGraph] is nullptr");
    compute_graph = GraphUtils::GetComputeGraph(model.GetGraph());
    for (const auto &node : compute_graph->GetDirectNode()) {
      GELOGI("Node %s set owner graph", node->GetName().c_str());
      GE_CHECK_NOTNULL(node);
      if (node->SetOwnerComputeGraph(compute_graph) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E19999", "SetOwnerComputeGraph failed for node:%s", node->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Invoke][SetGraph]Node %s set owner graph failed", node->GetName().c_str());
        return false;
      }
    }
    return true;
  } else {
    REPORT_CALL_ERROR("E19999", "Get Model failed from ModelDef:%s.", file);
    GELOGE(GRAPH_FAILED, "[Get][Model] failed from ModelDef:%s", file);
    return false;
  }
}

// Printing protocol messages in text format is useful for debugging and human editing of messages.
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::WriteProtoToTextFile(
    const google::protobuf::Message &proto, const char *real_path) {
#ifdef FMK_SUPPORT_DUMP
  const int FILE_AUTHORITY = 0600;
  int fd = mmOpen2(real_path, M_WRONLY | M_CREAT | O_TRUNC, FILE_AUTHORITY);
  if (fd < 0) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(GRAPH_FAILED, "[Open][File] failed for %s, reason:%s", real_path, strerror(errno));
    return;
  }
  google::protobuf::io::FileOutputStream *output = new (std::nothrow) FileOutputStream(fd);
  if (output == nullptr) {
    REPORT_CALL_ERROR("E19999", "create FileOutputStream failed.");
    GELOGE(GRAPH_FAILED, "[Create][FileOutputStream] Output is nullptr");
    if (mmClose(fd) != 0) {
      REPORT_CALL_ERROR("E19999", "close FileOutputStream failed, reason:%s.", strerror(errno));
      GELOGE(GRAPH_FAILED, "[Close][FileOutputStream] failed, reason:%s", strerror(errno));
    }
    return;
  }
  bool ret = google::protobuf::TextFormat::Print(proto, output);
  if (!ret) {
    REPORT_CALL_ERROR("E19999", "write file:%s failed.", real_path);
    GELOGE(GRAPH_FAILED, "[Invoke][Print] Fail to write the file: %s", real_path);
    delete output;
    output = nullptr;
    GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                     REPORT_CALL_ERROR("E19999", "close FileOutputStream failed, reason:%s.", strerror(errno));
                     return, "[Close][FileOutputStream] failed, reason:%s", strerror(errno));
    return;
  }
  delete output;
  output = nullptr;
  GE_CHK_BOOL_EXEC(mmClose(fd) == 0,
                   REPORT_CALL_ERROR("E19999", "close FileOutputStream failed, reason:%s.", strerror(errno));
                   return, "[Close][FileOutputStream] failed, reason:%s.", strerror(errno));

  FILE *file = fopen(real_path, "rb");
  if (file == nullptr) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed, errormessage:%s", real_path, strerror(errno));
    GELOGE(GRAPH_FAILED, "[Invoke][FOpen] fail to open the file: %s, %s", real_path, strerror(errno));
    return;
  }
  if (fseek(file, 0L, SEEK_END) == 0) {
    long fileSize = ftell(file);
    thread_local long max_dump_file_size = 0;
    if (max_dump_file_size == 0) {
      string opt = "0";
      // Can not check return value
      (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_SIZE, opt);
      max_dump_file_size = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
    }
    if (max_dump_file_size != 0 && fileSize != -1 && fileSize > max_dump_file_size) {
      GELOGW("[WriteProto][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
             fileSize, max_dump_file_size);
      GE_IF_BOOL_EXEC(remove(real_path) != 0, GELOGW("[WriteProto][RemovePath] Remove path %s failed", real_path));
      GE_CHK_BOOL_EXEC(fclose(file) == 0,
                       REPORT_CALL_ERROR("E19999", "close file:%s failed, error:%s", real_path, strerror(errno));
                       return, "[FClose][File] %s failed error:%s", real_path, strerror(errno));
      return;
    }
  }
  GE_CHK_BOOL_EXEC(fclose(file) == 0,
                   REPORT_CALL_ERROR("E19999", "close file:%s failed error:%s", real_path, strerror(errno));
                   return, "[FClose][File] %s failed error:%s", real_path, strerror(errno));
#else
  GELOGW("[Write][Proto] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::ReadProtoFromTextFile(
    const char *file, google::protobuf::Message *proto) {
  if (file == nullptr || proto == nullptr) {
    REPORT_INNER_ERROR("E19999", "param file or proto is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] incorrect parameter. file path or message is invalid");
    return false;
  }
  std::ifstream fs(file, std::ifstream::in);
  if (!fs.is_open()) {
    REPORT_CALL_ERROR("E19999", "open file:%s failed.", file);
    GELOGE(GRAPH_FAILED, "[Invoke][OpenFile]proto file '%s' open fail.", file);
    return false;
  }
  google::protobuf::io::IstreamInputStream input(&fs);
  bool ret = google::protobuf::TextFormat::Parse(&input, proto);
  if (!ret) {
    REPORT_INNER_ERROR("E19999", "parse proto from text ret fail, please check your text file '%s'.", file);
    GELOGE(GRAPH_FAILED, "[Parse][Proto] from text ret fail, please check your text file '%s'.", file);
  }
  fs.close();
  return ret;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGEGraphToOnnx(const ge::ComputeGraph &compute_graph,
                                                                                  const std::string &suffix) {
#ifdef FMK_SUPPORT_DUMP
  char dump_ge_graph[MMPA_MAX_PATH] = { 0x00 };
  INT32 res = mmGetEnv(kDumpGeGraph, dump_ge_graph, MMPA_MAX_PATH);
  int64_t dump_ge_graph_level =
      (res == EN_OK) ? std::strtol(dump_ge_graph, nullptr, kBaseOfIntegerValue) : OnnxUtils::NO_DUMP;
  if ((dump_ge_graph_level == OnnxUtils::NO_DUMP) || (dump_ge_graph_level >= OnnxUtils::DUMP_LEVEL_END)) {
    GELOGD("Skip DumpGEGraphToOnnx with dump_ge_graph_level %ld.", dump_ge_graph_level);
    return;
  }

  // dump the graph according to different graph level
  if (GraphUtils::MatchDumpStr(suffix)) {
    return;
  }

  // 1.Get ge::onnx::ModelProto from ge::Model
  ge::Model model("GE", "");
  std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = ComGraphMakeShared<ge::ComputeGraph>(compute_graph);
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(compute_graph_ptr)));
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "[Convert][GeModel] DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  static std::atomic_long atomic_file_index(0);
  auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump ge onnx file: %ld", file_index);

  thread_local long max_dump_file_num = 0;
  if (max_dump_file_num == 0) {
    string opt = "0";
    (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_NUM, opt);
    max_dump_file_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if (max_dump_file_num != 0 && file_index > max_dump_file_num) {
    GELOGW("[DumpGraph][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
           file_index, max_dump_file_num);
    return;
  }

  std::stringstream stream_file_name;
  GetDumpGraphPrefix(stream_file_name);
  if (mmAccess2(stream_file_name.str().c_str(), M_F_OK) != EN_OK) {
    int32_t ret = CreateDirectory(stream_file_name.str());
    if (ret != 0) {
      GELOGW("[DumpGraph][CreateDirectory] Create dump graph dir failed, path:%s", stream_file_name.str().c_str());
      stream_file_name.str("");
      stream_file_name << "./";
    }
  }

  stream_file_name << "ge_onnx_" << std::setw(kDumpGraphIndexWidth) << std::setfill('0') << file_index;
  stream_file_name << "_graph_" << compute_graph.GetGraphID();
  stream_file_name << "_" << suffix << ".pbtxt";
  std::string proto_file = stream_file_name.str();
  if ((proto_file.length()) >= kNameMax) {
    GELOGE(GRAPH_FAILED, "[Check][Param] File name is too longer!, file:%s", proto_file.c_str());
    return;
  }
  std::unique_ptr<char[]> real_path(new (std::nothrow) char[MMPA_MAX_PATH]{0});
  GE_CHECK_NOTNULL_EXEC(real_path, return);

  /// Returning nullptr means 3 case as follows:
  /// a.path is MMPA_MAX_PATH chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  if (mmRealPath(proto_file.c_str(), real_path.get(), MMPA_MAX_PATH) != EN_OK) {
    // For case a
    int err_num = errno;
    // linux: ENAMETOOLONG windows: ERANGE
    if (err_num == ENAMETOOLONG || err_num == ERANGE) {
      REPORT_CALL_ERROR("E19999", "RealPath failed for file:%s. reason:%s", proto_file.c_str(), strerror(errno));
      GELOGE(GRAPH_FAILED, "[Real][Path] failed for file:%s. reason:%s", proto_file.c_str(), strerror(errno));
      return;
    }
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.get());
#else
  GELOGW("[DumpGraph][Check] Need to define FMK_SUPPORT_DUMP for dump graph.");
#endif
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY void GraphUtils::DumpGrphToOnnx(const ge::ComputeGraph &compute_graph,
                                                                               const std::string &path,
                                                                               const std::string &suffix) {
  // 1.Get ge::onnx::ModelProto from ge::Model
  ge::Model model("GE", "");
  std::shared_ptr<ge::ComputeGraph> compute_graph_ptr = ComGraphMakeShared<ge::ComputeGraph>(compute_graph);
  model.SetGraph(GraphUtils::CreateGraphFromComputeGraph(std::const_pointer_cast<ComputeGraph>(compute_graph_ptr)));
  onnx::ModelProto model_proto;
  if (!OnnxUtils::ConvertGeModelToModelProto(model, model_proto)) {
    GELOGE(GRAPH_FAILED, "[Convert][GeModel] DumpGEGraphToOnnx failed.");
    return;
  }

  // 2.Set file name
  static std::atomic_long atomic_file_index(0);
  auto file_index = atomic_file_index.fetch_add(1);
  GELOGD("Start to dump ge onnx file: %ld", file_index);

  thread_local long max_dump_file_num = 0;
  if (max_dump_file_num == 0) {
    string opt = "0";
    (void)GetContext().GetOption(OPTION_GE_MAX_DUMP_FILE_NUM, opt);
    max_dump_file_num = std::strtol(opt.c_str(), nullptr, kBaseOfIntegerValue);
  }
  if (max_dump_file_num != 0 && file_index > max_dump_file_num) {
    GELOGW("[DumpGraph][Check] dump_graph_num exceeds max_dump_file_num, dump_graph_num=%ld, max_dump_file_num=%ld",
           file_index, max_dump_file_num);
    return;
  }

  std::stringstream stream_file_name;
  stream_file_name << path.c_str() << "/ge_onnx_" << std::setw(5) << std::setfill('0') << file_index;
  stream_file_name << "_graph_" << compute_graph.GetGraphID();
  stream_file_name << "_" << suffix << ".pbtxt";
  std::string proto_file = stream_file_name.str();
  if ((proto_file.length()) >= kNameMax) {
    GELOGE(GRAPH_FAILED, "[Check][Param] File name is too longer!, file:%s", proto_file.c_str());
    return;
  }
  std::unique_ptr<char[]> real_path(new (std::nothrow) char[MMPA_MAX_PATH]{0});
  if (real_path == nullptr) {
    GELOGE(GRAPH_FAILED, "[New][RealPath] failed.");
    return;
  }
  /// Returning nullptr means 3 case as follows:
  /// a.path is PATH_MAX chars or more
  /// b.the file does not exist
  /// c.the path has no permissions
  /// Distinguish between last the two cases in the function WriteProtoToTextFile call open()
  if (mmRealPath(proto_file.c_str(), real_path.get(), MMPA_MAX_PATH) != EN_OK) {
    // For case a
    if (errno == ENAMETOOLONG) {
      REPORT_CALL_ERROR("E19999", "RealPath failed for file:%s. reason:%s", proto_file.c_str(), strerror(errno));
      GELOGE(GRAPH_FAILED, "[Real][Path] failed for file:%s. reason:%s", proto_file.c_str(), strerror(errno));
      return;
    }
  }

  // 3. Serialize to file in current path
  GraphUtils::WriteProtoToTextFile(model_proto, real_path.get());
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY bool GraphUtils::LoadGEGraphFromOnnx(const char *file,
                                                                                    ge::ComputeGraph &compute_graph) {
  if (file == nullptr) {
    REPORT_INNER_ERROR("E19999", "param file is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] incorrect parameter. file path is invalid");
    return false;
  }
  onnx::ModelProto model_proto;
  // 1. Get ModelDef object from file generated by DumpGEGraphToOnnx()
  if (!ReadProtoFromTextFile(file, &model_proto)) {
    GELOGE(GRAPH_FAILED, "[Get][ModelDef] from file:%s failed", file);
    return false;
  }
  // 2.Convert onnx::ModelProto To ge::Model
  ge::Model model;
  if (!OnnxUtils::ConvertModelProtoToGeModel(model_proto, model)) {
    REPORT_CALL_ERROR("E19999", "ConvertModelProtoToGeModel failed, file:%s", file);
    GELOGE(GRAPH_FAILED, "[Convert][ModelDef] to Model failed");
    return false;
  }
  auto compute_graph_ptr = GraphUtils::GetComputeGraph(model.GetGraph());
  if (compute_graph_ptr == nullptr) {
    REPORT_CALL_ERROR("E19999", "GetComputeGraph failed, file:%s", file);
    GELOGE(GRAPH_FAILED, "[Get][ComputeGraph] from Model failed");
    return false;
  }
  compute_graph = *(compute_graph_ptr);
  return true;
}

namespace {
using InNodesToOut = std::unordered_map<NodePtr, std::unordered_set<NodePtr>>;

inline std::string GetNodeNameByAnchor(const Anchor *anchor) {
  if (anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "param anchor is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Anchor is nullptr");
    return "Null";
  }
  auto node = anchor->GetOwnerNode();
  return node == nullptr ? "Null" : node->GetName();
}

graphStatus ReplaceOutDataAnchor(const OutDataAnchorPtr &new_anchor, const OutDataAnchorPtr &old_anchor,
                                 InNodesToOut *in_nodes_to_out = nullptr) {
  if (new_anchor == nullptr || old_anchor == nullptr) {
    REPORT_INNER_ERROR("E19999", "param new_anchor or old_anchor is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] new_anchor or old_anchor is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto new_node = new_anchor->GetOwnerNode();
  for (const auto &peer_in_anchor : old_anchor->GetPeerInDataAnchors()) {
    auto ret = peer_in_anchor->Unlink(old_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to unlink old anchor link from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
                        GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Remove][Link] Failed to unlink old anchor link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(old_anchor.get()).c_str(), old_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
    ret = peer_in_anchor->LinkFrom(new_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "[Create][Link] Failed to relink new anchors from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
                        GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Create][Link] Failed to relink new anchors from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(new_anchor.get()).c_str(), new_anchor->GetIdx(),
             GetNodeNameByAnchor(peer_in_anchor.get()).c_str(), peer_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }

    if (in_nodes_to_out != nullptr) {
      (*in_nodes_to_out)[new_node].insert(peer_in_anchor->GetOwnerNode());
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus RelinkDataIO(const NodePtr &node, const std::vector<int> &io_map, InNodesToOut &in_nodes_to_out) {
  GE_CHECK_NOTNULL(node);
  auto in_data_anchors = node->GetAllInDataAnchors();
  auto out_data_anchors = node->GetAllOutDataAnchors();
  if (out_data_anchors.size() < io_map.size()) {
    REPORT_INNER_ERROR("E19999", "param io_map size:%zu > the actual size:%zu, node:%s type:%s",
                       io_map.size(), out_data_anchors.size(), node->GetName().c_str(), node->GetType().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] The io_map specified for node %s type %s is larger %zu than "
           "the actual size %zu", node->GetName().c_str(), node->GetType().c_str(),
           io_map.size(), out_data_anchors.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0; i < out_data_anchors.size(); ++i) {
    auto out_data_anchor = out_data_anchors.at(i);
    if (out_data_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to relink for node %s type %s, the out data anchor "
                         "at index %zu is null", node->GetName().c_str(), node->GetType().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to relink for node %s type %s, the out data anchor "
             "at index %zu is null", node->GetName().c_str(), node->GetType().c_str(), i);
      return GRAPH_FAILED;
    }

    int in_index = -1;
    if (i < io_map.size()) {
      in_index = io_map.at(i);
    }
    if (in_index < 0) {
      out_data_anchor->UnlinkAll();
      continue;
    }

    if (in_index >= static_cast<int>(in_data_anchors.size())) {
      REPORT_INNER_ERROR("E19999", "Failed to relink for node %s type %s, invalid index %d specified for input(%zu)",
                         node->GetName().c_str(), node->GetType().c_str(), in_index, in_data_anchors.size());
      GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to relink for node %s type %s, invalid index %d "
             "specified for input(%zu)", node->GetName().c_str(), node->GetType().c_str(),
             in_index, in_data_anchors.size());
      return GRAPH_PARAM_INVALID;
    }
    auto in_anchor = in_data_anchors.at(in_index);
    if (in_anchor == nullptr) {
      GELOGW("[Relink][Check] %d\'th in_data_anchor of node %s type %s is null, ignore it.", in_index,
             node->GetName().c_str(), node->GetType().c_str());
      continue;
    }
    auto peer_out_anchor = in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (peer_out_anchor->Unlink(in_anchor) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed relink node %s type %s, failed to unlink the data link "
                        "from %s(%d) to it at input-index %d", node->GetName().c_str(), node->GetType().c_str(),
                        GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(), in_index);
      GELOGE(GRAPH_FAILED, "[Remove][Link] Failed relink node %s type %s, failed to unlink the data link "
             "from %s(%d) to it at input-index %d", node->GetName().c_str(), node->GetType().c_str(),
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(), in_index);
      return GRAPH_FAILED;
    }
    auto ret = ReplaceOutDataAnchor(peer_out_anchor, out_data_anchor, &in_nodes_to_out);
    if (ret != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Replace][OutDataAnchor] Failed to relink node %s type %s for relinking data anchors",
             node->GetName().c_str(), node->GetType().c_str());
      return GRAPH_FAILED;
    }
  }

  for (const auto &in_anchor : node->GetAllInDataAnchors()) {
    in_anchor->UnlinkAll();
  }
  return GRAPH_SUCCESS;
}

InNodesToOut GetFullConnectIONodes(const NodePtr &node) {
  InNodesToOut in_nodes_to_out;
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Node is nullptr");
    return in_nodes_to_out;
  }
  auto in_nodes_list = node->GetInNodes();
  auto out_nodes_list = node->GetOutNodes();
  auto out_nodes = std::unordered_set<NodePtr>(out_nodes_list.begin(), out_nodes_list.end());

  for (const auto &in_node : in_nodes_list) {
    in_nodes_to_out.insert(std::make_pair(in_node, out_nodes));
  }
  return in_nodes_to_out;
}

graphStatus RelinkControlNodeIfNeed(const NodePtr &node, InNodesToOut &in_nodes_to_out,
                                    InNodesToOut &connected_data_in_to_out) {
  GE_CHECK_NOTNULL(node);
  for (const auto &in_node_to_out : in_nodes_to_out) {
    auto &in_node = in_node_to_out.first;
    GE_CHECK_NOTNULL(in_node);
    auto &connected_data_out = connected_data_in_to_out[in_node];
    for (const auto &out_node : in_node_to_out.second) {
      GE_CHECK_NOTNULL(out_node);
      if (connected_data_out.count(out_node) == 0) {
        GE_CHECK_NOTNULL(in_node->GetOutControlAnchor());
        if (in_node->GetOutControlAnchor()->IsLinkedWith(out_node->GetInControlAnchor())) {
          continue;
        }
        // Some pass, such as SameTransdataBreadFusionPass will generate a ring, so add a
        // ring breaking operation here, and notice, this is an operation which will be
        // delete later, so do not use this interface to break a ring
        if (in_node == out_node) {
          GELOGW("[Relink][CtrlNode] There is a cycle between %s to %s when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          continue;
        }
        auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), out_node->GetInControlAnchor());
        if (ret != GRAPH_SUCCESS) {
          REPORT_CALL_ERROR("E19999", "Add ControlEdge from %s to %s failed, when isolating node %s type %s",
                            in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                            node->GetType().c_str());
          GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed, when isolating node %s type %s",
                 in_node->GetName().c_str(), out_node->GetName().c_str(), node->GetName().c_str(),
                 node->GetType().c_str());
          return GRAPH_FAILED;
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceOutDataAnchors(const Node::Vistor<OutDataAnchorPtr> &new_outs,
                                  const Node::Vistor<OutDataAnchorPtr> &old_outs, const std::vector<int> &outputs_map) {
  auto new_out_size = new_outs.size();
  if (new_out_size < outputs_map.size()) {
    REPORT_INNER_ERROR("E19999", "Failed to replace out data anchors, the actual size %zu is less than "
                       "the mapping size %zu", new_out_size, outputs_map.size());
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to replace out data anchors, the actual size %zu is less than "
           "the mapping size %zu", new_out_size, outputs_map.size());
    return GRAPH_PARAM_INVALID;
  }
  for (size_t i = 0; i < new_out_size; ++i) {
    auto &new_out_anchor = new_outs.at(i);
    if (new_out_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace out data anchors, "
                         "the out data anchor on new node is null, index %zu", i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace out data anchors, "
             "the out data anchor on new node is null, index %zu", i);
      return GRAPH_FAILED;
    }
    if (i >= outputs_map.size()) {
      continue;
    }
    auto old_index = outputs_map.at(i);
    if (old_index < 0) {
      continue;
    }

    const OutDataAnchorPtr &old_out_anchor = old_outs.at(old_index);
    if (old_out_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace out data anchors, "
                         "the out data anchor on old node is null, index %d", old_index);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace out data anchors, "
             "the out data anchor on old node is null, index %d", old_index);
      return GRAPH_FAILED;
    }
    auto ret = ReplaceOutDataAnchor(new_out_anchor, old_out_anchor);
    if (ret != GRAPH_SUCCESS) {
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

graphStatus ReplaceInDataAnchors(const Node::Vistor<InDataAnchorPtr> &new_ins,
                                 const Node::Vistor<InDataAnchorPtr> &old_ins, const std::vector<int> &inputs_map) {
  auto new_in_size = new_ins.size();
  if (new_in_size < inputs_map.size()) {
    REPORT_INNER_ERROR("E19999", "Failed to replace in data anchors, "
                       "the actual size %zu is less than the mapping size %zu", new_in_size, inputs_map.size());
    GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace in data anchors, "
           "the actual size %zu is less than the mapping size %zu", new_in_size, inputs_map.size());
    return GRAPH_PARAM_INVALID;
  }

  for (size_t i = 0; i < new_in_size; ++i) {
    auto &new_in_anchor = new_ins.at(i);
    if (new_in_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace in data anchors, "
                         "the out data anchor on new node is null, index %zu", i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace in data anchors, "
             "the out data anchor on new node is null, index %zu", i);
      return GRAPH_FAILED;
    }
    if (i >= inputs_map.size()) {
      continue;
    }
    auto old_index = inputs_map.at(i);
    if (old_index < 0) {
      continue;
    }
    const InDataAnchorPtr &old_in_anchor = old_ins.at(old_index);
    if (old_in_anchor == nullptr) {
      REPORT_INNER_ERROR("E19999", "Failed to replace in data anchors, "
                         "the out data anchor on old node is null, index %d", old_index);
      GELOGE(GRAPH_FAILED, "[Check][Param] Failed to replace in data anchors, "
             "the out data anchor on old node is null, index %d", old_index);
      return GRAPH_FAILED;
    }

    auto peer_out_anchor = old_in_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      continue;
    }
    auto ret = peer_out_anchor->Unlink(old_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                        GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Remove][Link] Failed to unlink old anchors, unlink from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
    ret = peer_out_anchor->LinkTo(new_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Failed to link new anchors, link from %s(%d) to %s(%d)",
                        GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
                        GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      GELOGE(GRAPH_FAILED, "[Create][Link]Failed to link new anchors, link from %s(%d) to %s(%d)",
             GetNodeNameByAnchor(peer_out_anchor.get()).c_str(), peer_out_anchor->GetIdx(),
             GetNodeNameByAnchor(old_in_anchor.get()).c_str(), old_in_anchor->GetIdx());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus ReplaceControlAnchors(const NodePtr &new_node, const NodePtr &old_node) {
  GE_CHECK_NOTNULL(new_node);
  GE_CHECK_NOTNULL(new_node->GetInControlAnchor());
  GE_CHECK_NOTNULL(old_node);
  GE_CHECK_NOTNULL(old_node->GetInControlAnchor());
  auto peer_out_anchors = old_node->GetInControlAnchor()->GetPeerAnchors();
  auto new_in_control_anchor = new_node->GetInControlAnchor();
  auto exists_out_anchors = new_in_control_anchor->GetPeerAnchors();
  auto exists_out_anchors_set = std::set<AnchorPtr>(exists_out_anchors.begin(), exists_out_anchors.end());
  for (const auto &peer_out_anchor : peer_out_anchors) {
    if (peer_out_anchor == nullptr) {
      continue;
    }
    if (exists_out_anchors_set.count(peer_out_anchor) > 0) {
      continue;
    }
    auto ret = GraphUtils::AddEdge(peer_out_anchor, new_in_control_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add edge from %s to %s failed, ret:%d",
                        peer_out_anchor->GetOwnerNode()->GetName().c_str(),
                        new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             peer_out_anchor->GetOwnerNode()->GetName().c_str(),
             new_in_control_anchor->GetOwnerNode()->GetName().c_str(), ret);
      return GRAPH_FAILED;
    }
  }
  auto old_out_control_anchor = old_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(old_out_control_anchor);
  auto peer_in_anchors = old_out_control_anchor->GetPeerAnchors();
  auto new_out_control_anchor = new_node->GetOutControlAnchor();
  GE_CHECK_NOTNULL(new_out_control_anchor);
  auto exists_in_anchors = new_out_control_anchor->GetPeerAnchors();
  auto exists_in_anchors_set = std::set<AnchorPtr>(exists_in_anchors.begin(), exists_in_anchors.end());
  for (const auto &peer_in_anchor : peer_in_anchors) {
    if (peer_in_anchor == nullptr) {
      continue;
    }
    if (exists_in_anchors_set.count(peer_in_anchor) > 0) {
      continue;
    }
    auto ret = GraphUtils::AddEdge(new_out_control_anchor, peer_in_anchor);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "AddEdge from %s to %s failed, ret:%d",
                        new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
                        peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
      GELOGE(GRAPH_FAILED, "[Add][Edge] from %s to %s failed, ret:%d",
             new_out_control_anchor->GetOwnerNode()->GetName().c_str(),
             peer_in_anchor->GetOwnerNode()->GetName().c_str(), ret);
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}
}  // namespace

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNode(const NodePtr &node,
                                                                                   const std::vector<int> &io_map) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Failed to isolate node(null)");
    return GRAPH_PARAM_INVALID;
  }

  /// We must get full connections info before re-link data io, because the data
  /// edges may be unlinked when relink data io
  auto in_nodes_to_out = GetFullConnectIONodes(node);

  InNodesToOut data_in_to_out;
  auto ret = RelinkDataIO(node, io_map, data_in_to_out);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Relink][DataIO] failed, node %s type %s", node->GetName().c_str(), node->GetType().c_str());
    return ret;
  }

  ret = RelinkControlNodeIfNeed(node, in_nodes_to_out, data_in_to_out);
  if (ret != GRAPH_SUCCESS) {
    return ret;
  }
  NodeUtils::UnlinkAll(*node);

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::IsolateNode(const NodePtr &node, const std::initializer_list<int> &io_map) {
  return IsolateNode(node, std::vector<int>(io_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::IsolateNodeOneIO(const NodePtr &node) {
  if (node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr, check invalid.");
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] incorrect parameter. node is invalid");
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllInDataAnchorsSize() != 1) {
    return GRAPH_PARAM_INVALID;
  }
  if (node->GetAllOutDataAnchorsSize() != 1) {
    return GRAPH_PARAM_INVALID;
  }
  return IsolateNode(node, {0});
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeAnchors(const NodePtr &new_node, const NodePtr &old_node, const std::vector<int> &inputs_map,
                               const std::vector<int> &outputs_map) {
  if ((new_node == nullptr) || (old_node == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param new_node or old_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto ret = ReplaceNodeDataAnchors(new_node, old_node, inputs_map, outputs_map);
  if (ret != GRAPH_SUCCESS) {
    // The error log was printed in `ReplaceNodeDataAnchors`
    return GRAPH_FAILED;
  }
  ret = ReplaceControlAnchors(new_node, old_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Replace][ControlAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", old_node->GetName().c_str(), old_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::ReplaceNodeAnchors(
    const NodePtr &new_node, const NodePtr &old_node, const std::initializer_list<int> inputs_map,
    const std::initializer_list<int> outputs_map) {
  return ReplaceNodeAnchors(new_node, old_node, std::vector<int>(inputs_map), std::vector<int>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node,
                                   std::initializer_list<int> inputs_map, std::initializer_list<int> outputs_map) {
  return ReplaceNodeDataAnchors(new_node, old_node, std::vector<int>(inputs_map), std::vector<int>(outputs_map));
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus
GraphUtils::ReplaceNodeDataAnchors(const NodePtr &new_node, const NodePtr &old_node, const std::vector<int> &inputs_map,
                                   const std::vector<int> &outputs_map) {
  if (new_node == nullptr || old_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param new_node or old_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }

  auto ret = ReplaceOutDataAnchors(new_node->GetAllOutDataAnchors(), old_node->GetAllOutDataAnchors(), outputs_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Replace][OutDataAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", old_node->GetName().c_str(), old_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  ret = ReplaceInDataAnchors(new_node->GetAllInDataAnchors(), old_node->GetAllInDataAnchors(), inputs_map);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Replace][InDataAnchors] failed when replace node from old node %s type %s "
           "to new node %s type %s", old_node->GetName().c_str(), old_node->GetType().c_str(),
           new_node->GetName().c_str(), new_node->GetType().c_str());
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyInCtrlEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto src_ctrl_in_nodes = src_node->GetInControlNodes();
  if (src_ctrl_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<NodePtr> exist_in_ctrl_nodes_set;
  auto exist_in_ctrl_nodes = dst_node->GetInControlNodes();
  if (!exist_in_ctrl_nodes.empty()) {
    exist_in_ctrl_nodes_set.insert(exist_in_ctrl_nodes.begin(), exist_in_ctrl_nodes.end());
  }

  auto dst_ctrl = dst_node->GetInControlAnchor();
  for (const auto &in_node : src_ctrl_in_nodes) {
    if (exist_in_ctrl_nodes_set.count(in_node) > 0) {
      continue;
    }
    auto ret = GraphUtils::AddEdge(in_node->GetOutControlAnchor(), dst_ctrl);
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add ControlEdge from %s to %s failed, when copy control dependencies from %s to %s",
                        in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
                        dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed, when copy control dependencies from %s to %s",
             in_node->GetName().c_str(), dst_node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveInCtrlEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_FAILED;
  }
  auto ret = CopyInCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][InCtrlEdges] failed, ret:%d, src_node:%s, dst_node:%s",
           ret, src_node->GetName().c_str(), dst_node->GetName().c_str());
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetInControlAnchor());
  src_node->GetInControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyOutCtrlEdges(const NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_FAILED;
  }
  auto out_ctrl_nodes = src_node->GetOutControlNodes();
  if (out_ctrl_nodes.empty()) {
    return GRAPH_SUCCESS;
  }

  std::unordered_set<Node *> exists_out_ctrl_nodes_set;
  for (const auto &node : dst_node->GetOutControlNodes()) {
    exists_out_ctrl_nodes_set.insert(node.get());
  }

  auto dst_out_ctrl = dst_node->GetOutControlAnchor();
  for (const auto &node : out_ctrl_nodes) {
    if (exists_out_ctrl_nodes_set.count(node.get()) > 0) {
      continue;
    }
    auto ret = GraphUtils::AddEdge(dst_out_ctrl, node->GetInControlAnchor());
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add ControlEdge from %s to %s failed when copy control dependencies from %s to %s",
                        dst_node->GetName().c_str(), node->GetName().c_str(), src_node->GetName().c_str(),
                        dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][ControlEdge] from %s to %s failed when copy control dependencies from %s to %s",
             dst_node->GetName().c_str(), node->GetName().c_str(), src_node->GetName().c_str(),
             dst_node->GetName().c_str());
      return ret;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::MoveOutCtrlEdges(NodePtr &src_node,
                                                                                        NodePtr &dst_node) {
  if (src_node == nullptr || dst_node == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Parameter is nullptr");
    return GRAPH_FAILED;
  }
  auto ret = CopyOutCtrlEdges(src_node, dst_node);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][OutCtrlEdges] failed, ret:%d", ret);
    return ret;
  }
  GE_CHECK_NOTNULL(src_node->GetOutControlAnchor());
  src_node->GetOutControlAnchor()->UnlinkAll();
  return GRAPH_SUCCESS;
}

///
/// Copy all in-data edges from `src_node` to `dst_node`.
/// @param src_node
/// @param dst_node
/// @return
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::CopyInDataEdges(const NodePtr &src_node,
                                                                                       NodePtr &dst_node) {
  if ((src_node == nullptr) || (dst_node == nullptr)) {
    REPORT_INNER_ERROR("E19999", "param src_node or dst_node is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param]] Parameter is nullptr");
    return GRAPH_PARAM_INVALID;
  }
  auto src_data_in_nodes = src_node->GetInDataNodes();
  if (src_data_in_nodes.empty()) {
    return GRAPH_SUCCESS;
  }
  for (const auto &in_data_anchor : src_node->GetAllInDataAnchors()) {
    auto input_desc = src_node->GetOpDesc()->GetInputDesc(in_data_anchor->GetIdx());
    auto ret =
        GraphUtils::AddEdge(in_data_anchor->GetPeerOutAnchor(), dst_node->GetInDataAnchor(in_data_anchor->GetIdx()));
    if (ret != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Add DataEdge from %s to %s failed when copy in data edge from %s to %s",
                        in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(),
                        dst_node->GetName().c_str(), src_node->GetName().c_str(), dst_node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][DataEdge] from %s to %s failed when copy in data edge from %s to %s",
             in_data_anchor->GetPeerOutAnchor()->GetOwnerNode()->GetName().c_str(), dst_node->GetName().c_str(),
             src_node->GetName().c_str(), dst_node->GetName().c_str());
      return ret;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::AppendInputNode(const ComputeGraphPtr &graph,
                                                                                       const NodePtr &node) {
  if (graph->AddInputNode(node) == nullptr) {
    REPORT_CALL_ERROR("E19999", "AddInputNode %s(%s) failed, graph:%s", node->GetName().c_str(),
                      node->GetType().c_str(), graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Add][InputNode] %s(%s) failed, graph:%s", node->GetName().c_str(),
           node->GetType().c_str(), graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  graph->SetInputSize(graph->GetInputSize() + 1);
  if (graph->impl_ == nullptr) {
    GELOGE(GRAPH_FAILED, "Graph impl is nullptr.");
    return GRAPH_FAILED;
  }
  graph->impl_->inputs_order_.emplace_back(node->GetName());
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ComputeGraphPtr GraphUtils::FindRootGraph(ComputeGraphPtr graph) {
  ComputeGraphPtr result = nullptr;
  while (graph != nullptr) {
    result = std::move(graph);
    graph = result->GetParentGraph();
  }
  return result;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyGraph(const Graph &src_graph, Graph &dst_graph) {
  std::string graph_name = dst_graph.GetName();
  if (graph_name.empty()) {
    graph_name = src_graph.GetName();
  }
  ComputeGraphPtr new_compute_graph = ComGraphMakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(new_compute_graph);
  ComputeGraphPtr src_compute_graph = GetComputeGraph(src_graph);
  GE_CHECK_NOTNULL(src_compute_graph);
  if (src_compute_graph->GetParentGraph() != nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][RootGraph] Only support copy root graph, current graph name:%s, "
                         "parent graph name:%s.", src_compute_graph->GetName().c_str(),
           src_compute_graph->GetParentGraph()->GetName().c_str());
    return GRAPH_FAILED;
  }
  int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  graphStatus ret = CopyComputeGraph(src_compute_graph, new_compute_graph,
                                     node_old_2_new, op_desc_old_2_new, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][Graph] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  Graph tmp_graph = CreateGraphFromComputeGraph(new_compute_graph);
  ret = CopyGraphImpl(src_graph, tmp_graph,
                      node_old_2_new, op_desc_old_2_new);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][GraphImpl] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  std::swap(dst_graph, tmp_graph);
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyOpAndSubgraph(const ComputeGraphPtr &src_compute_graph,
                                          ComputeGraphPtr &dst_compute_graph,
                                          std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                          std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                          std::unordered_map<std::string, NodePtr> &all_new_nodes,
                                          int32_t depth) {
  auto dst_root_compute_graph = FindRootGraph(dst_compute_graph);
  GE_CHECK_NOTNULL(dst_root_compute_graph);
  auto src_root_compute_graph = FindRootGraph(src_compute_graph);
  GE_CHECK_NOTNULL(src_root_compute_graph);
  for (const auto &n : src_compute_graph->GetDirectNode()) {
    OpDescPtr op_desc = AttrUtils::CopyOpDesc(n->GetOpDesc());
    if (op_desc == nullptr || op_desc->impl_ == nullptr) {
      REPORT_CALL_ERROR("E19999", "CopyOpDesc failed from node:%s", n->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Copy][OpDesc] from node:%s failed", n->GetName().c_str());
      return GRAPH_FAILED;
    }
    if (CopyTensorAttrs(op_desc, n) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Copy][TensorAttrs] from node:%s failed.", n->GetName().c_str());
      return GRAPH_FAILED;
    }

    if (n->GetType() == CONSTANT || n->GetType() == CONSTANTOP) {
      GeTensorPtr weight = nullptr;
      if (AttrUtils::MutableTensor(n->GetOpDesc(), ATTR_NAME_WEIGHTS, weight)) {
        GeTensor copy_weight = weight->Clone();
        if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, copy_weight)) {
          REPORT_CALL_ERROR("E19999", "copy ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
          GELOGE(INTERNAL_ERROR, "[Set][Tensor]copy ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
          return GRAPH_FAILED;
        }
        GELOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", op_desc->GetName().c_str());
      }
    }

    op_desc->SetName(n->GetName());
    NodePtr node = dst_compute_graph->AddNode(op_desc);
    if (node == nullptr) {
      REPORT_CALL_ERROR("E19999", "AddNode %s to graph:%s failed",
                        op_desc->GetName().c_str(), dst_compute_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Add][Node][%s] to graph:%s failed",
             op_desc->GetName().c_str(), dst_compute_graph->GetName().c_str());
      return GRAPH_FAILED;
    }
    all_new_nodes[node->GetName()] = node;
    node_old_2_new[n] = node;
    op_desc_old_2_new[n->GetOpDesc()] = op_desc;

    // copy subgraph from old graph to new graph
    const auto &subgraph_names = n->GetOpDesc()->GetSubgraphInstanceNames();
    for (auto name_iter = subgraph_names.rbegin(); name_iter != subgraph_names.rend(); ++name_iter) {
      auto src_subgraph = src_root_compute_graph->GetSubgraph(*name_iter);
      GE_CHECK_NOTNULL(src_subgraph);
      ComputeGraphPtr dst_subgraph = ComGraphMakeShared<ComputeGraph>(src_subgraph->GetName());
      GE_CHECK_NOTNULL(dst_subgraph);
      dst_subgraph->SetParentGraph(dst_compute_graph);
      std::map<ConstNodePtr, NodePtr> sub_node_old_2_new;
      std::map<ConstOpDescPtr, OpDescPtr> sub_op_desc_old_2_new;
      graphStatus ret = CopyComputeGraph(src_subgraph, dst_subgraph, sub_node_old_2_new,
                                         sub_op_desc_old_2_new, depth + 1);
      if (ret != GRAPH_SUCCESS) {
        GELOGE(GRAPH_FAILED, "[Copy][SubGraph] %s of parent node:%s failed.",
               src_subgraph->GetName().c_str(), node->GetName().c_str());
        return GRAPH_FAILED;
      }
      dst_root_compute_graph->AddSubGraph(dst_subgraph);
      dst_subgraph->SetParentNode(node);
      op_desc->impl_->subgraph_ir_names_to_type_ = n->GetOpDesc()->impl_->subgraph_ir_names_to_type_;
      op_desc->impl_->subgraph_names_to_index_ = n->GetOpDesc()->impl_->subgraph_names_to_index_;
      op_desc->impl_->subgraph_instance_names_ = n->GetOpDesc()->impl_->subgraph_instance_names_;
    }
  }
  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyComputeGraph(const ComputeGraphPtr &src_compute_graph,
                                         ComputeGraphPtr &dst_compute_graph,
                                         std::map<ConstNodePtr, NodePtr> &node_old_2_new,
                                         std::map<ConstOpDescPtr, OpDescPtr> &op_desc_old_2_new,
                                         int32_t depth) {
  GE_CHECK_NOTNULL(dst_compute_graph);
  GE_CHECK_NOTNULL(src_compute_graph);

  if (depth >= kMaxRecursionDepth) {
    REPORT_INNER_ERROR("E19999", "param depth:%d >= %d(allow max subgraphs)", depth, kMaxRecursionDepth);
    GELOGE(GRAPH_FAILED, "[Check][Param]exist too much subgraphs:%d > %d(allow max subgraphs)",
           depth, kMaxRecursionDepth);
    return GRAPH_FAILED;
  }
  // copy op and subgraph from old graph to new graph
  std::unordered_map<std::string, NodePtr> all_new_nodes;
  graphStatus ret = CopyOpAndSubgraph(src_compute_graph, dst_compute_graph,
                                      node_old_2_new, op_desc_old_2_new,
                                      all_new_nodes, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][OpAndSubGraph] failed.");
    return GRAPH_FAILED;
  }

  for (const auto &n : src_compute_graph->GetDirectNode()) {
    if (RelinkGraphEdges(n, "", all_new_nodes) != GRAPH_SUCCESS) {
      GELOGE(GRAPH_FAILED, "[Relink][Edges] failed.");
      return GRAPH_FAILED;
    }
  }
  // To keep subgraph consistent with the source graph
  std::vector<ComputeGraphPtr> new_subgraphs;
  auto old_subgraphs = src_compute_graph->GetAllSubgraphs();
  for (const auto &sub_graph : old_subgraphs) {
    auto new_subgraph = dst_compute_graph->GetSubgraph(sub_graph->GetName());
    GE_CHK_BOOL_EXEC(new_subgraph != nullptr, return GRAPH_FAILED,
                     "[Reorder][SubGraphs] can't find subgraph:%s in new graph.", sub_graph->GetName().c_str());
    GELOGD("Copy new subgraph:%s.", sub_graph->GetName().c_str());
    new_subgraphs.push_back(new_subgraph);
  }
  dst_compute_graph->SetAllSubgraphs(new_subgraphs);

  // copy members from old graph to new graph
  ret = CopyMembers(src_compute_graph, dst_compute_graph, all_new_nodes);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][Members] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
graphStatus GraphUtils::CopyMembers(const ComputeGraphPtr &src_compute_graph,
                                    ComputeGraphPtr &dst_compute_graph,
                                    const std::unordered_map<std::string, NodePtr> &all_new_nodes) {
  if (src_compute_graph == nullptr || src_compute_graph->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param src_compute_graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Src compute graph is nullptr.");
    return GRAPH_FAILED;
  }
  if (dst_compute_graph == nullptr || dst_compute_graph->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param dst_compute_graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Dst compute graph is nullptr.");
    return GRAPH_FAILED;
  }
  // copy info of output nodes from old graph to new graph.
  const std::vector<std::pair<NodePtr, int32_t>> &out_nodes_info = src_compute_graph->GetGraphOutNodesInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  for (const auto &info : out_nodes_info) {
    auto it = all_new_nodes.find(info.first->GetName());
    if (it == all_new_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find output node:%s failed.", info.first->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find output node:%s failed.", info.first->GetName().c_str());
      return GRAPH_FAILED;
    }
    new_out_nodes_info.emplace_back(it->second, info.second);
  }
  dst_compute_graph->SetGraphOutNodesInfo(new_out_nodes_info);

  // copy info of input nodes from old graph to new graph.
  const ComputeGraph::Vistor<NodePtr> &input_nodes = src_compute_graph->GetInputNodes();
  for (const auto &node : input_nodes) {
    auto it = all_new_nodes.find(node->GetName());
    if (it == all_new_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find input node:%s failed.", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find input node:%s failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    dst_compute_graph->AddInputNode(it->second);
  }

  // copy target info nodes from old graph to new graph.
  const std::vector<NodePtr> &src_traget_nodes_info = src_compute_graph->GetGraphTargetNodesInfo();
  std::vector<NodePtr> dst_traget_nodes_info;
  for (const auto &node : src_traget_nodes_info) {
    auto it = all_new_nodes.find(node->GetName());
    if (it == all_new_nodes.end()) {
      REPORT_INNER_ERROR("E19999", "Find target info node:%s failed.", node->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] Find target info node:%s failed.", node->GetName().c_str());
      return GRAPH_FAILED;
    }
    dst_traget_nodes_info.emplace_back(it->second);
  }
  dst_compute_graph->SetGraphTargetNodesInfo(dst_traget_nodes_info);

  // copy attr from old graph to new graph.
  std::shared_ptr<proto::GraphDef> graph_proto = ComGraphMakeShared<proto::GraphDef>();
  if (src_compute_graph->impl_->attrs_.GetProtoMsg() != nullptr) {
    *graph_proto->mutable_attr() = *src_compute_graph->impl_->attrs_.GetProtoMsg();
    dst_compute_graph->impl_->attrs_ = ProtoAttrMapHelper(graph_proto, graph_proto->mutable_attr());
  }

  // copy other members from old graph to new graph.
  dst_compute_graph->impl_->data_format_ = src_compute_graph->impl_->data_format_;
  dst_compute_graph->impl_->is_unknown_shape_graph_ = src_compute_graph->impl_->is_unknown_shape_graph_;
  dst_compute_graph->impl_->need_iteration_ = src_compute_graph->impl_->need_iteration_;
  dst_compute_graph->impl_->is_summary_graph_ = src_compute_graph->impl_->is_summary_graph_;
  dst_compute_graph->impl_->is_valid_flag_ = src_compute_graph->impl_->is_valid_flag_;
  dst_compute_graph->impl_->input_size_ = src_compute_graph->impl_->input_size_;
  dst_compute_graph->impl_->output_size_ = src_compute_graph->impl_->output_size_;
  dst_compute_graph->impl_->direct_nodes_size_ = src_compute_graph->impl_->direct_nodes_size_;
  dst_compute_graph->impl_->inputs_order_ = src_compute_graph->impl_->inputs_order_;
  dst_compute_graph->impl_->op_name_map_ = src_compute_graph->impl_->op_name_map_;
  dst_compute_graph->impl_->out_nodes_map_ = src_compute_graph->impl_->out_nodes_map_;
  dst_compute_graph->impl_->params_share_map_ = src_compute_graph->impl_->params_share_map_;

  return GRAPH_SUCCESS;
}

///
/// Make a copy of ComputeGraph.
/// @param graph: original graph.
/// @param prefix: node name prefix of new graph.
/// @param output_nodes: output nodes of new graph.
/// @return ComputeGraphPtr
///
GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
ComputeGraphPtr GraphUtils::CloneGraph(const ComputeGraphPtr &graph, const std::string &prefix,
                                       std::vector<NodePtr> &input_nodes, std::vector<NodePtr> &output_nodes) {
  GE_CHK_BOOL_EXEC(graph != nullptr, REPORT_INNER_ERROR("E19999", "param graph is nullptr, check invalid.");
                   return nullptr, "[Check][Param] Original graph is null");
  ComputeGraphPtr new_graph = ComGraphMakeShared<ComputeGraph>(graph->GetName());
  GE_CHK_BOOL_EXEC(new_graph != nullptr,
                   REPORT_CALL_ERROR("E19999", "create computegraph %s failed.", graph->GetName().c_str());
                   return nullptr, "[Create][ComputeGraph] %s failed", graph->GetName().c_str());

  std::unordered_map<std::string, NodePtr> all_new_nodes;
  for (const auto &n : graph->GetDirectNode()) {
    OpDescPtr op_desc = AttrUtils::CopyOpDesc(n->GetOpDesc());
    GE_CHK_BOOL_EXEC(op_desc != nullptr,
                     REPORT_CALL_ERROR("E19999", "Create node:%s failed.", n->GetOpDesc()->GetName().c_str());
                     return nullptr, "[Create][Node] %s failed", n->GetOpDesc()->GetName().c_str());

    if (CopyTensorAttrs(op_desc, n) != GRAPH_SUCCESS) {
      return nullptr;
    }

    if (n->GetType() == CONSTANT || n->GetType() == CONSTANTOP) {
      GeTensorPtr weight = nullptr;
      if (!AttrUtils::MutableTensor(n->GetOpDesc(), ATTR_NAME_WEIGHTS, weight)) {
        GELOGI("Can not find attr ATTR_NAME_WEIGHTS for node:%s.", n->GetName().c_str());
        continue;
      }
      GeTensor copy_weight = weight->Clone();
      if (!AttrUtils::SetTensor(op_desc, ATTR_NAME_WEIGHTS, copy_weight)) {
        REPORT_CALL_ERROR("E19999", "Clone ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
        GELOGE(INTERNAL_ERROR, "[Set][Tensor] Clone ATTR_NAME_WEIGHTS for node:%s failed.", op_desc->GetName().c_str());
        return nullptr;
      }
      GELOGD("Clone ATTR_NAME_WEIGHTS for node:%s success.", op_desc->GetName().c_str());
    }

    op_desc->SetName(n->GetName() + prefix);
    NodePtr node = new_graph->AddNode(op_desc);
    GE_CHK_BOOL_EXEC(node != nullptr,
                     REPORT_CALL_ERROR("E19999", "add node %s to graph:%s failed",
                                       op_desc->GetName().c_str(), new_graph->GetName().c_str());
                     return nullptr, "[Add][Node] [%s] to graph:%s failed",
                     op_desc->GetName().c_str(), new_graph->GetName().c_str());
    all_new_nodes[node->GetName()] = node;

    if (node->GetType() == DATA) {
      input_nodes.emplace_back(node);
    } else if (node->GetType() == NETOUTPUT) {
      output_nodes.emplace_back(node);
    }
  }

  for (const auto &n : graph->GetDirectNode()) {
    if (RelinkGraphEdges(n, prefix, all_new_nodes) != GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  std::string session_graph_id;
  if (AttrUtils::GetStr(*graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id)) {
    bool ret = AttrUtils::SetStr(*new_graph, ATTR_NAME_SESSION_GRAPH_ID, session_graph_id);
    if (!ret) {
      REPORT_CALL_ERROR("E19999", "set attr ATTR_NAME_SESSION_GRAPH_ID failed, ret:%d, graph:%s",
                        ret, new_graph->GetName().c_str());
      GELOGE(GRAPH_FAILED, "[Set][Attr] ATTR_NAME_SESSION_GRAPH_ID failed, ret:%d, graph:%s.",
             ret, new_graph->GetName().c_str());
      return nullptr;
    }
  }

  // copy info of output nodes from old graph to new graph.
  std::vector<std::pair<NodePtr, int32_t>> out_nodes_info = graph->GetGraphOutNodesInfo();
  std::vector<std::pair<NodePtr, int32_t>> new_out_nodes_info;
  for (const auto &info : out_nodes_info) {
    auto it = all_new_nodes.find(info.first->GetName());
    if (it != all_new_nodes.end()) {
      new_out_nodes_info.emplace_back(it->second, info.second);
    }
  }
  new_graph->SetGraphOutNodesInfo(new_out_nodes_info);
  return new_graph;
}

///
/// Copy tensor attribute to new node.
/// @param [in] dst_node: cloned node.
/// @param [in] src_node: original node.
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::CopyTensorAttrs(const OpDescPtr &dst_desc, const NodePtr &src_node) {
  if (dst_desc == nullptr) {
    REPORT_INNER_ERROR("E19999", "param dst_desc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input param dst node not valid");
    return GRAPH_FAILED;
  }
  if (src_node == nullptr || src_node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "param src_node is nullptr or it's opdesc is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input param src node not valid");
    return GRAPH_FAILED;
  }

  const auto &src_desc = src_node->GetOpDesc();
  dst_desc->CopyAttrsFrom(*src_desc);

  for (uint32_t i = 0; i < src_node->GetAllInDataAnchorsSize(); ++i) {
    auto input_desc = dst_desc->MutableInputDesc(i);
    if (input_desc == nullptr) {
      continue;
    }
    input_desc->CopyAttrsFrom(src_desc->GetInputDesc(i));
  }

  for (uint32_t i = 0; i < src_node->GetAllOutDataAnchorsSize(); ++i) {
    auto output_desc = dst_desc->MutableOutputDesc(i);
    if (output_desc == nullptr) {
      REPORT_INNER_ERROR("E19999", "Param dst node:%s not valid, output_desc[%d] is nullptr",
                         dst_desc->GetName().c_str(), i);
      GELOGE(GRAPH_FAILED, "[Check][Param] Param dst node:%s not valid", dst_desc->GetName().c_str());
      return GRAPH_FAILED;
    }
    output_desc->CopyAttrsFrom(src_desc->GetOutputDesc(i));
  }

  return GRAPH_SUCCESS;
}

///
/// Relink all edges for cloned ComputeGraph.
/// @param [in] node: original node.
/// @param [in] prefix: node name prefix of new node.
/// @param [in] all_nodes: all nodes in new graph.
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::RelinkGraphEdges(const NodePtr &node, const string &prefix,
                                         const std::unordered_map<string, NodePtr> &all_nodes) {
  if (node == nullptr || node->GetOpDesc() == nullptr) {
    REPORT_INNER_ERROR("E19999", "param node is nullptr or it's opdesc is nullptr. check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Input node not valid");
    return GRAPH_FAILED;
  }

  auto it = all_nodes.find(node->GetName() + prefix);
  if (it == all_nodes.end()) {
    REPORT_INNER_ERROR("E19999", "all_nodes not contain node:%s.", node->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] not found", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  const auto &new_node = it->second;

  // traversing from the parent node can be completely restored in the original one-to-many order.
  for (const auto &out_anchor : node->GetAllOutDataAnchors()) {
    GE_CHK_BOOL_EXEC(out_anchor != nullptr,
                     REPORT_INNER_ERROR("E19999", "out data anchor is null, node:%s.", node->GetName().c_str());
                     return GRAPH_FAILED, "[Check][Param] Out data anchor is null, node:%s", node->GetName().c_str());
    for (const auto &peer_in_anchor : out_anchor->GetPeerInDataAnchors()) {
      GE_CHECK_NOTNULL(peer_in_anchor);
      GE_CHK_BOOL_EXEC(peer_in_anchor->GetOwnerNode() != nullptr,
                       REPORT_INNER_ERROR("E19999", "Peer in node:%s is null", node->GetName().c_str());
                       return GRAPH_FAILED, "Peer in node:%s is null", node->GetName().c_str());
      it = all_nodes.find(peer_in_anchor->GetOwnerNode()->GetName() + prefix);
      if (it == all_nodes.end()) {
        REPORT_INNER_ERROR("E19999", "all_nodes not contain node[%s]",
                           peer_in_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] not found", peer_in_anchor->GetOwnerNode()->GetName().c_str());
        return GRAPH_FAILED;
      }
      const auto &new_peer_in_node = it->second;
      auto ret = GraphUtils::AddEdge(new_node->GetOutAnchor(out_anchor->GetIdx()),
                                     new_peer_in_node->GetInAnchor(peer_in_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "add data edge from %s to %s failed",
                                         new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
                       return GRAPH_FAILED, "[Invoke][AddEdge] link data edge failed[%s to %s]",
                       new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
    }
  }

  if (node->GetOutControlAnchor() != nullptr) {
    for (const auto &peer_in_control_anchor : node->GetOutControlAnchor()->GetPeerAnchors()) {
      GE_CHECK_NOTNULL(peer_in_control_anchor);
      GE_CHK_BOOL_EXEC(peer_in_control_anchor->GetOwnerNode() != nullptr,
                       REPORT_INNER_ERROR("E19999", "Peer out node is null");
                       return GRAPH_FAILED, "[Invoke][GetOwnerNode] Peer out node is null");
      it = all_nodes.find(peer_in_control_anchor->GetOwnerNode()->GetName() + prefix);
      if (it == all_nodes.end()) {
        REPORT_INNER_ERROR("E19999", "all_nodes not contain node:%s",
                           peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] node[%s] not found",
               peer_in_control_anchor->GetOwnerNode()->GetName().c_str());
        return GRAPH_FAILED;
      }
      const auto &new_peer_in_node = it->second;
      auto ret = GraphUtils::AddEdge(new_node->GetOutControlAnchor(),
                                     new_peer_in_node->GetInAnchor(peer_in_control_anchor->GetIdx()));
      GE_CHK_BOOL_EXEC(ret == GRAPH_SUCCESS,
                       REPORT_CALL_ERROR("E19999", "add control edge from %s to %s failed.",
                                         new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
                       return GRAPH_FAILED, "[Invoke][AddEdge] link control edge failed[%s to %s]",
                       new_node->GetName().c_str(), new_peer_in_node->GetName().c_str());
    }
  }
  return GRAPH_SUCCESS;
}

///
/// Get reference-mapping of all data_anchors in graph
/// @param [in] graph
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::GetRefMapping(const ComputeGraphPtr &graph,
                                      std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                      std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(graph);
  for (const auto &node : graph->GetAllNodes()) {
    // in_data_anchor
    if (HandleInAnchorMapping(graph, node, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Find ref_mapping for in_data_anchors of node %s failed.", node->GetName().c_str());
      GE_LOGE("[Invoke][HandleInAnchorMapping] Find ref_mapping for in_data_anchors of node %s failed.",
              node->GetName().c_str());
      return GRAPH_FAILED;
    }

    // out_data_anchor
    if (HandleOutAnchorMapping(node, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E19999", "Find ref_mapping for out_data_anchors of node %s failed.", node->GetName().c_str());
      GE_LOGE("[Invoke][HandleInAnchorMapping] Find ref_mapping for out_data_anchors of node %s failed.",
              node->GetName().c_str());
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY
NodePtr GraphUtils::FindNodeFromAllNodes(ComputeGraphPtr &graph, const std::string &name) {
  auto root_graph = FindRootGraph(graph);
  if (root_graph == nullptr) {
    REPORT_INNER_ERROR("E19999", "param graph is nullptr,check invalid.");
    GE_LOGE("[Check][Param] param graph is nullptr, check invalid");
    return nullptr;
  }

  for (const auto &node : root_graph->GetAllNodes()) {
    if (node == nullptr) {
      continue;
    }
    if (node->GetName() == name) {
      return node;
    }
  }

  return nullptr;
}

///
/// Get reference-mapping for in_data_anchors of node
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleInAnchorMapping(const ComputeGraphPtr &graph, const NodePtr &node,
                                              std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                              std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  if (node->GetOwnerComputeGraph()->GetName() != graph->GetName()) {
    // when curr graph is subgraph , to handle subgraph input/output ref mapping
    if (NodeUtils::IsSubgraphOutput(node)) {
      return HandleSubgraphOutput(node, symbol_to_anchors, anchor_to_symbol);
    }

    if (NodeUtils::IsSubgraphInput(node)) {
      return HandleSubgraphInput(node, symbol_to_anchors, anchor_to_symbol);
    }
  }

  const std::string &type = node->GetType();
  if ((type == MERGE) || (type == STREAMMERGE)) {
    return HandleMergeInput(node, symbol_to_anchors, anchor_to_symbol);
  }

  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    NodeIndexIO cur_node_info(node, in_data_anchor->GetIdx(), kIn);
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      const std::string &symbol = cur_node_info.ToString();
      GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
      symbol_to_anchors[symbol] = { cur_node_info };
      anchor_to_symbol[symbol] = symbol;
    } else {
      NodeIndexIO exist_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
      if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
        GE_LOGE("[Update][SymbolMapping] failed.");
        return GRAPH_FAILED;
      }
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Get reference-mapping for out_data_anchors of node
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleOutAnchorMapping(const NodePtr &node,
                                               std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                               std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  for (const auto &out_data_anchor : node->GetAllOutDataAnchors()) {
    NodeIndexIO cur_node_info(node, out_data_anchor->GetIdx(), kOut);
    if (anchor_to_symbol.find(cur_node_info.ToString()) != anchor_to_symbol.end()) {
      continue;
    }

    int32_t reuse_in_index = -1;
    bool reuse_input_flag = IsRefFromInput(out_data_anchor, reuse_in_index);
    if (reuse_input_flag && (node->GetInDataAnchor(reuse_in_index) != nullptr)) {
      NodeIndexIO exist_node_info(node, reuse_in_index, kIn);
      if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
        GE_LOGE("[Update][SymbolMapping] failed.");
        return GRAPH_FAILED;
      }
    } else {
      if (reuse_input_flag) {
        GELOGW("[GetRefMapping][Check] Invalid reuse_input attr on output %d of node %s, please check attr reuse_input "
               "and reuse_input_index", out_data_anchor->GetIdx(), node->GetName().c_str());
      }
      const std::string &symbol = cur_node_info.ToString();
      GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
      symbol_to_anchors.emplace(std::make_pair(symbol, std::list<NodeIndexIO>{ cur_node_info }));
      anchor_to_symbol.emplace(std::make_pair(symbol, symbol));
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Handle input of subgraph
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleSubgraphInput(const NodePtr &node,
                                            std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                            std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  GE_CHECK_NOTNULL(node->GetOpDesc());

  // Data in subgraph
  uint32_t index = 0;
  if (!ge::AttrUtils::GetInt(node->GetOpDesc(), ATTR_NAME_PARENT_NODE_INDEX, index)) {
    REPORT_CALL_ERROR("E19999", "Get  Attr ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    GE_LOGE("[Get][Attr] ATTR_NAME_PARENT_NODE_INDEX failed, node:%s.", node->GetName().c_str());
    return GRAPH_FAILED;
  }
  NodePtr parent_node = node->GetOwnerComputeGraph()->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);
  InDataAnchorPtr parent_in_anchor = parent_node->GetInDataAnchor(index);
  GE_CHECK_NOTNULL(parent_in_anchor);
  OutDataAnchorPtr peer_out_anchor = parent_in_anchor->GetPeerOutAnchor();
  if (peer_out_anchor != nullptr) {
    // Data has and only has one input
    NodeIndexIO cur_node_info(node, 0, kIn);
    NodeIndexIO exist_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
    if (UpdateRefMapping(cur_node_info, exist_node_info, symbol_to_anchors, anchor_to_symbol) != GRAPH_SUCCESS) {
      GE_LOGE("[Update][SymbolMapping] failed.");
      return GRAPH_FAILED;
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Handle input of Merge op
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleMergeInput(const NodePtr &node,
                                         std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                         std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  std::vector<NodeIndexIO> exist_node_infos;
  std::vector<NodeIndexIO> cur_node_infos;
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    auto peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    if (peer_out_anchor == nullptr) {
      std::string next_name;
      if (AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_NEXT_ITERATION, next_name) && !next_name.empty()) {
        ComputeGraphPtr graph = node->GetOwnerComputeGraph();
        GE_CHECK_NOTNULL(graph);
        ge::NodePtr next_node = FindNodeFromAllNodes(graph, next_name);
        GE_CHECK_NOTNULL(next_node);
        // NextIteration has and only has one output
        peer_out_anchor = next_node->GetOutDataAnchor(0);
        GE_CHECK_NOTNULL(peer_out_anchor);
        cur_node_infos.emplace_back(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn));
        cur_node_infos.emplace_back(NodeIndexIO(next_node, peer_out_anchor->GetIdx(), kOut));
      }
    } else {
      cur_node_infos.emplace_back(NodeIndexIO(node, in_data_anchor->GetIdx(), kIn));
      exist_node_infos.emplace_back(NodeIndexIO(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut));
    }
  }

  size_t anchor_nums = 0;
  NodeIndexIO max_node_index_io(nullptr, 0, kOut);
  for (const auto &temp_node_info : exist_node_infos) {
    auto iter1 = anchor_to_symbol.find(temp_node_info.ToString());
    if (iter1 != anchor_to_symbol.end()) {
      const std::string &temp_symbol = iter1->second;
      auto iter2 = symbol_to_anchors.find(temp_symbol);
      if (iter2 != symbol_to_anchors.end()) {
        if (iter2->second.size() > anchor_nums) {
          max_node_index_io = temp_node_info;
          anchor_nums = iter2->second.size();
        }
      }
    }
  }

  std::string symbol;
  for (const auto &temp_node_info : exist_node_infos) {
    if ((UnionSymbolMapping(max_node_index_io, temp_node_info, symbol_to_anchors, anchor_to_symbol, symbol) !=
         GRAPH_SUCCESS) ||
        symbol.empty()) {
      GE_LOGE("[Union][SymbolMap] anchor1:%s & anchor2:%s failed.", max_node_index_io.ToString().c_str(),
              temp_node_info.ToString().c_str());
      return GRAPH_FAILED;
    }
  }

  auto iter = symbol_to_anchors.find(symbol);
  if (iter != symbol_to_anchors.end()) {
    for (const auto &temp_node_info : cur_node_infos) {
      GELOGD("Add anchor %s, symbol %s.", temp_node_info.ToString().c_str(), symbol.c_str());
      iter->second.emplace_back(temp_node_info);
      anchor_to_symbol.emplace(std::make_pair(temp_node_info.ToString(), symbol));
    }
  }

  return GRAPH_SUCCESS;
}

///
/// Handle output of subgraph
/// @param [in] node
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::HandleSubgraphOutput(const NodePtr &node,
                                             std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                             std::map<std::string, std::string> &anchor_to_symbol) {
  GE_CHECK_NOTNULL(node);
  ComputeGraphPtr owner_graph = node->GetOwnerComputeGraph();
  GE_CHECK_NOTNULL(owner_graph);
  NodePtr parent_node = owner_graph->GetParentNode();
  GE_CHECK_NOTNULL(parent_node);

  OpDescPtr op_desc = node->GetOpDesc();
  GE_CHECK_NOTNULL(op_desc);
  for (const auto &in_data_anchor : node->GetAllInDataAnchors()) {
    OutDataAnchorPtr peer_out_anchor = in_data_anchor->GetPeerOutAnchor();
    GE_CHECK_NOTNULL(peer_out_anchor);

    GeTensorDesc in_tensor = op_desc->GetInputDesc(in_data_anchor->GetIdx());
    uint32_t index = 0;
    if (!ge::AttrUtils::GetInt(in_tensor, ATTR_NAME_PARENT_NODE_INDEX, index)) {
      continue;
    }
    GE_CHECK_NOTNULL(parent_node->GetOutDataAnchor(index));
    // Union symbol of peer_out_anchor & parent_out_anchor
    NodeIndexIO peer_node_info(peer_out_anchor->GetOwnerNode(), peer_out_anchor->GetIdx(), kOut);
    NodeIndexIO parent_node_info(parent_node, index, kOut);
    std::string symbol;
    if ((UnionSymbolMapping(peer_node_info, parent_node_info, symbol_to_anchors, anchor_to_symbol,
                            symbol) != GRAPH_SUCCESS) || symbol.empty()) {
      GE_LOGE("[Union][SymbolMap] anchor1:%s, and anchor2:%s failed.",
              peer_node_info.ToString().c_str(), parent_node_info.ToString().c_str());
      return GRAPH_FAILED;
    }

    NodeIndexIO cur_node_info(node, in_data_anchor->GetIdx(), kIn);
    GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
    symbol_to_anchors[symbol].emplace_back(cur_node_info);
    anchor_to_symbol.emplace(std::make_pair(cur_node_info.ToString(), symbol));
  }

  return GRAPH_SUCCESS;
}

///
/// Union ref-mapping
/// @param [in] exist_node_info1
/// @param [in] exist_node_info2
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @param [out] symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::UnionSymbolMapping(const NodeIndexIO &exist_node_info1, const NodeIndexIO &exist_node_info2,
                                           std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                           std::map<std::string, std::string> &anchor_to_symbol, std::string &symbol) {
  const std::string &symbol1 = anchor_to_symbol[exist_node_info1.ToString()];
  const std::string &symbol2 = anchor_to_symbol[exist_node_info2.ToString()];
  if (symbol1 == symbol2) {
    symbol = symbol1;
    GELOGI("no need to union.");
    return GRAPH_SUCCESS;
  }

  auto iter1 = symbol_to_anchors.find(symbol1);
  auto iter2 = symbol_to_anchors.find(symbol2);
  if ((iter1 == symbol_to_anchors.end()) || (iter2 == symbol_to_anchors.end())) {
    REPORT_INNER_ERROR("E19999", "symbol %s or %s not exist.", symbol1.c_str(), symbol2.c_str());
    GE_LOGE("[Check][Param] symbol %s or %s not exist.", symbol1.c_str(), symbol2.c_str());
    return GRAPH_FAILED;
  }

  auto &max_iter = (iter1->second.size() > iter2->second.size() ? iter1 : iter2);
  auto &min_iter = (iter1->second.size() > iter2->second.size() ? iter2 : iter1);
  symbol = (iter1->second.size() > iter2->second.size() ? symbol1 : symbol2);
  std::string min_symbol = (iter1->second.size() > iter2->second.size() ? symbol2 : symbol1);
  for (auto &node_index_io : min_iter->second) {
    GELOGD("Update anchor %s, symbol %s.", node_index_io.ToString().c_str(), symbol.c_str());
    max_iter->second.emplace_back(node_index_io);
    auto iter = anchor_to_symbol.find(node_index_io.ToString());
    if (iter == anchor_to_symbol.end()) {
      REPORT_INNER_ERROR("E19999", "anchor %s not exist in anchor_to_symbol.", node_index_io.ToString().c_str());
      GE_LOGE("[Check][Param] anchor %s not exist in anchor_to_symbol.", node_index_io.ToString().c_str());
      return GRAPH_FAILED;
    }
    if (iter->second != min_symbol) {
      GELOGW("[GetRefMapping][Check] not expected symbol of anchor %s, expect %s but %s exactly.", iter->first.c_str(),
             min_symbol.c_str(), iter->second.c_str());
    }
    iter->second = symbol;
  }

  GELOGI("Union symbol %s and %s succ.", symbol.c_str(), min_symbol.c_str());
  symbol_to_anchors.erase(min_iter);
  return GRAPH_SUCCESS;
}

///
/// Update symbol mapping with a new reference pair
/// @param [in] cur_node_info
/// @param [in] exist_node_info
/// @param [out] symbol_to_anchors
/// @param [out] anchor_to_symbol
/// @return success: GRAPH_SUCESS
///
graphStatus GraphUtils::UpdateRefMapping(const NodeIndexIO &cur_node_info, const NodeIndexIO &exist_node_info,
                                         std::map<std::string, std::list<NodeIndexIO>> &symbol_to_anchors,
                                         std::map<std::string, std::string> &anchor_to_symbol) {
  auto iter1 = anchor_to_symbol.find(exist_node_info.ToString());
  if (iter1 == anchor_to_symbol.end()) {
    REPORT_INNER_ERROR("E19999", "data_anchor %s is not visible before data_anchor %s, maybe TopoSorting is missing.",
                       exist_node_info.ToString().c_str(), cur_node_info.ToString().c_str());
    GE_LOGE("[Check][Param] data_anchor %s is not visible before data_anchor %s, maybe TopoSorting is missing.",
            exist_node_info.ToString().c_str(), cur_node_info.ToString().c_str());
    return GRAPH_FAILED;
  }

  const std::string &symbol = iter1->second;
  auto iter2 = symbol_to_anchors.find(symbol);
  if (iter2 == symbol_to_anchors.end()) {
    REPORT_INNER_ERROR("E19999", "symbol %s not exist in symbol_to_anchors.", symbol.c_str());
    GE_LOGE("[Check][Param] symbol %s not found.", symbol.c_str());
    return GRAPH_FAILED;
  }
  GELOGD("Add anchor %s, symbol %s.", cur_node_info.ToString().c_str(), symbol.c_str());
  iter2->second.emplace_back(cur_node_info);
  anchor_to_symbol.emplace(std::make_pair(cur_node_info.ToString(), symbol));

  return GRAPH_SUCCESS;
}

///
/// Check if out_data_anchor is reference of input
/// @param [in] out_data_anchor
/// @param [out] reuse_in_index
/// @return bool
///
bool GraphUtils::IsRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index) {
  if (out_data_anchor == nullptr) {
    GELOGW("[Check][Param] out_data_anchor is null");
    return false;
  }
  int32_t output_index = out_data_anchor->GetIdx();

  // pass-through op
  NodePtr node = out_data_anchor->GetOwnerNode();
  const std::string &type = node->GetType();
  const std::set<std::string> pass_through_set = { NETOUTPUT, WHILE, _WHILE, STATELESSWHILE };
  if ((pass_through_set.count(type) > 0) || (NodeUtils::IsSubgraphInput(node))) {
    reuse_in_index = output_index;
    GELOGI("Pass-Through node name[%s] index[%u].", node->GetName().c_str(), reuse_in_index);
    return true;
  }

  // Merge op 0th output
  bool is_merge_op = (type == MERGE) && (output_index == 0);
  if (is_merge_op) {
    reuse_in_index = 0;
    GELOGI("Merge name[%s] output_index[0].", node->GetName().c_str());
    return true;
  }

  // ref op
  // op_desc of node should not be null
  OpDescPtr op_desc = node->GetOpDesc();
  bool is_ref = false;
  (void)ge::AttrUtils::GetBool(op_desc, ATTR_NAME_REFERENCE, is_ref);
  if (is_ref) {
    const string &output_name = op_desc->GetOutputNameByIndex(output_index);
    for (const auto &input_name : op_desc->GetAllInputNames()) {
      if (!input_name.empty() && (output_name == input_name)) {
        reuse_in_index = op_desc->GetInputIndexByName(input_name);
        GELOGI("Reference name[%s] output[%s][%d] ref to input[%s][%d].", op_desc->GetName().c_str(),
               output_name.c_str(), output_index, input_name.c_str(), reuse_in_index);
        return true;
      }
    }
  }

  // reuse input
  auto output_op_desc = op_desc->GetOutputDescPtr(output_index);
  bool reuse_input = false;
  if (output_op_desc != nullptr) {
    if ((TensorUtils::GetReuseInput(*output_op_desc, reuse_input) == GRAPH_SUCCESS) && reuse_input) {
      uint32_t reuse_input_index = 0;
      if (TensorUtils::GetReuseInputIndex(*output_op_desc, reuse_input_index) == GRAPH_SUCCESS) {
        reuse_in_index = static_cast<int32_t>(reuse_input_index);
        GELOGI("ReuseInput name[%s] output[%d] reuse input[%d].", op_desc->GetName().c_str(),
               output_index, reuse_in_index);
        return true;
      }
    }
  }
  // nopadding reuse
  return IsNoPaddingRefFromInput(out_data_anchor, reuse_in_index);
}

bool GraphUtils::IsNoPaddingRefFromInput(const OutDataAnchorPtr &out_data_anchor, int32_t &reuse_in_index) {
  NodePtr node = out_data_anchor->GetOwnerNode();
  // nopadding means output[0] reuse input[0], but as history reason,
  // other output index also return true for mem assign in block_mem_assigner
  bool attr_reuse = false;
  bool is_input_continuous = false;
  bool is_out_continuous = false;
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_INPUT, is_input_continuous);
  (void)ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT, is_out_continuous);
  bool get_reuse_flag = ge::AttrUtils::GetBool(node->GetOpDesc(), ATTR_NAME_OUTPUT_REUSE_INPUT, attr_reuse);
  bool is_no_padding_reuse_input = (is_input_continuous || is_out_continuous) && get_reuse_flag && attr_reuse;
  if (is_no_padding_reuse_input) {
    reuse_in_index = 0;
    GELOGI("Nopadding ReuseInput name[%s] output[%d] reuse input[%d].", node->GetName().c_str(),
           out_data_anchor->GetIdx(), reuse_in_index);
    return true;
  }
  return false;
}

bool GraphUtils::IsNodeInGraphRecursively(const ComputeGraphPtr &graph, const Node &node) {
  auto parent_graph = node.GetOwnerComputeGraph();
  while (parent_graph != nullptr) {
    if (parent_graph == graph) {
      return true;
    }
    parent_graph = parent_graph->GetParentGraph();
  }
  return false;
}

///
/// Determine if the graph is a UNKNOWN_SHAPE graph based on whether the graph and all subgraphs
/// of the graph have UNKNOWN_SHAPE operators or not.
/// Note: This function will only look 'down' from the graph, not 'up'. For example, the following
/// scenario (K for known shape, U for unknown shape), ROOT graph is UNKNOWN_SHAPE while SUB graph is KNOWN_SHAPE
/// ROOT graph:      A -----> B -----> C
///                  K    subgraph     U
///                           |
///                           V
/// SUB graph:          D --> E --> F
///                     K     K     K
/// @param [in] graph
/// @return bool
///
bool GraphUtils::IsUnknownShapeGraph(const ComputeGraphPtr &graph) {
  if (graph == nullptr) {
    GELOGW("[Check][Param] Input graph is nullptr.");
    return false;
  }
  for (const auto &node : graph->GetDirectNode()) {
    bool is_unknown = false;
    auto ret = NodeUtils::GetNodeUnknownShapeStatus(*node, is_unknown);
    if (ret != GRAPH_SUCCESS) {
      GELOGW("[Check][UnknownGraph] Get unknown status failed, node name:%s, type:%s", node->GetName().c_str(),
             node->GetType().c_str());
      continue;
    }
    if (is_unknown) {
      GELOGD("Node %s, type %s is unknown shape in graph %s.",
             node->GetName().c_str(), node->GetType().c_str(), graph->GetName().c_str());
      return true;
    }
  }
  GELOGD("Graph %s does not have unknown shape node.", graph->GetName().c_str());
  return false;
}

///
/// @brief Add node to graph
/// @param [in] op_desc
/// @return ComputeGraphBuilder
///
ComputeGraphBuilder& ComputeGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  nodes_.emplace_back(op_desc);
  return *this;
}

///
/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return ComputeGraphBuilder
///
ComputeGraphBuilder& ComputeGraphBuilder::AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                                      const std::string &dst_name, uint32_t in_anchor_ind) {
  data_links_.emplace_back(std::make_pair(std::make_pair(src_name, out_anchor_ind),
                                          std::make_pair(dst_name, in_anchor_ind)));
  return *this;
}

///
/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return ComputeGraphBuilder
///
ComputeGraphBuilder& ComputeGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  ctrl_links_.emplace_back(std::make_pair(src_name, dst_name));
  return *this;
}

///
/// @brief Build nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void ComputeGraphBuilder::BuildNodes(graphStatus &error_code, std::string &error_msg) {
  if (owner_graph_ == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return;
  }

  std::string node_name;
  for (auto &op_desc : nodes_) {
    if (op_desc == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "op_desc is NULL.";
      return;
    }

    node_name = op_desc->GetName();
    NodePtr node = owner_graph_->AddNode(op_desc);
    if (node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "Add node " + node_name + " failed.";
      return;
    }

    GELOGD("Add node name:%s, type:%s.", node_name.c_str(), op_desc->GetType().c_str());
    node_names_[node_name] = node;
  }

  GELOGD("BuildNodes succ.");
}

///
/// @brief Build data-links
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void ComputeGraphBuilder::BuildDataLinks(graphStatus &error_code, std::string &error_msg) {
  for (auto &pair : data_links_) {
    std::string src_name = pair.first.first;
    uint32_t out_ind = pair.first.second;
    std::string dst_name = pair.second.first;
    uint32_t in_ind = pair.second.second;
    std::string log_msg = "Add data-edge ";
    log_msg.append(src_name).append(":").append(std::to_string(out_ind)).append("->")
            .append(dst_name).append(":").append(std::to_string(in_ind));

    auto src_iter = node_names_.find(src_name);
    auto dst_iter = node_names_.find(dst_name);
    if ((src_iter == node_names_.end()) || (dst_iter == node_names_.end())) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node not exist in graph.";
      return;
    }

    NodePtr src_node = node_names_[src_name];
    NodePtr dst_node = node_names_[dst_name];
    if ((src_node == nullptr) || (dst_node == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node is NULL.";
      return;
    }

    if (GraphUtils::AddEdge(src_node->GetOutDataAnchor(out_ind), dst_node->GetInDataAnchor(in_ind)) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed.";
      return;
    }

    GELOGD("%s succ.", log_msg.c_str());
  }

  GELOGD("BuildDataLinks succ.");
}

///
/// @brief Build ctrl-links
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void ComputeGraphBuilder::BuildCtrlLinks(graphStatus &error_code, std::string &error_msg) {
  for (auto &pair : ctrl_links_) {
    std::string src_name = pair.first;
    std::string dst_name = pair.second;
    std::string log_msg = "Add ctrl-edge ";
    log_msg.append(src_name).append("->").append(dst_name);

    auto src_iter = node_names_.find(src_name);
    auto dst_iter = node_names_.find(dst_name);
    if ((src_iter == node_names_.end()) || (dst_iter == node_names_.end())) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node not exist in graph.";
      return;
    }

    NodePtr src_node = node_names_[src_name];
    NodePtr dst_node = node_names_[dst_name];
    if ((src_node == nullptr) || (dst_node == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed: node is NULL.";
      return;
    }

    if (GraphUtils::AddEdge(src_node->GetOutControlAnchor(), dst_node->GetInControlAnchor()) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = log_msg + " failed.";
      return;
    }

    GELOGD("%s succ.", log_msg.c_str());
  }

  GELOGD("BuildCtrlLinks succ.");
}

/// @brief Get node with name
/// @param [in] name
/// @return NodePtr
///
NodePtr ComputeGraphBuilder::GetNode(const std::string &name) {
  auto iter = node_names_.find(name);
  if (iter == node_names_.end()) {
    REPORT_INNER_ERROR("E19999", "node %s not exist.", name.c_str());
    GE_LOGE("[Check][Param] node %s not exist.", name.c_str());
    return nullptr;
  }
  return iter->second;
}

/// @brief Get all nodes
/// @return std::vector<NodePtr>
///
std::vector<NodePtr> ComputeGraphBuilder::GetAllNodes() {
  std::vector<NodePtr> nodes;
  for (const auto &iter : node_names_) {
    nodes.emplace_back(iter.second);
  }
  return nodes;
}

///
/// @brief Add node to graph
/// @param [in] op_desc
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  ComputeGraphBuilder::AddNode(op_desc);
  return *this;
}

///
/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                                        const std::string &dst_name, uint32_t in_anchor_ind) {
  ComputeGraphBuilder::AddDataLink(src_name, out_anchor_ind, dst_name, in_anchor_ind);
  return *this;
}

///
/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  ComputeGraphBuilder::AddControlLink(src_name, dst_name);
  return *this;
}

///
/// @brief Set index_th input anchor for graph
/// @param [in] index
/// @param [in] node_names
/// @param [in] anchor_inds
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetInput(uint32_t index, const std::vector<std::string> &node_names,
                                                     const std::vector<uint32_t> &anchor_inds) {
  graph_inputs_[index] = std::make_pair(node_names, anchor_inds);
  return *this;
}

///
/// @brief Set index_th input of graph as useless
/// @param [in] index
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetUselessInput(uint32_t index) {
  graph_inputs_[index] = std::make_pair(std::vector<std::string>(), std::vector<uint32_t>());
  return *this;
}

///
/// @brief Add output anchor for graph
/// @param [in] owner_node_name
/// @param [in] anchor_ind
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddOutput(const std::string &owner_node_name, uint32_t anchor_ind) {
  graph_outputs_.emplace_back(std::make_pair(owner_node_name, anchor_ind));
  return *this;
}

///
/// @brief Add target for graph
/// @param [in] target_name
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::AddTarget(const std::string &target_name) {
  graph_targets_.emplace_back(target_name);
  return *this;
}

///
/// @brief Set parent-node of graph
/// @param [in] parent_node
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetParentNode(const NodePtr &parent_node) {
  parent_node_ = parent_node;
  return *this;
}

///
/// @brief Set mapping-relation of parent-node in_anchor_ind & Data-node
/// @param [in] input_mapping: index_of_graph_input -> in_anchor_index_of_parent_node
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetInputMapping(const std::map<uint32_t, uint32_t> &input_mapping) {
  for (auto &item : input_mapping) {
    input_mapping_[item.first] = item.second;
  }
  return *this;
}

///
/// @brief Set mapping-relation of parent-node out_anchor_ind & NetOutput-node out_anchor_ind
/// @param [in] output_mapping: index_of_graph_output -> out_anchor_index_of_parent_node
/// @return CompleteGraphBuilder
///
CompleteGraphBuilder& CompleteGraphBuilder::SetOutputMapping(const std::map<uint32_t, uint32_t> &output_mapping) {
  for (auto &item : output_mapping) {
    output_mapping_[item.first] = item.second;
  }
  return *this;
}

///
/// @brief Build graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return ComputeGraphPtr
///
ComputeGraphPtr CompleteGraphBuilder::Build(graphStatus &error_code, std::string &error_msg) {
  owner_graph_ = shared_ptr<ComputeGraph>(new (std::nothrow) ComputeGraph(name_));
  if (owner_graph_ == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return nullptr;
  }

  BuildNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildDataLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildCtrlLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  AddDataNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  if (retval_flag_) {
    AddRetValNodes(error_code, error_msg);
    if (error_code != GRAPH_SUCCESS) {
      return nullptr;
    }
    BuildGraphTargets(error_code, error_msg);
    if (error_code != GRAPH_SUCCESS) {
      return nullptr;
    }
  } else {
    AddNetOutputNode(error_code, error_msg);
    if (error_code != GRAPH_SUCCESS) {
      return nullptr;
    }
  }

  PostProcess(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  return owner_graph_;
}

///
/// @brief Add data nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::AddDataNodes(graphStatus &error_code, std::string &error_msg) {
  for (auto &input : graph_inputs_) {
    NodePtr data_node = AddDataNode(input.first, error_code, error_msg);
    if (data_node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNodes failed: add node Data:" + std::to_string(input.first) +  + " failed.";
      return;
    }

    if (owner_graph_->AddInputNode(data_node) == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNodes failed: add input node Data:" + std::to_string(input.first) +  + " failed.";
      return;
    }

    // useless input
    std::vector<std::string> input_names = input.second.first;
    std::vector<uint32_t> anchor_indes = input.second.second;
    if (input_names.size() != anchor_indes.size()) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNodes failed: num of input_names and indexs not equal.";
      return;
    }
    if (input_names.empty()) {
      continue;
    }

    size_t input_num = input_names.size();
    for (size_t i = 0; i < input_num; i++) {
      std::string input_name = input_names[i];
      uint32_t ind = anchor_indes[i];
      auto iter = node_names_.find(input_name);
      if (iter == node_names_.end()) {
        error_code = GRAPH_FAILED;
        error_msg = "AddDataNodes failed: node " + input_name + " not exist in graph.";
        return;
      }

      NodePtr in_node = node_names_[input_name];
      if (in_node == nullptr) {
        error_code = GRAPH_FAILED;
        error_msg = "AddDataNodes failed: node " + input_name + " is NULL.";
        return;
      }

      if (GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), in_node->GetInDataAnchor(ind)) != GRAPH_SUCCESS) {
        error_code = GRAPH_FAILED;
        error_msg = "AddDataNodes failed: add data-edge Data:" + std::to_string(input.first) + ":0->" +
                    input_name + ":" + std::to_string(ind) + " failed.";
        return;
      }
    }

    GELOGD("AddDataNodes : Add %u input succ.", input.first);
  }

  GELOGD("AddDataNodes succ.");
}

///
/// @brief Add data node
/// @param [in] index
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
NodePtr CompleteGraphBuilder::AddDataNode(uint32_t index, graphStatus &error_code, std::string &error_msg) {
  std::string data_name = "Data_" + std::to_string(index);
  OpDescBuilder op_desc_builder(data_name, "Data");
  OpDescPtr op_desc = op_desc_builder.AddInput("x")
                                     .AddOutput("y")
                                     .Build();
  if (op_desc == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "AddDataNode failed: create op_desc " + data_name + " failed.";
    return nullptr;
  }

  auto index_iter = input_mapping_.find(index);
  if (index_iter != input_mapping_.end()) {
    if (!ge::AttrUtils::SetInt(op_desc, ATTR_NAME_PARENT_NODE_INDEX, index_iter->second)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNode failed: set attr ATTR_NAME_PARENT_NODE_INDEX for " + data_name + " failed.";
      return nullptr;
    }
  }
  if (parent_node_ != nullptr) {
    // op_desc should not be null
    const auto &parent_desc = parent_node_->GetOpDesc()->GetInputDesc(index_iter->second);
    if ((op_desc->UpdateInputDesc(0, parent_desc) != GRAPH_SUCCESS) ||
        (op_desc->UpdateOutputDesc(0, parent_desc) != GRAPH_SUCCESS)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddDataNode failed: update tensor_desc for " + data_name + " failed.";
      return nullptr;
    }
  }

  NodePtr data_node = owner_graph_->AddNode(op_desc);
  if (data_node == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "AddDataNode failed: add node " + data_name + " failed.";
    return nullptr;
  }
  node_names_[data_name] = data_node;

  return data_node;
}

///
/// @brief Add RetVal nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::AddRetValNodes(graphStatus &error_code, std::string &error_msg) {
  size_t output_num = graph_outputs_.size();
  for (size_t i = 0; i < output_num; i++) {
    int32_t index = graph_outputs_[i].second;
    auto out_iter = node_names_.find(graph_outputs_[i].first);
    if (out_iter == node_names_.end()) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode failed: node " + graph_outputs_[i].first + " not exist in graph.";
      return;
    }
    NodePtr node = out_iter->second;
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode failed: node is NULL.";
      return;
    }

    std::string name = node->GetName() + "_RetVal_"+ std::to_string(index);
    OpDescPtr ret_val_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(name, FRAMEWORKOP));
    if (ret_val_desc == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: op_desc is NULL.";
      return;
    }
    ge::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(index);
    if ((ret_val_desc->AddInputDesc(tensor) != GRAPH_SUCCESS) ||
        (ret_val_desc->AddOutputDesc(tensor) != GRAPH_SUCCESS)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add input_desc / output_desc failed.";
      return;
    }

    if (!(ge::AttrUtils::SetStr(ret_val_desc, ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, "_RetVal") &&
          ge::AttrUtils::SetInt(ret_val_desc, RETVAL_ATTR_NAME_INDEX, i))) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: set FRAMEWORK_ORIGINAL_TYPE / RETVAL_ATTR_NAME_INDEX failed.";
      return;
    }
    auto iter = output_mapping_.find(i);
    if (iter != output_mapping_.end()) {
      if (!ge::AttrUtils::SetInt(ret_val_desc, ATTR_NAME_PARENT_NODE_INDEX, iter->second)) {
        error_code = GRAPH_FAILED;
        error_msg = "AddRetValNode " + name + " failed: set attr PARENT_NODE_INDEX failed.";
        return;
      }
    }

    NodePtr ret_val_node = owner_graph_->AddNode(ret_val_desc);
    if (ret_val_node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add node failed.";
      return;
    }

    if (GraphUtils::AddEdge(node->GetOutDataAnchor(index), ret_val_node->GetInDataAnchor(0)) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddRetValNode " + name + " failed: add data-edge " +
                  node->GetName() + ":" + std::to_string(index) + "->" + ret_val_node->GetName() + ":0 failed.";
      return;
    }
  }

  GELOGD("AddRetValNodes succ.");
}

///
/// @brief Build target-nodes for graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::BuildGraphTargets(graphStatus &error_code, std::string &error_msg) {
  std::vector<NodePtr> target_nodes;
  for (const std::string &target_name : graph_targets_) {
    auto target_iter = node_names_.find(target_name);
    if ((target_iter == node_names_.end()) || (target_iter->second == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "BuildGraphTargets failed: target_node " + target_name + " not exist in graph.";
      return;
    }
    target_nodes.emplace_back(target_iter->second);
  }
  owner_graph_->SetGraphTargetNodesInfo(target_nodes);
}

///
/// @brief Add NetOutput node
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::AddNetOutputNode(graphStatus &error_code, std::string &error_msg) {
  std::string log_msg = "AddNetOutputNode name:" + std::string(NODE_NAME_NET_OUTPUT) + ", type:" + NETOUTPUT;
  OpDescPtr net_output_desc = shared_ptr<OpDesc>(new (std::nothrow) OpDesc(NODE_NAME_NET_OUTPUT, NETOUTPUT));
  if (net_output_desc == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = log_msg + " failed: op_desc is NULL.";
    return;
  }

  size_t output_num = graph_outputs_.size();
  std::vector<OutDataAnchorPtr> peer_out_anchors(output_num);
  for (size_t i = 0; i < output_num; i++) {
    int32_t index = graph_outputs_[i].second;
    auto out_iter = node_names_.find(graph_outputs_[i].first);
    if (out_iter == node_names_.end()) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: node " + graph_outputs_[i].first + " not exist in graph.";
      return;
    }
    NodePtr node = out_iter->second;
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: node is NULL.";
      return;
    }

    ge::GeTensorDesc tensor = node->GetOpDesc()->GetOutputDesc(index);
    uint32_t update_index = i;
    auto iter = output_mapping_.find(i);
    if (iter != output_mapping_.end()) {
      update_index = iter->second;
    }
    if (!ge::AttrUtils::SetInt(tensor, ATTR_NAME_PARENT_NODE_INDEX, update_index)) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: set attr PARENT_NODE_INDEX failed.";
      return;
    }
    if (net_output_desc->AddInputDesc(tensor) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add input_desc ailed.";
      return;
    }
    peer_out_anchors[i] = node->GetOutDataAnchor(index);
  }

  BuildNetOutputNodeWithLink(net_output_desc, peer_out_anchors, error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return;
  }

  GELOGD("%s succ.", log_msg.c_str());
}

///
/// @brief Build NetOutput nodes with data & ctrl edges
/// @param [in] net_output_desc
/// @param [in] peer_out_anchors
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::BuildNetOutputNodeWithLink(const OpDescPtr &net_output_desc,
                                                      const std::vector<OutDataAnchorPtr> &peer_out_anchors,
                                                      graphStatus &error_code, std::string &error_msg) {
  std::string log_msg = "AddNetOutputNode name:" + std::string(NODE_NAME_NET_OUTPUT) + ", type:" + NETOUTPUT;
  NodePtr net_output = owner_graph_->AddNode(net_output_desc);
  if (net_output == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = log_msg + " failed: add NetOutput node failed.";
    return;
  }

  size_t output_num = graph_outputs_.size();
  for (size_t i = 0; i < output_num; i++) {
    if (GraphUtils::AddEdge(peer_out_anchors[i], net_output->GetInDataAnchor(i)) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add data-edge " +
                  peer_out_anchors[i]->GetOwnerNode()->GetName() + ":" + std::to_string(peer_out_anchors[i]->GetIdx()) +
                  "->" + NODE_NAME_NET_OUTPUT + ":" + std::to_string(i) + " failed.";
      return;
    }
  }
  for (const std::string &target_name : graph_targets_) {
    auto target_iter = node_names_.find(target_name);
    if ((target_iter == node_names_.end()) || (target_iter->second == nullptr)) {
      error_code = GRAPH_FAILED;
      error_msg = "BuildGraphTargets failed: target_node " + target_name + " not exist in graph.";
      return;
    }
    const auto &target_node = target_iter->second;
    if (GraphUtils::AddEdge(target_node->GetOutControlAnchor(), net_output->GetInControlAnchor()) != GRAPH_SUCCESS) {
      error_code = GRAPH_FAILED;
      error_msg = "AddNetOutputNode failed: add ctrl-edge " +
                  target_node->GetName() + "->" + NODE_NAME_NET_OUTPUT + " failed.";
      return;
    }
  }
}

///
/// @brief process after build
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void CompleteGraphBuilder::PostProcess(graphStatus &error_code, std::string &error_msg) {
  if (parent_node_ != nullptr) {
    owner_graph_->SetParentNode(parent_node_);
    owner_graph_->SetParentGraph(parent_node_->GetOwnerComputeGraph());
    // ATTR_NAME_SESSION_GRAPH_ID
    std::string graph_id;
    if (!AttrUtils::GetStr(parent_node_->GetOwnerComputeGraph(), ATTR_NAME_SESSION_GRAPH_ID, graph_id)) {
      error_code = GRAPH_FAILED;
      error_msg = "Get attr session_graph_id failed.";
      return;
    }
    if (!AttrUtils::SetStr(owner_graph_, ATTR_NAME_SESSION_GRAPH_ID, graph_id)) {
      error_code = GRAPH_FAILED;
      error_msg = "Set attr session_graph_id failed.";
      return;
    }
  }

  // refresh node name
  for (const NodePtr &node : owner_graph_->GetDirectNode()) {
    if ((node->GetOpDesc() == nullptr) || (node->GetType() == VARIABLE) || (node->GetType() == VARIABLEV2)) {
      continue;
    }
    node->GetOpDesc()->SetName(owner_graph_->GetName() + "/" + node->GetName());
  }
}

///
/// @brief Add node to graph
/// @param [in] op_desc
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddNode(const OpDescPtr &op_desc) {
  ComputeGraphBuilder::AddNode(op_desc);
  return *this;
}

///
/// @brief Add data-link among nodes in graph
/// @param [in] src_name
/// @param [in] out_anchor_ind
/// @param [in] dst_name
/// @param [in] in_anchor_ind
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddDataLink(const std::string &src_name, uint32_t out_anchor_ind,
                                                      const std::string &dst_name, uint32_t in_anchor_ind) {
  ComputeGraphBuilder::AddDataLink(src_name, out_anchor_ind, dst_name, in_anchor_ind);
  return *this;
}

///
/// @brief Add ctrl-link among nodes in graph
/// @param [in] src_name
/// @param [in] dst_name
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddControlLink(const std::string &src_name, const std::string &dst_name) {
  ComputeGraphBuilder::AddControlLink(src_name, dst_name);
  return *this;
}

///
/// @brief Set owner graph
/// @param [in] graph
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::SetOwnerGraph(const ComputeGraphPtr &graph) {
  owner_graph_ = graph;
  return *this;
}

///
/// @brief Add exist node
/// @param [in] node
/// @return PartialGraphBuilder
///
PartialGraphBuilder& PartialGraphBuilder::AddExistNode(const NodePtr &node) {
  exist_nodes_.emplace_back(node);
  return *this;
}

///
/// @brief Build partial graph
/// @param [out] error_code
/// @param [out] error_msg
/// @return ComputeGraphPtr
///
ComputeGraphPtr PartialGraphBuilder::Build(graphStatus &error_code, std::string &error_msg) {
  if (owner_graph_ == nullptr) {
    error_code = GRAPH_FAILED;
    error_msg = "graph is NULL.";
    return nullptr;
  }

  BuildNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildExistNodes(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildDataLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  BuildCtrlLinks(error_code, error_msg);
  if (error_code != GRAPH_SUCCESS) {
    return nullptr;
  }

  return owner_graph_;
}

///
/// @brief Build exist nodes
/// @param [out] error_code
/// @param [out] error_msg
/// @return void
///
void PartialGraphBuilder::BuildExistNodes(graphStatus &error_code, std::string &error_msg) {
  std::string node_name;
  for (auto &node : exist_nodes_) {
    if (node == nullptr) {
      error_code = GRAPH_FAILED;
      error_msg = "Build exist nodes failed: node is NULL.";
      return;
    }

    node_name = node->GetName();
    if (node->GetOwnerComputeGraph() != owner_graph_) {
      error_code = GRAPH_FAILED;
      error_msg = "Build exist nodes failed: node " + node_name + " not belongs to this graph.";
      return;
    }

    GELOGD("Add exist_node name:%s.", node_name.c_str());
    node_names_[node_name] = node;
  }

  GELOGD("Build exist nodes succ.");
}

GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY graphStatus GraphUtils::TopologicalSortingByName(
        const ge::ComputeGraphPtr &compute_graph, vector<NodePtr> &node_vec) {
  if (compute_graph == nullptr || compute_graph->impl_ == nullptr) {
    REPORT_INNER_ERROR("E19999", "Param compute_graph is nullptr, check invalid");
    GELOGE(GRAPH_FAILED, "[Check][Param] Compute graph or impl is nullptr.");
    return GRAPH_FAILED;
  }
  std::vector<NodePtr> stack_input;
  std::map<NodePtr, uint32_t> map_in_edge_num;
  graphStatus ret = compute_graph->SortNodes(stack_input, map_in_edge_num);
  if (ret != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "SortNodes failed, ret:%d, graph:%s", ret, compute_graph->GetName().c_str());
    GELOGE(GRAPH_FAILED, "[Sort][Nodes] failed, ret:%d, graph:%s.", ret, compute_graph->GetName().c_str());
    return GRAPH_FAILED;
  }
  const size_t non_user_input_index = stack_input.size() - compute_graph->impl_->inputs_order_.size() - 1;
  std::sort(stack_input.begin(), stack_input.begin() + non_user_input_index,
            [](const NodePtr &a, const NodePtr &b) -> bool { return (a->GetName() > b->GetName()); });

  std::queue<NodePtr> stack;
  NodePtr cur_node = nullptr;
  std::map<string, NodePtr> name_node_map;
  vector<string> nodes_name;
  while (!stack_input.empty() || !stack.empty()) {
    if (!stack.empty()) {
      cur_node = stack.front();
      stack.pop();
    } else {
      cur_node = stack_input.back();
      stack_input.pop_back();
    }
    node_vec.emplace_back(cur_node);
    compute_graph->CollectBreadthOutNode(cur_node, map_in_edge_num, name_node_map);
    for (const auto &iter : name_node_map) {
      nodes_name.emplace_back(iter.first);
    }
    std::sort(nodes_name.begin(), nodes_name.end());
    for (const auto &iter : nodes_name) {
      stack.push(name_node_map[iter]);
    }
    name_node_map.clear();
    nodes_name.clear();
  }
  // If they are not equal, there is a closed loop
  if (node_vec.size() != compute_graph->GetDirectNodesSize()) {
    std::set<Node *> itered_nodes_set;
    for (auto &node : node_vec) {
      itered_nodes_set.insert(node.get());
    }
    REPORT_INNER_ERROR("E19999", "Failed to do topo sorting total %zu, itered %zu, exist closed loop in graph.",
                       compute_graph->GetDirectNodesSize(), node_vec.size());
    GE_LOGE("[Check][Param] Failed to do topo sorting total %zu, itered %zu, exist closed loop in graph.",
            compute_graph->GetDirectNodesSize(), node_vec.size());
    for (auto &node : compute_graph->impl_->nodes_) {
      if (itered_nodes_set.count(node.get()) == 0) {
        REPORT_INNER_ERROR("E19999", "The node %s does not itered when topological sorting", node->GetName().c_str());
        GE_LOGE("[Check][Param] The node %s does not itered when topological sorting", node->GetName().c_str());
      }
    }
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

}  // namespace ge
