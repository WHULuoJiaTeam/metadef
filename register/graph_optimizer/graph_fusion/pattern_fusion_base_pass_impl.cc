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

#include "register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.h"
#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/fusion_common/graph_pass_util.h"

namespace fe {
PatternFusionBasePassImpl::PatternFusionBasePassImpl() {}

PatternFusionBasePassImpl::~PatternFusionBasePassImpl() {
  for (auto pattern : patterns_) {
    if (pattern != nullptr) {
      delete pattern;
      pattern = nullptr;
    }
  }
}

void PatternFusionBasePassImpl::GetPatterns(std::vector<FusionPattern *> &patterns) { patterns = patterns_; }

void PatternFusionBasePassImpl::SetPatterns(const std::vector<FusionPattern *> &patterns) { patterns_ = patterns; }

void PatternFusionBasePassImpl::SetOpsKernelInfoStore(const OpsKernelInfoStorePtr ops_kernel_info_store_ptr) {
  ops_kernel_info_store_ptr_ = ops_kernel_info_store_ptr;
}

bool PatternFusionBasePassImpl::CheckOpSupported(const ge::OpDescPtr &op_desc_ptr) const {
  std::string un_supported_reason;

  if (ops_kernel_info_store_ptr_ == nullptr) {
    un_supported_reason = "opsKernelInfoStorePtr in PatternFusionBasePass is nullptr.";
    return false;
  }

  return ops_kernel_info_store_ptr_->CheckSupported(op_desc_ptr, un_supported_reason);
}

bool PatternFusionBasePassImpl::CheckOpSupported(const ge::NodePtr &node) const {
  std::string un_supported_reason;

  if (ops_kernel_info_store_ptr_ == nullptr) {
    un_supported_reason = "opsKernelInfoStorePtr in PatternFusionBasePass is nullptr.";
    return false;
  }

  return ops_kernel_info_store_ptr_->CheckSupported(node, un_supported_reason);
}

bool PatternFusionBasePassImpl::IsNodesExist(const ge::NodePtr current_node, std::vector<ge::NodePtr> &nodes) {
  return find(nodes.begin(), nodes.end(), current_node) != nodes.end();
}

bool PatternFusionBasePassImpl::IsMatched(const std::shared_ptr<OpDesc> op_desc, const ge::NodePtr node,
                                          const Mapping &mapping) {
  if ((op_desc == nullptr) || (node == nullptr)) {
    GELOGD("opDesc or node could not be null");
    return false;
  }

  const auto iter = mapping.find(op_desc);

  // check op_desc does not exist in mapping
  return (iter != mapping.end()) && (find(iter->second.begin(), iter->second.end(), node) != iter->second.end());
}

void PatternFusionBasePassImpl::DumpMappings(const FusionPattern &pattern, const Mappings &mappings) const {
  std::ostringstream oss;
  oss << std::endl << "Mappings of pattern ";
  oss << pattern.GetName() << ":" << std::endl;
  for (size_t i = 0; i < mappings.size(); i++) {
    const Mapping &mapping = mappings[i];
    oss << " Mapping " << (i + 1) << "/" << mappings.size() << ":" << std::endl;
    for (const auto &item : mapping) {
      const std::shared_ptr<OpDesc> op_desc = item.first;
      const ge::NodePtr node = item.second[0];
      if ((op_desc != nullptr) && (node != nullptr)) {
        oss << "    " << op_desc->id << " -> " << node->GetName() << std::endl;
      }
    }
  }
  GELOGD("%s", oss.str().c_str());
}

bool PatternFusionBasePassImpl::IsOpTypeExist(const std::string &type, const std::vector<std::string> &types) {
  return find(types.begin(), types.end(), type) != types.end();
}

bool PatternFusionBasePassImpl::MatchFromOutput(const ge::NodePtr output_node,
    const std::shared_ptr<OpDesc> output_op_desc, Mapping &mapping) const {
  if ((output_node == nullptr) || (output_op_desc == nullptr)) {
    GELOGW("[Match][Output] output node/op_desc is null, pattern matching failed");
    return false;
  }

  std::vector<ge::NodePtr> candidate_nodes = {output_node};
  std::vector<std::shared_ptr<OpDesc>> candidate_op_descs = {output_op_desc};

  // store the nodes matched
  mapping[output_op_desc].push_back(output_node);

  // match candidate node one by one
  while ((!candidate_nodes.empty()) && (!candidate_op_descs.empty())) {
    // get the first candidate node
    const bool result = MatchFromOutput(candidate_nodes, candidate_op_descs, mapping);
    if (!result) {
      return false;
    }

    // current op is matched successfully, thus remove it from candidate list
    (void)candidate_nodes.erase(candidate_nodes.cbegin());
    (void)candidate_op_descs.erase(candidate_op_descs.cbegin());

    // the sizes of candidate_nodes and candidate_op_descs should always keep the same
    if (candidate_nodes.size() != candidate_op_descs.size()) {
      GELOGW("[Match][Output] candidate_nodes_num != candidate_op_descs_num, pattern matching failed.");
      return false;
    }
  }

  // if candidate_nodes(or candidate_op_descs) is empty, the matching is done
  // successfully
  return candidate_op_descs.empty();
}

bool PatternFusionBasePassImpl::MatchFromOutput(std::vector<ge::NodePtr> &candidate_nodes,
                                                std::vector<std::shared_ptr<OpDesc>> &candidate_op_descs,
                                                Mapping &mapping) const {
  if (candidate_nodes.empty() || candidate_op_descs.empty()) {
    GELOGW("[Match][Output] candidate_nodes or candidate_op_descs is empty, pattern matching failed.");
    return false;
  }
  const ge::NodePtr node = candidate_nodes.front();
  std::shared_ptr<OpDesc> op_desc = candidate_op_descs.front();
  const std::string op_id = op_desc->id;
  // add the input nodes into candidate list
  const std::vector<std::shared_ptr<OpDesc>> * const inputs_desc = FusionPattern::GetInputs(op_desc);
  if (inputs_desc == nullptr) {
    GELOGW("[Match][Output] Get input_desc of op %s failed, pattern matching failed.", op_id.c_str());
    return false;
  }

  if (inputs_desc->empty()) {
    return true;
  }

  if (node->GetInDataNodes().empty()) {
    GELOGW("[Match][Output] in data nodes of op %s is empty, pattern matching failed.", op_id.c_str());
    return false;
  }

  // set flag for edge using
  const std::unique_ptr<bool[]> usage_flags(new (std::nothrow) bool[inputs_desc->size()]{});

  // order the input edges, and the order should also be the rule of pattern
  // setting
  std::vector<ge::InDataAnchorPtr> in_anchors;
  GetInDataAnchors(node, in_anchors);
  if (in_anchors.empty()) {
    GELOGW("[Match][Output] in data anchors of op %s is empty, pattern matching failed.", op_id.c_str());
    return false;
  }

  std::sort(in_anchors.begin(), in_anchors.end(),
            [](const ge::InDataAnchorPtr a, const ge::InDataAnchorPtr b) { return a->GetIdx() < b->GetIdx(); });

  for (const auto &in_anchor : in_anchors) {
    const ge::NodePtr input_node = in_anchor->GetPeerOutAnchor()->GetOwnerNode();
    for (uint32_t j = 0U; j < inputs_desc->size(); j++) {
      const std::shared_ptr<OpDesc> input_desc = inputs_desc->at(static_cast<size_t>(j));
      if (input_desc == nullptr) {
        GELOGW("[Match][Output] input_desc %u of op %s is null, pattern matching failed.", j, op_id.c_str());
        return false;
      }

      const bool cond =
          (IsOpTypeExist(ge::NodeUtils::GetNodeType(*input_node), input_desc->types) || input_desc->types.empty()) &&
          ((!usage_flags[static_cast<size_t>(j)]) || input_desc->repeatable);
      if (!cond) {
        continue;
      }
      // some nodes might be the input of multiple nodes, we use
      // IsMatched() to avoid repeat
      if (!IsMatched(input_desc, input_node, mapping)) {
        candidate_nodes.push_back(input_node);
        candidate_op_descs.push_back(input_desc);
        // store the matched node
        mapping[input_desc].push_back(input_node);
      }
      usage_flags[static_cast<size_t>(j)] = true;
      break;
    }
  }

  // return false if not all edges are matched
  if (!MatchAllEdges(inputs_desc->size(), usage_flags)) {
    GELOGW("[Match][Output] not all inputs of op %s are matched, pattern matching failed.", op_id.c_str());
    return false;
  }
  return true;
}

bool PatternFusionBasePassImpl::MatchAllEdges(const size_t &input_size, const std::unique_ptr<bool[]> &usage_flags) {
  for (size_t i = 0; i != input_size; i++) {
    if (!usage_flags[i]) {
      return false;
    }
  }
  return true;
}

void PatternFusionBasePassImpl::GetInDataAnchors(const ge::NodePtr &node,
                                                 std::vector<ge::InDataAnchorPtr> &in_anchor_vec) {
  for (const auto in_anchor : node->GetAllInDataAnchors()) {
    if ((in_anchor == nullptr) || (in_anchor->GetPeerOutAnchor() == nullptr) ||
        (in_anchor->GetPeerOutAnchor()->GetOwnerNode() == nullptr)) {
      continue;
    }
    in_anchor_vec.push_back(in_anchor);
  }
}

bool PatternFusionBasePassImpl::GetMatchOutputNodes(const ge::ComputeGraph &graph, const FusionPattern &pattern,
                                                    std::vector<ge::NodePtr> &matched_output_nodes) const {
  const std::shared_ptr<FusionPattern::OpDesc> output_op_desc = pattern.GetOutput();
  if (output_op_desc == nullptr) {
    GELOGW("[Get][Output] output op_desc is null, pattern matching failed");
    return false;
  }

  NodeMapInfoPtr node_map_info = nullptr;
  // get nodes by type from node
  if (GraphPassUtil::GetOpTypeMapToGraph(node_map_info, graph) == SUCCESS) {
    for (auto &OutOpType : output_op_desc->types) {
      const auto iter = node_map_info->node_type_map->find(OutOpType);
      if (iter != node_map_info->node_type_map->end()) {
        for (auto iter_node = iter->second.cbegin(); iter_node != iter->second.cend(); iter_node++) {
          const ge::NodePtr node_ptr = iter_node->second;

          if (node_ptr->GetInDataNodes().empty() && node_ptr->GetOutAllNodes().empty()) {
            continue;
          }
          if (ge::NodeUtils::GetNodeType(*node_ptr) == OutOpType) {
            matched_output_nodes.push_back(node_ptr);
          }
        }
      }
    }
  } else {  // for each graph to find type
    for (ge::NodePtr &n : graph.GetDirectNode()) {
      if (IsOpTypeExist(ge::NodeUtils::GetNodeType(*n), output_op_desc->types)) {
        matched_output_nodes.push_back(n);
      }
    }
  }

  if (matched_output_nodes.empty()) {
    return false;
  }
  return true;
}
}
