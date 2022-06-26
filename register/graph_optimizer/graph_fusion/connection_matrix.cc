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

#include "register/graph_optimizer/graph_fusion/connection_matrix.h"
#include "graph/debug/ge_log.h"

namespace fe {
ConnectionMatrix::ConnectionMatrix(const ge::ComputeGraph &graph) {
  const auto direct_nodes = graph.GetDirectNode();
  size_ = direct_nodes.size();
  bit_maps.reserve(size_);
  int64_t index_loop = 0;
  for (const auto &node : direct_nodes) {
    name_to_index_[node->GetName()] = index_loop;
    bit_maps.emplace_back(size_);
    index_loop++;
  }
};

ConnectionMatrix::~ConnectionMatrix() {
  bit_maps.clear();
  name_to_index_.clear();
}

Status ConnectionMatrix::Generate(const ge::ComputeGraph &graph) {
  for (auto &node : graph.GetDirectNode()) {
    const auto inputs = node->GetInAllNodes();
    SetConnectivity(inputs, node);
  }
  return ge::GRAPH_SUCCESS;
}

void ConnectionMatrix::Update(const ge::ComputeGraph &graph, const vector<ge::NodePtr> &fusion_nodes) {
  ge::LargeBitmap new_bit_vector(graph.GetDirectNode().size());
  new_bit_vector.SetValues(0U);
  for (size_t i = 0; i < fusion_nodes.size(); i++) {
    new_bit_vector.Or(GetBitMap(fusion_nodes[i]));
  }
  for (auto &node : graph.GetDirectNode()) {
    bool is_connected_to_fusion = false;
    for (size_t i = 0U; i < fusion_nodes.size(); i++) {
      if (GetBitMap(node).GetBit(static_cast<size_t>(GetIndex(fusion_nodes[i])))) {
        is_connected_to_fusion = true;
        break;
      }
    }
    if (is_connected_to_fusion) {
      GetBitMap(node).Or(new_bit_vector);
    }
  }
}

void ConnectionMatrix::SetConnectivity(const ge::Node::Vistor<ge::NodePtr> &inputs, const ge::NodePtr &node) {
  ge::LargeBitmap &bitmap = GetBitMap(node);
  if (std::find(inputs.begin(), inputs.end(), node) == inputs.end()) {
    bitmap.SetValues(0);
  }

  bitmap.SetBit(static_cast<size_t>(GetIndex(node)));
  for (const ge::NodePtr &input : inputs) {
    if (input != node) {
      bitmap.Or(GetBitMap(input));
    }
  }
}

int64_t ConnectionMatrix::GetIndex(const ge::NodePtr &node) const {
  const auto iter = name_to_index_.find(node->GetName());
  if (iter != name_to_index_.end()) {
    return iter->second;
  } else {
    GELOGW("node %s is not found in name_to_index_", node->GetName().c_str());
    return 0;
  }
}

bool ConnectionMatrix::IsConnected(const ge::NodePtr &a, const ge::NodePtr &b) const {
  return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

const ge::LargeBitmap &ConnectionMatrix::GetBitMap(const ge::NodePtr &node) const {
  return bit_maps[static_cast<uint64_t>(GetIndex(node))];
}

ge::LargeBitmap &ConnectionMatrix::GetBitMap(const ge::NodePtr &node) {
  return bit_maps[static_cast<uint64_t>(GetIndex(node))];
}
}
