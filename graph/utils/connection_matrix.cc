/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "graph/utils/connection_matrix.h"
#include "connection_matrix_impl.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"

namespace ge {
ConnectionMatrix::ConnectionMatrix(const ComputeGraphPtr &graph)
    : impl_(ComGraphMakeUnique<ConnectionMatrixImpl>(graph)) {}

bool ConnectionMatrix::IsConnected(const NodePtr &a, const NodePtr &b) const {
  if (impl_ == nullptr) {
    return false;
  }
  return impl_->IsConnected(a, b);
}

void ConnectionMatrix::SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->SetConnectivity(inputs, node);
}

graphStatus ConnectionMatrix::Generate(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(impl_);
  return impl_->Generate(graph);
}

void ConnectionMatrix::Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes) {
  if (impl_ == nullptr) {
    return;
  }
  impl_->Update(graph, fusion_nodes);
}

ConnectionMatrixImpl::ConnectionMatrixImpl(const ComputeGraphPtr &graph) : graph_(graph) {
  const auto direct_nodes = graph->GetDirectNode();
  size_ = direct_nodes.size();
  bit_maps_.reserve(size_);
  int64_t index_loop = 0;
  for (const auto &node : direct_nodes) {
    name_to_index_[node->GetName()] = index_loop;
    bit_maps_.emplace_back(size_);
    index_loop++;
  }
};

ConnectionMatrixImpl::~ConnectionMatrixImpl() {
  bit_maps_.clear();
  name_to_index_.clear();
}

graphStatus ConnectionMatrixImpl::Generate(const ComputeGraphPtr &graph) {
  if (graph_ == nullptr) {
    graph_ = graph;
  }
  for (auto &node : graph->GetDirectNode()) {
    const auto inputs = node->GetInAllNodes();
    SetConnectivity(inputs, node);
  }
  return GRAPH_SUCCESS;
}

void ConnectionMatrixImpl::Update(const ComputeGraphPtr &graph, const vector<NodePtr> &fusion_nodes) {
  if (graph_ == nullptr) {
    return;
  }
  if (graph != graph_) {
    GELOGW("Input graph %s is not the same one %s when contribute connection matrix.", graph->GetName().c_str(),
           graph_->GetName().c_str());
    return;
  }
  LargeBitmap new_bit_vector(graph->GetDirectNode().size());
  new_bit_vector.SetValues(0U);
  for (size_t i = 0U; i < fusion_nodes.size(); i++) {
    new_bit_vector.Or(GetBitMap(fusion_nodes[i]));
  }
  for (auto &node : graph->GetDirectNode()) {
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

void ConnectionMatrixImpl::SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node) {
  LargeBitmap &bitmap = GetBitMap(node);
  if (std::find(inputs.begin(), inputs.end(), node) == inputs.end()) {
    bitmap.SetValues(0U);
  }

  bitmap.SetBit(static_cast<size_t>(GetIndex(node)));
  for (const NodePtr &input : inputs) {
    if (input != node) {
      bitmap.Or(GetBitMap(input));
    }
  }
}

int64_t ConnectionMatrixImpl::GetIndex(const NodePtr &node) const {
  const auto iter = name_to_index_.find(node->GetName());
  if (iter != name_to_index_.end()) {
    return iter->second;
  } else {
    GELOGW("node %s is not found in name_to_index_", node->GetName().c_str());
    return 0;
  }
}

bool ConnectionMatrixImpl::IsConnected(const NodePtr &a, const NodePtr &b) const {
  return GetBitMap(b).GetBit(static_cast<size_t>(GetIndex(a)));
}

const LargeBitmap &ConnectionMatrixImpl::GetBitMap(const NodePtr &node) const {
  return bit_maps_[static_cast<uint64_t>(GetIndex(node))];
}

LargeBitmap &ConnectionMatrixImpl::GetBitMap(const NodePtr &node) {
  return bit_maps_[static_cast<uint64_t>(GetIndex(node))];
}
} // namespace ge
