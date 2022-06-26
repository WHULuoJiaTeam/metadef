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
#ifndef GRAPH_CONNECTION_MATRIX_IMPL_H_
#define GRAPH_CONNECTION_MATRIX_IMPL_H_

#include "graph/debug/ge_attr_define.h"
#include "graph/node.h"
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "common/large_bm.h"

namespace ge {
class ConnectionMatrixImpl {
public:
  explicit ConnectionMatrixImpl(const ComputeGraphPtr &graph);

  ~ConnectionMatrixImpl();

  bool IsConnected(const NodePtr &a, const NodePtr &b) const;

  // inputs are all input nodes of parameter node.
  // if there is a path between A->B, then B will own A's
  // connectivity. The reason is ---
  // If some node can reach A, than it can also reach B.
  void SetConnectivity(const Node::Vistor<NodePtr> &inputs, const NodePtr &node);

  /* Computes the connectivity between two nodes in the
   * computation. The returned ConnectivityMatrix is constructed such that
   * ConnectivityMatrix::IsConnected(a, b) returns true iff there exists a
   * directed path (from producer to consumer) from 'a' to 'b'. Both data
   * connection and control connection are considered for connectivity.
   * A node is connected to itself. */
  graphStatus Generate(const ComputeGraphPtr &graph);

  // update reachablity map for fused nodes.
  void Update(const ComputeGraphPtr &graph, const std::vector<NodePtr> &fusion_nodes);

private:
  ConnectionMatrixImpl() = delete;
  int64_t GetIndex(const NodePtr &node) const;

  const LargeBitmap &GetBitMap(const NodePtr &node) const;

  LargeBitmap &GetBitMap(const NodePtr &node);

  size_t size_;

  std::vector<LargeBitmap> bit_maps_;

  std::map<std::string, int64_t> name_to_index_;

  ComputeGraphPtr graph_;
};
}
#endif  // GRAPH_CONNECTION_MATRIX_H_
