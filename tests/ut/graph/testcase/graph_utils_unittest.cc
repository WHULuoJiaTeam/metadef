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

#include "graph/utils/graph_utils.h"
#include "graph/op_desc_impl.h"
#include "graph_builder_utils.h"
#include "graph/debug/ge_op_types.h"

#undef private
#undef protected

namespace ge {
class UtestGraphUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

/*
*               var                               var
*  atomicclean  |  \                             |   \
*         \\    |   assign                       |   assign
*          \\   |   //         =======>          |   //
*           allreduce                         identity  atomicclean
*             |                                 |       //
*            netoutput                        allreduce
*                                               |
*                                           netoutput
 */
TEST_F(UtestGraphUtils, InsertNodeBefore_DoNotMoveCtrlEdgeFromAtomicClean) {
  // build test graph
  auto builder = ut::GraphBuilder("test");
  auto var = builder.AddNode("var", VARIABLE, 0, 1);
  auto assign = builder.AddNode("assign", "Assign", 1, 1);
  auto allreduce = builder.AddNode("allreduce", "HcomAllReduce", 1, 1);
  auto atomic_clean = builder.AddNode("atomic_clean", ATOMICADDRCLEAN, 0, 0);
  auto netoutput1 = builder.AddNode("netoutput", NETOUTPUT, 1, 0);
  auto identity = builder.AddNode("identity", "Identity", 1, 1);

  builder.AddDataEdge(var, 0, assign, 0);
  builder.AddDataEdge(var,0,allreduce,0);
  builder.AddControlEdge(assign, allreduce);
  builder.AddControlEdge(atomic_clean, allreduce);
  auto graph = builder.GetGraph();

  // insert identity before allreduce
  GraphUtils::InsertNodeBefore(allreduce->GetInDataAnchor(0), identity, 0, 0);

  // check assign control-in on identity
  ASSERT_EQ(identity->GetInControlNodes().at(0)->GetName(), "assign");
  // check atomicclean control-in still on allreuce
  ASSERT_EQ(allreduce->GetInControlNodes().at(0)->GetName(), "atomic_clean");
}
}  // namespace ge
