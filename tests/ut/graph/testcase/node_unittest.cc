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

#include "graph/node.h"
#include "graph_builder_utils.h"

#undef private
#undef protected

namespace ge {
class UtestNode : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestNode, GetInDataAnchor) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();

  auto data_node = graph->FindNode("Data");
  auto in_data_anchor0 = data_node->GetInDataAnchor(0);
  EXPECT_NE(in_data_anchor0, nullptr);

  auto in_data_anchor1 = data_node->GetInDataAnchor(1);
  EXPECT_EQ(in_data_anchor1, nullptr);
}
}
