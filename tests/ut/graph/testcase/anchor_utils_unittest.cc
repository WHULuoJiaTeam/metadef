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
#include "test_structs.h"
#include "func_counter.h"
#include "graph/anchor.h"
#include "graph/utils/anchor_utils.h"
#include "graph/node.h"
#include "graph_builder_utils.h"

namespace ge {
namespace {
class SubAnchor : public Anchor
{
public:

  SubAnchor(const NodePtr& owner_node, const int32_t idx);

  virtual ~SubAnchor();

  virtual bool Equal(const AnchorPtr anchor) const;

};

SubAnchor::SubAnchor(const NodePtr &owner_node, const int32_t idx):Anchor(owner_node,idx){}

SubAnchor::~SubAnchor() = default;

bool SubAnchor::Equal(const AnchorPtr anchor) const{
  return true;
}
}

using SubAnchorPtr = std::shared_ptr<SubAnchor>;

class AnchorUtilsUt : public testing::Test {};

TEST_F(AnchorUtilsUt, GetStatus) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  InDataAnchorPtr inanch = std::make_shared<InDataAnchor>(node, 111);
  EXPECT_NE(AnchorUtils::GetStatus(inanch), ANCHOR_RESERVED);
  EXPECT_EQ(AnchorUtils::GetStatus(inanch), ANCHOR_SUSPEND);
  EXPECT_EQ(AnchorUtils::GetStatus(nullptr), ANCHOR_RESERVED);
}

TEST_F(AnchorUtilsUt, SetStatus) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  InDataAnchorPtr inanch = std::make_shared<InDataAnchor>(node, 111);
  EXPECT_EQ(AnchorUtils::SetStatus(inanch,  ANCHOR_DATA), GRAPH_SUCCESS);
  EXPECT_EQ(AnchorUtils::SetStatus(inanch,  ANCHOR_RESERVED), GRAPH_FAILED);
  EXPECT_EQ(AnchorUtils::SetStatus(nullptr, ANCHOR_DATA), GRAPH_FAILED);
}


TEST_F(AnchorUtilsUt, GetIdx) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto node = builder.AddNode("Data", "Data", 1, 1);
  InDataAnchorPtr inanch = std::make_shared<InDataAnchor>(node, 111);
  EXPECT_EQ(AnchorUtils::GetIdx(inanch), 111);

  ut::GraphBuilder builder2 = ut::GraphBuilder("graph");
  auto node2 = builder2.AddNode("Data", "Data", 2, 2);
  OutControlAnchorPtr outanch = std::make_shared<OutControlAnchor>(node2, 22);
  EXPECT_EQ(AnchorUtils::GetIdx(outanch), 22);

  SubAnchorPtr sanch = std::make_shared<SubAnchor>(node, 444);
  EXPECT_EQ(AnchorUtils::GetIdx(sanch), -1);

}

}  // namespace ge
