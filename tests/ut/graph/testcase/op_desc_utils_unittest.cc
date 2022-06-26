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

#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"

#undef private
#undef protected

namespace ge {
class UtestOpDescUtils : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDescUtils, SetWeight) {
  ut::GraphBuilder builder = ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto const1 = builder.AddNode("const1", "Const", 1, 1);
  auto addn = builder.AddNode("addn", "AddN", 2, 1);

  int32_t weight[1] = {1};
  GeTensorDesc weight_desc(GeShape({1}), FORMAT_NHWC, DT_INT32);
  GeTensorPtr tensor0 = std::make_shared<GeTensor>(weight_desc, (uint8_t *)weight, sizeof(weight));
  OpDescUtils::SetWeights(const1, {tensor0});

  builder.AddDataEdge(data, 0, addn, 0);
  builder.AddDataEdge(const1, 0, addn, 1);
  auto graph = builder.GetGraph();

  auto addn_node = graph->FindNode("addn");
  ge::GeTensorPtr tensor = std::make_shared<GeTensor>();
  std::vector<uint8_t> value{1, 2, 3};
  std::vector<int64_t> shape{3};
  tensor->MutableTensorDesc().SetShape(GeShape(shape));
  tensor->SetData(value);
  tensor->MutableTensorDesc().SetDataType(DT_UINT8);

  map<int, ge::GeTensorPtr> weight0;
  weight0[-1] = tensor;
  auto ret = ge::OpDescUtils::SetWeights(*addn_node, weight0);
  EXPECT_NE(ret, 0);

  map<int, ge::GeTensorPtr> weight1;
  weight1[1] = tensor;
  ret = ge::OpDescUtils::SetWeights(*addn_node, weight1);
  EXPECT_EQ(ret, 0);
  auto const_node = graph->FindNode("const1");
  auto const_tensor = OpDescUtils::MutableWeights(const_node);
  EXPECT_EQ(const_tensor[0]->MutableData().size(), 3);
  auto in_nodes = addn_node->GetInAllNodes();
  EXPECT_EQ(in_nodes.size(), 2);

  map<int, ge::GeTensorPtr> weight2;
  weight2[2] = tensor;
  ret = ge::OpDescUtils::SetWeights(*addn_node, weight2);
  EXPECT_EQ(ret, 0);
  auto in_nodes1 = addn_node->GetInAllNodes();
  EXPECT_EQ(in_nodes1.size(), 3);
}
}
