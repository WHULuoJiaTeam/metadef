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
#include "graph/op_desc.h"
#include "graph/ge_tensor.h"
#undef private
#undef protected

namespace ge {
class UtestOpDesc : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpDesc, TestCommonVerifyOnDummyShape) {
  auto tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(GeShape({-3}));
  tensor_desc->SetFormat(FORMAT_NCHW);
  tensor_desc->SetDataType(DT_FLOAT);
  auto op_desc = std::make_shared<OpDesc>("test", "Identity");
  op_desc->AddInputDesc(tensor_desc->Clone());
  op_desc->AddOutputDesc(tensor_desc->Clone());

  EXPECT_EQ(GRAPH_SUCCESS, op_desc->CommonVerify());
}
}
