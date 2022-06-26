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
#include "graph/op_desc.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"

namespace ge {
class UtestGeAttrValue : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestGeAttrValue, GetAllAttrsStr) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  op_desc->SetAttr("i", GeAttrValue::CreateFrom<GeAttrValue::INT>(1));
  auto tensor = std::make_shared<GeTensor>();
  op_desc->SetAttr("value", GeAttrValue::CreateFrom<GeAttrValue::TENSOR>(tensor));
  op_desc->SetAttr("input_desc", GeAttrValue::CreateFrom<GeAttrValue::TENSOR_DESC>(GeTensorDesc()));
  string attr = AttrUtils::GetAllAttrsStr(op_desc);
  string res = "i:\x18\x1;input_desc:td {\n  dtype: DT_FLOAT\n  layout: \"ND\"\n  attr {\n    key: \"origin_format\"\n    value {\n      s: \"ND\"\n    }\n  }\n  has_out_attr: true\n  device_type: \"NPU\"\n}\n;value:dtype: DT_FLOAT\nlayout: \"ND\"\nattr {\n  key: \"origin_format\"\n  value {\n    s: \"ND\"\n  }\n}\nhas_out_attr: true\ndevice_type: \"NPU\"\n;";
  EXPECT_EQ(res, attr);
}
}
