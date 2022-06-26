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
#include "graph/op_desc.h"
#include "graph/ge_attr_value.h"
#include "graph/utils/attr_utils.h"
#include <string>

namespace ge {
class UtestGeAttrValue : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};


TEST_F(UtestGeAttrValue, GetAttrsStrAfterRid) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>();
  EXPECT_EQ(AttrUtils::GetAttrsStrAfterRid(op_desc, {}), "");

  std::set<std::string> names ={"qazwsx", "d"};
  op_desc->SetAttr("qazwsx", GeAttrValue::CreateFrom<int64_t>(132));
  op_desc->SetAttr("xswzaq", GeAttrValue::CreateFrom<int64_t>(123));
  auto tensor = GeTensor();
  op_desc->SetAttr("value", GeAttrValue::CreateFrom<GeTensor>(tensor));
  std::string res = AttrUtils::GetAttrsStrAfterRid(op_desc, names);
  EXPECT_TRUE(res.find("qazwsx") == string::npos);
  EXPECT_TRUE(res.find("xswzaq") != string::npos);
}

TEST_F(UtestGeAttrValue, GetAllAttrsStr) {
 // 属性序列化
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  EXPECT_EQ(AttrUtils::GetAllAttrsStr(op_desc), "");
  op_desc->SetAttr("seri_i", GeAttrValue::CreateFrom<int64_t>(1));
  auto tensor = GeTensor();
  op_desc->SetAttr("seri_value", GeAttrValue::CreateFrom<GeTensor>(tensor));
  op_desc->SetAttr("seri_input_desc", GeAttrValue::CreateFrom<GeTensorDesc>(GeTensorDesc()));
  string attr = AttrUtils::GetAllAttrsStr(op_desc);

  EXPECT_TRUE(attr.find("seri_i") != string::npos);
  EXPECT_TRUE(attr.find("seri_value") != string::npos);
  EXPECT_TRUE(attr.find("seri_input_desc") != string::npos);

}
TEST_F(UtestGeAttrValue, GetAllAttrs) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);
  op_desc->SetAttr("i", GeAttrValue::CreateFrom<int64_t>(100));
  op_desc->SetAttr("input_desc", GeAttrValue::CreateFrom<GeTensorDesc>(GeTensorDesc()));
  auto attrs = AttrUtils::GetAllAttrs(op_desc);
  EXPECT_EQ(attrs.size(), 2);
  int64_t attr_value = 0;
  EXPECT_EQ(attrs["i"].GetValue(attr_value), GRAPH_SUCCESS);
  EXPECT_EQ(attr_value, 100);

}

TEST_F(UtestGeAttrValue, TrySetExists) {
  string name = "const";
  string type = "Constant";
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  EXPECT_TRUE(op_desc);

  int64_t attr_value = 0;

  EXPECT_FALSE(AttrUtils::GetInt(op_desc, "i", attr_value));
  op_desc->TrySetAttr("i", GeAttrValue::CreateFrom<int64_t>(100));
  EXPECT_TRUE(AttrUtils::GetInt(op_desc, "i", attr_value));
  EXPECT_EQ(attr_value, 100);

  op_desc->TrySetAttr("i", GeAttrValue::CreateFrom<int64_t>(102));
  attr_value = 0;
  AttrUtils::GetInt(op_desc, "i", attr_value);
  EXPECT_EQ(attr_value, 100);
}


TEST_F(UtestGeAttrValue, SetGetListInt) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  EXPECT_TRUE(op_desc);

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "li1", std::vector<int64_t>({1,2,3,4,5})));
  std::vector<int64_t> li1_out0;
  EXPECT_TRUE(AttrUtils::GetListInt(op_desc, "li1", li1_out0));
  EXPECT_EQ(li1_out0, std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(UtestGeAttrValue, SetListIntGetByGeAttrValue) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  EXPECT_TRUE(op_desc);

  EXPECT_TRUE(AttrUtils::SetListInt(op_desc, "li1", std::vector<int64_t>({1,2,3,4,5})));
  auto names_to_value = AttrUtils::GetAllAttrs(op_desc);
  auto iter = names_to_value.find("li1");
  EXPECT_NE(iter, names_to_value.end());

  std::vector<int64_t> li1_out;
  auto &ge_value = iter->second;
  EXPECT_EQ(ge_value.GetValue(li1_out), GRAPH_SUCCESS);
  EXPECT_EQ(li1_out, std::vector<int64_t>({1,2,3,4,5}));

  li1_out.clear();
  EXPECT_EQ(ge_value.GetValue<std::vector<int64_t>>(li1_out), GRAPH_SUCCESS);
  EXPECT_EQ(li1_out, std::vector<int64_t>({1,2,3,4,5}));
}

TEST_F(UtestGeAttrValue, SetGetAttr_GeTensor) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("const1", "Identity");
  GeTensorDesc td;
  td.SetShape(GeShape(std::vector<int64_t>({1,100})));
  td.SetOriginShape(GeShape(std::vector<int64_t>({1,100})));
  td.SetDataType(DT_FLOAT);
  td.SetFormat(FORMAT_ND);
  float data[100];
  for (size_t i = 0; i < 100; ++i) {
    data[i] = 1.0 * i;
  }
  auto tensor = std::make_shared<GeTensor>(td, reinterpret_cast<const uint8_t *>(data), sizeof(data));
  EXPECT_NE(tensor, nullptr);

  EXPECT_TRUE(AttrUtils::SetTensor(op_desc, "t", tensor));
  tensor = nullptr;

  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "t", tensor));
  EXPECT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->GetData().GetSize(), sizeof(data));
  auto attr_data = reinterpret_cast<const float *>(tensor->GetData().GetData());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(attr_data[i], data[i]);
  }
  tensor = nullptr;

  EXPECT_TRUE(AttrUtils::MutableTensor(op_desc, "t", tensor));
  EXPECT_NE(tensor, nullptr);

  EXPECT_EQ(tensor->GetData().GetSize(), sizeof(data));
  attr_data = reinterpret_cast<const float *>(tensor->GetData().GetData());
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_FLOAT_EQ(attr_data[i], data[i]);
  }
  tensor = nullptr;
}

TEST_F(UtestGeAttrValue, GetStr) {
  OpDescPtr op_desc = std::make_shared<OpDesc>("Add", "Add");
  EXPECT_TRUE(op_desc);

  std::string add_info = "add_info";
  AttrUtils::SetStr(op_desc, "compile_info_key", add_info);
  const std::string *s2 = AttrUtils::GetStr(op_desc, "compile_info_key");
  EXPECT_NE(s2, nullptr);
  EXPECT_EQ(*s2, add_info);
}

TEST_F(UtestGeAttrValue, SetNullObjectAttr) {
  OpDescPtr op_desc(nullptr);
  EXPECT_EQ(AttrUtils::SetStr(op_desc, "key", "value"), false);
  EXPECT_EQ(AttrUtils::SetInt(op_desc, "key", 0), false);
  EXPECT_EQ(AttrUtils::SetTensorDesc(op_desc, "key", GeTensorDesc()), false);
  GeTensorPtr ge_tensor;
  EXPECT_EQ(AttrUtils::SetTensor(op_desc, "key", ge_tensor), false);
  ConstGeTensorPtr const_ge_tensor;
  EXPECT_EQ(AttrUtils::SetTensor(op_desc, "key", const_ge_tensor), false);
  EXPECT_EQ(AttrUtils::SetBool(op_desc, "key", true), false);
  EXPECT_EQ(AttrUtils::SetBytes(op_desc, "key", Buffer()), false);
  EXPECT_EQ(AttrUtils::SetFloat(op_desc, "key", 1.0), false);
  EXPECT_EQ(AttrUtils::SetGraph(op_desc, "key", nullptr), false);
  EXPECT_EQ(AttrUtils::SetDataType(op_desc, "key", DT_UINT8), false);
  EXPECT_EQ(AttrUtils::SetListDataType(op_desc, "key", {DT_UINT8}), false);
  EXPECT_EQ(AttrUtils::SetListListInt(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListInt(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListTensor(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListBool(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListFloat(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListBytes(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListGraph(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListListFloat(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListTensorDesc(op_desc, "key", {}), false);
  EXPECT_EQ(AttrUtils::SetListStr(op_desc, "key", {}), false);
  std::vector<Buffer> buffer;
  EXPECT_EQ(AttrUtils::SetZeroCopyListBytes(op_desc, "key", buffer), false);
}
TEST_F(UtestGeAttrValue, GetNullObjectAttr) {
  OpDescPtr op_desc(nullptr);
  std::string value;
  EXPECT_EQ(AttrUtils::GetStr(op_desc, "key", value), false);
  int64_t i;
  EXPECT_EQ(AttrUtils::GetInt(op_desc, "key", i), false);
  GeTensorDesc ge_tensor_desc;
  EXPECT_EQ(AttrUtils::GetTensorDesc(op_desc, "key", ge_tensor_desc), false);
  ConstGeTensorPtr const_ge_tensor;
  EXPECT_EQ(AttrUtils::GetTensor(op_desc, "key", const_ge_tensor), false);
  bool flag;
  EXPECT_EQ(AttrUtils::GetBool(op_desc, "key", flag), false);
  Buffer buffer;
  EXPECT_EQ(AttrUtils::GetBytes(op_desc, "key", buffer), false);
  float j;
  EXPECT_EQ(AttrUtils::GetFloat(op_desc, "key", j), false);
  ComputeGraphPtr compute_graph;
  EXPECT_EQ(AttrUtils::GetGraph(op_desc, "key", compute_graph), false);
  DataType data_type;
  EXPECT_EQ(AttrUtils::GetDataType(op_desc, "key", data_type), false);
  std::vector<DataType> data_types;
  EXPECT_EQ(AttrUtils::GetListDataType(op_desc, "key", data_types), false);
  std::vector<std::vector<int64_t>> ints_list;
  EXPECT_EQ(AttrUtils::GetListListInt(op_desc, "key", ints_list), false);
  std::vector<int64_t> ints;
  EXPECT_EQ(AttrUtils::GetListInt(op_desc, "key", ints), false);
  std::vector<ConstGeTensorPtr> tensors;
  EXPECT_EQ(AttrUtils::GetListTensor(op_desc, "key", tensors), false);
  std::vector<bool> flags;
  EXPECT_EQ(AttrUtils::GetListBool(op_desc, "key", flags), false);
  std::vector<float> floats;
  EXPECT_EQ(AttrUtils::GetListFloat(op_desc, "key", floats), false);
  std::vector<Buffer> buffers;
  EXPECT_EQ(AttrUtils::GetListBytes(op_desc, "key", buffers), false);
  std::vector<ComputeGraphPtr> graphs;
  EXPECT_EQ(AttrUtils::GetListGraph(op_desc, "key", graphs), false);
  std::vector<std::vector<float>> floats_list;
  EXPECT_EQ(AttrUtils::GetListListFloat(op_desc, "key", floats_list), false);
  std::vector<GeTensorDesc> tensor_descs;
  EXPECT_EQ(AttrUtils::GetListTensorDesc(op_desc, "key", tensor_descs), false);
  std::vector<std::string> strings;
  EXPECT_EQ(AttrUtils::GetListStr(op_desc, "key", strings), false);
  EXPECT_EQ(AttrUtils::GetZeroCopyListBytes(op_desc, "key", buffers), false);
}
}
