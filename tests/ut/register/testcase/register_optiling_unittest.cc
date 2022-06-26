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
#include "register/op_tiling_registry.h"
#include "op_tiling/op_tiling.cc"
#include "common/sgt_slice_type.h"
#include "graph_builder_utils.h"
using namespace std;
using namespace ge;
using namespace ffts;
namespace optiling {
using ByteBuffer = std::stringstream;
class RegisterOpTilingUT : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(RegisterOpTilingUT, byte_buffer_test) {
  ByteBuffer stream;
  std::string str = "test";

  ByteBuffer &str_stream1 = ByteBufferPut(stream, str);

  string value;
  ByteBuffer &str_stream2 = ByteBufferGet(stream, str);

  char *dest = nullptr;
  size_t size = ByteBufferGetAll(stream, dest, 2);
  cout << size << endl;
}

TEST_F(RegisterOpTilingUT, op_run_info_test) {
  std::shared_ptr<utils::OpRunInfo> run_info = make_shared<utils::OpRunInfo>(8, true, 64);
  int64_t work_space;
  graphStatus ret = run_info->GetWorkspace(0, work_space);
  EXPECT_EQ(ret, GRAPH_FAILED);
  vector<int64_t> work_space_vec = {10, 20, 30, 40};
  run_info->SetWorkspaces(work_space_vec);
  ret = run_info->GetWorkspace(1, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 20);
  EXPECT_EQ(run_info->GetWorkspaceNum(), 4);
  string str = "test";
  run_info->AddTilingData(str);

  const ByteBuffer &tiling_data = run_info->GetAllTilingData();

  std::shared_ptr<utils::OpRunInfo> run_info_2 = make_shared<utils::OpRunInfo>(*run_info);
  ret = run_info_2->GetWorkspace(2, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 30);

  utils::OpRunInfo run_info_3 = *run_info;
  ret = run_info_3.GetWorkspace(3, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 40);

  utils::OpRunInfo &run_info_4 = *run_info;
  ret = run_info_4.GetWorkspace(0, work_space);
  EXPECT_EQ(ret, GRAPH_SUCCESS);
  EXPECT_EQ(work_space, 10);
}

TEST_F(RegisterOpTilingUT, op_compile_info_test) {
  std::shared_ptr<utils::OpCompileInfo> compile_info = make_shared<utils::OpCompileInfo>();
  string str_key = "key";
  string str_value = "value";
  AscendString key(str_key.c_str());
  AscendString value(str_value.c_str());
  compile_info->SetKey(key);
  compile_info->SetValue(value);

  std::shared_ptr<utils::OpCompileInfo> compile_info_2 = make_shared<utils::OpCompileInfo>(key, value);
  EXPECT_EQ(compile_info_2->GetKey() == key, true);
  EXPECT_EQ(compile_info_2->GetValue() == value, true);

  std::shared_ptr<utils::OpCompileInfo> compile_info_3 = make_shared<utils::OpCompileInfo>(str_key, str_value);
  EXPECT_EQ(compile_info_3->GetKey() == key, true);
  EXPECT_EQ(compile_info_3->GetValue() == value, true);

  std::shared_ptr<utils::OpCompileInfo> compile_info_4 = make_shared<utils::OpCompileInfo>(*compile_info);
  EXPECT_EQ(compile_info_4->GetKey() == key, true);
  EXPECT_EQ(compile_info_4->GetValue() == value, true);

  utils::OpCompileInfo compile_info_5 = *compile_info;
  EXPECT_EQ(compile_info_5.GetKey() == key, true);
  EXPECT_EQ(compile_info_5.GetValue() == value, true);

  utils::OpCompileInfo &compile_info_6 = *compile_info;
  EXPECT_EQ(compile_info_6.GetKey() == key, true);
  EXPECT_EQ(compile_info_6.GetValue() == value, true);
}

TEST_F(RegisterOpTilingUT, te_op_paras_test) {
  OpDescPtr op_desc = make_shared<OpDesc>("relu", OP_TYPE_DYNAMIC_ATOMIC_ADDR_CLEAN);
  GeShape shape({1,4,1,1});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  int32_t attr_value = 1024;
  AttrUtils::SetInt(op_desc, "some_int_attr", attr_value);
  vector<int64_t> attr_vec = {11, 22, 33, 44};
  AttrUtils::SetListInt(op_desc, "some_int_vec", attr_vec);
  TeOpParas op_param;
  op_param.op_type = op_desc->GetType();
  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  size_t size = 0;
  op_param.var_attrs.GetData("some_int_attr", "xxx", size);
  op_param.var_attrs.GetData("some_int_attr", "Int32", size);
  op_param.var_attrs.GetData("some_int_vec", "ListInt32", size);
}

bool op_tiling_stub(const Operator &op, const utils::OpCompileInfo &compile_info, utils::OpRunInfo &run_info) {
  return true;
}

REGISTER_OP_TILING_V2(ReluV2, op_tiling_stub);

TEST_F(RegisterOpTilingUT, OpFftsCalculateV2_1) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapPtr slice_info_ptr;
  slice_info_ptr = std::make_shared<ThreadSliceMap>();
  DimRange dim;
  dim.lower = 0;
  dim.higher = 1;
  vector<DimRange> vec_1;
  vec_1.push_back(dim);
  vector<vector<DimRange>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->thread_scope_id = 1;
  slice_info_ptr->thread_id = 2222;
  slice_info_ptr->thread_mode = true;
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  (void)node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  GeShape shape({4,1,3,4,16});
  GeTensorDesc tensor_desc(shape, ge::FORMAT_NCHW, ge::DT_FLOAT);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  std::vector<OpRunInfoV2> op_run_info;
  ge::graphStatus ret = OpFftsCalculateV2(*node, op_run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  auto dstAnchor = node->GetInDataAnchor(0);
  ge::AnchorUtils::SetStatus(dstAnchor, ge::ANCHOR_DATA);
  ret = OpFftsCalculateV2(*node, op_run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

// slice instance over
TEST_F(RegisterOpTilingUT, OpFftsCalculateV2_2) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapPtr slice_info_ptr;
  slice_info_ptr = std::make_shared<ThreadSliceMap>();
  DimRange dim;
  dim.lower = 0;
  dim.higher = 1;
  vector<DimRange> vec_1;
  vec_1.push_back(dim);
  vector<vector<DimRange>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->thread_scope_id = 1;
  slice_info_ptr->thread_id = 2222;
  slice_info_ptr->thread_mode = true;
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 4;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  (void)node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  GeShape shape({4,1,3,4,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  string compile_info_key = "compile_info_key";
  string compile_info_json = "compile_info_json";
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_KEY, compile_info_key);
  (void)ge::AttrUtils::SetStr(op_desc, COMPILE_INFO_JSON, compile_info_json);
  std::vector<OpRunInfoV2> op_run_info;
  ge::graphStatus ret = OpFftsCalculateV2(*node, op_run_info);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, PostProcCalculateV2_SUCCESS) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  Operator op = OpDescUtils::CreateOperatorFromNode(node);
  OpDescPtr op_desc = node->GetOpDesc();
  (void)ge::AttrUtils::SetStr(op_desc, "_alias_engine_name", "TEST");
  std::vector<int64_t> workspaces = { 1, 2, 3};
  OpRunInfoV2 run_info;
  run_info.SetWorkspaces(workspaces);
  workspaces.emplace_back(5);
  op_desc->SetWorkspaceBytes(workspaces);
  ge::graphStatus ret = PostProcCalculateV2(op, run_info);
  EXPECT_EQ(ret, ge::GRAPH_SUCCESS);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBySliceInfo1) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapPtr slice_info_ptr;
  slice_info_ptr = std::make_shared<ThreadSliceMap>();
  DimRange dim;
  dim.lower = 0;
  dim.higher = 1;
  vector<DimRange> vec_1;
  vec_1.push_back(dim);
  vector<vector<DimRange>> vec_2;
  vector<vector<DimRange>> vec_3;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  vec_3.push_back(vec_1);
  slice_info_ptr->thread_scope_id = 1;
  slice_info_ptr->thread_id = 2222;
  slice_info_ptr->thread_mode = true;
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_3);
  (void)node->GetOpDesc()->SetExtAttr(ffts::kAttrSgtStructInfo, slice_info_ptr);
  GeShape shape({4,1,3,4,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  vector<vector<int64_t>> ori_shape;
  vector<uint32_t> tmp = {};
  vector<uint32_t> tmp1 = { 0, 2};
  vector<vector<uint32_t>> in_out_idx;
  in_out_idx.push_back(tmp);
  in_out_idx.push_back(tmp1);
  auto ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 2, ori_shape, in_out_idx);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  op_desc->AddOutputDesc("y", tensor_desc);
  ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 0, ori_shape, in_out_idx);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

TEST_F(RegisterOpTilingUT, UpDateNodeShapeBySliceInfo2) {
  auto root_builder = ut::GraphBuilder("root");
  const auto &node = root_builder.AddNode("relu", "ReluV2", 1, 1);
  OpDescPtr op_desc = node->GetOpDesc();
  ThreadSliceMapPtr slice_info_ptr;
  slice_info_ptr = std::make_shared<ThreadSliceMap>();
  DimRange dim;
  dim.lower = 0;
  dim.higher = 1;
  vector<DimRange> vec_1;
  vec_1.push_back(dim);
  vector<vector<DimRange>> vec_2;
  vec_2.push_back(vec_1);
  vec_2.push_back(vec_1);
  slice_info_ptr->thread_scope_id = 1;
  slice_info_ptr->thread_id = 2222;
  slice_info_ptr->thread_mode = true;
  slice_info_ptr->parallel_window_size = 2;
  slice_info_ptr->slice_instance_num = 2;
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->input_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  slice_info_ptr->output_tensor_slice.push_back(vec_2);
  GeShape shape({4,1,3,4,16});
  GeTensorDesc tensor_desc(shape);
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddOutputDesc("y", tensor_desc);
  vector<vector<int64_t>> ori_shape;
  vector<uint32_t> tmp = {0, 1, 2};
  vector<uint32_t> tmp1 = { 0, 2};
  vector<vector<uint32_t>> in_out_idx;
  in_out_idx.push_back(tmp);
  in_out_idx.push_back(tmp1);
  auto ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 0, ori_shape, in_out_idx);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  vector<vector<uint32_t>> in_out_idx1;
  in_out_idx1.push_back(tmp1);
  in_out_idx1.push_back(tmp);
  ret = UpDateNodeShapeBySliceInfo(slice_info_ptr, op_desc, 1, ori_shape, in_out_idx);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
  ret = UpDateNodeShapeBack(op_desc, ori_shape, in_out_idx);
  EXPECT_EQ(ret, ge::GRAPH_FAILED);
}

}  // namespace ge
