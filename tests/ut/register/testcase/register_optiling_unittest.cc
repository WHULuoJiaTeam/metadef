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
#include "op_tiling.cpp"
#include "graph_builder_utils.h"
namespace optiling {
using ByteBuffer = std::stringstream;
class UtestOpTilingRegister : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

TEST_F(UtestOpTilingRegister, compile_info_test) {
  std::string key_str = "test_key";
  std::string val_str = "test_value";
  optiling::utils::OpCompileInfo compile_info(key_str, val_str);
  key_str = "key_0";
  val_str = "val_0";
  compile_info.SetKey(key_str);
  compile_info.SetValue(val_str);
  EXPECT_EQ(compile_info.GetKey(), key_str);
  EXPECT_EQ(compile_info.GetValue(), val_str);
}

TEST_F(UtestOpTilingRegister, run_info_test) {
  int temp = 2;
  std::string str = "test";
  int64_t temp1;
  ByteBuffer str_stream;
  optiling::utils::OpRunInfo run_info;
  run_info.AddTilingData(str.c_str());
  str += "_test1_test2";
  run_info.AddTilingData(str.c_str());
  run_info.SetTilingKey(uint32_t(temp));
  run_info.SetClearAtomic(true);
  run_info.SetBlockDim(uint32_t(temp));
  run_info.AddWorkspace(int64_t(temp));

  EXPECT_EQ(run_info.GetBlockDim(), uint32_t(temp));
  EXPECT_EQ(run_info.GetClearAtomic(), true);
  EXPECT_EQ(run_info.GetTilingKey(), uint32_t(temp));
  run_info.GetWorkspace(size_t(0), temp1);
  EXPECT_EQ(temp1, int64_t(temp));
  str_stream = run_info.GetAllTilingData();
  EXPECT_EQ(str_stream.str(), "test_test1_test2"+str);
}

TEST_F(UtestOpTilingRegister, OpParaCalculateV2_test_unregistered) {
  int temp = 2;
  ge::ut::GraphBuilder builder = ge::ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();
  auto data_node = graph->FindNode("Data");
  optiling::utils::OpRunInfo run_info(uint32_t(temp), true, uint32_t(temp));
  EXPECT_EQ(ge::GRAPH_FAILED, OpParaCalculateV2(*data, run_info));
}

TEST_F(UtestOpTilingRegister, OpAtomicCalculateV2_test_unregistered) {
  int temp = 2;
  ge::ut::GraphBuilder builder = ge::ut::GraphBuilder("graph");
  auto data = builder.AddNode("Data", "Data", 1, 1);
  auto graph = builder.GetGraph();
  auto data_node = graph->FindNode("Data");
  optiling::utils::OpRunInfo run_info(uint32_t(temp), true, uint32_t(temp));
  EXPECT_EQ(ge::GRAPH_FAILED, OpAtomicCalculateV2(*data, run_info));
}

TEST_F(UtestOpTilingRegister, TbeOpTilingPyInterfaceEx2_test_unregistered) {
  int temp_num = 3;
  std::string type_str = "conv";
  std::string temp_str = "temp";
  std::string inputs_str = "[{"shape":[8, 32, 28, 16], "format":"NC1HWC0"}]";
  std::string outputs_str = "[{"shape":[8, 32, 28, 16], "format":"NC1HWC0"}]";
  const char *optype = temp.c_str();
  const char *compile_info = temp_str.c_str();
  const char *inputs = inputs_str.c_str();
  const char *outputs = outputs_str.c_str();
  char *run_info_json;
  size_t run_info_len;
  const char *compile_info_hash;
  uint64_t *elapse;
  int res = TbeOpTilingPyInterfaceEx2(optype, compile_info, inputs, outputs,
                                      run_info_json, run_info_len, compile_info_hash, elapse);
  EXPECT_EQ(res, 0);
}

}  // namespace ge
