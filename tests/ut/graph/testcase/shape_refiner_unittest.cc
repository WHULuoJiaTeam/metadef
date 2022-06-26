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
#include "graph/compute_graph.h"
#include "graph/shape_refiner.h"
#include "graph/operator_factory_impl.h"
#include "graph/utils/tensor_utils.h"

namespace ge {
class UtestShapeRefiner : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};

static NodePtr CreateNode(const ComputeGraphPtr &graph, const string &name, const string &type, int in_num, int out_num) {
  OpDescPtr op_desc = std::make_shared<OpDesc>(name, type);
  op_desc->SetStreamId(0);
  static int32_t index = 0;
  op_desc->SetId(index++);

  GeTensorDesc tensor(GeShape(), FORMAT_NCHW, DT_FLOAT);
  tensor.SetOriginFormat(FORMAT_NCHW);
  tensor.SetOriginDataType(DT_FLOAT);
  TensorUtils::SetSize(tensor, 512);
  vector<int64_t> input_offset;
  for (int i = 0; i < in_num; i++) {
    op_desc->AddInputDesc(tensor);
    input_offset.emplace_back(1024);
  }
  op_desc->SetInputOffset(input_offset);

  vector<int64_t> output_offset;
  for (int i = 0; i < out_num; i++) {
    op_desc->AddOutputDesc(tensor);
    output_offset.emplace_back(1024);
  }
  op_desc->SetOutputOffset(output_offset);

  op_desc->SetWorkspace({});
  op_desc->SetWorkspaceBytes({});
  op_desc->SetOpKernelLibName("DNN_VM_RTS_OP_STORE");

  const auto stub_func = [](Operator &op) { return GRAPH_SUCCESS; };
  op_desc->AddInferFunc(stub_func);
  op_desc->AddInferFormatFunc(stub_func);
  op_desc->AddVerifierFunc(stub_func);

  return graph->AddNode(op_desc);
}

TEST_F(UtestShapeRefiner, infer_shape_and_type_for_running) {
  const auto graph = std::make_shared<ComputeGraph>("test_infer_shape");
  auto enter1 = CreateNode(graph, "enter", "Enter", 1, 1);

  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(enter1, true), GRAPH_SUCCESS);

  auto infershape_funcs_back = OperatorFactoryImpl::operator_infershape_funcs_;
  OperatorFactoryImpl::operator_infershape_funcs_.reset(new (std::nothrow) std::map<string, InferShapeFunc>());
  OperatorFactoryImpl::operator_infershape_funcs_->emplace("Merge", [](Operator &op) { return GRAPH_SUCCESS; });
  auto merge1 = CreateNode(graph, "merge1", "StreamMerge", 2, 2);
  merge1->GetOpDesc()->AddInferFunc(nullptr);
  EXPECT_EQ(ShapeRefiner::InferShapeAndTypeForRunning(merge1, true), GRAPH_SUCCESS);
  OperatorFactoryImpl::operator_infershape_funcs_ = infershape_funcs_back;
}
} // namespace ge
