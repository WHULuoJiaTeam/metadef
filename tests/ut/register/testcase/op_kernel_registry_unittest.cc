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

#include "register/op_kernel_registry.h"
#include "register/host_cpu_context.h"
#include "graph/debug/ge_log.h"

#define protected public
#define private public

namespace ge {
class UtestOpKernelRegistry : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestOpKernelRegistry, IsRegisteredTest) {
    OpKernelRegistry &op_registry = OpKernelRegistry::GetInstance();
    std::string op_type = "registry";
    bool ret = op_registry.IsRegistered(op_type);
    EXPECT_EQ(ret, false);
}

TEST_F(UtestOpKernelRegistry, HostCpuOpTest) {
  OpKernelRegistry op_registry;
  std::string op_type = "registry";
  OpKernelRegistry::CreateFn fn = nullptr;
  op_registry.RegisterHostCpuOp(op_type, fn);
  std::unique_ptr<HostCpuOp> host_cpu = op_registry.CreateHostCpuOp(op_type);
  EXPECT_EQ(host_cpu, nullptr);
}

TEST_F(UtestOpKernelRegistry, HostCpuOpRegistrarTest) {
  OpKernelRegistry op_registry;
  HostCpuOpRegistrar host_strar(nullptr, []()->::ge::HostCpuOp* {return nullptr;});
  std::string op_type = "registry";
  std::unique_ptr<HostCpuOp> host_cpu = op_registry.CreateHostCpuOp(op_type);
  EXPECT_EQ(host_cpu, nullptr);
}

} // namespace ge
