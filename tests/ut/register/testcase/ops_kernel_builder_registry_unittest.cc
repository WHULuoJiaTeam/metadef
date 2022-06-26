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

#include "register/ops_kernel_builder_registry.h"
#include "graph/debug/ge_log.h"

#define protected public
#define private public

namespace ge {
class UtestOpsKernelBuilderRegistry : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestOpsKernelBuilderRegistry, GetAllKernelBuildersTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    OpsKernelBuilderPtr opsptr = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.kernel_builders_.insert(pair<std::string, OpsKernelBuilderPtr>("ops1", opsptr));
    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 1);
    EXPECT_EQ(ops_map["ops1"], opsptr);
}

TEST_F(UtestOpsKernelBuilderRegistry, RegisterTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    std::string name = "register1";
    OpsKernelBuilderPtr instance = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name, instance);
    std::map<std::string, OpsKernelBuilderPtr> kernel_builders_;
    kernel_builders_ = ops_registry.GetAll();
    EXPECT_EQ(kernel_builders_.size(), 1);
    EXPECT_EQ(kernel_builders_["register1"], instance);
}

TEST_F(UtestOpsKernelBuilderRegistry, UnregisterTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    std::string name1 = "register1";
    OpsKernelBuilderPtr opsPtr1 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name1, opsPtr1);

    std::string name2 = "register2";
    OpsKernelBuilderPtr opsPtr2 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name2, opsPtr2);

    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 2);
    EXPECT_EQ(ops_map["register1"], opsPtr1);
    EXPECT_EQ(ops_map["register2"], opsPtr2);

    ops_registry.Unregister("register1");
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 1);
    EXPECT_EQ(ops_map.count("register1"), 0);
}

TEST_F(UtestOpsKernelBuilderRegistry, UnregisterAllTest) {
    ge::OpsKernelBuilderRegistry ops_registry;
    std::string name1 = "register1";
    OpsKernelBuilderPtr opsPtr1 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name1, opsPtr1);

    std::string name2 = "register2";
    OpsKernelBuilderPtr opsPtr2 = std::shared_ptr<OpsKernelBuilder>();
    ops_registry.Register(name2, opsPtr2);

    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 2);
    EXPECT_EQ(ops_map["register1"], opsPtr1);
    EXPECT_EQ(ops_map["register2"], opsPtr2);

    ops_registry.UnregisterAll();
    ops_map = ops_registry.GetAll();
    EXPECT_EQ(ops_map.size(), 0);
    EXPECT_EQ(ops_map.count("register1"), 0);
    EXPECT_EQ(ops_map.count("register2"), 0);
}

TEST_F(UtestOpsKernelBuilderRegistry, OpsKernelBuilderRegistrarTest) {
    std::string name = "register";
    ge::OpsKernelBuilderRegistrar::CreateFn fn = nullptr;
    OpsKernelBuilderRegistrar ops_rar(name, fn);
    std::map<std::string, OpsKernelBuilderPtr> ops_map;
    ops_map = OpsKernelBuilderRegistry::GetInstance().GetAll();
    EXPECT_EQ(ops_map.size(), 1);
}

} // namespace ge
