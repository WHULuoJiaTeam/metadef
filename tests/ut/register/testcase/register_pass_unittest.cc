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

#include "inc/external/register/register_pass.h"
#include "graph/debug/ge_log.h"
#include "register/custom_pass_helper.h"

#define protected public
#define private public

namespace ge {
class UtestRegisterPass : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestRegisterPass, GetPriorityTest) {
  ge::PassRegistrationData pass_data;
  pass_data.impl_ = nullptr;
  int32_t ret = pass_data.GetPriority();
  EXPECT_EQ(ret, INT_MAX);
}

TEST_F(UtestRegisterPass, GetPriorityImplIsNotNull) {
  ge::PassRegistrationData pass_data("register_pass");
  int32_t ret = pass_data.GetPriority();
  EXPECT_EQ(ret, INT_MAX);
}

TEST_F(UtestRegisterPass, GetPassNameTest) {
  ge::PassRegistrationData pass_data("registry");
  std::string name = pass_data.GetPassName();
  EXPECT_EQ(name, "registry");

  pass_data.impl_ = nullptr;
  name = pass_data.GetPassName();
  EXPECT_EQ(name, "");
}

TEST_F(UtestRegisterPass, PriorityTest) {
  ge::PassRegistrationData pass_data("registry");
  int32_t priority = -1;
  pass_data.Priority(priority);
  int32_t ret = pass_data.GetPriority();
  EXPECT_EQ(ret, INT_MAX);

  priority = 2;
  pass_data.Priority(priority);
  ret = pass_data.GetPriority();
  EXPECT_EQ(ret, 2);
}

TEST_F(UtestRegisterPass, CustomPassFnTest) {
  CustomPassFunc custom_pass_fn = nullptr;
  ge::PassRegistrationData pass_data("registry");
  pass_data.CustomPassFn(custom_pass_fn);
  auto ret = pass_data.GetCustomPassFn();
  EXPECT_EQ(ret, nullptr);

  custom_pass_fn = std::function<Status(ge::GraphPtr &)>();
  pass_data.impl_ = nullptr;
  pass_data.CustomPassFn(custom_pass_fn);
  ret = pass_data.GetCustomPassFn();
  EXPECT_EQ(ret, nullptr);
}

TEST_F(UtestRegisterPass, CustomPassHelperRunTest) {
  PassRegistrationData pass_data("registry");
  ge::PassReceiver pass_receiver(pass_data);
  CustomPassHelper cust_helper;
  auto graph = std::make_shared<Graph>("test");
  bool ret = cust_helper.Run(graph);
  EXPECT_EQ(ret, SUCCESS);

  PassRegistrationData pass_data2("registry2");
  cust_helper.registration_datas_.insert(pass_data2);
  auto graph2 = std::make_shared<Graph>("test2");
  ret = cust_helper.Run(graph2);
  EXPECT_EQ(ret, SUCCESS);
}
} // namespace ge
