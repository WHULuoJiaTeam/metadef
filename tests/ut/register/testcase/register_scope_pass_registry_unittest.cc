/**
 * Copyright 2021-2021 Huawei Technologies Co., Ltd
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
#include <gmock/gmock.h>
#include <vector>
#include <iostream>

#define private public
#define protected public
#include "external/register/scope/scope_fusion_pass_register.h"
#include "register/scope/scope_graph_impl.h"
#include "register/scope/scope_pass_impl.h"
#include "register/scope/scope_pass_registry_impl.h"
#undef private
#undef protected

using namespace ge;
namespace {
class ScopeBasePassChild : public ScopeBasePass {
protected:
  std::vector<ScopeFusionPatterns> DefinePatterns() {
  std::vector<ScopeFusionPatterns> ret;
    return ret;
  }
  std::string PassName() {
    return std::string("passName");
  }
  Status LastMatchScopesAndOPs(std::shared_ptr<ScopeGraph> &scope_graph,
                               std::vector<ScopesResult> &results) {
    return SUCCESS;
  }
  void GenerateFusionResult(const std::vector<Scope *> &scopes,
                            FusionScopesResult *fusion_rlt) {
    return;
  }
};
}

class UtestScopePassRegistry : public testing::Test {
public:
  std::unique_ptr<ScopeBasePass> scoPassPtr1;
  std::unique_ptr<ScopeBasePass> scoPassPtr2;

protected:
  void SetUp() {
    const char *regName1 = "regScoFusnPass1";
    REGISTER_SCOPE_FUSION_PASS(regName1, ScopeBasePassChild, true);
    scoPassPtr1 = 
      ScopeFusionPassRegistry::GetInstance().impl_->CreateScopeFusionPass(std::string(regName1));

    const char *regName2 = "regScoFusnPass2";
    REGISTER_SCOPE_FUSION_PASS(regName2, ScopeBasePassChild, true);
    scoPassPtr2 = 
      ScopeFusionPassRegistry::GetInstance().impl_->CreateScopeFusionPass(std::string(regName2));
  }
  void TearDown() {}
};

TEST_F(UtestScopePassRegistry, ScopeFusionPassRegistryRecreat) {
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  scope_graph->Init();
  std::vector<ScopesResult> results;
  EXPECT_EQ(scoPassPtr2->LastMatchScopesAndOPs(scope_graph, results), SUCCESS);

  REGISTER_SCOPE_FUSION_PASS("regScoFusnPass2", ScopeBasePassChild, true);
  EXPECT_EQ(scoPassPtr2->LastMatchScopesAndOPs(scope_graph, results), SUCCESS);
}

TEST_F(UtestScopePassRegistry, ScopeFusionPassRegistryRegister) {
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  scope_graph->Init();
  std::vector<ScopesResult> results;
  EXPECT_EQ(scoPassPtr1->LastMatchScopesAndOPs(scope_graph, results), SUCCESS);
  EXPECT_EQ(scoPassPtr2->LastMatchScopesAndOPs(scope_graph, results), SUCCESS);

  std::vector<std::string> nameList = ScopeFusionPassRegistry::GetInstance().impl_->GetAllRegisteredPasses();
  EXPECT_EQ(nameList[0]=="regScoFusnPass1", true);
  EXPECT_EQ(nameList[1]=="regScoFusnPass2", true);
}

TEST_F(UtestScopePassRegistry, SetPassEnableFlag) {
  bool retBool;
  std::unique_ptr<ScopeBasePass> scoPassPtr;
  const char *regName = "regScoFusnPass";
  REGISTER_SCOPE_FUSION_PASS(nullptr, ScopeBasePassChild, true);   // test for fail
  REGISTER_SCOPE_FUSION_PASS(regName, ScopeBasePassChild, true);
  scoPassPtr = ScopeFusionPassRegistry::GetInstance().impl_->CreateScopeFusionPass(std::string(regName));

  retBool = ScopeFusionPassRegistry::GetInstance().impl_->SetPassEnableFlag(std::string(regName), false);
  EXPECT_EQ(retBool, true);
  retBool = ScopeFusionPassRegistry::GetInstance().impl_->SetPassEnableFlag(std::string("test"), false);
  EXPECT_EQ(retBool, false);
}

TEST_F(UtestScopePassRegistry, GetCreateFnWithDisableFlag) {
  bool retBool;
  const char *regName = "regScoFusnPass1";
  
  std::shared_ptr<ScopeGraph> scope_graph = std::make_shared<ScopeGraph>();
  scope_graph->Init();
  std::vector<ScopesResult> results;

  std::unique_ptr<ScopeBasePass> scoPassPtr1 = 
    ScopeFusionPassRegistry::GetInstance().impl_->CreateScopeFusionPass(std::string(regName));
  EXPECT_EQ(scoPassPtr1->LastMatchScopesAndOPs(scope_graph, results), SUCCESS);

  retBool = ScopeFusionPassRegistry::GetInstance().impl_->SetPassEnableFlag(std::string(regName), false);
  ASSERT_EQ(retBool, true);

  std::unique_ptr<ScopeBasePass> scoPassPtr2 = 
    ScopeFusionPassRegistry::GetInstance().impl_->CreateScopeFusionPass(std::string(regName));
  EXPECT_EQ(scoPassPtr2, nullptr);
}
