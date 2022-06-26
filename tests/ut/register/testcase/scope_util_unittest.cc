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
#include <gmock/gmock.h>
#include <vector>

#define private public
#define protected public
#include "inc/external/register/scope/scope_fusion_pass_register.h"
#undef private
#undef protected

using namespace std;
using namespace testing;

namespace ge {

class ScopeUtilUt : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(ScopeUtilUt, StringReplaceAll) {
  AscendString str = ScopeUtil::StringReplaceAll("123456", "123", "abc");
  EXPECT_EQ(str.GetString(), string("abc456"));

  string str2 = ScopeUtil::StringReplaceAll(string("abc456"), string("456"), string("def"));
  EXPECT_EQ(str2, string("abcdef"));
}

TEST_F(ScopeUtilUt, FreeScopePatterns) {
  std::vector<std::vector<ScopePattern *>> scoPatternSub;
  std::vector<ScopePattern *> scoPatternSubSub1;
  std::vector<ScopePattern *> scoPatternSubSub2;
  ScopePattern *scoPattern1;
  ScopePattern *scoPattern2;

  scoPattern1 = new ScopePattern();
  scoPattern2 = new ScopePattern();
  scoPatternSubSub1.push_back(scoPattern1);
  scoPatternSubSub2.push_back(scoPattern2);

  scoPatternSub.push_back(scoPatternSubSub1);
  scoPatternSub.push_back(scoPatternSubSub2);

  ScopeUtil::FreeScopePatterns(scoPatternSub);
  EXPECT_EQ(scoPatternSub.size(), 0);
}

}  // namespace ge
