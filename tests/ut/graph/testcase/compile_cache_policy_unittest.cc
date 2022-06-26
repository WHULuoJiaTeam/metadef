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
#include "graph/compile_cache_policy/compile_cache_policy.h"
#include "graph/compile_cache_policy/compile_cache_state.h"

namespace ge {
class UtestCompileCachePolicy : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_1) {
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY, ge::AgingPolicyType::AGING_POLICY_LRU);
  ASSERT_NE(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPSuccess_2) {
  auto mp_ptr = PolicyManager::GetInstance().GetMatchPolicy(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY);
  auto ap_ptr = PolicyManager::GetInstance().GetAgingPolicy(ge::AgingPolicyType::AGING_POLICY_LRU);
  auto ccp = ge::CompileCachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_NE(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPFailed_1) {
  auto mp_ptr = PolicyManager::GetInstance().GetMatchPolicy(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY);
  auto ap_ptr = nullptr;
  auto ccp = ge::CompileCachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_EQ(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, CreateCCPFailed_2) {
  auto mp_ptr = nullptr;
  auto ap_ptr = nullptr;
  auto ccp = ge::CompileCachePolicy::Create(mp_ptr, ap_ptr);
  ASSERT_EQ(ccp, nullptr);
}

TEST_F(UtestCompileCachePolicy, AddSameCache) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  CacheItemId cache_id_same = ccp->AddCache(cache_desc);
  ASSERT_EQ(cache_id_same, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentOptypeCache) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  cache_desc.SetOpType("another_op");
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentUniqueIdCache) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  cache_desc.SetScopeId({1, 2});
  ASSERT_EQ(cache_desc.scope_id_.size(), 2);
  ASSERT_EQ(cache_desc.scope_id_[0], 1);
  ASSERT_EQ(cache_desc.scope_id_[1], 2);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  cache_desc.SetScopeId({1, 3});
  ASSERT_EQ(cache_desc.scope_id_.size(), 2);
  ASSERT_EQ(cache_desc.scope_id_[0], 1);
  ASSERT_EQ(cache_desc.scope_id_[1], 3);
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentBinarySizeCache) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentBinaryValueCache) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  uint8_t value_another = 8;
  BinaryHolder holder1(&value_another, 1);
  cache_desc.other_desc_.clear();
  cache_desc.AddBinary(holder1);
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, AddDifferentTensorFormatCache) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  cache_desc.tensor_info_args_vec_[0].format_ = ge::FORMAT_NCHW;
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);
}

TEST_F(UtestCompileCachePolicy, CacheFindFailBecauseRangeFirst) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  CompileCacheDesc cache_desc_match = cache_desc;
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  SmallVector<int64_t, kDefaultDimsNum> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  ASSERT_EQ(tensor_info.shape_.size(), 2);
  ASSERT_EQ(tensor_info.shape_[0], -1);
  ASSERT_EQ(tensor_info.shape_[1], -1);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{0,5};
  tensor_info.SetShape(shape_match);
  cache_desc_match.AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id_find, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, CacheFindFailBecauseRangeSecond) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  CompileCacheDesc cache_desc_match = cache_desc;
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{5,11};
  tensor_info.SetShape(shape_match);
  cache_desc_match.AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id_find, KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, CacheFindSuccessBecauseUnknownRank) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  CompileCacheDesc cache_desc_match = cache_desc;
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-2};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  cache_desc.AddTensorInfo(tensor_info);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{5,11};
  tensor_info.SetShape(shape_match);
  cache_desc_match.AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id_find, cache_id);
}


TEST_F(UtestCompileCachePolicy, CacheFindSuccessCommonTest) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  CompileCacheDesc cache_desc_match = cache_desc;
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<int64_t> shape_match{5,5};
  tensor_info.SetShape(shape_match);
  cache_desc_match.AddTensorInfo(tensor_info);
  CacheItemId cache_id_find = ccp->FindCache(cache_desc_match);
  ASSERT_EQ(cache_id, cache_id_find);
  ge::CompileCachePolicy ccp1;
  ASSERT_EQ(ccp1.FindCache(cache_desc_match), KInvalidCacheItemId);
}

TEST_F(UtestCompileCachePolicy, CacheDelTest) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  std::vector<int64_t> shape{-1,-1};
  tensor_info.SetShape(shape);
  tensor_info.SetOriginShape(shape);
  std::vector<std::pair<int64_t, int64_t>> ranges{{1,10}, {1,10}};
  tensor_info.SetShapeRange(ranges);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  cache_desc.AddBinary(holder);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  CacheItemId cache_id_same = ccp->AddCache(cache_desc);
  ASSERT_EQ(cache_id_same, cache_id);

  cache_desc.SetOpType("another_op");
  CacheItemId cache_id_another = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id_another, -1);
  ASSERT_NE(cache_id_another, cache_id);

  std::vector<CacheItemId> delete_item{cache_id, cache_id_another};
  std::vector<CacheItemId> delete_item_ret = ccp->DeleteCache(delete_item);
  ASSERT_EQ(delete_item_ret.size(), 2);
}

TEST_F(UtestCompileCachePolicy, AgingCacheSuccess_1) {
  CompileCacheDesc cache_desc;
  cache_desc.SetOpType("test_op");
  TensorInfoArgs tensor_info(ge::FORMAT_ND, ge::FORMAT_ND, ge::DT_FLOAT16);
  cache_desc.AddTensorInfo(tensor_info);
  uint8_t value = 9;
  uint8_t *data = &value;
  BinaryHolder holder(data, 1);
  BinaryHolder holder_new;
  holder_new = holder;
  ASSERT_NE(holder_new.GetDataPtr(), nullptr);
  cache_desc.AddBinary(holder_new);
  auto ccp = ge::CompileCachePolicy::Create(ge::MatchPolicyType::MATCH_POLICY_EXACT_ONLY,
      ge::AgingPolicyType::AGING_POLICY_LRU);
  CacheItemId cache_id = ccp->AddCache(cache_desc);
  ASSERT_NE(cache_id, -1);

  std::vector<CacheItemId> del_item = ccp->DoAging();
  ASSERT_EQ(cache_id, del_item[0]);
}
}
