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
#include "graph/compile_cache_policy/compile_cache_hasher.h"
namespace ge {
class UtestCompileCachePolicyHasher : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderConstrutFromPtr) {
  uint8_t data1[2] = {0U,1U};
  BinaryHolder holder1(data1, sizeof(data1));
  const uint8_t *dataPtr = holder1.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size1 = holder1.GetDataLen();
  ASSERT_EQ(size1, sizeof(data1));
  ASSERT_EQ(dataPtr[0], 0);
  ASSERT_EQ(dataPtr[1], 1);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderCopyConstrut) {
  uint8_t data1[2] = {0U,1U};
  BinaryHolder holder1(data1, sizeof(data1));
  const uint8_t *dataPtr = holder1.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size1 = holder1.GetDataLen();
  ASSERT_EQ(size1, sizeof(data1));

  BinaryHolder holder2 = holder1;
  ASSERT_EQ((holder1 != holder2), false);
  ASSERT_NE(holder1.GetDataPtr(), holder2.GetDataPtr());
  BinaryHolder holder3(holder1);
  ASSERT_NE(holder1.GetDataPtr(), holder3.GetDataPtr());
  ASSERT_EQ((holder1 != holder3), false);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderMoveConstrut) {
  uint8_t data1[2] = {0U,1U};
  BinaryHolder holder1(data1, sizeof(data1));

  BinaryHolder holder2 = std::move(holder1);
  ASSERT_EQ(holder1.GetDataPtr(), nullptr);
  ASSERT_NE(holder2.GetDataPtr(), nullptr);
  ASSERT_EQ(holder1.GetDataLen(), 0);
  ASSERT_EQ(holder2.GetDataLen(), sizeof(data1));

  const uint8_t *dataPtr = holder2.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size2 = holder2.GetDataLen();
  ASSERT_EQ(size2, sizeof(data1));
  ASSERT_EQ(dataPtr[0], 0);
  ASSERT_EQ(dataPtr[1], 1);

  BinaryHolder holder3(std::move(holder2));
  ASSERT_EQ(holder2.GetDataPtr(), nullptr);
  ASSERT_NE(holder3.GetDataPtr(), nullptr);
  ASSERT_EQ(holder2.GetDataLen(), 0);
  ASSERT_EQ(holder3.GetDataLen(), sizeof(data1));
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHoldercreateFromUniquePtr) {
  auto data_ptr = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[10]);
  ASSERT_NE(data_ptr, nullptr);
  const uint8_t *data_ptr_real = data_ptr.get();
  auto holder2 = BinaryHolder::createFrom(std::move(data_ptr), 10);
  ASSERT_NE(holder2->GetDataPtr(), nullptr);
  ASSERT_EQ(holder2->GetDataPtr(), data_ptr_real);
  ASSERT_EQ(holder2->GetDataLen(), 10);
  ASSERT_EQ(data_ptr, nullptr);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHoldercreateFromUniquePtrFail) {
  auto holder2 = BinaryHolder::createFrom(nullptr, 0);
  ASSERT_EQ(holder2->GetDataPtr(), nullptr);
  ASSERT_EQ(holder2->GetDataLen(), 0);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderEqual) {
  uint8_t data1[8] = {0U,1U,2U,3U,4U,5U,6U,7U};
  uint8_t data2[8] = {0U,1U,2U,3U,4U,5U,6U,7U};

  BinaryHolder holder1(data1, sizeof(data1));
  const uint8_t *dataPtr = holder1.GetDataPtr();
  ASSERT_NE(dataPtr, nullptr);
  size_t size1 = holder1.GetDataLen();
  ASSERT_EQ(size1, sizeof(data1));

  BinaryHolder holder2(data2, sizeof(data2));
  ASSERT_EQ((holder1 != holder2), false);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderDiffBecauseLength) {
  uint8_t data1[8] = {0U,1U,2U,3U,4U,5U,6U,7U};
  uint8_t data2[9] = {0U,1U,2U,3U,4U,5U,7U,9U,11U};

  BinaryHolder holder1(data1, sizeof(data1));
  ASSERT_EQ(holder1.GetDataLen(), sizeof(data1));
  BinaryHolder holder2(data2, sizeof(data2));
  ASSERT_EQ(holder2.GetDataLen(), sizeof(data2));
  ASSERT_EQ((holder1 != holder2), true);
}

TEST_F(UtestCompileCachePolicyHasher, TestBinaryHolderDiffBecauseVaule) {
  uint8_t data1[8] = {0U,1U,2U,3U,4U,5U,6U,7U};
  uint8_t data2[8] = {1U,1U,2U,3U,4U,5U,6U,7U};

  BinaryHolder holder1(data1, sizeof(data1));
  ASSERT_EQ(holder1.GetDataLen(), sizeof(data1));
  BinaryHolder holder2(data2, sizeof(data2));
  ASSERT_EQ(holder2.GetDataLen(), sizeof(data2));
  ASSERT_EQ((holder1 != holder2), true);
}

TEST_F(UtestCompileCachePolicyHasher, TestGetCacheDescHashWithoutShape) {
  CompileCacheDesc cacheDesc;
  cacheDesc.SetOpType("1111");
  TensorInfoArgs tensor_info_args(FORMAT_ND, FORMAT_ND, DT_BF16);
  cacheDesc.AddTensorInfo(tensor_info_args);
  CacheHashKey id = CompileCacheHasher::GetCacheDescHashWithoutShape(cacheDesc);
  cacheDesc.SetOpType("2222");
  CacheHashKey id_another = CompileCacheHasher::GetCacheDescHashWithoutShape(cacheDesc);
  ASSERT_NE(id, id_another);
}
}
