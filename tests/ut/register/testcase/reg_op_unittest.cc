/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "graph/operator_reg.h"
#include <gtest/gtest.h>

#include "graph/utils/op_desc_utils.h"
namespace ge {
class RegisterOpUnittest : public testing::Test {};

REG_OP(AttrIrNameRegSuccess1)
    .ATTR(AttrInt, Int, 0)
    .ATTR(AttrFloat, Float, 0.0)
    .ATTR(AttrBool, Bool, true)
    .ATTR(AttrTensor, Tensor, Tensor())
    .ATTR(AttrType, Type, DT_INT32)
    .ATTR(AttrString, String, "")
    .ATTR(AttrAscendString, AscendString, "")
    .ATTR(AttrListInt, ListInt, {})
    .ATTR(AttrListFloat, ListFloat, {})
    .ATTR(AttrListBool, ListBool, {})
    .ATTR(AttrListTensor, ListTensor, {})
    .ATTR(AttrListType, ListType, {})
    .ATTR(AttrListString, ListString, {})
    .ATTR(AttrListAscendString, ListAscendString, {})
    .ATTR(AttrBytes, Bytes, {})
    .ATTR(AttrListListInt, ListListInt, {})

    .REQUIRED_ATTR(ReqAttrInt, Int)
    .REQUIRED_ATTR(ReqAttrFloat, Float)
    .REQUIRED_ATTR(ReqAttrBool, Bool)
    .REQUIRED_ATTR(ReqAttrTensor, Tensor)
    .REQUIRED_ATTR(ReqAttrType, Type)
    .REQUIRED_ATTR(ReqAttrString, String)
    .REQUIRED_ATTR(ReqAttrAscendString, AscendString)

    .REQUIRED_ATTR(ReqAttrListInt, ListInt)
    .REQUIRED_ATTR(ReqAttrListFloat, ListFloat)
    .REQUIRED_ATTR(ReqAttrListBool, ListBool)
    .REQUIRED_ATTR(ReqAttrListTensor, ListTensor)
    .REQUIRED_ATTR(ReqAttrListType, ListType)
    .REQUIRED_ATTR(ReqAttrListString, ListString)
    .REQUIRED_ATTR(ReqAttrListAscendString, ListAscendString)
    .REQUIRED_ATTR(ReqAttrBytes, Bytes)
    .REQUIRED_ATTR(ReqAttrListListInt, ListListInt)

    //.ATTR(AttrNamedAttrs, NamedAttrs, NamedAttrs())
    //.ATTR(AttrListNamedAttrs, ListNamedAttrs, {})
    .OP_END_FACTORY_REG(AttrIrNameRegSuccess1);

TEST_F(RegisterOpUnittest, AttrIrNameRegSuccess) {
  auto op = OperatorFactory::CreateOperator("AttrIrNameRegSuccess1Op", "AttrIrNameRegSuccess1");
  auto op_desc = OpDescUtils::GetOpDescFromOperator(op);
  ASSERT_NE(op_desc, nullptr);
  const auto &ir_names = op_desc->GetIrAttrNames();
  EXPECT_EQ(ir_names,
            std::vector<std::string>({"AttrInt",
                                      "AttrFloat",
                                      "AttrBool",
                                      "AttrTensor",
                                      "AttrType",
                                      "AttrString",
                                      "AttrAscendString",
                                      "AttrListInt",
                                      "AttrListFloat",
                                      "AttrListBool",
                                      "AttrListTensor",
                                      "AttrListType",
                                      "AttrListString",
                                      "AttrListAscendString",
                                      "AttrBytes",
                                      "AttrListListInt",
                                      "ReqAttrInt",
                                      "ReqAttrFloat",
                                      "ReqAttrBool",
                                      "ReqAttrTensor",
                                      "ReqAttrType",
                                      "ReqAttrString",
                                      "ReqAttrAscendString",
                                      "ReqAttrListInt",
                                      "ReqAttrListFloat",
                                      "ReqAttrListBool",
                                      "ReqAttrListTensor",
                                      "ReqAttrListType",
                                      "ReqAttrListString",
                                      "ReqAttrListAscendString",
                                      "ReqAttrBytes",
                                      "ReqAttrListListInt"}));
}
}  // namespace ge