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

#include "graph/utils/type_utils.h"
#include <gtest/gtest.h>

namespace ge {
class UtestTypeUtils : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestTypeUtils, IsFormatValid) {
  ASSERT_TRUE(TypeUtils::IsFormatValid(FORMAT_NCHW));
  ASSERT_FALSE(TypeUtils::IsFormatValid(FORMAT_END));
}

TEST_F(UtestTypeUtils, IsInternalFormat) {
  ASSERT_TRUE(TypeUtils::IsInternalFormat(FORMAT_FRACTAL_Z));
  ASSERT_FALSE(TypeUtils::IsInternalFormat(FORMAT_RESERVED));
}

TEST_F(UtestTypeUtils, FormatToSerialString) {
  ASSERT_EQ(TypeUtils::FormatToSerialString(FORMAT_NCHW), "NCHW");
  ASSERT_EQ(TypeUtils::FormatToSerialString(FORMAT_END), "END");
  ASSERT_EQ(TypeUtils::FormatToSerialString(static_cast<Format>(GetFormatFromSub(FORMAT_FRACTAL_Z, 1))), "FRACTAL_Z:1");
  ASSERT_EQ(TypeUtils::FormatToSerialString(static_cast<Format>(GetFormatFromSub(FORMAT_END, 1))), "END:1");
}

TEST_F(UtestTypeUtils, SerialStringToFormat) {
  ASSERT_EQ(TypeUtils::SerialStringToFormat("NCHW"), FORMAT_NCHW);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("INVALID"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:1"), GetFormatFromSub(FORMAT_FRACTAL_Z, 1));
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:01"), GetFormatFromSub(FORMAT_FRACTAL_Z, 1));
  ASSERT_EQ(TypeUtils::SerialStringToFormat("INVALID:1"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:1%"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:"), FORMAT_RESERVED);  // invalid_argument exception
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:65535"), GetFormatFromSub(FORMAT_FRACTAL_Z, 0xffff));
  ASSERT_EQ(TypeUtils::SerialStringToFormat("FRACTAL_Z:65536"), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtils, DataFormatToFormat) {
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW"), FORMAT_NCHW);
  ASSERT_EQ(TypeUtils::DataFormatToFormat("INVALID"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW:1"), GetFormatFromSub(FORMAT_NCHW, 1));
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW:01"), GetFormatFromSub(FORMAT_NCHW, 1));
  ASSERT_EQ(TypeUtils::DataFormatToFormat("INVALID:1"), FORMAT_RESERVED);
  ASSERT_EQ(TypeUtils::DataFormatToFormat("NCHW:1%"), FORMAT_RESERVED);
}

TEST_F(UtestTypeUtils, IsDataTypeValid) {
  ASSERT_EQ(TypeUtils::IsDataTypeValid(DT_MAX), false);
}
}
