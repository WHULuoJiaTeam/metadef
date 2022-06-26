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

#ifndef METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_
#define METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_

#include <securec.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

#include "mmpa/mmpa_api.h"
#include "graph/def_types.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/debug/log.h"

namespace ge {
/**
 * @ingroup domi_calibration
 * @brief  Initializes an input array to a specified value
 * @param [in]  n        array initialization length
 * @param [in]  alpha    initialization value
 * @param [out]  output  array to be initialized
 * @return      Status
 */
template<typename Dtype>
Status NnSet(const int32_t n, const Dtype alpha, Dtype *const output) {
  GE_CHECK_NOTNULL(output);

  if (alpha == 0) {
    if ((sizeof(Dtype) * static_cast<size_t>(n)) < SECUREC_MEM_MAX_LEN) {
      const errno_t err =
          memset_s(output, sizeof(Dtype) * static_cast<size_t>(n), 0, sizeof(Dtype) * static_cast<size_t>(n));
      GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
    } else {
      const uint64_t size = static_cast<uint64_t>(sizeof(Dtype) * static_cast<size_t>(n));
      const uint64_t step = SECUREC_MEM_MAX_LEN - (SECUREC_MEM_MAX_LEN % sizeof(Dtype));
      const uint64_t times = size / step;
      const uint64_t remainder = size % step;
      uint64_t i = 0U;
      while (i < times) {
        const errno_t err = memset_s(ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), step, 0, step);
        GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
        i++;
      }
      if (remainder != 0U) {
        const errno_t err =
            memset_s(ValueToPtr(PtrToValue(output) + (i * (step / sizeof(Dtype)))), remainder, 0, remainder);
        GE_CHK_BOOL_RET_STATUS(err == EOK, PARAM_INVALID, "memset_s err");
      }
    }
  }

  for (int32_t i = 0; i < n; ++i) {
    output[i] = alpha;
  }
  return SUCCESS;
}

template<typename T>
class IntegerChecker {
 public:
  template<typename T1>
  static bool Compat(const T1 v) {
    static_assert(((sizeof(T) <= sizeof(uint64_t)) && (sizeof(T1) <= sizeof(uint64_t))),
                  "IntegerChecker can only check integers less than 64 bits");
    if (v >= static_cast<T1>(0)) {
      return static_cast<uint64_t>(v) <= static_cast<uint64_t>(std::numeric_limits<T>::max());
    }
    return static_cast<int64_t>(v) >= static_cast<int64_t>(std::numeric_limits<T>::min());
  }
};
}  // end namespace ge

#define REQUIRE_COMPAT(T, v)                                                                                           \
  do {                                                                                                                 \
    if (!IntegerChecker<T>::Compat((v))) {                                                                             \
      std::stringstream ss;                                                                                            \
      ss << #v << " value " << (v) << " out of " << #T << " range [" << std::numeric_limits<T>::min() << ","           \
         << std::numeric_limits<T>::max() << "]";                                                                      \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                                                      \
      return ge::FAILED;                                                                                               \
    }                                                                                                                  \
  } while (false)

#define REQUIRE_COMPAT_FOR_CHAR(T, v)                                                                                  \
  do {                                                                                                                 \
    if (!IntegerChecker<T>::Compat((v))) {                                                                             \
      std::stringstream ss;                                                                                            \
      ss << #v << " value " << static_cast<int32_t>(v) << " out of " << #T << " range ["                               \
         << static_cast<int32_t>(std::numeric_limits<T>::min()) << ","                                                 \
         << static_cast<int32_t>(std::numeric_limits<T>::max()) << "]";                                                \
      GELOGE(ge::FAILED, "%s", ss.str().c_str());                                                                      \
      return ge::FAILED;                                                                                               \
    }                                                                                                                  \
  } while (false)

#define REQUIRE_COMPAT_INT8(v) REQUIRE_COMPAT_FOR_CHAR(int8_t, (v))
#define REQUIRE_COMPAT_UINT8(v) REQUIRE_COMPAT_FOR_CHAR(uint8_t, (v))
#define REQUIRE_COMPAT_INT16(v) REQUIRE_COMPAT(int16_t, (v))
#define REQUIRE_COMPAT_UINT16(v) REQUIRE_COMPAT(uint16_t, (v))
#define REQUIRE_COMPAT_INT32(v) REQUIRE_COMPAT(int32_t, (v))
#define REQUIRE_COMPAT_UINT32(v) REQUIRE_COMPAT(uint32_t, (v))
#define REQUIRE_COMPAT_INT64(v) REQUIRE_COMPAT(int64_t, (v))
#define REQUIRE_COMPAT_UINT64(v) REQUIRE_COMPAT(uint64_t, (v))
#define REQUIRE_COMPAT_SIZE_T(v) REQUIRE_COMPAT(size_t, (v))

#endif  // METADEF_CXX_INC_GRAPH_UTILS_MATH_UTIL_H_
