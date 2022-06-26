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

#ifndef COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
#define COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_

#include <memory.h>
#include <functional>
#include <vector>
#include "graph/types.h"

namespace transformer {

/* Pad dimension according to reshape type */
bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, std::vector<int64_t> &dims);

} // namespace transformer

#endif //COMMON_UTILS_TRANSFORMER_INC_EXPAND_DIMENSION_H_
