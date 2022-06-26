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

#include "expand_dimension.h"
#include <unordered_set>
#include "framework/common/debug/ge_log.h"

namespace transformer {
static const int32_t NCHW_DIM_N = 0;
static const int32_t NCHW_DIM_C = 1;
static const int32_t NCHW_DIM_H = 2;
static const int32_t NCHW_DIM_W = 3;

static const int32_t NHWC_DIM_N = 0;
static const int32_t NHWC_DIM_H = 1;
static const int32_t NHWC_DIM_W = 2;
static const int32_t NHWC_DIM_C = 3;

static const int32_t HWCN_DIM_H = 0;
static const int32_t HWCN_DIM_W = 1;
static const int32_t HWCN_DIM_C = 2;
static const int32_t HWCN_DIM_N = 3;

static const int32_t CHWN_DIM_C = 0;
static const int32_t CHWN_DIM_H = 1;
static const int32_t CHWN_DIM_W = 2;
static const int32_t CHWN_DIM_N = 3;

static const int32_t NDHWC_DIM_N = 0;
static const int32_t NDHWC_DIM_D = 1;
static const int32_t NDHWC_DIM_H = 2;
static const int32_t NDHWC_DIM_W = 3;
static const int32_t NDHWC_DIM_C = 4;

static const int32_t NCDHW_DIM_N = 0;
static const int32_t NCDHW_DIM_C = 1;
static const int32_t NCDHW_DIM_D = 2;
static const int32_t NCDHW_DIM_H = 3;
static const int32_t NCDHW_DIM_W = 4;

static const int32_t DHWCN_DIM_D = 0;
static const int32_t DHWCN_DIM_H = 1;
static const int32_t DHWCN_DIM_W = 2;
static const int32_t DHWCN_DIM_C = 3;
static const int32_t DHWCN_DIM_N = 4;

static const int32_t DHWNC_DIM_D = 0;
static const int32_t DHWNC_DIM_H = 1;
static const int32_t DHWNC_DIM_W = 2;
static const int32_t DHWNC_DIM_N = 3;
static const int32_t DHWNC_DIM_C = 4;

static const size_t DIMENSION_NUM_FOUR = 4;
static const size_t DIMENSION_NUM_FIVE = 5;
static const size_t DIMENSION_NUM_TWO = 2;
static const std::string RESHAPE_TYPE_FORBIDDEN = "FORBIDDEN";

static const std::map<ge::Format, size_t> FULL_SIZE_OF_FORMAT {
    {ge::FORMAT_NCHW, DIMENSION_NUM_FOUR},
    {ge::FORMAT_NHWC, DIMENSION_NUM_FOUR},
    {ge::FORMAT_HWCN, DIMENSION_NUM_FOUR},
    {ge::FORMAT_CHWN, DIMENSION_NUM_FOUR},
    {ge::FORMAT_NDHWC, DIMENSION_NUM_FIVE},
    {ge::FORMAT_NCDHW, DIMENSION_NUM_FIVE},
    {ge::FORMAT_DHWCN, DIMENSION_NUM_FIVE},
    {ge::FORMAT_ND, DIMENSION_NUM_FOUR}
};

static const std::map<size_t, std::map<ge::Format, std::string>> DEFAULT_RESHAPE_TYPE {
    {0, {{ge::FORMAT_NCHW, ""}, {ge::FORMAT_NHWC, ""}, {ge::FORMAT_HWCN, ""}, {ge::FORMAT_CHWN, ""},
         {ge::FORMAT_NDHWC, ""}, {ge::FORMAT_NCDHW, ""}, {ge::FORMAT_DHWCN, ""}}},

    {1, {{ge::FORMAT_NCHW, "C"}, {ge::FORMAT_NHWC, "C"}, {ge::FORMAT_HWCN, "C"}, {ge::FORMAT_CHWN, "C"},
         {ge::FORMAT_NDHWC, "C"}, {ge::FORMAT_NCDHW, "C"}, {ge::FORMAT_DHWCN, "C"}}},

    {2, {{ge::FORMAT_NCHW, "CH"}, {ge::FORMAT_NHWC, "HW"}, {ge::FORMAT_HWCN, "CN"}, {ge::FORMAT_CHWN, "WN"},
         {ge::FORMAT_NDHWC, "WC"}, {ge::FORMAT_NCDHW, "HW"}, {ge::FORMAT_DHWCN, "CN"}}},

    {3, {{ge::FORMAT_NCHW, "CHW"}, {ge::FORMAT_NHWC, "HWC"}, {ge::FORMAT_HWCN, "WCN"}, {ge::FORMAT_CHWN, "HWN"},
         {ge::FORMAT_NDHWC, "HWC"}, {ge::FORMAT_NCDHW, "DHW"}, {ge::FORMAT_DHWCN, "WCN"}}},

    {4, {{ge::FORMAT_NDHWC, "DHWC"}, {ge::FORMAT_NCDHW, "CDHW"}, {ge::FORMAT_DHWCN, "HWCN"}}}
};

static const std::map<ge::Format, std::map<std::string, int32_t>> AXIS_INDEX_OF_FORMAT {
    {ge::FORMAT_NCHW, {{"N", NCHW_DIM_N}, {"C", NCHW_DIM_C}, {"H", NCHW_DIM_H}, {"W", NCHW_DIM_W}}},
    {ge::FORMAT_HWCN, {{"N", HWCN_DIM_N}, {"C", HWCN_DIM_C}, {"H", HWCN_DIM_H}, {"W", HWCN_DIM_W}}},
    {ge::FORMAT_NHWC, {{"N", NHWC_DIM_N}, {"C", NHWC_DIM_C}, {"H", NHWC_DIM_H}, {"W", NHWC_DIM_W}}},
    {ge::FORMAT_CHWN, {{"N", CHWN_DIM_N}, {"C", CHWN_DIM_C}, {"H", CHWN_DIM_H}, {"W", CHWN_DIM_W}}},
    {ge::FORMAT_NDHWC,
     {{"N", NDHWC_DIM_N}, {"C", NDHWC_DIM_C}, {"H", NDHWC_DIM_H}, {"W", NDHWC_DIM_W}, {"D", NDHWC_DIM_D}}},
    {ge::FORMAT_NCDHW,
     {{"N", NCDHW_DIM_N}, {"C", NCDHW_DIM_C}, {"H", NCDHW_DIM_H}, {"W", NCDHW_DIM_W}, {"D", NCDHW_DIM_D}}},
    {ge::FORMAT_DHWCN,
     {{"N", DHWCN_DIM_N}, {"C", DHWCN_DIM_C}, {"H", DHWCN_DIM_H}, {"W", DHWCN_DIM_W}, {"D", DHWCN_DIM_D}}},
    {ge::FORMAT_DHWNC,
     {{"N", DHWNC_DIM_N}, {"C", DHWNC_DIM_C}, {"H", DHWNC_DIM_H}, {"W", DHWNC_DIM_W}, {"D", DHWNC_DIM_D}}}
};

static const std::map<ge::Format, std::unordered_set<std::string>> ALL_VALID_RESHAPE_TYPE {
        {ge::FORMAT_NCHW, {
                              "N", "C", "H", "W",
                              "NC", "NH", "NW", "CH", "CW", "HW",
                              "NCH", "NCW", "NHW", "CHW"
                          }},
        {ge::FORMAT_NHWC, {
                              "N", "H", "W", "C",
                              "NH", "NW", "NC", "HW", "HC", "WC",
                              "NHW", "NHC", "NWC", "HWC"
                          }},
        {ge::FORMAT_HWCN, {
                              "H", "W", "C", "N",
                              "HW", "HC", "HN", "WC", "WN", "CN",
                              "HWC", "HWN", "HCN", "WCN"
                           }},
        {ge::FORMAT_CHWN, {
                              "C", "H", "W", "N",
                              "CH", "CW", "CN", "HW", "HN", "WN",
                              "CHW", "CHN", "CWN", "HWN"
                           }},
        {ge::FORMAT_NDHWC, {
                              "N", "D", "H", "W", "C",
                              "ND", "NH", "NW", "NC", "DH", "DW", "DC", "HW", "HC", "WC",
                              "NDH", "NDW", "NDC", "NHW", "NHC", "NWC", "DHW", "DHC", "DWC", "HWC",
                              "NDHW", "NDHC", "NDWC", "NHWC", "DHWC"
                           }},
        {ge::FORMAT_NCDHW, {
                               "N", "C", "D", "H", "W",
                               "NC", "ND", "NH", "NW", "CD", "CH", "CW", "DH", "DW", "HW",
                               "NCD", "NCH", "NCW", "NDH", "NDW", "NHW", "CDH", "CDW", "CHW", "DHW",
                               "NCDH", "NCDW", "NCHW", "NDHW", "CDHW"
                          }},
        {ge::FORMAT_DHWCN, {
                               "D", "H", "W", "C", "N",
                               "DH", "DW", "DC", "DN", "HW", "HC", "HN", "WC", "WN", "CN",
                               "DHW", "DHC", "DHN", "DWC", "DWN", "DCN", "HWC", "HWN", "HCN", "WCN",
                               "DHWC", "DHWN", "DHCN", "DWCN", "HWCN"
                         }}
};

bool GetDefaultReshapeType(const ge::Format &original_format, size_t old_dims_size, std::string &reshape_type) {
  auto rsp_tp_all_format = DEFAULT_RESHAPE_TYPE.find(old_dims_size);
  if (rsp_tp_all_format == DEFAULT_RESHAPE_TYPE.end()) {
    GELOGW("dim size %zu is invalid.", old_dims_size);
    return false;
  }

  auto iter_rsp_tp = rsp_tp_all_format->second.find(original_format);
  if (iter_rsp_tp == rsp_tp_all_format->second.end()) {
    GELOGW("Cannot find default reshape type for %u.", original_format);
    return false;
  }

  reshape_type = iter_rsp_tp->second;
  return true;
}

bool IsExpandNecessary(std::vector<int64_t> &dims, const ge::Format &original_format, const ge::Format &final_format,
                       const std::string &reshape_type, size_t &full_size) {
  /* 1. Check whether the old dim size is full. Full size is not necessary for expand. */
  size_t old_dims_size = dims.size();
  auto iter_full_size = FULL_SIZE_OF_FORMAT.find(original_format);
  if (iter_full_size == FULL_SIZE_OF_FORMAT.end()) {
    GELOGW("Original Format %u is invalid.", original_format);
    return false;
  } else {
    if (old_dims_size >= iter_full_size->second) {
      return false;
    }
  }
  /* 2. Check whether the final format does not need expanding demension. */
  bool no_need_reshape_flag = reshape_type == RESHAPE_TYPE_FORBIDDEN || final_format == ge::FORMAT_FRACTAL_NZ ||
                              (original_format == ge::FORMAT_ND && final_format == ge::FORMAT_FRACTAL_Z);
  if (no_need_reshape_flag) {
    return false;
  }
  full_size = iter_full_size->second;
  return true;
}

void ExpandByReshapeType(std::vector<int64_t> &dims, const std::string &op_type, const ge::Format &original_format,
                         size_t full_size, const uint32_t &tensor_index, const std::string &reshape_type) {
  GELOGD("Expand tensor %u of %s by reshape type %s.", tensor_index, op_type.c_str(), reshape_type.c_str());
  auto old_dims_size = dims.size();
  if (reshape_type == "CN") {
    /* If the reshape type is CN, we will consider the original format is HWCN. */
    std::vector<int64_t> new_dims;
    if (old_dims_size < DIMENSION_NUM_TWO) {
      GELOGW("old dims size %zu is less than 2. Reshape type is %s.", dims.size(), reshape_type.c_str());
      return;
    }
    new_dims.push_back(1);
    new_dims.push_back(1);
    new_dims.push_back(dims[0]);
    new_dims.push_back(dims[1]);
    dims.swap(new_dims);
    /* In this case the final format must be HWCN, we just return true */
    return;
  } else {
    /* Build a array with all 1 of full size. Then we will substitute some of the 1 with the original axis value. */
    std::vector<int64_t> new_dims;
    for (size_t i = 0; i < full_size; i++) {
      new_dims.emplace_back(1);
    }

    auto iter_axis_name_index = AXIS_INDEX_OF_FORMAT.find(original_format);
    if (iter_axis_name_index == AXIS_INDEX_OF_FORMAT.end()) {
      GELOGW("Cannot find axis index name map value of original format %u of tensor %u of %s.",
             original_format, tensor_index, op_type.c_str());
      return;
    }
    for (size_t i = 0; i < old_dims_size; i++) {
      /* The length of reshape type is larger than the dims. */
      std::string axis_str(1, reshape_type.at(i));
      auto iter_axis_index = iter_axis_name_index->second.find(axis_str);
      if (iter_axis_index == iter_axis_name_index->second.end()) {
        GELOGW("Invalid reshape type %s for tensor %u of %s.", reshape_type.c_str(), tensor_index, op_type.c_str());
        return;
      }
      int32_t index = iter_axis_index->second;
      if (index < 0 || index >= (int32_t)full_size) {
        GELOGW("Index of %s is %d which is larger than the full size %zu.", axis_str.c_str(), index, full_size);
        return;
      }
      new_dims[index] = dims[i];
    }
    dims.swap(new_dims);
  }
}

bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, std::vector<int64_t> &dims) {
  /* 1. Check expanding necessary. */
  size_t full_size = 0;
  if (!IsExpandNecessary(dims, original_format, final_format, reshape_type, full_size)) {
    return true;
  }

  /* 2. Check whether the reshape type is consistent with the original format.
   * If not consistent, just return and report a warning. */
  std::string valid_reshape_type = reshape_type;
  size_t old_dims_size = dims.size();
  auto iter_format = ALL_VALID_RESHAPE_TYPE.find(original_format);
  if (iter_format != ALL_VALID_RESHAPE_TYPE.end()) {
    auto iter_reshape_type = iter_format->second.find(reshape_type);
    if (iter_reshape_type == iter_format->second.end()) {
      if (!GetDefaultReshapeType(original_format, old_dims_size, valid_reshape_type)) {
        return true;
      }
      GELOGI("Get default reshape type %s for op %s tensor %u original format %u is invalid.",
             valid_reshape_type.c_str(), op_type.c_str(), tensor_index, original_format);
    }
  }

  /* 3. Check whether the dimension of original shape is less than or equal to
   * the length of reshape type. If the dimension of original shape if larger,
   * we cannot find suitable posotion for all axis in original shape and we just return. */
  if (old_dims_size > valid_reshape_type.length()) {
    GELOGW("Dimension %zu of tensor %u of %s is larger than the length of reshape type which is %zu.",
           old_dims_size, tensor_index, op_type.c_str(), valid_reshape_type.length());
    return true;
  }

  /* 4. Expand dimension. */
  ExpandByReshapeType(dims, op_type, original_format, full_size, tensor_index, valid_reshape_type);
  return true;
}

} // namespace transformer
