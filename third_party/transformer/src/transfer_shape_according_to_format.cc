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

#include "transfer_shape_according_to_format.h"
#include <algorithm>
#include "framework/common/debug/ge_log.h"

namespace transformer {
using namespace ge;

inline bool CheckInt64MulOverflow(int64_t a, int64_t b) {
  if (a > 0) {
    if (b > 0) {
      if (a > (INT64_MAX / b)) {
        return false;
      }
    } else {
      if (b < (INT64_MIN / a)) {
        return false;
      }
    }
  } else {
    if (b > 0) {
      if (a < (INT64_MIN / b)) {
        return false;
      }
    } else {
      if ((a != 0) && (b < (INT64_MAX / a))) {
        return false;
      }
    }
  }
  return true;
}

#define TRANS_INT64_MULCHECK(a, b)                        \
  if (CheckInt64MulOverflow((a), (b)) != true) {       \
    return false;                                         \
  }

namespace {
  static std::unique_ptr<AxisUtil> axisutil_object(new(std::nothrow) AxisUtil());
  static std::map<ge::Format, GetNewShapeByAxisValueAndFormatPtr> getNewShapeFuncMap = {
    {ge::FORMAT_NCHW, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue)},
    {ge::FORMAT_NHWC, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue)},
    {ge::FORMAT_NC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue)},
    {ge::FORMAT_NDC1HWC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNDC1HWC0ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzShapeByAxisValue)},
    {ge::FORMAT_HWCN, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue)},
    {ge::FORMAT_C1HWNCoC0, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_NZ, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNzShapeByAxisValue)},
    {ge::FORMAT_NC1HWC0_C04, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_C04, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzC04ShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_G, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzGShapeByAxisValue)},
    {ge::FORMAT_CHWN, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetCHWNShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_3D, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFz3DShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE,
        std::make_shared<GetNewShapeByAxisValueAndFormat>(
            ShapeTransferAccordingToFormat::GetFz3DTransposeShapeByAxisValue)},
    {ge::FORMAT_FRACTAL_ZN_LSTM, std::make_shared<GetNewShapeByAxisValueAndFormat>(
        ShapeTransferAccordingToFormat::GetFzLstmShapeByAxisValue)}};

    static std::map<ge::DataType, uint32_t> mapOfDtypeAndC0 = {
    {ge::DT_FLOAT16, SHAPE_NUMBER_16}, {ge::DT_FLOAT, SHAPE_NUMBER_16},  {ge::DT_INT8, SHAPE_NUMBER_32},
    {ge::DT_INT16, SHAPE_NUMBER_16},   {ge::DT_INT32, SHAPE_NUMBER_16},  {ge::DT_INT64, SHAPE_NUMBER_16},
    {ge::DT_UINT8, SHAPE_NUMBER_32},   {ge::DT_UINT16, SHAPE_NUMBER_16}, {ge::DT_UINT32, SHAPE_NUMBER_16},
    {ge::DT_UINT64, SHAPE_NUMBER_16},  {ge::DT_BOOL, SHAPE_NUMBER_16}};
}
ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat(void) {}

bool ShapeTransferAccordingToFormat::GetNDC1HWC0ShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                 const vector<int64_t> &axis_value, const vector<int64_t> &nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  new_shape.push_back(axis_value[AXIS_N]);
  new_shape.push_back(axis_value[AXIS_D]);
  new_shape.push_back(axis_value[AXIS_C1]);
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  new_shape.push_back(axis_value[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNCHWShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                             const vector<int64_t>& axis_value,
                                                             const vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  new_shape.push_back(axis_value[AXIS_N]);
  new_shape.push_back(axis_value[AXIS_C]);
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNHWCShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                             const vector<int64_t>& axis_value,
                                                             const vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  new_shape.push_back(axis_value[AXIS_N]);
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  new_shape.push_back(axis_value[AXIS_C]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNC1HWC0ShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                                const vector<int64_t>& axis_value,
                                                                const vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  if (impl_type == EN_IMPL_HW_TBE || impl_type == EN_IMPL_CUSTOM_TBE || impl_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
    new_shape.push_back(axis_value[AXIS_N]);
    new_shape.push_back(axis_value[AXIS_C1]);
    new_shape.push_back(axis_value[AXIS_H]);
    new_shape.push_back(axis_value[AXIS_W]);
    new_shape.push_back(axis_value[AXIS_C0]);
  } else {
    new_shape.push_back(axis_value[AXIS_N]);
    new_shape.push_back(axis_value[AXIS_C]);
    new_shape.push_back(axis_value[AXIS_H]);
    new_shape.push_back(axis_value[AXIS_W]);
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                           const vector<int64_t>& axis_value,
                                                           const vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  if (nd_value.size() == SIZE_OF_CN) {
    auto size_of_original_vec = nd_value.size();
    new_shape = nd_value;
    /* size_of_original_vec - 1 mean the last value of original vec
     * size_of_original_vec - 2 mean the second last value of original vec */
    new_shape[size_of_original_vec - MINUS_VALUE_ONE] =
        DivisionCeiling(nd_value[size_of_original_vec - MINUS_VALUE_ONE], SHAPE_NUMBER_16);
    new_shape[size_of_original_vec - MINUS_VALUE_TWO] =
        DivisionCeiling(nd_value[size_of_original_vec - MINUS_VALUE_TWO], axis_value[AXIS_C0]);
    new_shape.push_back(SHAPE_NUMBER_16);
    new_shape.push_back(axis_value[AXIS_C0]);
  } else {
    if (impl_type == EN_IMPL_HW_TBE || impl_type == EN_IMPL_CUSTOM_TBE ||
        impl_type == EN_IMPL_NON_PERSISTENT_CUSTOM_TBE) {
      bool has_unknown_shape = axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                               axis_value[AXIS_C1] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_G] == UNKNOWN_SHAPE_VALUE;
      int64_t hwc1 = UNKNOWN_SHAPE_VALUE;
      int64_t axis_n_val = axis_value[AXIS_N];
      if (!has_unknown_shape) {
        int64_t group_val = axis_value[AXIS_G];
        int64_t axis_c1_val = axis_value[AXIS_C1];
        int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
        int64_t axis_c_val = axis_value[AXIS_C];
        if (group_val > GROUPS_DEFAULT_VALUE && axis_n_val >= group_val) {
          int64_t enlarge_value =
              GetAsisEnlargeValue(axis_c_val, axis_n_val / group_val, axis_value[AXIS_C0], group_val);
          axis_g_val = DivisionCeiling(group_val, enlarge_value);
          TRANS_INT64_MULCHECK(axis_c_val, enlarge_value);
          axis_c_val *= enlarge_value;
          TRANS_INT64_MULCHECK(axis_n_val / group_val, enlarge_value);
          axis_n_val = (axis_n_val / group_val) * enlarge_value;
          axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
        }
        TRANS_INT64_MULCHECK(axis_g_val, axis_c1_val);
        int64_t g_c1_val = axis_g_val * axis_c1_val;
        TRANS_INT64_MULCHECK(g_c1_val, axis_value[AXIS_H]);
        g_c1_val *= axis_value[AXIS_H];
        TRANS_INT64_MULCHECK(g_c1_val, axis_value[AXIS_W]);
        hwc1 = g_c1_val * axis_value[AXIS_W];
      }
      new_shape.push_back(hwc1);
      new_shape.push_back(DivisionCeiling(axis_n_val, NI));
      new_shape.push_back(NI);
      new_shape.push_back(axis_value[AXIS_C0]);
    } else {
      new_shape.push_back(axis_value[AXIS_N]);
      new_shape.push_back(axis_value[AXIS_C]);
      new_shape.push_back(axis_value[AXIS_H]);
      new_shape.push_back(axis_value[AXIS_W]);
    }
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetHWCNShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                             const vector<int64_t>& axis_value,
                                                             const vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  new_shape.push_back(axis_value[AXIS_C]);
  new_shape.push_back(axis_value[AXIS_N]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetC1HWNCoC0ShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                                  const vector<int64_t>& axis_value,
                                                                  const vector<int64_t>& nd_value) {
  CHECK(axis_value.empty(), GELOGD("AxisValue is empty!"), return true);
  /* axis_value is initialized as a size 6 vector. */
  new_shape.push_back(axis_value[AXIS_C1]);
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  new_shape.push_back(axis_value[AXIS_N]);
  new_shape.push_back(axis_value[AXIS_Co]);
  new_shape.push_back(axis_value[AXIS_C0]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetNzShapeByAxisValue(vector<int64_t>& new_shape, const int64_t& impl_type,
                                                           const vector<int64_t>& axis_value,
                                                           const vector<int64_t>& nd_value) {
  CHECK(nd_value.empty(), GELOGD("nd_value is empty!"), return true);
  CHECK(axis_value.empty() || axis_value.size() <= AXIS_C0,
        GELOGD("AxisValue is empty or its size %zu <= AXIS_C0[%u]", axis_value.size(), AXIS_C0), return true);
  uint32_t size_of_original_vec = nd_value.size();
  if (size_of_original_vec < MINIMUM_NZ_SHAPE_DIM_NUM) {
    GELOGD("nd_value's dim num is less than 2!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_shape = nd_value;

  /* size_of_original_vec - 1 mean the last value of original vec
   * size_of_original_vec - 2 mean the second last value of original vec */
  new_shape[size_of_original_vec - MINUS_VALUE_ONE] =
      DivisionCeiling(nd_value[size_of_original_vec - MINUS_VALUE_TWO], (int64_t)SHAPE_NUMBER_16);

  new_shape[size_of_original_vec - MINUS_VALUE_TWO] =
      DivisionCeiling(nd_value[size_of_original_vec - MINUS_VALUE_ONE], axis_value[AXIS_C0]);
  new_shape.push_back(SHAPE_NUMBER_16);
  new_shape.push_back(axis_value[AXIS_C0]);
  return true;
}

bool CheckInputParam(const ShapeAndFormat& shapeAndFormatInfo, ge::Format primary_new_format) {
  bool invalid_format =
      (shapeAndFormatInfo.oldFormat == ge::FORMAT_RESERVED || shapeAndFormatInfo.oldFormat >= ge::FORMAT_END) ||
      (primary_new_format == ge::FORMAT_RESERVED || primary_new_format >= ge::FORMAT_END);
  if (invalid_format) {
    GELOGE(GRAPH_FAILED, "Old format %u or new format %u is invalid!", shapeAndFormatInfo.oldFormat,
           primary_new_format);
    return false;
  }

  if (shapeAndFormatInfo.currentDataType == ge::DT_UNDEFINED ||
      shapeAndFormatInfo.currentDataType >= ge::DT_MAX) {
    GELOGE(GRAPH_FAILED, "currentDataType %u is invalid!", shapeAndFormatInfo.currentDataType);
    return false;
  }
  if (axisutil_object == nullptr) {
    return false;
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(ShapeAndFormat& shapeAndFormatInfo, int64_t* c) {
  /* The default new shape is old shape */
  shapeAndFormatInfo.newShape = shapeAndFormatInfo.oldShape;
  ge::Format primary_new_format =
      static_cast<Format>(GetPrimaryFormat(shapeAndFormatInfo.newFormat));
  if (!CheckInputParam(shapeAndFormatInfo, primary_new_format)) {
    return false;
  }

  if (!axisutil_object->HasAxisValueFunc(shapeAndFormatInfo.oldFormat)) {
    return true;
  }

  auto iterGetNewShapeFunc = getNewShapeFuncMap.find(primary_new_format);
  if (iterGetNewShapeFunc == getNewShapeFuncMap.end()) {
    GELOGD("Can not get new shape of new format %u!", primary_new_format);
    return true;
  }
  GELOGD("Original format is %u, new format %u", shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat);
  GetNewShapeByAxisValueAndFormatPtr getNewShapeFunc = iterGetNewShapeFunc->second;

  vector<int64_t> axis_value;
  for (uint32_t i = 0; i < AXIS_BOTTOM; i++) {
    axis_value.push_back(1);
  }

  int64_t group = static_cast<int64_t>(ge::GetSubFormat(shapeAndFormatInfo.newFormat));
  if (group > GROUPS_DEFAULT_VALUE) {
    axis_value[AXIS_G] = group;
  }

  vector<int64_t> nd_value;
  uint32_t c0;
  if (mapOfDtypeAndC0.empty()) {
    c0 = SHAPE_NUMBER_16;
  } else {
    auto iterGetC0 = mapOfDtypeAndC0.find(shapeAndFormatInfo.currentDataType);
    if (iterGetC0 == mapOfDtypeAndC0.end()) {
      GELOGE(GRAPH_FAILED, "Dtype is not support.");
      return true;
    }
    c0 = iterGetC0->second;
  }

  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  if (shapeAndFormatInfo.newFormat == ge::FORMAT_NC1HWC0_C04) {
    c0 = SHAPE_DIM_VALUE_C04;
  }
  if (axisutil_object == nullptr) {
    return false;
  }
  bool ret = axisutil_object->GetAxisValueByOriginFormat(
      shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.oldShape, c0, axis_value, nd_value);
  if (ret != true && shapeAndFormatInfo.newFormat != ge::FORMAT_FRACTAL_NZ) {
    return true;
  }

  shapeAndFormatInfo.newShape.clear();
  (*getNewShapeFunc)(shapeAndFormatInfo.newShape, shapeAndFormatInfo.opImplType, axis_value, nd_value);
  if (c != nullptr) {
    *c = axis_value[AXIS_C];
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetCHWNShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                                             const vector<int64_t> &axis_value,
                                                             const vector<int64_t> &nd_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  new_shape.push_back(axis_value[AXIS_C]);
  new_shape.push_back(axis_value[AXIS_H]);
  new_shape.push_back(axis_value[AXIS_W]);
  new_shape.push_back(axis_value[AXIS_N]);
  return true;
}

bool ShapeTransferAccordingToFormat::GetFz3DShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                                             const vector<int64_t> &axis_value,
                                                             const vector<int64_t> &nd_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }

  /* axis_value is initialized as a size 6 vector. */
  bool has_unknown_shape = axis_value[AXIS_D] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C1] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_G] == UNKNOWN_SHAPE_VALUE;

  int64_t gdhwc1 = UNKNOWN_SHAPE_VALUE;
  int64_t group_val = axis_value[AXIS_G];
  int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
  int64_t axis_n_val = axis_value[AXIS_N];
  int64_t axis_c_val = axis_value[AXIS_C];
  int64_t axis_c1_val = axis_value[AXIS_C1];
  if (!has_unknown_shape) {
    if (group_val > GROUPS_DEFAULT_VALUE && axis_n_val >= group_val) {
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val / group_val,
                                                  axis_value[AXIS_C0], group_val);
      axis_g_val = DivisionCeiling(group_val, enlarge_value);
      TRANS_INT64_MULCHECK(axis_c_val, enlarge_value);
      axis_c_val *= enlarge_value;
      TRANS_INT64_MULCHECK(axis_n_val / group_val, enlarge_value);
      axis_n_val = (axis_n_val / group_val) * enlarge_value;
      axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
    }
    TRANS_INT64_MULCHECK(axis_g_val, axis_c1_val);
    int64_t g_c1_val = axis_g_val * axis_c1_val;
    TRANS_INT64_MULCHECK(g_c1_val, axis_value[AXIS_D]);
    g_c1_val *= axis_value[AXIS_D];
    TRANS_INT64_MULCHECK(g_c1_val, axis_value[AXIS_H]);
    g_c1_val *= axis_value[AXIS_H];
    TRANS_INT64_MULCHECK(g_c1_val, axis_value[AXIS_W]);
    gdhwc1 = g_c1_val * axis_value[AXIS_W];
  }
  new_shape.push_back(gdhwc1);
  new_shape.push_back(DivisionCeiling(axis_n_val, NI));
  new_shape.push_back(NI);
  new_shape.push_back(axis_value[AXIS_C0]);

  return true;
}

bool ShapeTransferAccordingToFormat::GetFz3DTransposeShapeByAxisValue(vector<int64_t> &new_shape,
                                                                      const int64_t &impl_type,
                                                                      const vector<int64_t> &axis_value,
                                                                      const vector<int64_t> &nd_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  for (auto ele : axis_value) {
    GELOGI("value is %ld", ele);
  }
  int64_t n1 = DivisionCeiling(axis_value[AXIS_N], NI);
  int64_t dhwn1 = n1 * axis_value[AXIS_H] * axis_value[AXIS_W] * axis_value[AXIS_D];
  if (n1 == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
      axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_D] == UNKNOWN_SHAPE_VALUE) {
    dhwn1 = UNKNOWN_SHAPE_VALUE;
  }

  new_shape.push_back(dhwn1);
  if (axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    new_shape.push_back(UNKNOWN_SHAPE_VALUE);
  } else {
    new_shape.push_back(axis_value[AXIS_C1]);
  }
  new_shape.push_back(NI);
  new_shape.push_back(axis_value[AXIS_C0]);

  return true;
}

bool ShapeTransferAccordingToFormat::GetFzLstmShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                                               const vector<int64_t> &axis_value,
                                                               const vector<int64_t> &nd_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  int64_t h = axis_value[AXIS_N] / LSTM_NI;
  int64_t i = axis_value[AXIS_C] - h;
  int64_t first_element_of_fz_lstm = DivisionCeiling(i, NI) + DivisionCeiling(h, NI);

  int64_t second_element_of_fz_lstm = LSTM_NI * DivisionCeiling(h, NI);
  if (axis_value[AXIS_N] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    first_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
    second_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
  }
  new_shape.push_back(first_element_of_fz_lstm);
  new_shape.push_back(second_element_of_fz_lstm);
  new_shape.push_back(NI);
  new_shape.push_back(NI);
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzC04ShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                                              const vector<int64_t> &axis_value,
                                                              const vector<int64_t> &nd_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  if (impl_type == EN_IMPL_HW_TBE || impl_type == EN_IMPL_CUSTOM_TBE) {
    int64_t x = SHAPE_DIM_VALUE_C04 * axis_value[AXIS_H] * axis_value[AXIS_W];
    new_shape.push_back(DivisionCeiling(x, X0));


    new_shape.push_back(DivisionCeiling(axis_value[AXIS_N], NI));
    new_shape.push_back(NI);
    new_shape.push_back(X0);
  } else {
    new_shape.push_back(axis_value[AXIS_N]);
    new_shape.push_back(axis_value[AXIS_C]);
    new_shape.push_back(axis_value[AXIS_H]);
    new_shape.push_back(axis_value[AXIS_W]);
  }
  return true;
}

bool ShapeTransferAccordingToFormat::GetFzGShapeByAxisValue(vector<int64_t> &new_shape, const int64_t &impl_type,
                                                            const vector<int64_t> &axis_value,
                                                            const vector<int64_t> &nd_value) {
  if (axis_value.empty()) {
    GELOGW("AxisValue is empty!");
    return true;
  }
  /* axis_value is initialized as a size 6 vector. */
  int64_t new_c = axis_value[AXIS_C] * axis_value[AXIS_G];
  int64_t new_c1 = DivisionCeiling(new_c, axis_value[AXIS_C0]);
  int64_t hwc1 = new_c1 * axis_value[AXIS_H] * axis_value[AXIS_W];
  new_shape.push_back(hwc1);
  new_shape.push_back(DivisionCeiling(axis_value[AXIS_N], NI));
  new_shape.push_back(NI);
  new_shape.push_back(axis_value[AXIS_C0]);
  return true;
}

int64_t GetGreatestCommonDivisor(int64_t x, int64_t y) {
  if (y == 0) {
    return x;
  }
  return GetGreatestCommonDivisor(y, x % y);
}

int64_t GetLeastCommonMultiple(int64_t x, int64_t y) {
  if (x == 0 || y == 0) {
    return 0;
  }
  return (x * y) / GetGreatestCommonDivisor(x, y);
}

int64_t ShapeTransferAccordingToFormat::GetAsisEnlargeValue(const int64_t& cin, const int64_t& cout, const int64_t& c0,
                                                            const int64_t& group) {
  if (cin == 0 || cout == 0) {
    return 0;
  }
  int64_t tmp = GetLeastCommonMultiple(GetLeastCommonMultiple(cin, c0) / cin, GetLeastCommonMultiple(cout, NI) / cout);
  return std::min(tmp, group);
}
} // namespace transformer
