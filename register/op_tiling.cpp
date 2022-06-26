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

#include "register/op_tiling.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <nlohmann/json.hpp>
#include <type_traits>
#include <typeinfo>

#include "common/util/error_manager/error_manager.h"
#include "external/graph/operator.h"
#include "external/graph/operator_factory.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "securec.h"

#define LOG_ENABLED(loglvl) CheckLogLevel(GE_MODULE_NAME, loglvl)

namespace optiling {

using DataBuf = std::tuple<const uint8_t *, size_t>;

class AnyValueBase {
 public:
  virtual ~AnyValueBase() = default;
  virtual DataBuf GetDataBuf() const = 0;
};

template<typename VT>
class AnyValue : public AnyValueBase {
 public:
  explicit AnyValue(const VT &value) : value_(value) {}
  ~AnyValue() override = default;
  DataBuf GetDataBuf() const override {
    return DataBuf(reinterpret_cast<const uint8_t *>(&value_), sizeof(value_));
  }

 private:
  VT value_;
};

template<typename VT>
class AnyVecValue : public AnyValueBase {
 public:
  explicit AnyVecValue(std::vector<VT> &value) : value_(std::move(value)) {}
  ~AnyVecValue() override = default;
  DataBuf GetDataBuf() const override {
    return DataBuf(reinterpret_cast<const uint8_t *>(value_.data()), sizeof(VT) * value_.size());
  }

 private:
  vector<VT> value_;
};

template<typename T, typename Enabled = void>
struct Getter;

template<typename T>
struct Getter<T, typename std::enable_if<std::is_integral<T>::value>::type> {
  using ST = int64_t;
  static constexpr bool (*func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                int64_t &) = ge::AttrUtils::GetInt;
  static constexpr bool (*list_func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                     vector<int64_t> &) = ge::AttrUtils::GetListInt;
};
template<typename T>
struct Getter<T, typename std::enable_if<std::is_floating_point<T>::value>::type> {
  using ST = float;
  static constexpr bool (*func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                float &) = ge::AttrUtils::GetFloat;
  static constexpr bool (*list_func)(ge::AttrUtils::ConstAttrHolderAdapter &&, const string &,
                                     vector<float> &) = ge::AttrUtils::GetListFloat;
};

class TeOpVarAttrArgsImpl {
  using DataKeyType = std::pair<std::string, std::string>;

 public:
  explicit TeOpVarAttrArgsImpl(ge::OpDescPtr &op_desc) : op_desc_(op_desc){};
  ~TeOpVarAttrArgsImpl() = default;

  Status GetDataByName(const string &name, const string &dtype, DataBuf &data);

 private:
  template<typename T>
  Status GetNodeAttrDataIntListList(const std::string &name, DataBuf &data) {
    std::vector<std::vector<int64_t>> value;
    bool res = ge::AttrUtils::GetListListInt(op_desc_, name, value);
    if (!res) {
      GE_LOGE("attr not found. %s", name.c_str());
      return domi::FAILED;
    }

    std::vector<T> dest;
    for (const auto &vec : value) {
      for (auto elem : vec) {
        dest.emplace_back(static_cast<T>(elem));
      }
    }
    auto dest_ptr = std::make_shared<AnyVecValue<T>>(dest);
    data_map_.emplace(name + '_' + typeid(T).name(), dest_ptr);
    data = dest_ptr->GetDataBuf();
    GELOGI("IntListList attr found. %s", name.c_str());
    return domi::SUCCESS;
  }

  template<typename T, bool IsList = false, typename std::enable_if<!IsList, bool>::type = true>
  Status GetNodeAttrDataTmpl(const std::string &name, DataBuf &data) {
    auto func = Getter<T>::func;
    typename Getter<T>::ST value;
    bool res = func(op_desc_, name, value);
    if (!res) {
      GE_LOGE("attr not found. %s", name.c_str());
      return domi::FAILED;
    }

    auto dest_ptr = std::make_shared<AnyValue<T>>(static_cast<T>(value));
    data_map_.emplace(name + '_' + typeid(T).name(), dest_ptr);
    data = dest_ptr->GetDataBuf();
    GELOGI("Single attr found. %s", name.c_str());
    return domi::SUCCESS;
  }

  template<typename T, bool IsList = false, typename std::enable_if<IsList, bool>::type = true>
  Status GetNodeAttrDataTmpl(const std::string &name, DataBuf &data) {
    auto func = Getter<T>::list_func;
    std::vector<typename Getter<T>::ST> value;
    bool res = func(op_desc_, name, value);
    if (!res) {
      GE_LOGE("List attr not found. %s", name.c_str());
      return domi::FAILED;
    }

    std::vector<T> dest;
    for (auto elem : value) {
      dest.emplace_back(static_cast<T>(elem));
    }
    auto dest_ptr = std::make_shared<AnyVecValue<T>>(dest);
    data_map_.emplace(name + '_' + typeid(T).name(), dest_ptr);
    data = dest_ptr->GetDataBuf();
    GELOGI("attr found. %s", name.c_str());
    return domi::SUCCESS;
  }

 private:
  static std::map<std::string, std::function<Status(TeOpVarAttrArgsImpl *, const std::string &, DataBuf &)>>
      data_getter_;
  ge::OpDescPtr op_desc_;
  std::map<std::string, std::shared_ptr<AnyValueBase>> data_map_;
};

class VarAttrHelper {
 public:
  static void InitTeOpVarAttr(ge::OpDescPtr &op_desc, TeOpVarAttrArgs &attr);
};

const char *COMPILE_INFO_JSON = "compile_info_json";
const char *COMPILE_INFO_KEY = "compile_info_key";
const char *ATOMIC_COMPILE_INFO_JSON = "_atomic_compile_info_json";
const char *ATOMIC_COMPILE_INFO_KEY = "_atomic_compile_info_key";

const std::map<ge::DataType, std::string> DATATYPE_STRING_MAP{{ge::DT_FLOAT, "float32"},
                                                              {ge::DT_FLOAT16, "float16"},
                                                              {ge::DT_INT8, "int8"},
                                                              {ge::DT_INT16, "int16"},
                                                              {ge::DT_INT32, "int32"},
                                                              {ge::DT_INT64, "int64"},
                                                              {ge::DT_UINT8, "uint8"},
                                                              {ge::DT_UINT16, "uint16"},
                                                              {ge::DT_UINT32, "uint32"},
                                                              {ge::DT_UINT64, "uint64"},
                                                              {ge::DT_BOOL, "bool"},
                                                              {ge::DT_DOUBLE, "double"},
                                                              {ge::DT_DUAL, "dual"},
                                                              {ge::DT_DUAL_SUB_INT8, "dual_sub_int8"},
                                                              {ge::DT_DUAL_SUB_UINT8, "dual_sub_uint8"}};

std::map<std::string, std::function<Status(TeOpVarAttrArgsImpl *, const std::string &, DataBuf &)>>
    TeOpVarAttrArgsImpl::data_getter_ = {{"Int8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int8_t>},
                                         {"Int16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int16_t>},
                                         {"Int32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int32_t>},
                                         {"Int64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int64_t>},
                                         {"UInt8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint8_t>},
                                         {"UInt16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint16_t>},
                                         {"UInt32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint32_t>},
                                         {"UInt64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint64_t>},
                                         {"Float", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<float>},
                                         {"ListInt8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int8_t, true>},
                                         {"ListInt16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int16_t, true>},
                                         {"ListInt32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int32_t, true>},
                                         {"ListInt64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<int64_t, true>},
                                         {"ListUInt8", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint8_t, true>},
                                         {"ListUInt16", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint16_t, true>},
                                         {"ListUInt32", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint32_t, true>},
                                         {"ListUInt64", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<uint64_t, true>},
                                         {"ListFloat", &TeOpVarAttrArgsImpl::GetNodeAttrDataTmpl<float, true>}};

void VarAttrHelper::InitTeOpVarAttr(ge::OpDescPtr &op_desc, TeOpVarAttrArgs &attr) {
  attr.impl_ = std::make_shared<TeOpVarAttrArgsImpl>(op_desc);
}

Status TeOpVarAttrArgsImpl::GetDataByName(const string &name, const string &dtype, DataBuf &data) {
  auto iter = data_getter_.find(dtype);
  if (iter == data_getter_.end()) {
    GE_LOGE("wrong dtype: %s", dtype.c_str());
    return domi::FAILED;
  } else {
    return iter->second(this, name, data);
  }
}

const uint8_t *TeOpVarAttrArgs::GetData(const std::string &name, const std::string &dtype, size_t &size) const {
  DataBuf data(nullptr, 0);
  auto rc = impl_->GetDataByName(name, dtype, data);
  if (rc == domi::SUCCESS) {
    GELOGI("attr found. %s, %s, %p, %ld", name.c_str(), dtype.c_str(), std::get<0>(data), std::get<1>(data));
  }
  size = std::get<1>(data);
  return std::get<0>(data);
}

bool FeedTeOpTensorArg(ge::OpDesc::Vistor<ge::GeTensorDescPtr> &tensor_desc, std::vector<TeOpTensorArg> &tensor_arg,
                       ge::OpDescPtr &op_desc) {
  size_t index = 0;
  for (auto &desc : tensor_desc) {
    TeOpTensorArg arg_tensor;
    TeOpTensor tensor;
    arg_tensor.arg_type = TA_SINGLE;
    tensor.shape = desc->GetShape().GetDims();
    if (tensor.shape.empty()) {
      tensor.shape = {1};
    }
    tensor.ori_shape = desc->GetOriginShape().GetDims();
    tensor.name = op_desc->GetInputNameByIndex(index);

    ge::Format primary_format = static_cast<ge::Format>(ge::GetPrimaryFormat(desc->GetFormat()));
    tensor.format = ge::TypeUtils::FormatToSerialString(primary_format);

    tensor.ori_format = ge::TypeUtils::FormatToSerialString(desc->GetOriginFormat());

    ge::DataType dtype = desc->GetDataType();
    auto dataTypeIter = DATATYPE_STRING_MAP.find(dtype);
    if (dataTypeIter == DATATYPE_STRING_MAP.end()) {
      GE_LOGE("datatype error %d", static_cast<int>(dtype));
      return false;
    }
    tensor.dtype = dataTypeIter->second;
    if (LOG_ENABLED(DLOG_INFO)) {
      std::stringstream shapestr;
      shapestr << "shape:[";
      for (auto &i : tensor.shape) {
        shapestr << i << ",";
      }
      shapestr << "], ori_shape:[";
      for (auto &i : tensor.ori_shape) {
        shapestr << i << ",";
      }
      shapestr << "], format:" << tensor.format;
      shapestr << ", ori_format:" << tensor.ori_format;
      shapestr << ", dtype: " << tensor.dtype;
      GELOGI("calling optiling shape info: %s", shapestr.str().c_str());
    }

    arg_tensor.tensor.emplace_back(tensor);
    tensor_arg.emplace_back(arg_tensor);
    index++;
  }
  return true;
}

void FeedTeOpConstTensor(const ge::Node &node, const ge::OpDescPtr &op_desc,
                         std::map<std::string, TeConstTensorData> &const_inputs) {
  ge::Operator op = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());
  std::vector<std::string> inferDepends = op_desc->GetOpInferDepends();

  for (auto &depend : inferDepends) {
    ge::Tensor data;
    ge::graphStatus rc = op.GetInputConstData(depend.c_str(), data);
    GELOGI("GetInputConstData: %s, %d", depend.c_str(), rc);
    if (rc != ge::GRAPH_SUCCESS) {
      continue;
    }

    const uint8_t *pbuf = data.GetData();
    size_t buflen = data.GetSize();

    GELOGI("Const input tensor data: %s, %p %zu", depend.c_str(), pbuf, buflen);
    const_inputs.emplace(depend, TeConstTensorData{pbuf, buflen, data});
  }
}

bool GetCompileInfo(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name,
                    OpCompileInfo &op_compile_info) {
  bool bres = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY, op_compile_info.key);
  if (!bres) {
    GE_LOGE("Can not find the attribute %s. op_type:%s, op_name:%s", COMPILE_INFO_KEY, op_type, op_name);
    return false;
  }

  bres = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON, op_compile_info.str);
  if (!bres) {
    GE_LOGE("Can not find the attribute %s. op_type:%s, op_name:%s", COMPILE_INFO_JSON, op_type, op_name);
    return false;
  }
  return true;
}

bool GetCompileInfoV2(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name,
                      optiling::utils::OpCompileInfo &op_compile_info) {
  std::string op_compile_info_key;
  std::string op_compile_info_json;
  bool bres = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_KEY, op_compile_info_key);
  if (!bres) {
    REPORT_CALL_ERROR("E19999",
                      "Can not find the attribute compile info key %s. "
                      "op_type:%s, op_name:%s",
                      COMPILE_INFO_KEY, op_type, op_name);
    return false;
  }
  ge::AscendString compile_info_key(op_compile_info_key.c_str());
  op_compile_info.SetKey(compile_info_key);
  bres = ge::AttrUtils::GetStr(op_desc, COMPILE_INFO_JSON, op_compile_info_json);
  if (!bres) {
    REPORT_CALL_ERROR("E19999",
                      "Can not find the attribute compile info json%s. "
                      "op_type:%s, op_name:%s",
                      COMPILE_INFO_JSON, op_type, op_name);
    return false;
  }
  ge::AscendString compile_info_value(op_compile_info_json.c_str());
  op_compile_info.SetValue(compile_info_value);
  return true;
}

bool GetAtomicCleanCompileInfo(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name,
                               OpCompileInfo &op_compile_info) {
  bool bres = ge::AttrUtils::GetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, op_compile_info.key);
  if (!bres) {
    GE_LOGE("Can not find the attribute %s. op_type:%s, op_name:%s", ATOMIC_COMPILE_INFO_KEY, op_type, op_name);
    return false;
  }

  bres = ge::AttrUtils::GetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, op_compile_info.str);
  if (!bres) {
    GE_LOGE("Can not find the attribute %s. op_type:%s, op_name:%s", ATOMIC_COMPILE_INFO_JSON, op_type, op_name);
    return false;
  }
  return true;
}

bool GetAtomicCleanCompileInfoV2(const ge::OpDescPtr &op_desc, const char *op_type, const char *op_name,
                                 optiling::utils::OpCompileInfo &op_compile_info) {
  std::string op_compile_info_key;
  std::string op_compile_info_json;
  bool bres = ge::AttrUtils::GetStr(op_desc, ATOMIC_COMPILE_INFO_KEY, op_compile_info_key);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "Can not find the attribute %s. op_type:%s, op_name:%s", ATOMIC_COMPILE_INFO_KEY,
                      op_type, op_name);
    return false;
  }
  ge::AscendString compile_info_key(op_compile_info_key.c_str());
  op_compile_info.SetKey(compile_info_key);

  bres = ge::AttrUtils::GetStr(op_desc, ATOMIC_COMPILE_INFO_JSON, op_compile_info_json);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "Can not find the attribute %s. op_type:%s, op_name:%s", ATOMIC_COMPILE_INFO_JSON,
                      op_type, op_name);
    return false;
  }
  ge::AscendString compile_info_value(op_compile_info_json.c_str());
  op_compile_info.SetValue(compile_info_value);
  return true;
}

void ParseShapeDesc(const nlohmann::json &shape, std::vector<TeOpTensor> &tensors) {
  TeOpTensor tensor;
  if (shape.contains("shape")) {
    tensor.shape = shape["shape"].get<vector<int64_t>>();
  }
  if (shape.contains("ori_shape")) {
    tensor.ori_shape = shape["ori_shape"].get<vector<int64_t>>();
  }
  if (shape.contains("format")) {
    tensor.format = shape["format"].get<std::string>();
  }
  if (shape.contains("ori_format")) {
    tensor.ori_format = shape["ori_format"].get<std::string>();
  }
  if (shape.contains("dtype")) {
    tensor.dtype = shape["dtype"].get<std::string>();
  }
  tensors.emplace_back(tensor);
}

void ParseShapeDescList(const nlohmann::json &shape_list, std::vector<TeOpTensorArg> &op_args) {
  for (const auto &elem : shape_list) {
    TeOpTensorArg tensor_arg;
    tensor_arg.arg_type = TA_NONE;

    if (elem.is_array()) {
      tensor_arg.arg_type = TA_LIST;
      for (const auto &shape : elem) {
        ParseShapeDesc(shape, tensor_arg.tensor);
      }
    } else {
      tensor_arg.arg_type = TA_SINGLE;
      ParseShapeDesc(elem, tensor_arg.tensor);
    }
    op_args.emplace_back(tensor_arg);
  }
}

template<typename T>
void GetConstDataPointer(const nlohmann::json &json_array, std::vector<uint8_t> &const_value) {
  std::vector<T> value = json_array.get<std::vector<T>>();
  uint8_t *pv_begin = reinterpret_cast<uint8_t *>(value.data());
  uint8_t *pv_end = pv_begin + value.size() * sizeof(T);
  const_value = std::move(std::vector<uint8_t>(pv_begin, pv_end));
}

bool CopyConstData(const std::string &dtype, const nlohmann::json &json_array, std::vector<uint8_t> &value) {
  if (dtype == "int8") {
    GetConstDataPointer<int8_t>(json_array, value);
  } else if (dtype == "uint8") {
    GetConstDataPointer<uint8_t>(json_array, value);
  } else if (dtype == "int16") {
    GetConstDataPointer<int16_t>(json_array, value);
  } else if (dtype == "uint16") {
    GetConstDataPointer<uint16_t>(json_array, value);
  } else if (dtype == "int32") {
    GetConstDataPointer<int32_t>(json_array, value);
  } else if (dtype == "uint32") {
    GetConstDataPointer<uint32_t>(json_array, value);
  } else if (dtype == "int64") {
    GetConstDataPointer<int64_t>(json_array, value);
  } else if (dtype == "uint64") {
    GetConstDataPointer<uint64_t>(json_array, value);
  } else if (dtype == "float32") {
    GetConstDataPointer<float>(json_array, value);
  } else if (dtype == "double") {
    GetConstDataPointer<double>(json_array, value);
  } else {
    GE_LOGE("Unknown dtype: %s", dtype.c_str());
    return false;
  }
  return true;
}

void ParseConstShapeDescV2(const nlohmann::json &shape_json, ge::Operator &op_para,
                           std::map<std::string, std::vector<uint8_t>> &const_values, ge::OpDescPtr op_desc) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    REPORT_CALL_ERROR("E19999", "const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  ge::GeShape ge_shape(shape);
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
  ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  ge::GeTensorDesc ge_tensor(ge_shape, ge_format, ge_dtype);
  ge_tensor.SetName(name);
  ge::GeTensor const_tensor(ge_tensor, res.first->second);
  ge::GeTensorPtr const_tensor_ptr = std::make_shared<ge::GeTensor>(const_tensor);
  ge::OpDescPtr const_op_desc = ge::OpDescUtils::CreateConstOp(const_tensor_ptr);
  ge::Operator const_op = ge::OpDescUtils::CreateOperatorFromOpDesc(const_op_desc);
  op_para.SetInput(name, const_op);
  return;
}

void ParseConstShapeDesc(const nlohmann::json &shape_json, std::map<std::string, TeConstTensorData> &const_tensors,
                         std::map<std::string, std::vector<uint8_t>> &const_values) {
  std::vector<int64_t> shape;
  std::string format_str;
  std::string dtype_str;

  if (!shape_json.contains("const_value")) {
    GELOGI("Not const tenosr");
    return;
  }
  if (!shape_json.contains("name")) {
    GE_LOGE("const tensor has no name");
    return;
  }
  std::string name = shape_json["name"];

  if (shape_json.contains("shape")) {
    shape = shape_json["shape"].get<vector<int64_t>>();
  }
  if (shape_json.contains("format")) {
    format_str = shape_json["format"].get<std::string>();
  }
  if (shape_json.contains("dtype")) {
    dtype_str = shape_json["dtype"].get<std::string>();
  }

  std::vector<uint8_t> value;
  bool bres = CopyConstData(dtype_str, shape_json["const_value"], value);
  if (!bres) {
    GE_LOGE("CopyConstData faild.  buffer is null");
    return;
  }
  auto res = const_values.emplace(name, std::move(value));
  if (res.first == const_values.end()) {
    return;  // CodeDEX complains 'CHECK_CONTAINER_EMPTY'
  }

  ge::Shape ge_shape(shape);
  std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
  dtype_str = "DT_" + dtype_str;
  ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
  std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
  ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
  ge::Tensor const_tensor(ge::TensorDesc(ge_shape, ge_format, ge_dtype), res.first->second);
  const_tensors.emplace(name, std::make_tuple(const_tensor.GetData(), const_tensor.GetSize(), const_tensor));
  return;
}

void ParseConstTensorListV2(const nlohmann::json &shape_list, ge::Operator &operator_para,
                            std::map<std::string, std::vector<uint8_t>> &const_values, ge::OpDescPtr op_desc) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDescV2(shape, operator_para, const_values, op_desc);
      }
    } else {
      ParseConstShapeDescV2(elem, operator_para, const_values, op_desc);
    }
  }
}

void ParseConstTensorList(const nlohmann::json &shape_list, std::map<std::string, TeConstTensorData> &const_tensors,
                          std::map<std::string, std::vector<uint8_t>> &const_values) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseConstShapeDesc(shape, const_tensors, const_values);
      }
    } else {
      ParseConstShapeDesc(elem, const_tensors, const_values);
    }
  }
}

std::string DumpByteBuffer(const ByteBuffer &buf) {
  static const char hex_digits[] = "0123456789ABCDEF";
  std::string str = buf.str();
  std::string output;
  output.reserve(str.size() * 2);
  for (unsigned char c : str) {
    output.push_back(hex_digits[c >> 4]);
    output.push_back(hex_digits[c & 15]);
  }
  return output;
}

bool DumpRunInfoV2(optiling::utils::OpRunInfo &run_info, char *run_info_json, size_t run_info_len) {
  if (run_info_json == nullptr) {
    REPORT_CALL_ERROR("E19999", "run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  std::vector<int64_t> workspaces;
  int64_t workspace;
  for (size_t i = 0; i < run_info.GetWorkspaceNum(); ++i) {
    (void) run_info.GetWorkspace(i, workspace);
    workspaces.push_back(workspace);
  }
  json_obj["block_dim"] = run_info.GetBlockDim();
  json_obj["workspaces"] = workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.GetAllTilingData());
  json_obj["clear_atomic"] = run_info.GetClearAtomic();
  json_obj["tiling_key"] = run_info.GetTilingKey();

  std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    REPORT_CALL_ERROR("E19999", "runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  auto rc = memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1);
  if (rc != EOK) {
    return false;
  }
  return true;
}

bool DumpRunInfo(const OpRunInfo &run_info, char *run_info_json, size_t run_info_len) {
  if (run_info_json == nullptr) {
    GE_LOGE("run_info buffer is null");
    return false;
  }

  nlohmann::json json_obj;
  json_obj["block_dim"] = run_info.block_dim;
  json_obj["workspaces"] = run_info.workspaces;
  json_obj["tiling_data"] = DumpByteBuffer(run_info.tiling_data);
  json_obj["clear_atomic"] = run_info.clear_atomic;
  json_obj["tiling_key"] = run_info.tiling_key;

  std::string str = json_obj.dump();
  if (str.size() >= run_info_len) {
    GE_LOGE("runinfo too large. %zu/%zu", str.size(), run_info_len);
    return false;
  }
  auto rc = memcpy_s(run_info_json, str.size() + 1, str.c_str(), str.size() + 1);
  if (rc != EOK) {
    return false;
  }
  return true;
}

extern "C" int TbeOpTilingPyInterfaceEx2BackUp(const char *optype, const char *compile_info, const char *inputs,
                                               const char *outputs, char *run_info_json, size_t run_info_len,
                                               const char *compile_info_hash, uint64_t *elapse,
                                               std::map<std::string, optiling::OpTilingFunc>::iterator iter) {
  if (optype == nullptr || compile_info == nullptr || inputs == nullptr || outputs == nullptr) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }

  std::chrono::time_point<std::chrono::steady_clock> before_tiling, after_tiling;
  std::string compile_info_str = compile_info;
  TeOpParas op_params;
  op_params.op_type = optype;
  std::map<std::string, std::vector<uint8_t>> const_values;
  try {
    nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescList(inputs_json, op_params.inputs);
    ParseShapeDescList(outputs_json, op_params.outputs);
    ParseConstTensorList(inputs_json, op_params.const_inputs, const_values);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }
  GELOGI("Optiling func found, op_type:%s, func:[%s:%p]", optype, iter->first.c_str(),
         iter->second.target<OpTilingFuncPtr>());

  OpCompileInfo op_compile_info{compile_info};
  if (compile_info_hash) {
    op_compile_info.key = compile_info_hash;
  }

  OpRunInfo run_info;
  if (elapse) {
    before_tiling = std::chrono::steady_clock::now();
  }

  bool rc = (iter->second)(op_params, op_compile_info, run_info);

  if (elapse) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    REPORT_CALL_ERROR("E19999", "Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse) {
    *elapse = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();
    *(elapse + 1) = last_op_tiling_perf;
    last_op_tiling_perf = -1;
  }

  GELOGI("Optiling succeed. op_type:%s", optype);
  DumpRunInfo(run_info, run_info_json, run_info_len);
  return 1;
}

void ParseShapeDescV2(const nlohmann::json &shape, ge::OpDescPtr &op_desc, std::string Flag) {
  ge::GeTensorDesc tensor;
  std::string name;
  if (shape.contains("shape")) {
    tensor.SetShape(ge::GeShape(shape["shape"].get<vector<int64_t>>()));
  }
  if (shape.contains("ori_shape")) {
    tensor.SetOriginShape(ge::GeShape(shape["ori_shape"].get<vector<int64_t>>()));
  }
  if (shape.contains("format")) {
    std::string format_str = shape["format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetFormat(ge_format);
  }
  if (shape.contains("ori_format")) {
    std::string format_str = shape["ori_format"].get<std::string>();
    std::transform(format_str.begin(), format_str.end(), format_str.begin(), ::toupper);
    ge::Format ge_format = ge::TypeUtils::SerialStringToFormat(format_str);
    tensor.SetOriginFormat(ge_format);
  }
  if (shape.contains("dtype")) {
    std::string dtype_str = shape["dtype"].get<std::string>();
    std::transform(dtype_str.begin(), dtype_str.end(), dtype_str.begin(), ::toupper);
    dtype_str = "DT_" + dtype_str;
    ge::DataType ge_dtype = ge::TypeUtils::SerialStringToDataType(dtype_str);
    tensor.SetDataType(ge_dtype);
  }
  if (shape.contains("name")) {
    name = shape["name"];
    tensor.SetName(name);
    Flag == "inputs" ? op_desc->AddInputDesc(name, tensor) : op_desc->AddOutputDesc(name, tensor);
  } else {
    Flag == "inputs" ? op_desc->AddInputDesc(tensor) : op_desc->AddOutputDesc(tensor);
  }
}

void ParseShapeDescListV2(const nlohmann::json &shape_list, ge::OpDescPtr &op_desc, std::string Flag) {
  for (const auto &elem : shape_list) {
    if (elem.is_array()) {
      for (const auto &shape : elem) {
        ParseShapeDescV2(shape, op_desc, Flag);
      }
    } else {
      ParseShapeDescV2(elem, op_desc, Flag);
    }
  }
}

extern "C" int TbeOpTilingPyInterfaceEx2New(const char *optype, const char *compile_info, const char *inputs,
                                            const char *outputs, char *run_info_json, size_t run_info_len,
                                            const char *compile_info_hash, uint64_t *elapse,
                                            std::map<std::string, optiling::utils::OpTilingFuncV2>::iterator iter) {
  if (optype == nullptr || compile_info == nullptr || inputs == nullptr || outputs == nullptr) {
    REPORT_CALL_ERROR("E19999", "optype/compile_info/inputs/outputs is null, %s, %s, %s, %s", optype, compile_info,
                      inputs, outputs);
    return 0;
  }

  GELOGI("Optiling func found, op_type:%s, func:[%s:%p]", optype, iter->first.c_str(),
         iter->second.target<optiling::utils::OpTilingFuncV2Ptr>());

  std::chrono::time_point<std::chrono::steady_clock> before_tiling, after_tiling;
  std::string compile_info_str = compile_info;
  std::string optype_str = optype;
  ge::OpDescPtr op_desc = std::make_shared<ge::OpDesc>("", optype_str);
  std::map<std::string, std::vector<uint8_t>> const_values;
  ge::Operator operator_param;
  try {
    nlohmann::json inputs_json = nlohmann::json::parse(inputs);
    nlohmann::json outputs_json = nlohmann::json::parse(outputs);
    ParseShapeDescListV2(inputs_json, op_desc, "inputs");
    ParseShapeDescListV2(outputs_json, op_desc, "outputs");
    operator_param = ge::OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    ParseConstTensorListV2(inputs_json, operator_param, const_values, op_desc);
  } catch (...) {
    REPORT_CALL_ERROR("E19999", "Failed to parse json_str. %s, %s, %s", compile_info, inputs, outputs);
    return 0;
  }

  optiling::utils::OpCompileInfo op_compile_info{"", compile_info};
  ge::AscendString opCompileInfoHash(compile_info_hash);
  if (compile_info_hash) {
    op_compile_info.SetKey(opCompileInfoHash);
  }

  optiling::utils::OpRunInfo run_info(uint32_t(0), false, uint32_t(0));
  if (elapse) {
    before_tiling = std::chrono::steady_clock::now();
  }

  bool rc = (iter->second)(operator_param, op_compile_info, run_info);

  if (elapse) {
    after_tiling = std::chrono::steady_clock::now();
  }
  if (!rc) {
    REPORT_CALL_ERROR("E19999", "Optiling failed. op_type:%s", optype);
    return 0;
  }

  if (elapse) {
    *elapse = std::chrono::duration_cast<std::chrono::microseconds>(after_tiling - before_tiling).count();
    *(elapse + 1) = last_op_tiling_perf;
    last_op_tiling_perf = -1;
  }

  GELOGI("Optiling succeed. op_type:%s", optype);
  DumpRunInfoV2(run_info, run_info_json, run_info_len);
  return 1;
}

extern "C" int TbeOpTilingPyInterfaceEx2(const char *optype, const char *compile_info, const char *inputs,
                                         const char *outputs, char *run_info_json, size_t run_info_len,
                                         const char *compile_info_hash, uint64_t *elapse) {
  auto &interf_2 = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf();
  auto &interf_1 = OpTilingRegistryInterf::RegisteredOpInterf();
  int flag = 1;
  auto iter_2 = interf_2.find(optype);
  auto iter_1 = interf_1.find(optype);
  if (iter_2 == interf_2.end()) {
    GELOGI("Optiling func[optype] in V2 not found, turn to find it in V1[optype]. "
           "op_type:%s",
           optype);
    flag = 0;
    if (iter_1 == interf_1.end()) {
      GELOGI("Optiling func[optype] in V1 not found, turn to find it in "
             "V2[Autotiling]. op_type:%s",
             optype);
      iter_2 = interf_2.find("AutoTiling");
      flag = 1;
      if (iter_2 == interf_2.end()) {
        GELOGI("Optiling func[AutoTiling] in V2 not found, turn to find it in "
               "V1[Autotiling]. op_type:%s",
               optype);
        iter_1 = interf_1.find("AutoTiling");
        flag = 0;
        if (iter_1 == interf_1.end()) {
          REPORT_CALL_ERROR("E19999", "Optiling func not found. op_type:%s", optype);
          return 0;
        }
      }
    }
  }

  return (flag == 1 ? TbeOpTilingPyInterfaceEx2New(optype, compile_info, inputs, outputs, run_info_json, run_info_len,
                                                   compile_info_hash, elapse, iter_2)
                    : TbeOpTilingPyInterfaceEx2BackUp(optype, compile_info, inputs, outputs, run_info_json,
                                                      run_info_len, compile_info_hash, elapse, iter_1));
}

extern "C" int TbeOpTilingPyInterfaceEx(const char *optype, const char *compile_info, const char *inputs,
                                        const char *outputs, char *run_info_json, size_t run_info_len,
                                        uint64_t *elapse) {
  return TbeOpTilingPyInterfaceEx2(optype, compile_info, inputs, outputs, run_info_json, run_info_len, nullptr, elapse);
}

extern "C" int TbeOpTilingPyInterface(const char *optype, const char *compile_info, const char *inputs,
                                      const char *outputs, char *run_info_json, size_t run_info_len) {
  return TbeOpTilingPyInterfaceEx(optype, compile_info, inputs, outputs, run_info_json, run_info_len, nullptr);
}

bool StructToClass_RunInfo(OpRunInfo &run_info_struct, optiling::utils::OpRunInfo &run_info_cls) {
  bool res = false;
  run_info_cls.InternelSetTiling(run_info_struct.tiling_data);
  run_info_cls.SetBlockDim(run_info_struct.block_dim);
  run_info_cls.SetClearAtomic(run_info_struct.clear_atomic);
  run_info_cls.SetTilingKey(run_info_struct.tiling_key);
  if (!run_info_struct.workspaces.empty()) {
    for (auto i : run_info_struct.workspaces) {
      run_info_cls.AddWorkspace(i);
    }
  } else {
    GELOGI("Null workspaces get from runinfo_struct.");
  }
  res = true;
  return res;
}

extern "C" ge::graphStatus OpParaCalculate(const ge::Node &node, OpRunInfo &run_info,
                                           std::map<std::string, optiling::OpTilingFunc>::iterator iter) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string op_type = op_desc->GetType();
  std::string op_name = op_desc->GetName();
  TeOpParas op_param;
  op_param.op_type = op_type;

  GELOGI("Do optiling, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());

  auto inputs = op_desc->GetAllInputsDescPtr();
  auto outputs = op_desc->GetAllOutputsDescPtr();

  bool bres = false;
  bres = FeedTeOpTensorArg(inputs, op_param.inputs, op_desc);
  if (!bres) {
    GE_LOGE("Do optiling, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  bres = FeedTeOpTensorArg(outputs, op_param.outputs, op_desc);
  if (!bres) {
    return ge::GRAPH_FAILED;
  }

  VarAttrHelper::InitTeOpVarAttr(op_desc, op_param.var_attrs);
  FeedTeOpConstTensor(node, op_desc, op_param.const_inputs);

  OpCompileInfo op_compile_info;
  bres = GetCompileInfo(op_desc, op_type.c_str(), op_name.c_str(), op_compile_info);
  if (!bres) {
    GE_LOGE("Failed to get compile_info, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }

  GELOGI("Optiling func found, op_type:%s, op_name:%s, func:[%s:%p]", op_type.c_str(), op_name.c_str(),
         iter->first.c_str(), iter->second.target<OpTilingFuncPtr>());
  bool rc = (iter->second)(op_param, op_compile_info, run_info);
  if (rc) {
    GELOGI("Optiling succeed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  } else {
    GE_LOGE("Optiling failed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  }
  return rc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TurnToOpParaCalculate(const ge::Node &node, optiling::utils::OpRunInfo &run_info,
                                      std::map<std::string, optiling::OpTilingFunc>::iterator iter) {
  OpRunInfo run_info_struct;
  run_info_struct.block_dim = run_info.GetBlockDim();
  run_info_struct.clear_atomic = run_info.GetClearAtomic();
  run_info_struct.tiling_key = run_info.GetTilingKey();
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string op_type = op_desc->GetType();
  std::string op_name = op_desc->GetName();
  if (OpParaCalculate(node, run_info_struct, iter) != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "OpParaCalculate failed, op_type[%s], op_name[%s]", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  if (!StructToClass_RunInfo(run_info_struct, run_info)) {
    REPORT_CALL_ERROR("E19999", "Trans struct to class failed, op_type[%s], op_name[%s].", op_type.c_str(),
                      op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

void checkTensordescShape(ge::OpDescPtr &op_desc, std::vector<size_t> &inputsIndexes,
                          std::vector<size_t> &outputsIndexes) {
  size_t input_size = op_desc->GetAllInputsSize();
  ge::GeTensorDesc tensor_temp;
  for (size_t i = 0; i < input_size; ++ i) {
    tensor_temp = op_desc->GetInputDesc(i);
    if (tensor_temp.GetShape().GetShapeSize() == 0) {
      inputsIndexes.push_back(i);
      tensor_temp.SetShape(ge::GeShape({1}));
      op_desc->UpdateInputDesc(i, tensor_temp);
    }
  }

  size_t output_size = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++ i) {
    tensor_temp = op_desc->GetOutputDesc(i);
    if (tensor_temp.GetShape().GetShapeSize() == 0) {
      outputsIndexes.push_back(i);
      tensor_temp.SetShape(ge::GeShape({1}));
      op_desc->UpdateOutputDesc(i, tensor_temp);
    }
  }
}

void backTraceTensordescShape(ge::OpDescPtr &op_desc, const std::vector<size_t> &inputsIndexes,
                              const std::vector<size_t> &outputsIndexes) {
  auto iter = inputsIndexes.begin();
  ge::GeTensorDesc tensor_temp;
  std::vector<int64_t> noneVec;
  while(iter != inputsIndexes.end()) {
    tensor_temp = op_desc->GetInputDesc(*iter);
    tensor_temp.SetShape(ge::GeShape(noneVec));
    op_desc->UpdateInputDesc(*iter, tensor_temp);
    ++ iter;
  }
  iter = outputsIndexes.begin();
  while(iter != outputsIndexes.end()) {
    tensor_temp = op_desc->GetOutputDesc(*iter);
    tensor_temp.SetShape(ge::GeShape(noneVec));
    op_desc->UpdateOutputDesc(*iter, tensor_temp);
    ++ iter;
  }
}

void addNameToTensordesc(ge::OpDescPtr &op_desc) {
  std::vector<std::string> inferDepends = op_desc->GetOpInferDepends();
  ge::GeTensorDesc tensor_temp;
  for (auto name : inferDepends) {
    tensor_temp = op_desc->GetInputDesc(name);
    tensor_temp.SetName(name);
    op_desc->UpdateInputDesc(name, tensor_temp);
  }
}

extern "C" ge::graphStatus OpParaCalculateNew(const ge::Node &node, optiling::utils::OpRunInfo &run_info,
                                              std::map<std::string, optiling::utils::OpTilingFuncV2>::iterator iter) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string op_type = op_desc->GetType();
  std::string op_name = op_desc->GetName();
  std::vector<size_t> inputsIndexes;
  std::vector<size_t> outputsIndexes;
  checkTensordescShape(op_desc, inputsIndexes, outputsIndexes);
  addNameToTensordesc(op_desc);
  ge::Operator op_param = ge::OpDescUtils::CreateOperatorFromNode(node.shared_from_this());
  GELOGI("Do optiling, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());

  optiling::utils::OpCompileInfo op_compile_info("", "");
  bool bres = GetCompileInfoV2(op_desc, op_type.c_str(), op_name.c_str(), op_compile_info);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "Failed to get compile_info, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    backTraceTensordescShape(op_desc, inputsIndexes, outputsIndexes);
    return ge::GRAPH_FAILED;
  }

  GELOGI("Optiling func found, op_type:%s, op_name:%s, func:[%s:%p]", op_type.c_str(), op_name.c_str(),
         iter->first.c_str(), iter->second.target<optiling::utils::OpTilingFuncV2Ptr>());
  bool rc = (iter->second)(op_param, op_compile_info, run_info);
  if (rc) {
    GELOGI("Optiling succeed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  } else {
    REPORT_CALL_ERROR("E19999", "Optiling failed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  }
  backTraceTensordescShape(op_desc, inputsIndexes, outputsIndexes);
  return rc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

extern "C" ge::graphStatus OpParaCalculateV2(const ge::Node &node, optiling::utils::OpRunInfo &run_info) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string optype = op_desc->GetType();
  auto &interf_2 = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf();
  auto &interf_1 = OpTilingRegistryInterf::RegisteredOpInterf();
  int flag = 1;
  auto iter_2 = interf_2.find(optype);
  auto iter_1 = interf_1.find(optype);
  if (iter_2 == interf_2.end()) {
    GELOGI("Optiling func[optype] in V2 not found, turn to find it in V1[optype]. "
           "op_type:%s",
           optype.c_str());
    flag = 0;
    if (iter_1 == interf_1.end()) {
      GELOGI("Optiling func[optype] in V1 not found, turn to find it in "
             "V2[Autotiling]. op_type:%s",
             optype.c_str());
      iter_2 = interf_2.find("AutoTiling");
      flag = 1;
      if (iter_2 == interf_2.end()) {
        GELOGI("Optiling func[AutoTiling] in V2 not found, turn to find it in "
               "V1[Autotiling]. op_type:%s",
               optype.c_str());
        iter_1 = interf_1.find("AutoTiling");
        flag = 0;
        if (iter_1 == interf_1.end()) {
          REPORT_CALL_ERROR("E19999", "Optiling func not found. op_type:%s", optype.c_str());
          return ge::GRAPH_FAILED;
        }
      }
    }
  }
  return (flag == 1 ? OpParaCalculateNew(node, run_info, iter_2) : TurnToOpParaCalculate(node, run_info, iter_1));
}

extern "C" ge::graphStatus OpAtomicCalculate(const ge::Node &node, OpRunInfo &run_info) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string op_type = "DynamicAtomicAddrClean";
  std::string op_name = op_desc->GetName();
  std::string origin_op_type = "DynamicAtomicAddrClean";
  TeOpParas op_param;
  op_param.op_type = op_type;

  GELOGI("Do Atomic optiling. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  std::vector<int64_t> atomic_output_indices;
  (void) ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  if (atomic_output_indices.empty()) {
    GE_LOGE("No ATOMIC_ATTR_OUTPUT_INDEX found, op_type:%s, op_name:%s", origin_op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }

  auto tensor = op_desc->MutableOutputDesc(atomic_output_indices[0]);
  if (tensor == nullptr) {
    GE_LOGE("Get MutableOutputDesc failed. op_type:%s, op_name:%s", origin_op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }

  int64_t clean_size = 0;
  auto res = ge::TensorUtils::GetSize(*tensor, clean_size);
  if (res != ge::GRAPH_SUCCESS) {
    GE_LOGE("Get size of tensor desc failed. op_type:%s, op_name:%s", origin_op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }

  GELOGI("Atomic clean size: %ld, op_type:%s, op_name:%s", clean_size, origin_op_type.c_str(), op_name.c_str());
  op_param.const_inputs.emplace("workspace_size",
                                TeConstTensorData(nullptr, static_cast<size_t>(clean_size), ge::Tensor()));

  auto &interf = OpTilingRegistryInterf::RegisteredOpInterf();
  auto iter = interf.find(op_type);
  if (iter == interf.end()) {
    GE_LOGE("Atomic optiling func not found. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }

  OpCompileInfo op_compile_info;
  bool bres = GetAtomicCleanCompileInfo(op_desc, op_type.c_str(), op_name.c_str(), op_compile_info);
  if (!bres) {
    GE_LOGE("Failed to get compile_info, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }

  bool rc = (iter->second)(op_param, op_compile_info, run_info);
  if (rc) {
    GELOGI("Atomic optiling succeed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  } else {
    GE_LOGE("Atomic optiling failed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  }

  return rc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}

ge::graphStatus TurnToOpAtomicCalculate(const ge::Node &node, optiling::utils::OpRunInfo &run_info) {
  OpRunInfo run_info_struct;
  run_info_struct.block_dim = run_info.GetBlockDim();
  run_info_struct.clear_atomic = run_info.GetClearAtomic();
  run_info_struct.tiling_key = run_info.GetTilingKey();
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string op_type = op_desc->GetType();
  std::string op_name = op_desc->GetName();
  if (OpAtomicCalculate(node, run_info_struct) != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "OpAtomicCalculate failed, op_type[%s], op_name[%s]", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  if (!StructToClass_RunInfo(run_info_struct, run_info)) {
    REPORT_CALL_ERROR("E19999", "Trans struct to class failed, op_type[%s], op_name[%s].", op_type.c_str(),
                      op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

bool checkOpRegistryInterf(std::map<std::string, optiling::utils::OpTilingFuncV2> &interf,
                           std::map<std::string, optiling::utils::OpTilingFuncV2>::iterator iter, std::string op_name,
                           std::string op_type) {
  if (iter == interf.end()) {
    GELOGI("Atomic optiling func on the new way is not found, turn "
           "to the old way, op_type:%s, op_name:%s",
           op_type.c_str(), op_name.c_str());
    return false;
  }
  return true;
}

extern "C" ge::graphStatus OpAtomicCalculateV2(const ge::Node &node, optiling::utils::OpRunInfo &run_info) {
  ge::OpDescPtr op_desc = node.GetOpDesc();
  std::string op_type = "DynamicAtomicAddrClean";
  std::string op_name = op_desc->GetName();
  std::string origin_op_type = "DynamicAtomicAddrClean";
  ge::Operator op_param(op_type);
  auto &interf = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf();
  auto iter = interf.find(op_type);
  if (!checkOpRegistryInterf(interf, iter, op_type, op_name)) {
    return TurnToOpAtomicCalculate(node, run_info);
  }
  GELOGI("Do Atomic optiling. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  std::vector<int64_t> atomic_output_indices;
  (void) ge::AttrUtils::GetListInt(op_desc, ge::ATOMIC_ATTR_OUTPUT_INDEX, atomic_output_indices);
  if (atomic_output_indices.empty()) {
    REPORT_CALL_ERROR("E19999", "No ATOMIC_ATTR_OUTPUT_INDEX found, op_type:%s, op_name:%s", origin_op_type.c_str(),
                      op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  ge::GeTensorDescPtr tensor = op_desc->MutableOutputDesc(atomic_output_indices[0]);
  if (tensor == nullptr) {
    REPORT_CALL_ERROR("E19999", "Get MutableOutputDesc failed. op_type:%s, op_name:%s", origin_op_type.c_str(),
                      op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  int64_t clean_size = 0;
  vector<int> workspace_list;
  auto res = ge::TensorUtils::GetSize(*tensor, clean_size);
  if (res != ge::GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get size of tensor desc failed. op_type:%s, op_name:%s", origin_op_type.c_str(),
                      op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  GELOGI("Atomic clean size: %ld, op_type:%s, op_name:%s", clean_size, origin_op_type.c_str(), op_name.c_str());
  workspace_list.push_back(clean_size);
  op_param.SetAttr(ATTR_NAME_ATOMIC_CLEAN_WORKSPACE, workspace_list);
  optiling::utils::OpCompileInfo op_compile_info("", "");
  bool bres = GetAtomicCleanCompileInfoV2(op_desc, op_type.c_str(), op_name.c_str(), op_compile_info);
  if (!bres) {
    REPORT_CALL_ERROR("E19999", "Failed to get compile_info, op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
    return ge::GRAPH_FAILED;
  }
  bool rc = (iter->second)(op_param, op_compile_info, run_info);
  if (rc) {
    GELOGI("Atomic optiling succeed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  } else {
    REPORT_CALL_ERROR("E19999", "Atomic optiling failed. op_type:%s, op_name:%s", op_type.c_str(), op_name.c_str());
  }
  return rc ? ge::GRAPH_SUCCESS : ge::GRAPH_FAILED;
}
}  // namespace optiling
