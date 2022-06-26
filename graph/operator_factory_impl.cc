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

#include "graph/operator_factory_impl.h"

#include <algorithm>

#include "debug/ge_log.h"
#include "graph/utils/mem_utils.h"

namespace ge {
std::shared_ptr<std::map<std::string, OpCreator>> OperatorFactoryImpl::operator_creators_;
std::shared_ptr<std::map<std::string, OpCreatorV2>> OperatorFactoryImpl::operator_creators_v2_;
std::shared_ptr<std::map<std::string, InferShapeFunc>> OperatorFactoryImpl::operator_infershape_funcs_;
std::shared_ptr<std::map<std::string, InferFormatFunc>> OperatorFactoryImpl::operator_inferformat_funcs_;
std::shared_ptr<std::map<std::string, VerifyFunc>> OperatorFactoryImpl::operator_verify_funcs_;
std::shared_ptr<std::map<std::string, InferDataSliceFunc>> OperatorFactoryImpl::operator_infer_data_slice_funcs_;
std::shared_ptr<std::map<std::string, InferValueRangePara>> OperatorFactoryImpl::operator_infer_value_range_paras_;
std::shared_ptr<std::map<std::string, InferAxisSliceFunc>> OperatorFactoryImpl::operator_infer_axis_slice_funcs_;
std::shared_ptr<std::map<std::string, InferAxisTypeInfoFunc>> OperatorFactoryImpl::operator_infer_axis_type_info_funcs_;

Operator OperatorFactoryImpl::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  if (operator_creators_v2_ != nullptr) {
    const std::map<std::string, ge::OpCreatorV2>::const_iterator
        it_v2 = operator_creators_v2_->find(operator_type);
    if (it_v2 != operator_creators_v2_->cend()) {
      return it_v2->second(operator_name.c_str());
    } else {
      GELOGW("[Create][Operator] No op_proto of [%s] registered by AscendString.", operator_type.c_str());
    }
  }
  if (operator_creators_ == nullptr) {
    return Operator();
  }
  const std::map<std::string, ge::OpCreator>::const_iterator it = operator_creators_->find(operator_type);
  if (it == operator_creators_->cend()) {
    GELOGW("[Create][Operator] No op_proto of [%s] registered by string.", operator_type.c_str());
    return Operator();
  }
  return it->second(operator_name);
}

graphStatus OperatorFactoryImpl::GetOpsTypeList(std::vector<std::string> &all_ops) {
  all_ops.clear();
  if (operator_creators_v2_ != nullptr) {
    all_ops.resize(operator_creators_v2_->size());
    (void)std::transform(
        operator_creators_v2_->begin(), operator_creators_v2_->end(), all_ops.begin(),
        [](const std::pair<std::string, OpCreatorV2> &operator_creator_v2) { return operator_creator_v2.first; });
    return GRAPH_SUCCESS;
  } else {
    GELOGW("[Get][OpsTypeList] Ops not registered by AscendString.");
  }

  if (operator_creators_ != nullptr) {
    all_ops.resize(operator_creators_->size());
    (void)std::transform(
        operator_creators_->begin(), operator_creators_->end(), all_ops.begin(),
        [](const std::pair<std::string, OpCreator> &operator_creator) { return operator_creator.first; });
  } else {
    REPORT_INNER_ERROR("E18888", "no operator creators found");
    GELOGE(GRAPH_FAILED, "[Check][Param] no operator creators found");
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}

bool OperatorFactoryImpl::IsExistOp(const std::string &operator_type) {
  if (operator_creators_v2_ != nullptr) {
    const std::map<std::string, ge::OpCreatorV2>::const_iterator it_v2 = operator_creators_v2_->find(operator_type);
    if (it_v2 != operator_creators_v2_->cend()) {
      return true;
    }
  }

  if (operator_creators_ == nullptr) {
    return false;
  }
  const std::map<std::string, ge::OpCreator>::const_iterator it = operator_creators_->find(operator_type);
  if (it == operator_creators_->cend()) {
    return false;
  }
  return true;
}

InferShapeFunc OperatorFactoryImpl::GetInferShapeFunc(const std::string &operator_type) {
  if (operator_infershape_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_infershape_funcs_->find(operator_type);
  if (it == operator_infershape_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

InferFormatFunc OperatorFactoryImpl::GetInferFormatFunc(const std::string &operator_type) {
  if (operator_inferformat_funcs_ == nullptr) {
    GELOGI("operator_inferformat_funcs_ is null");
    return nullptr;
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_inferformat_funcs_->find(operator_type);
  if (it == operator_inferformat_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

InferValueRangePara OperatorFactoryImpl::GetInferValueRangePara(const std::string &operator_type) {
  const InferValueRangePara ret_para;
  if (operator_infer_value_range_paras_ == nullptr) {
    GELOGI("operator_infervalue_paras_ is null, operator infer value registration is none");
    return ret_para;
  }
  const std::map<std::string, ge::InferValueRangePara>::const_iterator
      it = operator_infer_value_range_paras_->find(operator_type);
  if (it == operator_infer_value_range_paras_->end()) {
    GELOGI("optype[%s] has not registered infer value func", operator_type.c_str());
    return ret_para;
  }
  return it->second;
}

VerifyFunc OperatorFactoryImpl::GetVerifyFunc(const std::string &operator_type) {
  if (operator_verify_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_verify_funcs_->find(operator_type);
  if (it == operator_verify_funcs_->cend()) {
        return nullptr;
    }
    return it->second;
}

InferDataSliceFunc OperatorFactoryImpl::GetInferDataSliceFunc(const std::string &operator_type) {
  if (operator_infer_data_slice_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_infer_data_slice_funcs_->find(operator_type);
  if (it == operator_infer_data_slice_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

graphStatus OperatorFactoryImpl::RegisterOperatorCreator(const std::string &operator_type,
                                                         OpCreator const &op_creator) {
  if (operator_creators_ == nullptr) {
    operator_creators_ = MakeShared<std::map<std::string, OpCreator>>();
    GE_CHECK_NOTNULL(operator_creators_);
  }
  const std::map<std::string, ge::OpCreator>::const_iterator it = operator_creators_->find(operator_type);
  if (it != operator_creators_->cend()) {
    return GRAPH_FAILED;
  }
  (void)operator_creators_->emplace(operator_type, op_creator);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterOperatorCreator(const std::string &operator_type,
                                                         OpCreatorV2 const &op_creator) {
  if (operator_creators_v2_ == nullptr) {
    operator_creators_v2_ = MakeShared<std::map<std::string, OpCreatorV2>>();
    GE_CHECK_NOTNULL(operator_creators_v2_);
  }
  const std::map<std::string, ge::OpCreatorV2>::const_iterator it = operator_creators_v2_->find(operator_type);
  if (it != operator_creators_v2_->cend()) {
    return GRAPH_FAILED;
  }
  (void)operator_creators_v2_->emplace(operator_type, op_creator);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferShapeFunc(const std::string &operator_type,
                                                        InferShapeFunc const infer_shape_func) {
  if (operator_infershape_funcs_ == nullptr) {
    GELOGI("operator_infershape_funcs_ init");
    operator_infershape_funcs_ = MakeShared<std::map<std::string, InferShapeFunc>>();
    GE_CHECK_NOTNULL(operator_infershape_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_infershape_funcs_->find(operator_type);
  if (it != operator_infershape_funcs_->cend()) {
    GELOGW("[Register][InferFunc] op [%s] has already registered infer_func", operator_type.c_str());
    return GRAPH_FAILED;
  }
  GELOGD("Register infershape function of type: %s.", operator_type.c_str());
  (void)operator_infershape_funcs_->emplace(operator_type, infer_shape_func);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferFormatFunc(const std::string &operator_type,
                                                         InferFormatFunc const infer_format_func) {
  if (operator_inferformat_funcs_ == nullptr) {
    GELOGI("operator_inferformat_funcs_ init");
    operator_inferformat_funcs_ = MakeShared<std::map<std::string, InferFormatFunc>>();
    GE_CHECK_NOTNULL(operator_inferformat_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_inferformat_funcs_->find(operator_type);
  if (it != operator_inferformat_funcs_->cend()) {
    return GRAPH_FAILED;
  }
  (void)operator_inferformat_funcs_->emplace(operator_type, infer_format_func);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterVerifyFunc(const std::string &operator_type, VerifyFunc const verify_func) {
  if (operator_verify_funcs_ == nullptr) {
    GELOGI("operator_verify_funcs_ init");
    operator_verify_funcs_ = MakeShared<std::map<std::string, VerifyFunc>>();
    GE_CHECK_NOTNULL(operator_verify_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator it = operator_verify_funcs_->find(operator_type);
  if (it != operator_verify_funcs_->cend()) {
    return GRAPH_FAILED;
  }
  (void)operator_verify_funcs_->emplace(operator_type, verify_func);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferDataSliceFunc(const std::string &operator_type,
                                                            InferDataSliceFunc const infer_data_slice_func) {
  if (operator_infer_data_slice_funcs_ == nullptr) {
    GELOGI("operator_infer_data_slice_funcs_ init");
    operator_infer_data_slice_funcs_ = MakeShared<std::map<std::string, InferDataSliceFunc>>();
    GE_CHECK_NOTNULL(operator_infer_data_slice_funcs_);
  }
  const std::map<std::string, ge::InferShapeFunc>::const_iterator
      it = operator_infer_data_slice_funcs_->find(operator_type);
  if (it != operator_infer_data_slice_funcs_->cend()) {
    return GRAPH_FAILED;
  }
  (void)operator_infer_data_slice_funcs_->emplace(operator_type, infer_data_slice_func);
  return GRAPH_SUCCESS;
}

graphStatus OperatorFactoryImpl::RegisterInferValueRangeFunc(const std::string &operator_type) {
  return RegisterInferValueRangeFunc(operator_type, INPUT_HAS_VALUE_RANGE,
                                     true, nullptr);
}

graphStatus OperatorFactoryImpl::RegisterInferValueRangeFunc(const std::string &operator_type,
                                                             const WHEN_CALL when_call,
                                                             const bool use_cpu_kernel,
                                                             const InferValueRangeFunc &infer_value_range_func) {
  if (operator_infer_value_range_paras_ == nullptr) {
    GELOGI("operator_infervalue_paras_ init");
    operator_infer_value_range_paras_ = MakeShared<std::map<std::string, InferValueRangePara>>();
    GE_CHECK_NOTNULL(operator_infer_value_range_paras_);
  }
  const std::map<std::string, ge::InferValueRangePara>::const_iterator
      it = operator_infer_value_range_paras_->find(operator_type);
  if (it != operator_infer_value_range_paras_->cend()) {
    GELOGW("optype[%s] has registered infervalue func, no duplicate registration", operator_type.c_str());
    return GRAPH_FAILED;
  }
  InferValueRangePara tmp_para(when_call, use_cpu_kernel, infer_value_range_func);
  (void)operator_infer_value_range_paras_->emplace(operator_type, tmp_para);

  GELOGD("Optype[%s] infervalue func registered successfully, when_call = %d, use_cpu_kernel = %d",
         operator_type.c_str(), static_cast<int32_t>(when_call), static_cast<int32_t>(use_cpu_kernel));
  return GRAPH_SUCCESS;
}

InferAxisSliceFunc OperatorFactoryImpl::GetInferAxisSliceFunc(const std::string &operator_type) {
  if (operator_infer_axis_slice_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, InferAxisSliceFunc>::const_iterator
      it = operator_infer_axis_slice_funcs_->find(operator_type);
  if (it == operator_infer_axis_slice_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

graphStatus OperatorFactoryImpl::RegisterInferAxisSliceFunc(const std::string &operator_type,
                                                            const InferAxisSliceFunc &infer_axis_slice_func) {
  if (operator_infer_axis_slice_funcs_ == nullptr) {
    GELOGI("axis slice derivation funcs init");
    operator_infer_axis_slice_funcs_ = MakeShared<std::map<std::string, InferAxisSliceFunc>>();
    GE_CHECK_NOTNULL(operator_infer_axis_slice_funcs_);
  }
  const std::map<std::string, InferAxisSliceFunc>::const_iterator
      it = operator_infer_axis_slice_funcs_->find(operator_type);
  if (it != operator_infer_axis_slice_funcs_->cend()) {
    return GRAPH_FAILED;
  }
  (void)operator_infer_axis_slice_funcs_->emplace(operator_type, infer_axis_slice_func);
  return GRAPH_SUCCESS;
}

InferAxisTypeInfoFunc OperatorFactoryImpl::GetInferAxisTypeInfoFunc(const std::string &operator_type) {
  if (operator_infer_axis_type_info_funcs_ == nullptr) {
    return nullptr;
  }
  const std::map<std::string, InferAxisTypeInfoFunc>::const_iterator
      it = operator_infer_axis_type_info_funcs_->find(operator_type);
  if (it == operator_infer_axis_type_info_funcs_->cend()) {
    return nullptr;
  }
  return it->second;
}

graphStatus OperatorFactoryImpl::RegisterInferAxisTypeInfoFunc(const std::string &operator_type,
                                                               const InferAxisTypeInfoFunc &infer_axis_type_info_func) {
  if (operator_infer_axis_type_info_funcs_ == nullptr) {
    GELOGI("axis type info derivation funcs init");
    operator_infer_axis_type_info_funcs_ = MakeShared<std::map<std::string, InferAxisTypeInfoFunc>>();
    GE_CHECK_NOTNULL(operator_infer_axis_type_info_funcs_);
  }
  const std::map<std::string, InferAxisTypeInfoFunc>::const_iterator
      it = operator_infer_axis_type_info_funcs_->find(operator_type);
  if (it != operator_infer_axis_type_info_funcs_->cend()) {
    GELOGW("[Register][InferFunc] optype[%s] has registered axis type info func", operator_type.c_str());
    return GRAPH_FAILED;
  }
  (void)operator_infer_axis_type_info_funcs_->emplace(operator_type, infer_axis_type_info_func);
  return GRAPH_SUCCESS;
}
}  // namespace ge
