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

#include "graph/runtime_inference_context.h"
#include "graph/utils/tensor_adapter.h"
#include <cstdint>
#include "framework/common/debug/ge_log.h"

namespace ge {
std::map<std::string, std::unique_ptr<RuntimeInferenceContext>> RuntimeInferenceContext::contexts_;
std::mutex RuntimeInferenceContext::ctx_mu_;

graphStatus RuntimeInferenceContext::CreateContext(const std::string &context_id) {
  GELOGI("To create context. session id = %s", context_id.c_str());
  auto ctx = std::unique_ptr<RuntimeInferenceContext>(new (std::nothrow)RuntimeInferenceContext());
  if (ctx == nullptr) {
    REPORT_CALL_ERROR("E19999", "Failed to create instance of RuntimeInferenceContext. context_id = %s",
                      context_id.c_str());
    GELOGE(GRAPH_FAILED,
           "[Create][RuntimeInferenceContext] Failed to create instance of RuntimeInferenceContext. context_id = %s",
           context_id.c_str());
    return GRAPH_FAILED;
  }

  std::lock_guard<std::mutex> lk(ctx_mu_);
  auto emplace_ret = contexts_.emplace(context_id, std::move(ctx));
  if (!emplace_ret.second) {
    REPORT_INNER_ERROR("E19999", "save context failed as Old context not destroyed, context_id:%s",
                       context_id.c_str());
    GELOGE(GRAPH_FAILED, "[Add][Context] Old context not destroyed");
    return GRAPH_FAILED;
  }

  return GRAPH_SUCCESS;
}

void RuntimeInferenceContext::DestroyContext(const std::string &context_id) {
  GELOGI("To destroy context. session id = %s", context_id.c_str());
  std::lock_guard<std::mutex> lk(ctx_mu_);
  contexts_.erase(context_id);
}

graphStatus RuntimeInferenceContext::GetContext(const std::string &context_id, RuntimeInferenceContext **ctx) {
  std::lock_guard<std::mutex> lk(ctx_mu_);
  auto it = contexts_.find(context_id);
  if (it != contexts_.end()) {
    *ctx = it->second.get();
    return GRAPH_SUCCESS;
  }

  GELOGD("Runtime inference context not created. session id = %s", context_id.c_str());
  return GRAPH_FAILED;
}


graphStatus RuntimeInferenceContext::SetTensor(int64_t node_id, int output_id, Tensor &&tensor) {
  std::lock_guard<std::mutex> lk(mu_);
  auto &output_tensors = tensors_[node_id];
  if (static_cast<uint32_t>(output_id) >= output_tensors.size()) {
    output_tensors.resize(output_id + 1);
  }

  GELOGD("Set tensor for node_id = %ld, output_id = %d", node_id, output_id);
  output_tensors[output_id].reset(new (std::nothrow)Tensor(std::move(tensor)));
  if (output_tensors[output_id] == nullptr) {
    REPORT_CALL_ERROR("E19999", "[New][Tensor] Failed, node id = %ld, output index = %d", node_id, output_id);
    GELOGE(GRAPH_PARAM_INVALID, "[New][Tensor] Failed, node id = %ld, output index = %d", node_id, output_id);
    return GRAPH_FAILED;
  }

  auto &output_ge_tensors = ge_tensors_[node_id];
  if (static_cast<uint32_t>(output_id) >= output_ge_tensors.size()) {
    output_ge_tensors.resize(output_id + 1);
  }

  GELOGD("Set ge tensor successfully, node_id = %ld, output_id = %d", node_id, output_id);
  output_ge_tensors[output_id] = TensorAdapter::AsGeTensorPtr(tensor);
  return GRAPH_SUCCESS;
}

graphStatus RuntimeInferenceContext::GetTensor(int64_t node_id, int output_id, Tensor &tensor) {
  if (output_id < 0) {
    REPORT_INNER_ERROR("E19999", "param output_id:%d < 0, check invalid.", output_id);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Invalid output index: %d", output_id);
    return GRAPH_PARAM_INVALID;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto iter = tensors_.find(node_id);
  if (iter == tensors_.end()) {
    REPORT_INNER_ERROR("E19999", "Node not register. Id = %ld", node_id);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Node not register. Id = %ld", node_id);
    return INTERNAL_ERROR;
  }

  auto &output_tensors = iter->second;
  if (static_cast<uint32_t>(output_id) >= output_tensors.size()) {
    REPORT_INNER_ERROR("E19999", "Node output is not registered. node_id = %ld, output index = %d",
                       node_id, output_id);
    GELOGE(GRAPH_FAILED, "[Check][Param] Node output is not registered. node_id = %ld, output index = %d",
           node_id, output_id);
    return GRAPH_FAILED;
  }

  GELOGD("Get tensor for node_id = %ld, output_id = %d", node_id, output_id);
  auto &output_tensor = output_tensors[output_id];
  if (output_tensor == nullptr) {
    GELOGI("Node output is not registered. node_id = %ld, output index = %d", node_id, output_id);
    return GRAPH_FAILED;
  }
  tensor = *output_tensor;
  return GRAPH_SUCCESS;
}

graphStatus RuntimeInferenceContext::GetTensor(int64_t node_id, int output_id, GeTensorPtr &tensor) {
  if (output_id < 0) {
    REPORT_INNER_ERROR("E19999", "Invalid output index: %d", output_id);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Invalid output index: %d", output_id);
    return GRAPH_PARAM_INVALID;
  }

  std::lock_guard<std::mutex> lk(mu_);
  auto iter = ge_tensors_.find(node_id);
  if (iter == ge_tensors_.end()) {
    REPORT_INNER_ERROR("E19999", "Node not register. Id = %ld", node_id);
    GELOGE(INTERNAL_ERROR, "[Check][Param] Node not register. Id = %ld", node_id);
    return INTERNAL_ERROR;
  }

  auto &output_tensors = iter->second;
  if (static_cast<uint32_t>(output_id) >= output_tensors.size()) {
    REPORT_INNER_ERROR("E19999", "Node output is not registered. node_id = %ld, output index = %d",
                       node_id, output_id);
    GELOGE(GRAPH_FAILED, "[Check][Param] Node output is not registered. node_id = %ld, output index = %d",
           node_id, output_id);
    return GRAPH_FAILED;
  }

  GELOGD("Get ge tensor for node_id = %ld, output_id = %d", node_id, output_id);
  tensor = output_tensors[output_id];
  return GRAPH_SUCCESS;
}
} // namespace ge
