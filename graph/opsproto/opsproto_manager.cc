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

#include "graph/opsproto_manager.h"
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <iostream>
#include <sstream>
#include "debug/ge_util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/debug/ge_log.h"
#include "graph/types.h"
#include "graph/def_types.h"
#include "mmpa/mmpa_api.h"

namespace ge {
OpsProtoManager *OpsProtoManager::Instance() {
  static OpsProtoManager instance;
  return &instance;
}

bool OpsProtoManager::Initialize(const std::map<std::string, std::string> &options) {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (is_init_) {
    GELOGI("OpsProtoManager is already initialized.");
    return true;
  }

  const std::map<std::string, std::string>::const_iterator iter = options.find("ge.opsProtoLibPath");
  if (iter == options.end()) {
    GELOGW("[Initialize][CheckOption] Option \"ge.opsProtoLibPath\" not set");
    return false;
  }

  pluginPath_ = iter->second;
  LoadOpsProtoPluginSo(pluginPath_);

  is_init_ = true;

  return true;
}

void OpsProtoManager::Finalize() {
  const std::lock_guard<std::mutex> lock(mutex_);

  if (!is_init_) {
    GELOGI("OpsProtoManager is not initialized.");
    return;
  }

  for (const auto handle : handles_) {
    if (handle != nullptr) {
      if (mmDlclose(handle) != 0) {
        const char_t *error = mmDlerror();
        error = (error == nullptr) ? "" : error;
        GELOGW("[Finalize][CloseHandle] close handle failed, reason:%s", error);
        continue;
      }
      GELOGI("close opsprotomanager handler success");
    } else {
      GELOGW("[Finalize][CheckHandle] handler is null");
    }
  }

  is_init_ = false;
}

static std::vector<std::string> SplitStr(const std::string &str, const char_t delim) {
  std::vector<std::string> elems;
  if (str.empty()) {
    elems.emplace_back("");
    return elems;
  }

  std::stringstream str_stream(str);
  std::string item;

  while (getline(str_stream, item, delim)) {
    elems.push_back(item);
  }

  const auto str_size = str.size();
  if ((str_size > 0UL) && (str[str_size - 1UL] == delim)) {
    elems.emplace_back("");
  }

  return elems;
}

static void FindParserSo(const std::string &path, std::vector<std::string> &file_list) {
  // Lib plugin path not exist
  if (path.empty()) {
    GELOGI("realPath is empty");
    return;
  }
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(path.size() >= static_cast<size_t>(MMPA_MAX_PATH),
                                 REPORT_INNER_ERROR("E18888", "param path size:%zu >= max path:%d",
                                                    path.size(), MMPA_MAX_PATH);
                                 return, "[Check][Param] path is invalid");

  char_t resolved_path[MMPA_MAX_PATH] = {};

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  const INT32 result = mmRealPath(path.c_str(), &(resolved_path[0U]), MMPA_MAX_PATH);
  if (result != EN_OK) {
    GELOGW("[FindSo][Check] Get real_path for file %s failed, reason:%s", path.c_str(), strerror(errno));
    return;
  }

  const INT32 is_dir = mmIsDir(&(resolved_path[0U]));
  // Lib plugin path not exist
  if (is_dir != EN_OK) {
    GELOGW("[FindSo][Check] Open directory %s failed, maybe it is not exit or not a dir, errmsg:%s",
           &(resolved_path[0U]), strerror(errno));
    return;
  }

  mmDirent **entries = nullptr;
  const auto ret = mmScandir(&(resolved_path[0U]), &entries, nullptr, nullptr);
  if ((ret < EN_OK) || (entries == nullptr)) {
    GELOGW("[FindSo][Scan] Scan directory %s failed, ret:%d, reason:%s", &(resolved_path[0U]), ret, strerror(errno));
    return;
  }
  for (int32_t i = 0; i < ret; ++i) {
    const mmDirent *const dir_ent = *PtrAdd<mmDirent*>(entries, static_cast<size_t>(ret), static_cast<size_t>(i));
    const std::string name = std::string(dir_ent->d_name);
    if ((strncmp(name.c_str(), ".", 1U) == 0) || (strncmp(name.c_str(), "..", 2U) == 0)) {
      continue;
    }
    const std::string full_name = path + "/" + name;
    const std::string so_suff = ".so";

    if ((static_cast<int32_t>(dir_ent->d_type) != DT_DIR) && (name.size() >= so_suff.size()) &&
        (name.compare(name.size() - so_suff.size(), so_suff.size(), so_suff) == 0)) {
      file_list.push_back(full_name);
      GELOGI("OpsProtoManager Parse full name = %s \n", full_name.c_str());
    }
  }
  mmScandirFree(entries, ret);
  GELOGI("Found %d libs.", ret);
}

static void GetPluginSoFileList(const std::string &path, std::vector<std::string> &file_list) {
  // Support multi lib directory with ":" as delimiter
  const std::vector<std::string> v_path = SplitStr(path, ':');

  for (auto i = 0UL; i < v_path.size(); ++i) {
    FindParserSo(v_path[i], file_list);
    GELOGI("OpsProtoManager full name = %s", v_path[i].c_str());
  }
}

void OpsProtoManager::LoadOpsProtoPluginSo(const std::string &path) {
  if (path.empty()) {
    REPORT_INNER_ERROR("E18888", "filePath is empty. please check your text file.");
    GELOGE(GRAPH_FAILED, "[Check][Param] filePath is empty. please check your text file.");
    return;
  }
  std::vector<std::string> file_list;

  // If there is .so file in the lib path
  GetPluginSoFileList(path, file_list);

  // Not found any .so file in the lib path
  if (file_list.empty()) {
    GELOGW("[LoadSo][Check] OpsProtoManager can not find any plugin file in pluginPath: %s \n", path.c_str());
    return;
  }
  // Warning message
  GELOGW("[LoadSo][Check] Shared library will not be checked. Please make sure that the source of shared library is "
         "trusted.");

  // Load .so file
  for (const auto elem : file_list) {
    void *const handle = mmDlopen(elem.c_str(), static_cast<int32_t>(static_cast<uint32_t>(MMPA_RTLD_NOW) |
        static_cast<uint32_t>(MMPA_RTLD_GLOBAL)));
    if (handle == nullptr) {
      const char_t *error = mmDlerror();
      error = (error == nullptr) ? "" : error;
      GELOGW("[LoadSo][Open] OpsProtoManager dlopen failed, plugin name:%s. Message(%s).", elem.c_str(), error);
      continue;
    } else {
      // Close dl when the program exist, not close here
      GELOGI("OpsProtoManager plugin load %s success.", elem.c_str());
      handles_.push_back(handle);
    }
  }
}
}  // namespace ge
