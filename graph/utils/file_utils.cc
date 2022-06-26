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

#include "file_utils.h"

#include "debug/ge_log.h"
#include "mmpa/mmpa_api.h"

namespace ge {
std::string RealPath(const char *path) {
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(path == nullptr,
                                 REPORT_INNER_ERROR("E19999", "path is nullptr, check invalid");
                                 return "", "[Check][Param] path pointer is NULL.");
  GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(strlen(path) >= MMPA_MAX_PATH,
                                 ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                                                 {path, std::to_string(MMPA_MAX_PATH)});
                                 return "",
                                 "[Check][Param]Path[%s] len is too long, it must be less than %d",
                                 path, MMPA_MAX_PATH);

  // Nullptr is returned when the path does not exist or there is no permission
  // Return absolute path when path is accessible
  std::string res;
  char resolved_path[MMPA_MAX_PATH] = {0};
  if (mmRealPath(path, resolved_path, MMPA_MAX_PATH) == EN_OK) {
    res = resolved_path;
  } else {
    GELOGW("[Util][realpath] Get real_path for %s failed, reason:%s", path, strerror(errno));
  }

  return res;
}

/**
 *  @ingroup domi_common
 *  @brief Create directory, support to create multi-level directory
 *  @param [in] directory_path  Path, can be multi-level directory
 *  @return -1 fail
 *  @return 0 success
 */
int32_t CreateDirectory(const std::string &directory_path) {
  GE_CHK_BOOL_EXEC(!directory_path.empty(),
                   REPORT_INNER_ERROR("E19999", "directory path is empty, check invalid");
                   return -1, "[Check][Param] directory path is empty.");
  auto dir_path_len = directory_path.length();
  if (dir_path_len >= MMPA_MAX_PATH) {
    ErrorManager::GetInstance().ATCReportErrMessage("E19002", {"filepath", "size"},
                                                    {directory_path, std::to_string(MMPA_MAX_PATH)});
    GELOGW("[Util][mkdir] Path %s len is too long, it must be less than %d", directory_path.c_str(), MMPA_MAX_PATH);
    return -1;
  }
  char tmp_dir_path[MMPA_MAX_PATH] = {0};
  for (size_t i = 0; i < dir_path_len; i++) {
    tmp_dir_path[i] = directory_path[i];
    if ((tmp_dir_path[i] == '\\') || (tmp_dir_path[i] == '/')) {
      if (mmAccess2(tmp_dir_path, M_F_OK) != EN_OK) {
        int32_t ret = mmMkdir(tmp_dir_path, M_IRUSR | M_IWUSR | M_IXUSR);  // 700
        if (ret != 0) {
          if (errno != EEXIST) {
            REPORT_CALL_ERROR("E19999",
                              "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                              directory_path.c_str(), strerror(errno));
            GELOGW("[Util][mkdir] Create directory %s failed, reason:%s. Make sure the directory exists and writable.",
                   directory_path.c_str(), strerror(errno));
            return ret;
          }
        }
      }
    }
  }
  int32_t ret = mmMkdir(const_cast<char *>(directory_path.c_str()), M_IRUSR | M_IWUSR | M_IXUSR);  // 700
  if (ret != 0) {
    if (errno != EEXIST) {
      REPORT_CALL_ERROR("E19999",
                        "Can not create directory %s. Make sure the directory exists and writable. errmsg:%s",
                        directory_path.c_str(), strerror(errno));
      GELOGW("[Util][mkdir] Create directory %s failed, reason:%s. Make sure the directory exists and writable.",
             directory_path.c_str(), strerror(errno));
      return ret;
    }
  }
  return 0;
}

}
