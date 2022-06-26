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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_PASS_UTIL_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_PASS_UTIL_H_
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/type_utils.h"
#include "register/graph_optimizer/graph_optimize_register_error_codes.h"

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fe {
using NodeTypeMap = std::unordered_map<std::string, std::map<std::string, ge::NodePtr>>;
using NodeTypeMapPtr = std::shared_ptr<NodeTypeMap>;
struct NodeMapInfo {
  int64_t run_count;
  NodeTypeMapPtr node_type_map;
};
using NodeMapInfoPtr = std::shared_ptr<NodeMapInfo>;

/** @brief define graph pass, which provides two interface: 1. run pass;
* 2. record op names before fusion */
class GraphPassUtil {
 public:
  /** set outputdesc attr for data dump
   *
   * @param origin_index,usually is origin node output index
   *
   * @param fusion_index,usually is fusion node output index
   *
   * @param origin_node, usually is origin node
   *
   * @param fusion_node, usually is fusion node
   */
  static void SetOutputDescAttr(const uint32_t &origin_index, const uint32_t &fusion_index,
                                const ge::NodePtr &origin_node, const ge::NodePtr &fusion_node) {
    if (origin_node == nullptr || fusion_node == nullptr) {
      return;
    }

    if (fusion_node->GetOpDesc() == nullptr) {
      return;
    }

    const ge::OpDescPtr origin_op_desc = origin_node->GetOpDesc();
    if (origin_op_desc == nullptr) {
      return;
    }

    auto origin_node_output_desc = origin_node->GetOpDesc()->GetOutputDescPtr(origin_index);
    if (origin_node_output_desc == nullptr) {
      return;
    }

    auto fusion_node_output_desc = fusion_node->GetOpDesc()->MutableOutputDesc(fusion_index);
    if (fusion_node_output_desc == nullptr) {
      return;
    }

    SetOutputDescAttr(origin_node_output_desc, static_cast<int64_t>(origin_index), origin_op_desc,
                      fusion_node_output_desc);
  }

  static void SetOutputDescAttr(ge::ConstGeTensorDescPtr &origin_tensor_desc, const int64_t origin_index,
                                const ge::OpDescPtr &origin_op_desc, const ge::GeTensorDescPtr &target_tensor_desc) {
    if (origin_tensor_desc == nullptr || target_tensor_desc == nullptr || origin_op_desc == nullptr) {
      return;
    }

    // set origin name
    std::string original_name;
    if (!ge::AttrUtils::GetStr(origin_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, original_name) ||
        original_name.empty()) {
      std::vector<std::string> original_names;
      if (ge::AttrUtils::GetListStr(origin_op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names) &&
          !original_names.empty()) {
        original_name = original_names[0];
      } else {
        original_name = origin_op_desc->GetName();
      }
    }
    (void)ge::AttrUtils::SetStr(target_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, original_name);

    // set origin output index
    int64_t origin_output_index = 0;
    if (ge::AttrUtils::GetInt(origin_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index)) {
      (void)ge::AttrUtils::SetInt(target_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
    } else {
      (void)ge::AttrUtils::SetInt(target_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_index);
    }

    // set origin output data type
    const ge::DataType origin_data_type = GetDataDumpOriginDataType(origin_tensor_desc);
    if (origin_data_type != ge::DT_UNDEFINED) {
      SetDataDumpOriginDataType(origin_data_type, target_tensor_desc);
    } else {
      SetDataDumpOriginDataType(origin_tensor_desc->GetOriginDataType(), target_tensor_desc);

    }

    // set origin output format
    const ge::Format origin_format = GetDataDumpOriginFormat(origin_tensor_desc);
    if (origin_format != ge::FORMAT_RESERVED) {
      SetDataDumpOriginFormat(origin_format, target_tensor_desc);
    } else {
      SetDataDumpOriginFormat(origin_tensor_desc->GetOriginFormat(), target_tensor_desc);
    }
  }

  /** get origin format for data dump
   *
   * @param tensor_desc,usually is output_desc
   *
   * @return format of this tensor_desc
   */
  static ge::Format GetDataDumpOriginFormat(const ge::GeTensorDescPtr &tensor_desc) {
    std::string origin_format_str;
    if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format_str)) {
      // Can not get the certificate and it's not set,return directly
      return ge::FORMAT_RESERVED;
    }
    if (origin_format_str == "RESERVED") {
      return ge::FORMAT_RESERVED;
    }
    return ge::TypeUtils::SerialStringToFormat(origin_format_str);
  }

  static ge::Format GetDataDumpOriginFormat(ge::ConstGeTensorDescPtr &tensor_desc) {
    std::string origin_format_str;
    if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format_str)) {
      // Can not get the certificate and it's not set,return directly
      return ge::FORMAT_RESERVED;
    }
    if (origin_format_str == "RESERVED") {
      return ge::FORMAT_RESERVED;
    }
    return ge::TypeUtils::SerialStringToFormat(origin_format_str);
  }

  /** set origin format for data dump
   *
   * @param origin format
   *
   * @param tensor_desc,usually is output_desc
   */
  static void SetDataDumpOriginFormat(const ge::Format &origin_format, const ge::GeTensorDescPtr &tensor_desc) {
    std::string origin_format_str = "RESERVED";
    if (origin_format != ge::FORMAT_RESERVED) {
      origin_format_str = ge::TypeUtils::FormatToSerialString(origin_format);
    }
    (void)ge::AttrUtils::SetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format_str);
  }

  /** set origin datatype for data dump
   *
   * @param origin datatype
   *
   * @param tensor_desc,usually is output_desc
   */
  static void SetDataDumpOriginDataType(const ge::DataType origin_data_type, const ge::GeTensorDescPtr &tensor_desc) {
    std::string origin_data_type_str = "RESERVED";
    if (origin_data_type != ge::DT_UNDEFINED) {
      origin_data_type_str = ge::TypeUtils::DataTypeToSerialString(origin_data_type);
    }
    (void)ge::AttrUtils::SetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_data_type_str);
  }

  /** get origin datatype for data dump
   *
   * @param tensor_desc,usually is output_desc
   *
   * @return format of this tensor_desc
   */
  static ge::DataType GetDataDumpOriginDataType(const ge::GeTensorDescPtr &tensor_desc) {
    std::string origin_data_type_str;
    if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_data_type_str)) {
      return ge::DT_UNDEFINED;
    }
    if (origin_data_type_str == "RESERVED") {
      return ge::DT_UNDEFINED;
    }
    return ge::TypeUtils::SerialStringToDataType(origin_data_type_str);
  }

  static ge::DataType GetDataDumpOriginDataType(ge::ConstGeTensorDescPtr &tensor_desc) {
    std::string origin_data_type_str;
    if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_data_type_str)) {
      return ge::DT_UNDEFINED;
    }
    if (origin_data_type_str == "RESERVED") {
      return ge::DT_UNDEFINED;
    }
    return ge::TypeUtils::SerialStringToDataType(origin_data_type_str);
  }

  static void AddNodeFromOpTypeMap(const NodeMapInfoPtr &node_map_info, const ge::NodePtr &node_ptr) {
    if ((node_map_info == nullptr) || (node_ptr == nullptr)) {
      return;
    }
    NodeTypeMapPtr node_type_map = node_map_info->node_type_map;
    std::string real_op_type = ge::NodeUtils::GetNodeType(*node_ptr);
    const auto iter = node_type_map->find(real_op_type);
    if (iter != node_type_map->end()) {
      iter->second[node_ptr->GetName()] = node_ptr;
    } else {
      (void)node_type_map->emplace(std::make_pair(real_op_type,
          std::map<std::string, ge::NodePtr>{{node_ptr->GetName(), node_ptr}}));
    }
  }

  static Status GetOpTypeMapToGraph(NodeMapInfoPtr &node_map_info, const ge::ComputeGraph &graph) {
    node_map_info = graph.TryGetExtAttr("NodeMapInfo", node_map_info);
    if (node_map_info == nullptr) {
      return FAILED;
    }
    return SUCCESS;
  }

  static void RecordOriginalNames(const std::vector<ge::NodePtr> &original_nodes, const ge::NodePtr &node) {
    // 1. get the original_names
    std::vector<std::string> original_names;
    for (const ge::NodePtr &original_node : original_nodes) {
      if ((original_node == nullptr) || (original_node->GetOpDesc() == nullptr)) {
        return;
      }

      const ge::OpDescPtr origin_op_desc_ptr = original_node->GetOpDesc();
      std::vector<std::string> names_tmp;
      const bool is_has_attr =
              ge::AttrUtils::GetListStr(origin_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp) &&
              !names_tmp.empty();
      if (is_has_attr) {
        for (const auto &node_name : names_tmp) {
          if (!node_name.empty()) {
            original_names.push_back(node_name);
          }
        }
      } else {
        original_names.push_back(origin_op_desc_ptr->GetName());
      }
    }

    // 2. set the dump attr
    if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
      return;
    }
    ge::OpDescPtr node_op_desc_ptr = node->GetOpDesc();
    (void)ge::AttrUtils::SetListStr(node_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
  }

  static void AddNodeToNodeTypeMap(const NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                   const ge::NodePtr &node_ptr) {
    if ((node_type_map == nullptr) || (node_ptr == nullptr)) {
      return;
    }
    const auto iter = node_type_map->find(op_type);
    if (iter == node_type_map->end()) {
      (void)node_type_map->emplace(std::make_pair(op_type,
          std::map<std::string, ge::NodePtr>{{node_ptr->GetName(), node_ptr}}));
    } else {
      (void)iter->second.emplace(node_ptr->GetName(), node_ptr);
    }
  }

  static void RemoveNodeFromNodeTypeMap(NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                        const ge::NodePtr &node_ptr) {
    if ((node_type_map == nullptr) || (node_ptr == nullptr)) {
      return;
    }
    const auto iter = node_type_map->find(op_type);
    if (iter != node_type_map->end()) {
      (void)iter->second.erase(node_ptr->GetName());
    }
  }

  static void GetNodesFromNodeTypeMap(NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                      std::vector<ge::NodePtr> &nodes) {
    if (node_type_map == nullptr) {
      return;
    }

    const auto iter = node_type_map->find(op_type);
    if (iter == node_type_map->end()) {
      return;
    }
    if (iter->second.empty()) {
      return;
    }

    for (auto node_iter = iter->second.cbegin(); node_iter != iter->second.cend(); node_iter++) {
      nodes.push_back(node_iter->second);
    }
  }
};

}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_GRAPH_PASS_UTIL_H_
