/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include <gtest/gtest.h>
#include <iostream>
#define private public
#include "graph/graph.h"
#include "graph/compute_graph.h"
#include "graph/utils/graph_utils.h"
#include "inc/graph/operator_factory_impl.h"
#include "graph/utils/op_desc_utils.h"
#include "graph_builder_utils.h"
#include "register/graph_optimizer/fusion_common/pattern_fusion_base_pass.h"
#include "register/graph_optimizer/graph_fusion/fusion_pattern.h"
#include "register/graph_optimizer/fusion_common/fusion_statistic_recorder.h"

#undef private

using namespace ge;
class UtestFusionStatistics : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() {}
};



TEST_F(UtestFusionStatistics, test_01) {
  auto &fs_instance = fe::FusionStatisticRecorder::Instance();
  fe::FusionInfo fusion_info(0, "", "test_pass");

  fs_instance.UpdateGraphFusionMatchTimes(fusion_info);

  fs_instance.UpdateGraphFusionEffectTimes(fusion_info);

  fs_instance.UpdateBufferFusionMatchTimes(fusion_info);

  string session_graph_id = "0_1";
  std::map<std::string, fe::FusionInfo> graph_fusion_info_map;
  std::map<std::string, fe::FusionInfo> buffer_fusion_info_map;
  fs_instance.GetAndClearFusionInfo(session_graph_id, graph_fusion_info_map,
                                    buffer_fusion_info_map);

  fs_instance.GetFusionInfo(session_graph_id, graph_fusion_info_map,
                            buffer_fusion_info_map);

  fs_instance.ClearFusionInfo(session_graph_id);

  std::vector<string> session_graph_id_vec = {session_graph_id};
  fs_instance.GetAllSessionAndGraphIdList(session_graph_id_vec);
}

TEST_F(UtestFusionStatistics, test_02) {
  auto &fs_instance = fe::FusionStatisticRecorder::Instance();
  fe::FusionInfo fusion_info(0, "", "test_pass");

  fusion_info.AddEffectTimes(1);
  fusion_info.AddMatchTimes(1);
  fusion_info.GetEffectTimes();
  fusion_info.GetGraphId();
  fusion_info.GetMatchTimes();
  fusion_info.GetPassName();
  fusion_info.GetSessionId();
  fusion_info.SetEffectTimes(2);
  fusion_info.SetMatchTimes(2);

  fs_instance.UpdateGraphFusionMatchTimes(fusion_info);

  fs_instance.UpdateGraphFusionEffectTimes(fusion_info);

  fs_instance.UpdateBufferFusionMatchTimes(fusion_info);

  fs_instance.UpdateBufferFusionEffectTimes(fusion_info);
  string session_graph_id = "0_1";
  std::map<std::string, fe::FusionInfo> graph_fusion_info_map;
  std::map<std::string, fe::FusionInfo> buffer_fusion_info_map;
  fs_instance.GetAndClearFusionInfo(session_graph_id, graph_fusion_info_map,
                                    buffer_fusion_info_map);

  fs_instance.GetFusionInfo(session_graph_id, graph_fusion_info_map,
                            buffer_fusion_info_map);

  fs_instance.ClearFusionInfo(session_graph_id);

  std::vector<string> session_graph_id_vec = {session_graph_id};
  fs_instance.GetAllSessionAndGraphIdList(session_graph_id_vec);
}




