# Copyright 2019-2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

project(ut_register)

set(CMAKE_CXX_STANDARD 11)
# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
include_directories(${METADEF_DIR})
include_directories(${METADEF_DIR}/register)


include_directories(${METADEF_DIR}/third_party/fwkacllib/inc)
include_directories(${METADEF_DIR}/third_party/graphengine/inc)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/external)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/framework)

set(REGISTER_UT_FILES
    "testcase/register_unittest.cc"
    "testcase/register_prototype_unittest.cc"
    "testcase/register_scope_graph_unittest.cc"
    "testcase/register_scope_pass_unittest.cc"
    "testcase/register_scope_pass_registry_unittest.cc"
    "testcase/infer_data_slice_registry_unittest.cc"
    "testcase/infer_axis_slice_registry_unittest.cc"
    "testcase/register_op_tiling_v4_ut.cc"
    "testcase/register_op_tiling_v3_ut.cc"
    "testcase/register_op_tiling_v2_ut.cc"
    "testcase/register_op_tiling_v1_ut.cc"
    "testcase/register_optiling_unittest.cc"
    "testcase/op_tiling_attr_utils_test.cc"
    "graph_builder_utils.cc"
    "testcase/graph_fusion_clycle_detection_unittest.cc"
    "testcase/auto_mapping_util_unittest.cc"
    "testcase/tensor_assign_unittest.cc"
    "testcase/scope_pattern_unittest.cc"
    "testcase/scope_util_unittest.cc"
    "testcase/register_pass_unittest.cc"
    "testcase/ops_kernel_builder_registry_unittest.cc"
    "testcase/op_kernel_registry_unittest.cc"
    "testcase/fusion_statistics.cc"
    "testcase/register_graph_fusion.cc"
    "testcase/register_graph_fusion_2.cc"
    "testcase/register_graph_fusion_3.cc"
    "testcase/register_buffer_fusion.cc"
    "testcase/graph_pass_util_ut.cc"
    "testcase/reg_op_unittest.cc"
)
add_executable(ut_register
    ${REGISTER_UT_FILES}
)

target_compile_options(ut_register PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_register PRIVATE
    google=ascend_private
    $<$<STREQUAL:${ENABLE_OPEN_SRC},True>:ONLY_COMPILE_OPEN_SRC>
)

target_link_libraries(ut_register
    $<BUILD_INTERFACE:intf_pub>
    register_static graph_static
    gtest
    gtest_main
    slog_stub
    ascend_protobuf
    json
    c_sec
    error_manager_static
    mmpa_stub
    -lrt
    -ldl
    -lgcov
)
