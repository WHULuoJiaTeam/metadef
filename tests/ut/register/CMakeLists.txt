# Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
# Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

set(REGISTER_PROTO_LIST
    "${METADEF_DIR}/proto/tensorflow/attr_value.proto"
    "${METADEF_DIR}/proto/tensorflow/function.proto"
    "${METADEF_DIR}/proto/tensorflow/graph.proto"
    "${METADEF_DIR}/proto/tensorflow/graph_library.proto"
    "${METADEF_DIR}/proto/tensorflow/node_def.proto"
    "${METADEF_DIR}/proto/tensorflow/op_def.proto"
    "${METADEF_DIR}/proto/tensorflow/resource_handle.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor.proto"
    "${METADEF_DIR}/proto/tensorflow/tensor_shape.proto"
    "${METADEF_DIR}/proto/tensorflow/types.proto"
    "${METADEF_DIR}/proto/tensorflow/versions.proto"
)


protobuf_generate(ge REGISTER_PROTO_SRCS REGISTER_PROTO_HDRS ${REGISTER_PROTO_LIST})

############ libut_register_proto.a ############
add_library(ut_register_proto STATIC
    ${REGISTER_PROTO_HDRS}
    ${REGISTER_PROTO_SRCS}
)

target_compile_definitions(ut_register_proto PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ut_register_proto PRIVATE
    -O2
    -fno-common
)

target_link_libraries(ut_register_proto PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
)


set(REGISTER_SRC_FILES
    "${METADEF_DIR}/register/auto_mapping_util.cpp"
    "${METADEF_DIR}/register/graph_optimizer/buffer_fusion/buffer_fusion_pass_base.cc"
    "${METADEF_DIR}/register/graph_optimizer/buffer_fusion/buffer_fusion_pass_registry.cc"
    "${METADEF_DIR}/register/graph_optimizer/buffer_fusion/buffer_fusion_pattern.cc"
    "${METADEF_DIR}/register/graph_optimizer/fusion_statistic/fusion_statistic_recorder.cc"
    "${METADEF_DIR}/register/graph_optimizer/graph_fusion/fusion_pass_registry.cc"
    "${METADEF_DIR}/register/graph_optimizer/graph_fusion/fusion_pattern.cc"
    "${METADEF_DIR}/register/graph_optimizer/graph_fusion/graph_fusion_pass_base.cc"
    "${METADEF_DIR}/register/graph_optimizer/graph_fusion/pattern_fusion_base_pass.cc"
    "${METADEF_DIR}/register/graph_optimizer/graph_fusion/pattern_fusion_base_pass_impl.cc"
    "${METADEF_DIR}/register/host_cpu_context.cc"
    "${METADEF_DIR}/register/infer_data_slice_registry.cc"
    "${METADEF_DIR}/register/ops_kernel_builder_registry.cc"
    "${METADEF_DIR}/register/op_kernel_registry.cpp"
    "${METADEF_DIR}/register/op_tiling.cpp"
    "${METADEF_DIR}/register/op_tiling_registry.cpp"
    "${METADEF_DIR}/register/register.cpp"
    "${METADEF_DIR}/register/register_format_transfer.cc"
    "${METADEF_DIR}/register/register_pass.cpp"
    "${METADEF_DIR}/register/scope/scope_graph.cc"
    "${METADEF_DIR}/register/scope/scope_pass.cc"
    "${METADEF_DIR}/register/scope/scope_pass_registry.cc"
    "${METADEF_DIR}/register/scope/scope_pattern.cc"
    "${METADEF_DIR}/register/scope/scope_util.cc"
    "${METADEF_DIR}/register/tensor_assign.cpp"
    "${METADEF_DIR}/register/prototype_pass_registry.cc"
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${METADEF_DIR})
include_directories(${METADEF_DIR}/graph)
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/inc/external)
include_directories(${METADEF_DIR}/inc/register)
include_directories(${METADEF_DIR}/third_party/fwkacllib/inc)
include_directories(${METADEF_DIR}/third_party/graphengine/inc)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/external)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/framework)


set(REGISTER_UT_FILES
    "testcase/register_unittest.cc"
    "testcase/register_prototype_unittest.cc"
)

############ libut_metadef_register.a ############
add_library(ut_metadef_register STATIC
    ${REGISTER_SRC_FILES} ${REGISTER_PROTO_HDRS}
)

target_compile_definitions(ut_metadef_register PRIVATE
    google=ascend_private
)

target_compile_options(ut_metadef_register PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ut_metadef_register PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    json
)


add_executable(ut_register
    ${REGISTER_UT_FILES} ${REGISTER_PROTO_HDRS}
)

target_compile_options(ut_register PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_register PRIVATE
    google=ascend_private
)

target_link_libraries(ut_register
    $<BUILD_INTERFACE:intf_pub>
    ut_metadef_register ut_register_proto ut_metadef_graph ut_metadef_proto
    gtest
    gtest_main
    slog_stub
    ascend_protobuf
    c_sec
    error_manager_stub
    mmpa_stub
    -lrt
    -ldl
    -lgcov
)
