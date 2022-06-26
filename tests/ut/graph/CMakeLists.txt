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

project(ut_graph)

set(CMAKE_CXX_STANDARD 11)

set(PROTO_LIST
    "${METADEF_DIR}/proto/om.proto"
    "${METADEF_DIR}/proto/ge_ir.proto"
    "${METADEF_DIR}/proto/insert_op.proto"
    "${METADEF_DIR}/proto/task.proto"
    "${METADEF_DIR}/proto/dump_task.proto"
    "${METADEF_DIR}/proto/fwk_adapter.proto"
    "${METADEF_DIR}/proto/op_mapping.proto"
    "${METADEF_DIR}/proto/onnx/ge_onnx.proto"
)

protobuf_generate(ge PROTO_SRCS PROTO_HDRS ${PROTO_LIST})

############ libut_metadef_proto.a ############
add_library(ut_metadef_proto STATIC
    ${PROTO_HDRS}
    ${PROTO_SRCS}
)

target_compile_definitions(ut_metadef_proto PRIVATE
    PROTOBUF_INLINE_NOT_IN_HEADERS=0
    google=ascend_private
)

target_compile_options(ut_metadef_proto PRIVATE
    -O2
    -fno-common
)

target_link_libraries(ut_metadef_proto PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    ascend_protobuf
)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/inc)
include_directories(${METADEF_DIR}/inc/graph)
include_directories(${METADEF_DIR}/inc/external)
include_directories(${METADEF_DIR}/inc/external/graph)
include_directories(${METADEF_DIR}/graph)
include_directories(${METADEF_DIR}/third_party)
include_directories(${METADEF_DIR}/third_party/graphengine/inc)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/external)
include_directories(${METADEF_DIR}/third_party/graphengine/inc/external/ge)
include_directories(${METADEF_DIR}/third_party/fwkacllib/inc)
include_directories(${METADEF_DIR}/third_party/transformer/inc)
include_directories(${METADEF_DIR}/)
include_directories(${CMAKE_BINARY_DIR})
include_directories(${CMAKE_BINARY_DIR}/proto/ge)
include_directories(${CMAKE_BINARY_DIR}/proto/ge/proto)

set(UT_FILES
    "testcase/graph_unittest.cc"
    "testcase/types_unittest.cc"
    "testcase/tensor_unittest.cc"
    "testcase/type_utils_unittest.cc"
    "testcase/aligned_ptr_unittest.cc"
    "testcase/tensor_ut.cc"
    "testcase/node_utils_unittest.cc"
    "testcase/node_unittest.cc"
    "testcase/op_desc_utils_unittest.cc"
    "testcase/graph_builder_utils.cc"
    "testcase/operator_unittest.cc"
    "testcase/buffer_unittest.cc"
    "testcase/ge_attr_value_unittest.cc"
    "testcase/shape_refiner_unittest.cc"
    "testcase/graph_utils_unittest.cc"
    "testcase/runtime_inference_context_unittest.cc"
    "testcase/op_desc_unittest.cc"
)

set(GRAPH_SRC_FILES
    "${METADEF_DIR}/graph/aligned_ptr.cc"
    "${METADEF_DIR}/graph/anchor.cc"
    "${METADEF_DIR}/graph/ascend_string.cc"
    "${METADEF_DIR}/graph/attr_value.cc"
    "${METADEF_DIR}/graph/buffer.cc"
    "${METADEF_DIR}/graph/compute_graph.cc"
    "${METADEF_DIR}/graph/debug/graph_debug.cc"
    "${METADEF_DIR}/graph/detail/attributes_holder.cc"
    "${METADEF_DIR}/graph/format_refiner.cc"
    "${METADEF_DIR}/graph/ge_attr_define.cc"
    "${METADEF_DIR}/graph/ge_attr_value.cc"
    "${METADEF_DIR}/graph/ge_tensor.cc"
    "${METADEF_DIR}/graph/gnode.cc"
    "${METADEF_DIR}/graph/graph.cc"
    "${METADEF_DIR}/graph/inference_context.cc"
    "${METADEF_DIR}/graph/model.cc"
    "${METADEF_DIR}/graph/model_serialize.cc"
    "${METADEF_DIR}/graph/node.cc"
    "${METADEF_DIR}/graph/op_desc.cc"
    "${METADEF_DIR}/graph/operator.cc"
    "${METADEF_DIR}/graph/operator_factory.cc"
    "${METADEF_DIR}/graph/operator_factory_impl.cc"
    "${METADEF_DIR}/graph/opsproto/opsproto_manager.cc"
    "${METADEF_DIR}/graph/option/ge_context.cc"
    "${METADEF_DIR}/graph/option/ge_local_context.cc"
    "${METADEF_DIR}/graph/ref_relation.cc"
    "${METADEF_DIR}/graph/runtime_inference_context.cc"
    "${METADEF_DIR}/graph/shape_refiner.cc"
    "${METADEF_DIR}/graph/tensor.cc"
    "${METADEF_DIR}/graph/types.cc"
    "${METADEF_DIR}/graph/utils/anchor_utils.cc"
    "${METADEF_DIR}/graph/utils/ge_ir_utils.cc"
    "${METADEF_DIR}/graph/utils/graph_utils.cc"
    "${METADEF_DIR}/graph/utils/dumper/ge_graph_dumper.cc"
    "${METADEF_DIR}/graph/utils/node_utils.cc"
    "${METADEF_DIR}/graph/utils/op_desc_utils.cc"
    "${METADEF_DIR}/graph/utils/tensor_utils.cc"
    "${METADEF_DIR}/graph/utils/transformer_utils.cc"
    "${METADEF_DIR}/graph/utils/tuning_utils.cc"
    "${METADEF_DIR}/graph/utils/type_utils.cc"
    "${METADEF_DIR}/ops/op_imp.cpp"
    "${METADEF_DIR}/third_party/transformer/src/axis_util.cc"
    "${METADEF_DIR}/third_party/transformer/src/expand_dimension.cc"
    "${METADEF_DIR}/third_party/transformer/src/transfer_shape_according_to_format.cc"
)

############ libut_metadef_graph.a ############
add_library(ut_metadef_graph STATIC
    ${GRAPH_SRC_FILES} ${PROTO_HDRS}
)

target_compile_definitions(ut_metadef_graph PRIVATE
    google=ascend_private
)

target_compile_options(ut_metadef_graph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
    -Werror=format
)

target_link_libraries(ut_metadef_graph PRIVATE
    $<BUILD_INTERFACE:intf_pub>
    c_sec
    ascend_protobuf
    #json
)


add_executable(ut_graph ${UT_FILES} ${PROTO_HDRS})

target_compile_options(ut_graph PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_graph PRIVATE
    google=ascend_private
)

target_link_libraries(ut_graph
    $<BUILD_INTERFACE:intf_pub>
    ut_metadef_graph ut_metadef_proto
    gtest gtest_main slog_stub ascend_protobuf c_sec error_manager_stub mmpa_stub -lrt -ldl -lgcov
)
