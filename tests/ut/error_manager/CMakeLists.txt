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

project(ut_error_manager)

set(CMAKE_CXX_STANDARD 11)

# include directories
include_directories(${CMAKE_CURRENT_LIST_DIR})
include_directories(${METADEF_DIR}/error_manager)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos)
include_directories(${CMAKE_BINARY_DIR}/proto/metadef_protos/proto)
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
    "testcase/error_manager_unittest.cc"
)

add_executable(ut_error_manager ${UT_FILES})

target_compile_options(ut_error_manager PRIVATE
    -g --coverage -fprofile-arcs -ftest-coverage
)

target_compile_definitions(ut_error_manager PRIVATE
    google=ascend_private
)

target_link_libraries(ut_error_manager
    $<BUILD_INTERFACE:intf_pub>
    error_manager_static
    gtest gtest_main slog_stub ascend_protobuf c_sec mmpa_stub -lrt -ldl -lgcov
)
