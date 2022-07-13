# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

include(ExternalProject)

set(CUTLASS_PREFIX_DIR ${THIRD_PARTY_PATH}/cutlass)
set(CUTLASS_SOURCE_DIR ${THIRD_PARTY_PATH}/cutlass/src/extern_cutlass)
set(CUTLASS_REPOSITORY ${GIT_URL}/NVIDIA/cutlass.git)
# set(CUTLASS_TAG        v2.8.0)
# NOTE: need to checkout master branch to enable b2b
set(CUTLASS_TAG        master)

cache_third_party(extern_cutlass
    REPOSITORY    ${CUTLASS_REPOSITORY}
    TAG           ${CUTLASS_TAG}
    DIR           CUTLASS_SOURCE_DIR)


set(CUTLASS_INCLUDE_DIR ${CUTLASS_SOURCE_DIR}/include 
                        ${CUTLASS_SOURCE_DIR}/tools/util/include
                        ${CUTLASS_SOURCE_DIR}/examples/13_two_tensor_op_fusion)
                        
include_directories(${CUTLASS_INCLUDE_DIR})

ExternalProject_Add(
  extern_cutlass
  ${EXTERNAL_PROJECT_LOG_ARGS}
  ${SHALLOW_CLONE}
  "${CUTLASS_DOWNLOAD_CMD}"
  PREFIX          ${CUTLASS_PREFIX_DIR}
  SOURCE_DIR      ${CUTLASS_SOURCE_DIR}
  UPDATE_COMMAND ""
  CONFIGURE_COMMAND ""
  BUILD_COMMAND     ""
  INSTALL_COMMAND   ""
  TEST_COMMAND      ""
)

add_library(cutlass INTERFACE)

add_dependencies(cutlass extern_cutlass)
