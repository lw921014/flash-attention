cmake_minimum_required(VERSION 3.5)

project(aiak_inference)
set(PROJECT_VERSION_MAJOR 1)
set(PROJECT_VERSION_MINOR 1)
set(PROJECT_VERSION_PATCH 0)

# some build option
option(USE_ABI                   "Set -D_GLIBCXX_USE_CXX11_ABI=0 for compile process!"                   OFF)
option(WITH_TRTPLUGIN            "Compile Aiak Inference with TRT plugin and generate aiak trt plugin!"  ON)
option(WITH_TESTING              "Compile Aiak Inference with unit testing!"                             ON)
option(ENABLE_PRINT              "ENABLE_PRINT in kernel function!"                                      OFF)
option(WITH_COVERAGE             "Compile Aiak Inference with code coverage!"                            OFF)
option(GIT_URL                   "Git URL to clone dependent repos!"                                     ${GIT_URL})
option(CMAKE_PREFIX_PATH         "Cmake prefix path!"                                                    ${CMAKE_PREFIX_PATH})

message(STATUS "Aiak Inference version is ${PROJECT_VERSION_MAJOR}."
               "${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}!")
message(STATUS "CXX compiler: ${CMAKE_CXX_COMPILER}, version: "
               "${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C compiler: ${CMAKE_C_COMPILER}, version: "
               "${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}")
if(USE_ABI)
    message(STATUS "Set -D_GLIBCXX_USE_CXX11_ABI=0 for compile process!")
endif()

# some common build config
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(init)
include(flags)
include(generic)
include(third_party)  # download, build, install third_party
include(nvidia_gpu)

# enable cutlass
add_definitions(-DDEBUG_USING_CU) 

if(ENABLE_PRINT)
    add_definitions(-DENABLE_PRINT) 
    message(STATUS "Enable printf in cuda kernel function.")
endif()

include_directories(${TENSORRT_INCLUDE_DIR})
link_directories(${TENSORRT_LIBRARY})
link_directories(${TENSORRT_PLUGIN_LIBRARY})

add_library(trt_rt_lib SHARED IMPORTED GLOBAL)
set_property(TARGET trt_rt_lib PROPERTY IMPORTED_LOCATION ${TENSORRT_LIBRARY})


set(csrc_path ../csrc/flash_attn/)
# include
include_directories(./)
include_directories(${csrc_path})
include_directories(${csrc_path}/src)
include_directories(${csrc_path}/cutlass/include)
include_directories(/opt/conda/lib/python3.8/site-packages/torch/include)
include_directories(/opt/conda/lib/python3.8/site-packages/torch/include/torch/csrc/api/include)
include_directories(/opt/conda/include/python3.8/)

# lib
set(TORCH_LIB /opt/conda/lib/python3.8/site-packages/torch/lib/)

# src

function(fmha_test TARGET_NAME)
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)

    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    set(src_list ${cc_test_SRCS})
    list(APPEND src_list ${csrc_path}/fmha_api.cpp)
    list(APPEND src_list ${csrc_path}/src/fmha_block_dgrad_fp16_kernel_loop.sm80.cu)
    list(APPEND src_list ${csrc_path}/src/fmha_block_fprop_fp16_kernel.sm80.cu)
    list(APPEND src_list ${csrc_path}/src/fmha_dgrad_fp16_kernel_loop.sm80.cu)
    list(APPEND src_list ${csrc_path}/src/fmha_fprop_fp16_kernel.sm80.cu)

    cc_test(${TARGET_NAME}
	    SRCS ${src_list}
	    DEPS ${cc_test_DEPS} device_rt_lib)

    set(COMMON_DEPS ${TORCH_LIB}/libc10.so
                    ${TORCH_LIB}/libtorch.so
                    ${TORCH_LIB}/libtorch_cuda.so
                    ${TORCH_LIB}/libc10_cuda.so
                    ${TORCH_LIB}/libtorch_cpu.so
                    ${TORCH_LIB}/libtorch_global_deps.so
                    ${TORCH_LIB}/libtorch_cuda_linalg.so
                    ${TORCH_LIB}/libcaffe2_nvrtc.so
                    ${TORCH_LIB}/libbackend_with_compiler.so)
    target_link_libraries(${TARGET_NAME} ${COMMON_DEPS})
endfunction(cc_test)

# fmha_test(flash_attn_demo SRCS fwd.cu)
# fmha_test(flash_attn_sanity SRCS sanity_fwd.cu)
fmha_test(sanity_fwd_ele SRCS sanity_fwd_ele.cu)
