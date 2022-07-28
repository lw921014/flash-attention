function(common_link TARGET_NAME)
  if (WITH_PROFILER)
    target_link_libraries(${TARGET_NAME} gperftools::profiler)
  endif()
endfunction()


function(cc_test_build TARGET_NAME)
  list(APPEND UT_DEPS_LIB gtest_main gtest glog gflags)
  list(APPEND UT_DEPS_LIB glog)

  if(WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_executable(${TARGET_NAME} ${cc_test_SRCS})
    target_link_libraries(${TARGET_NAME} ${cc_test_DEPS} ${UT_DEPS_LIB} z pthread)
    add_dependencies(${TARGET_NAME} ${cc_test_DEPS} ${UT_DEPS_LIB})
  endif()
endfunction()

function(cc_test_run TARGET_NAME)
  if(WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    set(oneValueArgs "")
    set(multiValueArgs COMMAND ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    add_test(NAME ${TARGET_NAME}
	    COMMAND ${cc_test_COMMAND} ${cc_test_ARGS}
            WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  elseif(WITH_TESTING AND NOT TEST ${TARGET_NAME})
    add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_COMMAND} -E echo CI skip ${TARGET_NAME}.)
  endif()
endfunction()

function(cc_test TARGET_NAME)
    # The environment variable `CI_SKIP_CPP_TEST` is used to skip the compilation
    # and execution of test in CI. `CI_SKIP_CPP_TEST` is set to ON when no files
  # other than *.py are modified.
  if(WITH_TESTING AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS ARGS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cc_test_build(${TARGET_NAME}
	    SRCS ${cc_test_SRCS}
	    DEPS ${cc_test_DEPS})
    # we dont test hcom op, because it need complex configuration
    # with more than one machine
    cc_test_run(${TARGET_NAME} COMMAND ${TARGET_NAME} ARGS ${cc_test_ARGS})
  elseif(WITH_TESTING AND NOT TEST ${TARGET_NAME})
    add_test(NAME ${TARGET_NAME} COMMAND ${CMAKE_COMMAND} -E echo CI skip ${TARGET_NAME}.)
  endif()
endfunction(cc_test)

function(trt_plugin_test TARGET_NAME)
  if(WITH_TESTING AND WITH_TRTPLUGIN AND NOT "$ENV{CI_SKIP_CPP_TEST}" STREQUAL "ON")
    set(oneValueArgs "")
    set(multiValueArgs SRCS DEPS)
    cmake_parse_arguments(cc_test "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
    cc_test(${TARGET_NAME} SRCS ${cc_test_SRCS} DEPS ${cc_test_DEPS} trt_sample trt_rt_lib trt_plugin_rt_lib device_rt_lib)
    target_link_libraries(${TARGET_NAME} "-Wl,--no-as-needed"
                                         ${operators_shared_target}
                                         ${trt_plugin_shared_target}
                                         "-Wl,--as-needed")
    if(NOT ${cc_test_DEPS})
      add_dependencies(${TARGET_NAME} ${cc_test_DEPS})
    endif()
    if(NOT ${UT_DEPS_LIB})
      add_dependencies(${TARGET_NAME} ${UT_DEPS_LIB})
    endif()
  endif()
endfunction()

function(cc_library TARGET_NAME)
  set(options STATIC static SHARED shared INTERFACE interface)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(cc_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(WIN32)
      # add libxxx.lib prefix in windows
      set(${TARGET_NAME}_LIB_NAME "${CMAKE_STATIC_LIBRARY_PREFIX}${TARGET_NAME}${CMAKE_STATIC_LIBRARY_SUFFIX}" CACHE STRING "output library name for target ${TARGET_NAME}")
  endif(WIN32)
  if(cc_library_SRCS)
      if(cc_library_SHARED OR cc_library_shared) # build *.so
        add_library(${TARGET_NAME} SHARED ${cc_library_SRCS})
      elseif(cc_library_INTERFACE OR cc_library_interface)
        generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cc_library")
      else()
        add_library(${TARGET_NAME} STATIC ${cc_library_SRCS})
      endif()
    if(cc_library_DEPS)
      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
      common_link(${TARGET_NAME})
    endif()

  else(cc_library_SRCS)
    if(cc_library_DEPS)
      list(REMOVE_DUPLICATES cc_library_DEPS)

      generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:cc_library")

      target_link_libraries(${TARGET_NAME} ${cc_library_DEPS})
    else()
      message(FATAL_ERROR "Please specify source files or libraries in cc_library(${TARGET_NAME} ...).")
    endif()
  endif(cc_library_SRCS)
endfunction(cc_library)

# copied from paddle
# Modification of standard 'protobuf_generate_cpp()' with protobuf-lite support
# Usage:
#   aiak_inference_protobuf_generate_cpp(<proto_srcs> <proto_hdrs> <proto_files>)
function(aiak_inference_protobuf_generate_cpp SRCS HDRS)
  if(NOT ARGN)
    message(SEND_ERROR "Error: aiak_inference_protobuf_generate_cpp() called without any proto files")
    return()
  endif()

  set(${SRCS})
  set(${HDRS})

  foreach(FIL ${ARGN})
    get_filename_component(ABS_FIL ${FIL} ABSOLUTE)
    get_filename_component(FIL_WE ${FIL} NAME_WE)

    set(_protobuf_protoc_src "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.cc")
    set(_protobuf_protoc_hdr "${CMAKE_CURRENT_BINARY_DIR}/${FIL_WE}.pb.h")
    list(APPEND ${SRCS} "${_protobuf_protoc_src}")
    list(APPEND ${HDRS} "${_protobuf_protoc_hdr}")

    add_custom_command(
      OUTPUT "${_protobuf_protoc_src}"
             "${_protobuf_protoc_hdr}"

      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}"
      COMMAND ${PROTOBUF_PROTOC_EXECUTABLE}
      -I${CMAKE_CURRENT_SOURCE_DIR}
      --cpp_out "${CMAKE_CURRENT_BINARY_DIR}" ${ABS_FIL}
      # Set `EXTERN_PROTOBUF_DEPEND` only if need to compile `protoc.exe`.
      DEPENDS ${ABS_FIL} ${EXTERN_PROTOBUF_DEPEND}
      COMMENT "Running C++ protocol buffer compiler on ${FIL}"
      VERBATIM )
  endforeach()

  set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
  set(${SRCS} ${${SRCS}} PARENT_SCOPE)
  set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()


function(proto_library TARGET_NAME)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(proto_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  set(proto_srcs)
  set(proto_hdrs)
  aiak_inference_protobuf_generate_cpp(proto_srcs proto_hdrs ${proto_library_SRCS})
  cc_library(${TARGET_NAME} SRCS ${proto_srcs} DEPS ${proto_library_DEPS} protobuf)
endfunction()

macro(subdirlist result curdir)
  file(GLOB children RELATIVE ${curdir} ${curdir}/*)
  set(dirlist "")
  foreach(child ${children})
    if(IS_DIRECTORY ${curdir}/${child})
      list(APPEND dirlist ${child})
    endif()
  endforeach()
  set(${result} ${dirlist})
endmacro()

# auto add all subdir under ROOT_PATH except those in RMS
macro(add_all_subdir ROOT_PATH)
  subdirlist(SUBDIRS ${ROOT_PATH})
  set(multiValueArgs RMS)
  cmake_parse_arguments(add_all_subdir "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  list(REMOVE_DUPLICATES SUBDIRS)
  if(add_all_subdir_RMS)
    foreach(rm_dir ${add_all_subdir_RMS})
      list(REMOVE_ITEM SUBDIRS ${rm_dir})
      message(STATUS "Auto remove subdir ${rm_dir} from ${SUBDIRS}")
    endforeach()
  endif()
  foreach(subdir ${SUBDIRS})
    add_subdirectory(${subdir})
    message(STATUS "Auto exec add_subdirectory(${subdir}) under ${ROOT_PATH}")
  endforeach()
endmacro()
