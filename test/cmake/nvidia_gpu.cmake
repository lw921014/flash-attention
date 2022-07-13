# set cuda flags
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "-O2 -g -DNDEBUG")
set(CMAKE_CUDA_FLAGS_MINSIZEREL "-O1 -DNDEBUG")

function(nv_library TARGET_NAME)
  set(options STATIC static SHARED shared)
  set(oneValueArgs "")
  set(multiValueArgs SRCS DEPS)
  cmake_parse_arguments(nv_library "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})
  if(nv_library_SRCS)
    # Attention:
    # 1. cuda_add_library is deprecated after cmake v3.10, use add_library for CUDA please.
    # 2. cuda_add_library does not support ccache.
    # Reference: https://cmake.org/cmake/help/v3.10/module/FindCUDA.html
    if (nv_library_SHARED OR nv_library_shared) # build *.so
      add_library(${TARGET_NAME} SHARED ${nv_library_SRCS})
    else()
      add_library(${TARGET_NAME} STATIC ${nv_library_SRCS})
    endif()
    if (nv_library_DEPS)
      add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
      target_link_libraries(${TARGET_NAME} ${nv_library_DEPS})
    endif()
    # cpplint code style
    foreach(source_file ${nv_library_SRCS})
      string(REGEX REPLACE "\\.[^.]*$" "" source ${source_file})
      if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
        list(APPEND nv_library_HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/${source}.h)
      endif()
    endforeach()
  else(nv_library_SRCS)
    if (nv_library_DEPS)
      list(REMOVE_DUPLICATES nv_library_DEPS)
      generate_dummy_static_lib(LIB_NAME ${TARGET_NAME} FILE_PATH ${target_SRCS} GENERATOR "generic.cmake:nv_library")

      target_link_libraries(${TARGET_NAME} ${nv_library_DEPS})
      add_dependencies(${TARGET_NAME} ${nv_library_DEPS})
    else()
      message(FATAL "Please specify source file or library in nv_library.")
    endif()
  endif(nv_library_SRCS)
  if((CUDA_VERSION GREATER 9.2) AND (CUDA_VERSION LESS 11.0) AND (MSVC_VERSION LESS 1910))
    set_target_properties(${TARGET_NAME} PROPERTIES VS_USER_PROPS ${WIN_PROPS})
  endif()
endfunction(nv_library)

# cuda 
if (DEFINED ENV{ENV_CUDA_HOME})
  set(CUDA_TOOLKIT_ROOT_DIR $ENV{ENV_CUDA_HOME})
endif()
find_package(CUDA REQUIRED cudart)
enable_language(CUDA)
link_directories(${CUDA_CUDART_LIBRARY})
include_directories(${CUDA_INCLUDE_DIRS})
add_library(device_rt_lib STATIC IMPORTED GLOBAL)
set_property(TARGET device_rt_lib PROPERTY IMPORTED_LOCATION ${CUDA_CUDART_LIBRARY})

message(STATUS "Found CUDA (include: ${CUDA_INCLUDE_DIRS}, library: ${CUDA_CUDART_LIBRARY})")

# nvcc
set(CMAKE_CUDA_FLAGS)
include(cuda)

# nvtx
if (WITH_NVTX)
    find_package(NVTX QUIET)
    if (NVTX_FOUND)
        include_directories(${NVTX_PATH})
        add_definitions(-DWITH_NVTX)
        message(STATUS "Use NV gpu and start nvtx!")
    else()
        message(FATAL_ERROR "Use NV gpu and enable nvtx but dont find nvtx lib!")
    endif()
endif()
