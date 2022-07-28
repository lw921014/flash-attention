include(ExternalProject)
# Creat a target named "third_party", which can compile external dependencies on all platform(windows/linux/mac)

set(THIRD_PARTY_PATH  "${CMAKE_BINARY_DIR}/third_party" CACHE STRING
    "A path setting third party libraries download & build directories.")
set(THIRD_PARTY_CACHE_PATH     "${CMAKE_SOURCE_DIR}"    CACHE STRING
    "A path cache third party source code to avoid repeated download.")

set(THIRD_PARTY_BUILD_TYPE Release)
set(third_party_deps)

# cache funciton to avoid repeat download code of third_party.
# This function has 4 parameters, URL / REPOSITOR / TAG / DIR:
# 1. URL:           specify download url of 3rd party
# 2. REPOSITORY:    specify git REPOSITORY of 3rd party
# 3. TAG:           specify git tag/branch/commitID of 3rd party
# 4. DIR:           overwrite the original SOURCE_DIR when cache directory
#
# The function Return 1 PARENT_SCOPE variables:
#  - ${TARGET}_DOWNLOAD_CMD: Simply place "${TARGET}_DOWNLOAD_CMD" in ExternalProject_Add,
#                            and you no longer need to set any donwnload steps in ExternalProject_Add.
# For example:
#    Cache_third_party(${TARGET}
#            REPOSITORY ${TARGET_REPOSITORY}
#            TAG        ${TARGET_TAG}
#            DIR        ${TARGET_SOURCE_DIR})

FUNCTION(cache_third_party TARGET)
    SET(options "")
    SET(oneValueArgs URL REPOSITORY TAG DIR)
    SET(multiValueArgs "")
    cmake_parse_arguments(cache_third_party "${optionps}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

    STRING(REPLACE "extern_" "" TARGET_NAME ${TARGET})
    STRING(REGEX REPLACE "[0-9]+" "" TARGET_NAME ${TARGET_NAME})
    STRING(TOUPPER ${TARGET_NAME} TARGET_NAME)
    IF(cache_third_party_REPOSITORY)
        SET(${TARGET_NAME}_DOWNLOAD_CMD
                GIT_REPOSITORY  ${cache_third_party_REPOSITORY})
        IF(cache_third_party_TAG)
            LIST(APPEND   ${TARGET_NAME}_DOWNLOAD_CMD
                    GIT_TAG     ${cache_third_party_TAG})
        ENDIF()
    ELSEIF(cache_third_party_URL)
        SET(${TARGET_NAME}_DOWNLOAD_CMD
                URL             ${cache_third_party_URL})
    ELSE()
        MESSAGE(FATAL_ERROR    "Download link (Git repo or URL) must be specified for cache!")
    ENDIF()
    IF(WITH_TP_CACHE)
        IF(NOT cache_third_party_DIR)
            MESSAGE(FATAL_ERROR   "Please input the ${TARGET_NAME}_SOURCE_DIR for overwriting when -DWITH_TP_CACHE=ON")
        ENDIF()
        # Generate and verify cache dir for third_party source code
        SET(cache_third_party_REPOSITORY ${cache_third_party_REPOSITORY} ${cache_third_party_URL})
        IF(cache_third_party_REPOSITORY AND cache_third_party_TAG)
            STRING(MD5 HASH_REPO ${cache_third_party_REPOSITORY})
            STRING(MD5 HASH_GIT ${cache_third_party_TAG})
            STRING(SUBSTRING ${HASH_REPO} 0 8 HASH_REPO)
            STRING(SUBSTRING ${HASH_GIT} 0 8 HASH_GIT)
            STRING(CONCAT HASH ${HASH_REPO} ${HASH_GIT})
            # overwrite the original SOURCE_DIR when cache directory
            SET(${cache_third_party_DIR} ${THIRD_PARTY_CACHE_PATH}/third_party/${TARGET}_${HASH})
        ELSEIF(cache_third_party_REPOSITORY)
            STRING(MD5 HASH_REPO ${cache_third_party_REPOSITORY})
            STRING(SUBSTRING ${HASH_REPO} 0 16 HASH)
            # overwrite the original SOURCE_DIR when cache directory
            SET(${cache_third_party_DIR} ${THIRD_PARTY_CACHE_PATH}/third_party/${TARGET}_${HASH})
        ENDIF()

        IF(EXISTS ${${cache_third_party_DIR}})
            # judge whether the cache dir is empty
            FILE(GLOB files ${${cache_third_party_DIR}}/*)
            LIST(LENGTH files files_len)
            IF(files_len GREATER 0)
                list(APPEND ${TARGET_NAME}_DOWNLOAD_CMD DOWNLOAD_COMMAND "")
            ENDIF()
        ENDIF()
        SET(${cache_third_party_DIR} ${${cache_third_party_DIR}} PARENT_SCOPE)
    ENDIF()

    # Pass ${TARGET_NAME}_DOWNLOAD_CMD to parent scope, the double quotation marks can't be removed
    SET(${TARGET_NAME}_DOWNLOAD_CMD "${${TARGET_NAME}_DOWNLOAD_CMD}" PARENT_SCOPE)
ENDFUNCTION()

MACRO(UNSET_VAR VAR_NAME)
    UNSET(${VAR_NAME} CACHE)
    UNSET(${VAR_NAME})
ENDMACRO()

# Funciton to Download the dependencies during compilation
# This function has 2 parameters, URL / DIRNAME:
# 1. URL:           The download url of 3rd dependencies
# 2. NAME:          The name of file, that determin the dirname
#
FUNCTION(file_download_and_uncompress URL NAME)
  MESSAGE(STATUS "Download dependence[${NAME}] from ${URL}")
  SET(${NAME}_INCLUDE_DIR ${THIRD_PARTY_PATH}/${NAME}/data PARENT_SCOPE)
  ExternalProject_Add(
      extern_download_${NAME}
      ${EXTERNAL_PROJECT_LOG_ARGS}
      PREFIX                ${THIRD_PARTY_PATH}/${NAME}
      URL                   ${URL}
      DOWNLOAD_DIR          ${THIRD_PARTY_PATH}/${NAME}/data/
      SOURCE_DIR            ${THIRD_PARTY_PATH}/${NAME}/data/
      DOWNLOAD_NO_PROGRESS  1
      CONFIGURE_COMMAND     ""
      BUILD_COMMAND         ""
      UPDATE_COMMAND        ""
      INSTALL_COMMAND       ""
    )
  set(third_party_deps ${third_party_deps} extern_download_${NAME} PARENT_SCOPE)
ENDFUNCTION()


########################### include third_party according to flags ###############################
include(external/zlib)      # download, build, install zlib
include(external/gflags)    # download, build, install gflags
include(external/glog)      # download, build, install glog
# include(external/xxhash)    # download, build, install xxhash
# include(external/cutlass)    # download, build, install cutlass

# list(APPEND third_party_deps extern_protobuf extern_gflags extern_glog)
list(APPEND third_party_deps extern_gflags extern_glog)

# include(external/snappy)
# list(APPEND third_party_deps extern_snappy)

# include(external/leveldb)
# list(APPEND third_party_deps extern_leveldb)

# include(external/brpc)
# list(APPEND third_party_deps extern_brpc)

IF(WITH_TESTING)
    include(external/gtest)     # download, build, install gtest
    list(APPEND third_party_deps extern_gtest)
ENDIF()

add_custom_target(third_party ALL DEPENDS ${third_party_deps})
