# Attention: cmake will append these flags to compile command automatically.
# So if you want to add global option, change this file rather than flags.cmake

# Linux
# DEBUG:  default: "-g"
# RELEASE:  default: "-O3 -DNDEBUG"
# RELWITHDEBINFO: default: "-O2 -g -DNDEBUG"
# MINSIZEREL: default: "-O2 -g -DNDEBUG"

# config GIT_URL with gitlab mirrors to speed up dependent repos clone
# you can change it as https://github.com
find_package(Git REQUIRED)
if(NOT GIT_URL)
    set(GIT_URL "ssh://g@gitlab.baidu.com:8022/inf-gpt3/eccl-thirdparty")
endif()
message(STATUS "Aiak inference use ${GIT_URL} for load third party!")

if(NOT WIN32)
    set(CMAKE_C_FLAGS_DEBUG "-g -O0")
    set(CMAKE_C_FLAGS_RELEASE "-g -O0")
    set(CMAKE_C_FLAGS_RELWITHDEBINFO "-g -O0")
    set(CMAKE_C_FLAGS_MINSIZEREL "-g -O0")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-g -O0")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-g -O0")
    set(CMAKE_CXX_FLAGS_MINSIZEREL "-g -O0")

    if (USE_ABI)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_C_FLAGS_MINSIZEREL "${CMAKE_C_FLAGS_MINSIZEREL} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -D_GLIBCXX_USE_CXX11_ABI=0")
        set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -D_GLIBCXX_USE_CXX11_ABI=0")
        MESSAGE(STATUS "Build in ascend platform: add -D_GLIBCXX_USE_CXX11_ABI=0 in c/cxx flags")
    endif()
endif()
