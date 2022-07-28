#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <sys/time.h>
#include <fstream>
#include <string>
#include <thread>

#define CUDA_CHECK(cmd)                               \
    do {                                              \
        cudaError_t e = cmd;                          \
        if (e != cudaSuccess) {                       \
            printf("Failed: Cuda error %s:%d '%s'\n", \
                   __FILE__,                          \
                   __LINE__,                          \
                   cudaGetErrorString(e));            \
            exit(EXIT_FAILURE);                       \
        }                                             \
    } while (0)
    
inline int load_int_from_env(const std::string& env_var, int default_var) {
    char* str = getenv(env_var.c_str());
    if (str) {
        VLOG(3) << "Get [" << env_var << "] from env as " << atoi(str) << "!"
                << std::endl;
        return atoi(str);
    }
    VLOG(3) << "Can't get [" << env_var << "] from env, use default "
            << default_var << "!" << std::endl;
    return default_var;
}

template <typename T, bool same = false>
inline void gen_random_data(T* data, size_t len, T same_value = 1.0) {
    uint32_t seed = unsigned(time(nullptr));
    for (size_t i = 0; i < len; i++) {
        if (std::is_same<T, __half>::value) {
            if (same) {
                data[i] = __float2half(same_value);
            } else {
                data[i] = __float2half(static_cast<float>(
                    rand_r(&seed) / static_cast<float>(RAND_MAX)));
            }
        } else {
            if (same) {
                data[i] = static_cast<T>(same_value);
            } else {
                data[i] = static_cast<T>(rand_r(&seed) /
                                         static_cast<float>(RAND_MAX));
            }
        }
    }
}

template <typename T, bool isHost>
inline int dumpMem(void* ptr,
                   size_t count,
                   std::ostream& os,
                   size_t lda = 20,
                   const std::string& name = "test") {
    os << "name = " << name << ", count = " << count << ", lda = " << lda
       << ", position = " << (isHost ? "HOST" : "DEVICE")
       << ", type = " << (std::is_same<T, __half>::value ? "float16" : "other")
       << std::endl;
    T* cpu_mem = nullptr;
    if (isHost) {
        cpu_mem = static_cast<T*>(ptr);
    } else {
        cpu_mem = new T[count];
        CHECK(cpu_mem != nullptr);
        // NOTE : cat not use assert, because when use assert(function()),
        // function() will not execute
        CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(cpu_mem),
                              reinterpret_cast<void*>(ptr),
                              sizeof(T) * count,
                              cudaMemcpyDeviceToHost));
    }

    for (size_t i = 0; i < count; i++) {
        if (i % lda == 0) {
            os << std::endl;
        }
        os << "[" << i << " : ";
        if (std::is_same<T, __half>::value) {
            os << __half2float(cpu_mem[i]);
        } else if (std::is_same<T, float>::value) {
            os << float(cpu_mem[i]);
        } else {
            LOG(FATAL) << "Only support dump fp16 and float data.";
        }
        os << "], ";
    }

    if (!isHost) {
        delete[] cpu_mem;
    }

    os << std::endl << std::endl << std::endl;
    return 0;
}

template <typename T, bool isHost>
inline int dumpMemInfoFile(void* ptr,
                           size_t count,
                           const std::string& fileName,
                           size_t lda = 20) {
    std::ofstream file(fileName.c_str());

    int ret = dumpMem<T, isHost>(ptr, count, file, lda);
    VLOG(3) << "Dump " << (isHost ? "HOST" : "DEVICE") << " data in ["
            << fileName << (ret == 0 ? "] OK!" : "] failed!") << std::endl;
    return ret;
}

// used for GLOG_v > 3 to help check data
inline std::string get_dump_file_name(
    const std::string prefix = "data_b2b_gemm_process") {
    std::stringstream dump_data_file_name;
    dump_data_file_name << prefix << "_in_threadid_"
                        << std::this_thread::get_id() << ".data";
    return dump_data_file_name.str();
}
