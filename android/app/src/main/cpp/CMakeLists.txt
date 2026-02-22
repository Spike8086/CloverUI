cmake_minimum_required(VERSION 3.22.1)
project("clover-bridge")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv8.4-a+dotprod+i8mm+fp16")
# 设置 JNI 库的路径
set(JNI_LIBS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})

# 1. 导入 libggml-base.so (新版基础库)
add_library(ggml-base SHARED IMPORTED)
set_target_properties(ggml-base PROPERTIES IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-base.so)

# 2. 导入 libggml-cpu.so (新版 CPU 库)
add_library(ggml-cpu SHARED IMPORTED)
set_target_properties(ggml-cpu PROPERTIES IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml-cpu.so)

# 3. 导入 libggml.so
add_library(ggml SHARED IMPORTED)
set_target_properties(ggml PROPERTIES IMPORTED_LOCATION ${JNI_LIBS_DIR}/libggml.so)

# 4. 导入 libllama.so
add_library(llama SHARED IMPORTED)
set_target_properties(llama PROPERTIES IMPORTED_LOCATION ${JNI_LIBS_DIR}/libllama.so)

# 5. 编译我们的桥接代码
add_library(clover-bridge SHARED native-lib.cpp)

# 6. 查找 Android Log 库
find_library(log-lib log)

# 7. 链接所有库 (注意顺序：最上层的 llama 放前面，底层的 ggml-base 放后面)
target_link_libraries(clover-bridge
    llama
    ggml
    ggml-cpu
    ggml-base
    ${log-lib})

# 8. 强制 16 KB 对齐，防止 Android 15 闪退
set_target_properties(clover-bridge PROPERTIES LINK_FLAGS "-Wl,-z,max-page-size=16384")
