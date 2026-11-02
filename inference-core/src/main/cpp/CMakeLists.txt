cmake_minimum_required(VERSION 3.22.1)
project(yakuyomi_ncnn)
set(CMAKE_CXX_STANDARD 17)
set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ncnn/lib/cmake/ncnn)
find_package(ncnn REQUIRED)

# 夜讀 C++ 核心（yakuyomi-nightread/native）編進同一個 .so（引擎只有一個 .so：ncnn 是靜態庫，兩個 .so 各帶一份就有兩個 SimpleOMP 常駐池）。
# 產品的重繪還在 Kotlin，C++ 目前只給 sandbox 探針用；關掉＝-Pyakuyomi.nightNative=false。
option(YAKUYOMI_NIGHT_NATIVE "把夜讀 C++ 核心編進 libyakuyomi_ncnn.so" ON)
# 數值自測橋（test_jni.cpp＋案例產生器）只給 sandbox 探針：-Pyakuyomi.nightProbe=true。產品的 .so 不帶。
option(YAKUYOMI_NIGHT_PROBE "再編進夜讀的數值自測橋（只給 sandbox 探針）" OFF)

add_library(yakuyomi_ncnn SHARED ncnn_jni.cpp)
target_link_libraries(yakuyomi_ncnn ncnn log)

# 只匯出 JNI：靜態庫的符號用 --exclude-libs 藏起來，ncnn 標頭的 inline 落在 ncnn_jni.o（不算靜態庫）所以另用 version script 收。
target_link_options(yakuyomi_ncnn PRIVATE -Wl,--exclude-libs,ALL -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
set_target_properties(yakuyomi_ncnn PROPERTIES LINK_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)

if(YAKUYOMI_NIGHT_NATIVE)
  # 往上 4 層到 repo 根；引擎 checkout 與 fork 的巢狀 submodule（yakuyomi-engine/yakuyomi-nightread）都成立
  set(YAKUYOMI_NIGHT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../yakuyomi-nightread/native)
  if(NOT EXISTS ${YAKUYOMI_NIGHT_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "找不到 ${YAKUYOMI_NIGHT_DIR}：先 git submodule update --init，或用 -Pyakuyomi.nightNative=false 關掉夜讀 C++")
  endif()
  set(NR_PROBE_JNI ${YAKUYOMI_NIGHT_PROBE})
  # 逐位元旗標（-O2、-ffp-contract=off…）由夜讀自己的 CMake 釘在它的目標上，不影響 ncnn_jni.cpp
  add_subdirectory(${YAKUYOMI_NIGHT_DIR} nightread_native EXCLUDE_FROM_ALL)
  target_sources(yakuyomi_ncnn PRIVATE $<TARGET_OBJECTS:nightread_jni>)
  target_link_libraries(yakuyomi_ncnn nightread_core)
  if(YAKUYOMI_NIGHT_PROBE)
    target_sources(yakuyomi_ncnn PRIVATE $<TARGET_OBJECTS:nightread_probe_jni>)
    target_link_libraries(yakuyomi_ncnn nightread_testcases)
  endif()
endif()
