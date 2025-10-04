cmake_minimum_required(VERSION 3.16)
project(MyProject LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置第三方库根目录
set(THIRD_PARTY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty)

# =============== Eigen ===============
set(EIGEN3_INCLUDE_DIR ${THIRD_PARTY_DIR}/eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${EIGEN3_INCLUDE_DIR})

# =============== OpenCV ===============
set(OpenCV_DIR ${THIRD_PARTY_DIR}/opencv-install/share/opencv4)
find_package(OpenCV REQUIRED)

# =============== MNN ===============
set(MNN_ROOT ${THIRD_PARTY_DIR}/mnn-install)
add_library(mnn STATIC IMPORTED)
set_target_properties(mnn PROPERTIES
    IMPORTED_LOCATION ${MNN_ROOT}/lib/libMNN.a
    INTERFACE_INCLUDE_DIRECTORIES ${MNN_ROOT}/include
)

# =============== ONNX Runtime ===============
set(ONNXRUNTIME_ROOT ${THIRD_PARTY_DIR}/onnxruntime-linux-x64-1.16.1)
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_LOCATION ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so
    INTERFACE_INCLUDE_DIRECTORIES ${ONNXRUNTIME_ROOT}/include
)

# =============== 自动收集 src/ 下所有 .cpp 文件 ===============
file(GLOB_RECURSE SRC_FILES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

# =============== 通用 include 路径：只需 src 根目录 ===============
set(COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/src)

# =============== 辅助函数：为测试源文件创建可执行程序 ===============
function(add_test_executable TEST_SOURCE)
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE} ${SRC_FILES})
    target_include_directories(${TEST_NAME} PRIVATE ${COMMON_INCLUDE_DIRS})
    target_link_libraries(${TEST_NAME} PRIVATE
        ${OpenCV_LIBS}
        mnn
        eigen
        onnxruntime
    )
endfunction()

# =============== 自动构建 test/ 下所有测试程序 ===============
file(GLOB TEST_SOURCES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp
)

foreach(TEST_SRC ${TEST_SOURCES})
    add_test_executable(${TEST_SRC})
endforeach()