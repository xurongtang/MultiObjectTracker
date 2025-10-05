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

# =============== OpenCV ===============    静态库
set(OpenCV_DIR ${THIRD_PARTY_DIR}/opencv-install/lib/cmake/opencv4)
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

# =============== 构建静态核心库 ===============
add_library(mot_core STATIC ${SRC_FILES})
target_include_directories(mot_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(mot_core
    PUBLIC
        ${OpenCV_LIBS}
        mnn
        eigen
        onnxruntime
)

# =============== 可执行文件：强制静态链接 ===============
function(add_test_executable TEST_SOURCE)
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE
        mot_core
        -static-libstdc++
        -static-libgcc
    )
endfunction()

# =============== 构建所有测试程序 ===============
file(GLOB TEST_SOURCES
    LIST_DIRECTORIES false
    CONFIGURE_DEPENDS
    ${CMAKE_CURRENT_SOURCE_DIR}/test/*.cpp
)

foreach(TEST_SRC ${TEST_SOURCES})
    add_test_executable(${TEST_SRC})
endforeach()