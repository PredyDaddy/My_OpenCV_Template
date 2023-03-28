cmake_minimum_required(VERSION 3.10)

project(OpenCV_Face_Detect)

# 查找OpenCV
find_package(OpenCV REQUIRED)

# 查找gflags
find_package(gflags REQUIRED)

# 添加头文件路径
include_directories(${OpenCV_INCLUDE_DIRS} ${gflags_INCLUDE_DIRS})

# 添加可执行文件
add_executable(run_file src/main.cpp)

# 链接OpenCV和gflags库
target_link_libraries(run_file ${OpenCV_LIBS} ${gflags_LIBS})


