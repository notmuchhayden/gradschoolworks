cmake_minimum_required(VERSION 3.20)

project(HW1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 과제 3-3 번 문제
# add_executable(HW1_3_3
#     ../HW1_3_3.cpp
# )

add_executable(HW1_4_2
    ../HW1_4_2.cpp
)

#enable_testing()
#add_test(NAME HW1_3_3_runs COMMAND HW1_3_3)