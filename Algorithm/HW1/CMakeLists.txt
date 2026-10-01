cmake_minimum_required(VERSION 3.20)

project(HW1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 과제 3-3: 역전쌍 개수 계산
add_executable(HW1_3_3
    HW1_3_3.cpp
)

# 과제 4-2: 상담 선택
add_executable(HW1_4_2
    HW1_4_2.cpp
)

if(MSVC)
    target_compile_options(HW1_3_3 PRIVATE /utf-8)
    target_compile_options(HW1_4_2 PRIVATE /utf-8)
endif()
