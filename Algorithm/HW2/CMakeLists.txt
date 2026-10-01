cmake_minimum_required(VERSION 3.20)

project(HW2 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


add_executable(HW2_2_3
    HW2_2_3.cpp
)
if(MSVC)
    target_compile_options(HW2_2_3 PRIVATE /utf-8)
endif()
