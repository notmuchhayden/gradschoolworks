cmake_minimum_required(VERSION 3.16)

project(ArxStreamCipherExperiment LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(arx_stream_ciphers
    src/Salsa20.cpp
    src/ChaCha20.cpp
)

target_include_directories(arx_stream_ciphers PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
)

if(MSVC)
    target_compile_options(arx_stream_ciphers PRIVATE /W4)
else()
    target_compile_options(arx_stream_ciphers PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

add_executable(arx_stream_test
    src/TestMain.cpp
)

target_link_libraries(arx_stream_test PRIVATE arx_stream_ciphers)
target_compile_definitions(arx_stream_test PRIVATE
    ARX_PROJECT_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

if(MSVC)
    target_compile_options(arx_stream_test PRIVATE /W4)
else()
    target_compile_options(arx_stream_test PRIVATE
        -Wall
        -Wextra
        -Wpedantic
    )
endif()

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME arx_stream_cipher_validation
        COMMAND arx_stream_test
            "${CMAKE_CURRENT_SOURCE_DIR}/data_samples/gradient_512x512.bmp"
            3
    )
endif()
