cmake_minimum_required(VERSION 3.18)
project(breath_counter CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Torch REQUIRED)

add_library(imci INTERFACE)
target_include_directories(imci INTERFACE src)

add_executable(breath_counter src/main.cpp)
target_link_libraries(breath_counter PRIVATE imci "${TORCH_LIBRARIES}")

enable_testing()
add_executable(test_imci tests/test_imci.cpp)
target_link_libraries(test_imci PRIVATE imci)
add_test(NAME imci_thresholds COMMAND test_imci)
