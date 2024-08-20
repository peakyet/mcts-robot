cmake_minimum_required(VERSION 3.10)

# Project name and version
project(pomcp VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

# Specify the paths for include files
include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(matplotlib_cpp INTERFACE)
target_link_libraries(matplotlib_cpp INTERFACE
  Python3::Python
  Python3::Module
)
find_package(Python3 COMPONENTS NumPy)
if(Python3_NumPy_FOUND)
  target_link_libraries(matplotlib_cpp INTERFACE Python3::NumPy)
else()
  target_compile_definitions(matplotlib_cpp INTERFACE WITHOUT_NUMPY)
endif()

# Add source files
set(SOURCES
    src/main.cpp
)

# Define the executable target
add_executable(POMCP_Solver ${SOURCES})
target_link_libraries(POMCP_Solver matplotlib_cpp)

# Optional: Add compiler warnings for better code quality
target_compile_options(POMCP_Solver PRIVATE -Wall -Wextra -Wpedantic)
