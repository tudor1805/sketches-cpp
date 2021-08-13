set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Don't use e.g. GNU extension (like -std=gnu++11) for portability
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-O3 -Wall -Wextra -pedantic)

project(DDSketch_Examples VERSION 1.0)
add_executable(DDSketch_Examples ddsketch_examples.cpp)
