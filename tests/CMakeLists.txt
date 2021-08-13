set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Don't use e.g. GNU extension (like -std=gnu++11) for portability
set(CMAKE_CXX_EXTENSIONS OFF)
add_compile_options(-O3 -Wall -Wextra -pedantic)

if (RUN_COVERAGE)
  add_compile_options(--coverage)
  link_libraries(gcov)
endif()

if (UNIX)
  link_libraries(pthread)
endif()

link_libraries(gtest)

project(DDSketch_Unit_Tests VERSION 1.0)
add_executable(DDSketch_Unit_Tests ddsketch_unit_tests.cpp)
