cmake_minimum_required(VERSION 3.5)

project(DDSketch_CPP VERSION 1.0)
option(BUILD_EXAMPLES   "Build programs that illustrate the usage of the DDSketch algorithm" OFF)
option(BUILD_UNIT_TESTS "Build the unit tests" OFF)
option(RUN_COVERAGE     "Run code coverage" OFF)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if (BUILD_UNIT_TESTS OR RUN_COVERAGE)
  add_subdirectory(tests)
endif()
