cmake_minimum_required(VERSION 3.24)
project(hftcore VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(HFTCORE_BUILD_TESTS "Build the Catch2 unit tests" ON)
option(HFTCORE_BUILD_PYTHON "Build the pybind11 module" OFF)
option(HFTCORE_SANITIZE "Build with AddressSanitizer and UndefinedBehaviorSanitizer" OFF)

set(HFTCORE_WARNINGS -Wall -Wextra -Wpedantic -Werror)
if(HFTCORE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

include(FetchContent)
FetchContent_Declare(
  simdjson
  URL https://github.com/simdjson/simdjson/archive/refs/tags/v3.10.1.tar.gz
  URL_HASH SHA256=1e8f881cb2c0f626c56cd3665832f1e97b9d4ffc648ad9e1067c134862bba060
)
set(SIMDJSON_DEVELOPER_MODE OFF CACHE INTERNAL "")
set(SIMDJSON_BUILD_STATIC_LIB ON CACHE INTERNAL "")
set(BUILD_SHARED_LIBS OFF CACHE INTERNAL "")
FetchContent_MakeAvailable(simdjson)
set_target_properties(simdjson PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(hftcore_core STATIC src/version.cpp src/decimal.cpp src/timestamp.cpp src/bar_aggregator.cpp src/features.cpp src/market_engine.cpp src/replay.cpp src/frame_router.cpp src/itch.cpp src/order_book.cpp src/itch_book.cpp)
target_include_directories(hftcore_core PUBLIC include)
target_link_libraries(hftcore_core PRIVATE simdjson)
target_compile_options(hftcore_core PRIVATE ${HFTCORE_WARNINGS})
# Bit-identical results with the Python reference: no fused multiply-add.
target_compile_options(hftcore_core PUBLIC -ffp-contract=off)
set_target_properties(hftcore_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(HFTCORE_BUILD_PYTHON)
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(hftcore src/bindings.cpp)
  target_link_libraries(hftcore PRIVATE hftcore_core)
  target_compile_options(hftcore PRIVATE ${HFTCORE_WARNINGS})
  install(TARGETS hftcore LIBRARY DESTINATION .)
endif()

option(HFTCORE_BUILD_BENCH "Build benchmarks" ON)
if(HFTCORE_BUILD_BENCH)
  add_executable(hftcore_bench_aggregator bench/bench_aggregator.cpp)
  target_link_libraries(hftcore_bench_aggregator PRIVATE hftcore_core)
  target_compile_options(hftcore_bench_aggregator PRIVATE ${HFTCORE_WARNINGS})
  add_executable(hftcore_bench_engine bench/bench_engine.cpp)
  add_executable(hftcore_bench_frames bench/bench_frames.cpp)
  target_link_libraries(hftcore_bench_frames PRIVATE hftcore_core)
  target_link_libraries(hftcore_bench_engine PRIVATE hftcore_core)
  target_compile_options(hftcore_bench_engine PRIVATE ${HFTCORE_WARNINGS})
  add_executable(hftcore_bench_itch bench/bench_itch.cpp)
  target_link_libraries(hftcore_bench_itch PRIVATE hftcore_core)
  target_compile_options(hftcore_bench_itch PRIVATE ${HFTCORE_WARNINGS})
  add_executable(hftcore_itch_replay bench/itch_replay.cpp)
  target_link_libraries(hftcore_itch_replay PRIVATE hftcore_core)
  target_compile_options(hftcore_itch_replay PRIVATE ${HFTCORE_WARNINGS})
endif()

if(HFTCORE_BUILD_TESTS)
  include(FetchContent)
  FetchContent_Declare(
    Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    URL_HASH SHA256=c991b247a1a0d7bb9c39aa35faf0fe9e19764213f28ffba3109388e62ee0269c
  )
  FetchContent_MakeAvailable(Catch2)
  add_executable(hftcore_tests tests/test_version.cpp tests/test_decimal.cpp tests/test_timestamp.cpp tests/test_bar_aggregator.cpp tests/test_market_engine.cpp tests/test_id_set.cpp tests/test_itch.cpp tests/test_order_book.cpp tests/test_itch_book.cpp)
  target_link_libraries(hftcore_tests PRIVATE hftcore_core Catch2::Catch2WithMain)
  target_compile_options(hftcore_tests PRIVATE ${HFTCORE_WARNINGS})
  include(CTest)
  include(Catch)
  catch_discover_tests(hftcore_tests)
endif()
