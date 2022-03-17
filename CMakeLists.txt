cmake_minimum_required(VERSION 3.7)
project(pytorch-test)

# variables
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(ENABLE_PCH ON)

# compiler
if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /favor:INTEL64 /arch:SSE /arch:SSE2 /arch:AVX /arch:AVX2 /openmp")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /Wall /WX")
else()
  set(CXX_FLAGS "-march=native -fopenmp -Wall -Werror -Wfatal-errors -Wextra -Wpedantic -Wshadow -Wno-unused-parameter -Wno-missing-field-initializers")
  set(CXX_FLAGS_DEBUG "-g -Og")
  set(CXX_FLAGS_RELEASE "-Ofast")

  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${CXX_FLAGS_DEBUG}")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CXX_FLAGS_RELEASE}")

  set(CMAKE_C_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_C_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
  set(CMAKE_C_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
endif()

# ignore warnings from external sources
file(GLOB_RECURSE SRC_EXTERNAL "external/*")
set_source_files_properties(${SRC_EXTERNAL} PROPERTIES COMPILE_FLAGS "-w")

# ccache
find_program(CCACHE_FOUND ccache)
if(CCACHE_FOUND)
  set(CMAKE_CXX_COMPILER_LAUNCHER ccache)
endif()

# targets
add_executable(pytorch-test)

# status messages
message(STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER_ID} " " ${CMAKE_CXX_COMPILER_VERSION})
message(STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER_ID} " " ${CMAKE_C_COMPILER_VERSION})
message(STATUS "CMAKE_CXX_FLAGS: " ${CMAKE_CXX_FLAGS})
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: " ${CMAKE_CXX_FLAGS_DEBUG})
message(STATUS "CMAKE_CXX_FLAGS_RELEASE: " ${CMAKE_CXX_FLAGS_RELEASE})
message(STATUS "ENABLE_PROFILING: " ${ENABLE_PROFILING})
message(STATUS "ENABLE_SANITIZER: " ${ENABLE_SANITIZER})

# fmt
add_subdirectory(external/fmt)
include_directories("external/fmt/include")
target_link_libraries(pytorch-test fmt::fmt-header-only)

# torch
find_package(Torch REQUIRED HINTS "external/libtorch")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
target_link_libraries(pytorch-test "${TORCH_LIBRARIES}")

# custom
include_directories("src")

# sources
file(GLOB_RECURSE SRC CONFIGURE_DEPENDS
src/*.hpp src/*.cpp
)
target_sources(pytorch-test PRIVATE ${SRC})

if (ENABLE_PCH)
  # precompiled headers
  set(CMAKE_PCH_INSTANTIATE_TEMPLATES ON)
  file(GLOB_RECURSE SRC_PRECOMPILED CONFIGURE_DEPENDS src/Precompiled.hpp)
  message(STATUS "PCH: " ${SRC_PRECOMPILED})
  target_precompile_headers(pytorch-test PRIVATE ${SRC_PRECOMPILED})
endif()