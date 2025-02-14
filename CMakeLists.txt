cmake_minimum_required(VERSION 3.15...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(MODULE _cudakdtree_interface)

add_library("_${MODULE}" SHARED "src/cudakdtree_jax_binding/cudakdtree_interface.cu")
set_target_properties("_${MODULE}" PROPERTIES POSITION_INDEPENDENT_CODE ON
                                                CUDA_STANDARD 17)
target_include_directories("_${MODULE}" PUBLIC ${XLA_DIR})
install(TARGETS "_${MODULE}" LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
