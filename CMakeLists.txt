cmake_minimum_required(VERSION 3.18...3.30)
project(${SKBUILD_PROJECT_NAME} LANGUAGES CXX CUDA)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
execute_process(
  COMMAND "${Python_EXECUTABLE}"
          "-c" "from jax.extend import ffi; print(ffi.include_dir())"
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE XLA_DIR)
message(STATUS "XLA include directory: ${XLA_DIR}")

find_package(nanobind CONFIG REQUIRED)
find_package(CUDAToolkit REQUIRED)

# include_directories("../cudakdtree")
include_directories("extern/cudaKDTree")

set(MODULE _cudakdtree_interface)

nanobind_add_module(${MODULE} NB_STATIC "src/cudakdtree_jax_binding/cudakdtree_interface.cu")
target_include_directories(${MODULE} PUBLIC ${XLA_DIR})
target_link_libraries(${MODULE} PRIVATE CUDA::cudart)
target_compile_options(${MODULE} PRIVATE "$<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G;-src-in-ptx>")
install(TARGETS ${MODULE} LIBRARY DESTINATION ${SKBUILD_PROJECT_NAME})
