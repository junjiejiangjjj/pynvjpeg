cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

include(GNUInstallDirs)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

SET(PROJECT_NAME pynvjpeg)
PROJECT(${PROJECT_NAME} VERSION 0.1 LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_STANDARD)
    set(CMAKE_CUDA_STANDARD 11)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)    
endif()

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) 
link_directories(${CMAKE_CUDA_INCLUDE_DIRS})

find_library(CUDART_LIBRARY cudart ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(NVJPEG_LIBRARY nvjpeg ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include_directories(
  ${PROJECT_SOURCE_DIR}
  )

include(FetchContent)
FetchContent_Declare(
  pybind11_sources
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG v2.9
  )
FetchContent_GetProperties(pybind11_sources)
if(NOT pybind11_sources_POPULATED)
  FetchContent_Populate(pybind11_sources)
  add_subdirectory(
    ${pybind11_sources_SOURCE_DIR}
    ${pybind11_sources_BINARY_DIR}
    )
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX ${CMAKE_BINARY_DIR} CACHE PATH "" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

add_library(${PROJECT_NAME}
  SHARED
  nvjpeg_decoder.cpp
  jpeg_image.cpp
  cuda_util.cpp
  python_nvjpeg_decoder.cpp
  )

target_link_libraries(${PROJECT_NAME}
  PUBLIC
  ${CUDART_LIBRARY}
  ${NVJPEG_LIBRARY}
  pybind11::module)

set_target_properties(${PROJECT_NAME}
  PROPERTIES
  PREFIX "${PYTHON_MODULE_PREFIX}"
  SUFFIX "${PYTHON_MODULE_EXTENSION}"
  )

add_executable(test_decode test.cpp)
target_link_libraries(test_decode ${PROJECT_NAME} pybind11::embed)

include(CTest)
enable_testing()

add_test(
  NAME test_decode
  COMMAND $<TARGET_FILE:test_decode> ${CMAKE_CURRENT_SOURCE_DIR}/cat.jpg
  )

add_test(
  NAME python_test_short
  COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py $<TARGET_FILE:${PROJECT_NAME}>
  )
