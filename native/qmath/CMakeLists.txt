cmake_minimum_required(VERSION 3.20)
project(qcns_qmath_bridge LANGUAGES CXX)
find_package(OpenMP REQUIRED)
set(QMATH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../qcnspp/QMath")
set(QMATH_KERNELS
    ${QMATH_ROOT}/qmath/src/matrix_storage.cpp
    ${QMATH_ROOT}/qmath/src/dynamics/single_transformation.cpp
    ${QMATH_ROOT}/qmath/src/dynamics/controlled_transformation.cpp
    ${QMATH_ROOT}/qmath/src/dynamics/cnot.cpp
    ${QMATH_ROOT}/qmath/src/channels/dephasing.cpp
    ${QMATH_ROOT}/qmath/src/channels/amplitude_damping.cpp
    ${QMATH_ROOT}/qmath/src/channels/depolarization.cpp
    ${QMATH_ROOT}/qmath/src/quantum_info/reductions.cpp)
add_library(qcns_qmath SHARED bridge.cpp ${QMATH_KERNELS})
target_compile_features(qcns_qmath PRIVATE cxx_std_20)
target_include_directories(qcns_qmath PRIVATE ${QMATH_ROOT}/qmath/include)
target_link_libraries(qcns_qmath PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(qcns_qmath PROPERTIES PREFIX "" RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}")
