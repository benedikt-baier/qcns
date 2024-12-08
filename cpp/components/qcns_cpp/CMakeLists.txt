cmake_minimum_required(VERSION 3.10)
project(qcns_cpp)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)

# Find Eigen3 package
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(include ${EIGEN3_INCLUDE_DIR})

# Source files
set(SOURCES
    src/main.cpp
    src/linalg/matrix_utils.cpp
    src/linalg/operator_utils.cpp
    src/linalg/matrices.cpp
    src/qubits/QSystem.cpp
    src/qubits/Qubit.cpp
)

# Create a library
add_library(qcns_cpp ${SOURCES})

# Link external libraries (if any)
target_link_libraries(qcns_cpp Eigen3::Eigen)

# Add executable for testing or main application
# add_executable(project_name ${SOURCES})
