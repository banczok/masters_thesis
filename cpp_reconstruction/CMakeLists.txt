cmake_minimum_required(VERSION 3.10)
project(FlyingEdgesVolumeExtraction)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV not found. Please install it and set the correct path.")
endif()

# Find VTK
find_package(VTK REQUIRED)
if(NOT VTK_FOUND)
    message(FATAL_ERROR "VTK not found. Please install it and set the correct path.")
endif()

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${VTK_INCLUDE_DIRS})

# Add the executable
add_executable(${PROJECT_NAME} fyling_preds.cpp)

# Link libraries
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS} ${VTK_LIBRARIES})

# Enable VTK's auto-init for rendering modules
vtk_module_autoinit(
    TARGETS ${PROJECT_NAME}
    MODULES ${VTK_LIBRARIES}
)
