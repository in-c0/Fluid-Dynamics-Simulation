cmake_minimum_required(VERSION 3.16)
project(FluidDynamicsSimulation LANGUAGES CXX)

# Set up C++ and Fortran standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
# set(CMAKE_Fortran_COMPILER gfortran)

# Vulkan SDK detection from the official Vulkan sample CMakeLists.txt
find_package(Vulkan REQUIRED)

if(Vulkan_FOUND)
    if(NOT Vulkan_dxc_exe_FOUND)
        find_program(Vulkan_dxc_EXECUTABLE
                     NAMES dxc
                     HINTS
                     "$ENV{VULKAN_SDK}/Bin"
                     "$ENV{VULKAN_SDK}/bin")
    endif()
    if(Vulkan_dxc_EXECUTABLE)
        message(STATUS "Found DirectX Shader Compiler under ${Vulkan_dxc_EXECUTABLE}")
    else()
        message(STATUS "Couldn't find DirectX Shader Compiler executable, make sure it is present in Vulkan SDK or add it manually via Vulkan_dxc_EXECUTABLE cmake variable. HLSL shaders won't be compiled.")
    endif()
endif()


add_subdirectory(third_party/glfw)

add_executable(${PROJECT_NAME}
    src/main.cpp
    ${FORTRAN_SOURCES}
)

include_directories(
    ${Vulkan_INCLUDE_DIRS}
    ${GLFW_INCLUDE_DIRS}
    third_party/glfw/include
)

target_link_libraries(${PROJECT_NAME}
    ${Vulkan_LIBRARIES}
    glfw
    Vulkan::Vulkan
)
