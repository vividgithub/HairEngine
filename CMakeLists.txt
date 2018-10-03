# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.12)

macro(set_apple_cuda_rpath _TARGET)
    if (APPLE)
        # We need to add the path to the driver (libcuda.dylib) as an rpath,
        # so that the static cuda runtime can find it at runtime.
        set_property(TARGET ${_TARGET}
                PROPERTY
                BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
    endif()
endmacro()

project ("HairEngine" CXX)

# Expand the cmake module path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")

include(OptimizeForArchitecture)

# Optimize for architecture, we use to enable and Eigen vectorization
OptimizeForArchitecture()

# Set the root directory and the src directory
set(HAIRENGINE_ROOT_DIR ${PROJECT_SOURCE_DIR})  # Root directory
set(HAIRENGINE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/HairEngine")  # Source code directory

# Some of the dependencies
set(HAIRENGINE_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/External")  

set(CMAKE_CXX_STANDARD 14)

# All the common compile source files
set(
	HAIRENGINE_COMMON_SOURCE_FILES
	"${HAIRENGINE_SOURCE_DIR}/geo/hair.cpp"
	"${HAIRENGINE_EXTERNAL_DIR}/CompactNSearch/src/CompactNSearch.cpp"
    "${HAIRENGINE_SOURCE_DIR}/solver/sdf_collision_solver.cu"
	) 

# All the include directories
set(
	HAIRENGINE_INCLUDE_DIRS 
	"${HAIRENGINE_ROOT_DIR}/.."
	"${HAIRENGINE_EXTERNAL_DIR}/Eigen3" # Eigen3
	"${HAIRENGINE_EXTERNAL_DIR}/VPly/cpp" # VPly 
	"${HAIRENGINE_EXTERNAL_DIR}/CompactNSearch/include" # CompactNSearch
    "${HAIRENGINE_EXTERNAL_DIR}/cxxopts"
    "${HAIRENGINE_EXTERNAL_DIR}/inih"
	) 

# All the external libraries
set(HAIRENGINE_LIBRARIES) 

# Compile tags
set(HAIRENGINE_COMPILE_OPTIONS)

# Enabling OPENMP
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "/usr/local/lib")
if (CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
    # Handle Apple Clang openmp
    set(APPLE_OPENMP_ROOT $ENV{OPENMP_ROOT})
    if (NOT APPLE_OPENMP_ROOT)
        message(WARNING "Check using Apple Clang compiler, please set the environment \"OPENMP_ROOT\" correctly")
    endif()

    if (APPLE_OPENMP_ROOT)
        find_library(APPLE_OPENMP_LIB_PATH omp NAMES libomp HINTS ${APPLE_OPENMP_ROOT}/lib)
    else()
        find_library(APPLE_OPENMP_LIB_PATH omp NAMES libomp)
    endif()
    if (APPLE_OPENMP_LIB_PATH)
        message(STATUS "Find OpenMP lib path: " ${APPLE_OPENMP_LIB_PATH})
    else()
        message(FATAL_ERROR "Could not find OpenMP library path")
    endif()

    if (APPLE_OPENMP_ROOT)
        find_path(APPLE_OPENMP_INCLUDE_PATH omp.h HINTS ${APPLE_OPENMP_ROOT}/include)
    endif()
    if (APPLE_OPENMP_INCLUDE_PATH)
        message(STATUS "Find OpenMP include path: " ${APPLE_OPENMP_INCLUDE_PATH})
    else()
        message(FATAL_ERROR "Could not find OpenMP include path")
    endif()

    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "-Xpreprocessor -fopenmp -lomp")
    set(HAIRENGINE_LIBRARIES ${HAIRENGINE_LIBRARIES} ${APPLE_OPENMP_LIB_PATH})
    set(HAIRENGINE_INCLUDE_DIRS ${HAIRENGINE_INCLUDE_DIRS} ${APPLE_OPENMP_INCLUDE_PATH})
else()
    find_package(OpenMP REQUIRED)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    message(STATUS "OpenMp added with flags: " ${CMAKE_CXX_FLAGS} " with OpenMP Flags: " ${OpenMP_CXX_FLAGS})
endif()
set(HAIRENGINE_COMPILE_OPTIONS ${HAIRENGINE_COMPILE_OPTIONS} "-DHAIRENGINE_ENABLE_OPENMP")

option(HAIRENGINE_ENABLE_CUDA "Enable cuda for GPU acceleration" ON)
if (HAIRENGINE_ENABLE_CUDA)
    enable_language(CUDA)
    message(STATUS "Enable cuda...")

    set(CMAKE_CUDA_STANDARD 14)

    set(HAIRENGINE_INCLUDE_DIRS ${HAIRENGINE_INCLUDE_DIRS} ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    message(STATUS "Add cuda include directories: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
    message(STATUS "Add cuda library path: ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
    set(HAIRENGINE_COMPILE_OPTIONS ${HAIRENGINE_COMPILE_OPTIONS} HAIRENGINE_ENABLE_CUDA)
    set(HAIRENGINE_LIBRARIES ${HAIRENGINE_LIBRARIES} cudart)

    link_directories(${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()


# We use the main file to generate executable files for debugging in testing.
# But it will not exisit in the git project
if (EXISTS "${HAIRENGINE_SOURCE_DIR}/main/main.cpp")
	add_executable(HairEngineExecutable ${HAIRENGINE_COMMON_SOURCE_FILES} ${HAIRENGINE_SOURCE_DIR}/main/main.cpp)
	target_include_directories(HairEngineExecutable PUBLIC ${HAIRENGINE_INCLUDE_DIRS})
	target_link_libraries(HairEngineExecutable PUBLIC ${HAIRENGINE_LIBRARIES})
	target_compile_definitions(HairEngineExecutable PUBLIC ${HAIRENGINE_COMPILE_OPTIONS})

    set_apple_cuda_rpath(HairEngineExecutable)
else()
endif()

# If the source files is not empty, generate the library 
if (HAIRENGINE_COMMON_SOURCE_FILES)
	# Target for libHairEngine
	add_library(HairEngine ${HAIRENGINE_COMMON_SOURCE_FILES})
	target_include_directories(HairEngine PUBLIC ${HAIRENGINE_INCLUDE_DIRS})
	target_link_libraries(HairEngine PUBLIC ${HAIRENGINE_LIBRARIES})
	target_compile_definitions(HairEngine PUBLIC ${HAIRENGINE_COMPILE_OPTIONS})

    set_apple_cuda_rpath(HairEngine)
endif()