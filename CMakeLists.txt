# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("HairEngine")

# Expand the cmake module path
list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/CMake")

# Set the root directory and the src directory
set(HAIRENGINE_ROOT_DIR ${PROJECT_SOURCE_DIR})  # Root directory
set(HAIRENGINE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/HairEngine")  # Source code directory

# Some of the dependencies
set(HAIRENGINE_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/External")  

set(CMAKE_CXX_STANDARD 14)

# Options
option(HAIRENGINE_ENABLE_VPBRT "Enable VPbrt related class and functions" OFF)
option(HAIRENGINE_ENABLE_OPENMP "Enable openmp for parallism work, otherwise all the code that related to openmp will be sequential" ON)

# All the common compile source files
set(
	HAIRENGINE_COMMON_SOURCE_FILES
	"${HAIRENGINE_SOURCE_DIR}/geo/hair.cpp"
	) 

# All the include directories
set(
	HAIRENGINE_INCLUDE_DIRS 
	"${HAIRENGINE_ROOT_DIR}/.."
	"${HAIRENGINE_EXTERNAL_DIR}/Eigen3" # Eigen3
	"${HAIRENGINE_EXTERNAL_DIR}/VPly/cpp" # VPly
	) 

# All the external libraries
set(HAIRENGINE_LIBRARIES) 

# Compile tags
set(HAIRENGINE_COMPILE_OPTIONS)
if (HAIRENGINE_ENABLE_VPBRT) 
	set(HAIRENGINE_COMPILE_OPTIONS ${HAIRENGINE_COMPILE_OPTIONS} "-DHAIRENGINE_ENABLE_VPBRT")
endif()
if (HAIRENGINE_ENABLE_OPENMP)
	find_package(OpenMP REQUIRED)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}") 
	message(STATUS "OpenMp added with flags: " ${CMAKE_CXX_FLAGS} " with OpenMP Flags: " ${OpenMP_CXX_FLAGS})
	set(HAIRENGINE_COMPILE_OPTIONS ${HAIRENGINE_COMPILE_OPTIONS} "-DHAIRENGINE_ENABLE_OPENMP")
endif()

# We use the main file to generate executable files for debugging in testing.
# But it will not exisit in the git project
if (EXISTS "${HAIRENGINE_SOURCE_DIR}/main/main.cpp")
	add_executable(HairEngineExecutable ${HAIRENGINE_COMMON_SOURCE_FILES} ${HAIRENGINE_SOURCE_DIR}/main/main.cpp)
	target_include_directories(HairEngineExecutable PUBLIC ${HAIRENGINE_INCLUDE_DIRS})
	target_link_libraries(HairEngineExecutable PUBLIC ${HAIRENGINE_LIBRARIES})
	target_compile_definitions(HairEngineExecutable PUBLIC ${HAIRENGINE_COMPILE_OPTIONS})
else()
endif()

# If the source files is not empty, generate the library 
if (HAIRENGINE_COMMON_SOURCE_FILES)
	# Target for libHairEngine
	add_library(HairEngine ${HAIRENGINE_COMMON_SOURCE_FILES})
	target_include_directories(HairEngine PUBLIC ${HAIRENGINE_INCLUDE_DIRS})
	target_link_libraries(HairEngine PUBLIC ${HAIRENGINE_LIBRARIES})
	target_compile_definitions(HairEngine PUBLIC ${HAIRENGINE_COMPILE_OPTIONS})
endif()