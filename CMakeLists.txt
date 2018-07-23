# CMakeList.txt : Top-level CMake project file, do global configuration
# and include sub-projects here.
#
cmake_minimum_required (VERSION 3.8)

project ("HairEngine")

# Set the root directory and the src directory
set(HAIRENGINE_ROOT_DIR ${PROJECT_SOURCE_DIR})  # Root directory
set(HAIRENGINE_SOURCE_DIR "${PROJECT_SOURCE_DIR}/HairEngine")  # Source code directory
set(HAIRENGINE_EXTERNAL_DIR "${PROJECT_SOURCE_DIR}/External")  # Some of the dependencies


# All the common compile source files
set(HAIRENGINE_COMMON_SOURCE_FILES) 

# All the include directories
set(
	HAIRENGINE_INCLUDE_DIRS 
	"${HAIRENGINE_ROOT_DIR}/.."
	"${HAIRENGINE_EXTERNAL_DIR}"
	) 

# All the external libraries
set(HAIRENGINE_LIBRARIES) 

# We use the main file to generate executable files for debugging in testing.
# But it will not exisit in the git project
if (EXISTS "${HAIRENGINE_SOURCE_DIR}/main/main.cpp")
	add_executable(HairEngineExecutable ${HAIRENGINE_COMMON_SOURCE_FILES} ${HAIRENGINE_SOURCE_DIR}/main/main.cpp)
	target_include_directories(HairEngineExecutable PUBLIC ${HAIRENGINE_INCLUDE_DIRS})
	target_link_libraries(HairEngineExecutable PUBLIC ${HAIRENGINE_LIBRARIES})
else()
endif()

# If the source files is not empty, generate the library 
if (HAIRENGINE_COMMON_SOURCE_FILES)
	# Target for libHairEngine
	add_library(HairEngine ${HAIRENGINE_COMMON_SOURCE_FILES})
	target_include_directories(HairEngine PUBLIC ${HAIRENGINE_INCLUDE_DIRS})
	target_link_libraries(HairEngine PUBLIC ${HAIRENGINE_LIBRARIES})
endif()