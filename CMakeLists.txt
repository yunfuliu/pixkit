project(pixkit)

cmake_minimum_required (VERSION 2.6)


# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/bin")


# ============================================================================ #
# Where to install the pixkit library
set(PIXKIT_INSTALL_PREFIX "/usr/local/pixkit/lib" CACHE PATH "Installation Directory")
set(PIXKIT_INCLUDE_PREFIX "/usr/local/pixkit/include" CACHE PATH "Includes Directory")


# ============================================================================ #
# Find external libraries and dependencies
find_package(OpenCV REQUIRED)
#if (NOT APPLE)
#  find_package(OpenMP)
#endif ()
#find_package(Doxygen)


# ============================================================================ #
# Create projects (pixkit-image)
include_directories("pixkit-image/")

# pixkit-image libraries
set(PIXKIT_IMAGE_SRCS
	pixkit-image/pixkit-image.hpp
	pixkit-image/src/attack.cpp
	pixkit-image/src/edgedetection.cpp
	pixkit-image/src/enhancement.cpp
	pixkit-image/src/filtering.cpp
	pixkit-image/src/halftoning.cpp
	pixkit-image/src/qualityassessment.cpp
	pixkit-image/src/NADD2013/NADD.cpp
)
add_library(PIXKIT_IMAGE ${PIXKIT_IMAGE_SRCS})
target_link_libraries(PIXKIT_IMAGE ${OpenCV_LIBS})

# Create projects (pixkit-ml)
include_directories("pixkit-ml/")

# pixkit-ml libraries
set(PIXKIT_ML_SRCS
	pixkit-ml/pixkit-ml.hpp
	pixkit-ml/src/pixkit-ml.cpp
)
add_library(PIXKIT_ML ${PIXKIT_ML_SRCS})
target_link_libraries(PIXKIT_ML ${OpenCV_LIBS})


# ============================================================================ #
# Sample programs
add_executable(image_halftoning_dotdiffusion_GuoLiu2009
	examples-image/halftoning/dotdiffusion.GuoLiu2009/example_dotdiffusion.GuoLiu2009.cpp
)
target_link_libraries(image_halftoning_dotdiffusion_GuoLiu2009 PIXKIT_IMAGE)

# Sample programs
add_executable(image_halftoning_dotdiffusion_NADD2013
	examples-image/halftoning/dotdiffusion.NADD2013/example_dotdiffusion.NADD2013.cpp
)
target_link_libraries(image_halftoning_dotdiffusion_NADD2013 PIXKIT_IMAGE)


# ============================================================================ #
# Library installation
install(TARGETS PIXKIT_IMAGE DESTINATION ${PIXKIT_INSTALL_PREFIX})
install(TARGETS PIXKIT_ML DESTINATION ${PIXKIT_INSTALL_PREFIX})
install(FILES
    pixkit-image/pixkit-image.hpp
	pixkit-ml.hpp
    DESTINATION
    ${PIXKIT_INCLUDE_PREFIX}
)


# ============================================================================ #
# Generate Doxygen-based documentation project
#if(DOXYGEN_FOUND)
#  add_custom_target(akaze_documentation
#    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#    COMMENT "Generating API documentation with Doxygen" VERBATIM)
#endif(DOXYGEN_FOUND)
