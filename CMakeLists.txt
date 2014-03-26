project(pixkit)

cmake_minimum_required (VERSION 2.6)

set(PIXKIT_VERSION_MAJOR 0)
set(PIXKIT_VERSION_MINOR 4)
set(PIXKIT_VERSION_PATCH 0)
set(PIXKIT_VERSION_STRING ${PIXKIT_VERSION_MAJOR}${PIXKIT_VERSION_MINOR}${PIXKIT_VERSION_PATCH})



# ============================================================================ #
# Where to install the pixkit library
set(PIXKIT_INSTALL_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install/lib" CACHE PATH "Installation Directory")
set(PIXKIT_INCLUDE_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install/include" CACHE PATH "Includes Directory")
set(PIXKIT_SAMPLES_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install/samples" CACHE PATH "Samples Directory")
set(PIXKIT_TEST_PREFIX "${CMAKE_CURRENT_BINARY_DIR}/install/test" CACHE PATH "Samples Directory")



# ============================================================================ #
# Where to output binaries and libraries
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib")
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/samples")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PIXKIT_SAMPLES_PREFIX})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG "${CMAKE_CURRENT_BINARY_DIR}/bin")



# ============================================================================ #
# Find external libraries and dependencies
find_package(OpenCV REQUIRED)
#if (NOT APPLE)
#  find_package(OpenMP)
#endif ()
#find_package(Doxygen)



# ============================================================================ #
# Create projects
set(CMAKE_DEBUG_POSTFIX ${PIXKIT_VERSION_STRING}d)
set(CMAKE_RELEASE_POSTFIX ${PIXKIT_VERSION_STRING})
# Create projects (pixkit-image) ***********************************************
include_directories("modules/pixkit-image")

# pixkit-image libraries
set(PIXKIT_IMAGE_SRCS
	modules/pixkit-image/include/pixkit-image.hpp
	modules/pixkit-image/src/attack.cpp
	modules/pixkit-image/src/comp.cpp
	modules/pixkit-image/src/edgedetection.cpp
	modules/pixkit-image/src/enhancement.cpp
	modules/pixkit-image/src/filtering.cpp
	modules/pixkit-image/src/halftoning.cpp
	modules/pixkit-image/src/qualityassessment.cpp
	modules/pixkit-image/src/NADD2013/NADD.cpp
)
add_library(pixkit_image ${PIXKIT_IMAGE_SRCS})
target_link_libraries(pixkit_image ${OpenCV_LIBS})

# Create projects (pixkit-ml) **************************************************
#include_directories("modules/pixkit-ml/")

# pixkit-ml libraries
set(PIXKIT_ML_SRCS
	modules/pixkit-ml/include/pixkit-ml.hpp
	modules/pixkit-ml/src/fuzzyCMeans.cpp
	modules/pixkit-ml/src/KMeans.cpp
	modules/pixkit-ml/src/labeling.cpp
)
add_library(pixkit_ml ${PIXKIT_ML_SRCS})
target_link_libraries(pixkit_ml ${OpenCV_LIBS})



# ============================================================================ #
# Sample programs (image/halftoning)
add_executable(image_halftoning_dotdiffusion_GuoLiu2009
	examples-image/halftoning/dotdiffusion.GuoLiu2009/example_dotdiffusion.GuoLiu2009.cpp
)
target_link_libraries(image_halftoning_dotdiffusion_GuoLiu2009 pixkit_image)
add_executable(image_halftoning_dotdiffusion_NADD2013
	examples-image/halftoning/dotdiffusion.NADD2013/example_dotdiffusion.NADD2013.cpp
)
target_link_libraries(image_halftoning_dotdiffusion_NADD2013 pixkit_image)

# Sample programs (image/enhancement)
add_executable(image_enhancement_MaryKim2008
	examples-image/enhancement/enhancement.MaryKim2008/example_MaryKim2008.cpp
)
target_link_libraries(image_enhancement_MaryKim2008 pixkit_image)
add_executable(image_enhancement_WadudKabirDewanChae2007
	examples-image/enhancement/enhancement.WadudKabirDewanChae2007/example_WadudKabirDewanChae2007.cpp
)
target_link_libraries(image_enhancement_WadudKabirDewanChae2007 pixkit_image)



# ============================================================================ #
# Test programs (image/halftoning)
add_executable(test_image_halftoning
	modules/pixkit-image/test/test_halftoning.cpp
)
target_link_libraries(test_image_halftoning pixkit_image)

# set up property
set_target_properties(
	test_image_halftoning
	PROPERTIES
	RUNTIME_OUTPUT_DIRECTORY_RELEASE	${PIXKIT_TEST_PREFIX}
)



# ============================================================================ #
# Library installation
if(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(PIXKIT_ENV "x64")
else(CMAKE_SIZEOF_VOID_P EQUAL 8)
	set(PIXKIT_ENV "x86")
endif(CMAKE_SIZEOF_VOID_P EQUAL 8)
if(MSVC10)
	set(PIXKIT_COMPILER "vc10")
elseif(MSVC11)
	set(PIXKIT_COMPILER "vc11")
endif()
install(TARGETS pixkit_image DESTINATION "${PIXKIT_INSTALL_PREFIX}/${PIXKIT_ENV}/${PIXKIT_COMPILER}")
install(TARGETS pixkit_ml DESTINATION "${PIXKIT_INSTALL_PREFIX}/${PIXKIT_ENV}/${PIXKIT_COMPILER}")
install(FILES
    modules/pixkit-image/include/pixkit-image.hpp
	modules/pixkit-ml/include/pixkit-ml.hpp
    DESTINATION
    ${PIXKIT_INCLUDE_PREFIX}
)
install(FILES
    modules/python/unittest.py
    DESTINATION
    ${PIXKIT_TEST_PREFIX}
)



# ============================================================================ #
# Generate Doxygen-based documentation project
#if(DOXYGEN_FOUND)
#  add_custom_target(akaze_documentation
#    ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile
#    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
#    COMMENT "Generating API documentation with Doxygen" VERBATIM)
#endif(DOXYGEN_FOUND)