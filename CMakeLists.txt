SET(SRC
	main.cpp
    Mtcnn.cpp
)

SET(NCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packages/ncnn_20171225)

if (WIN32)
	SET(OpenCV_DIR C:/library/opencv-3.3.1/opencv/build)
	SET(OpenCV_LIBS ${OpenCV_DIR}/x64/vc14/lib/opencv_world331.lib)
	SET(NCNN_LIBS ${NCNN_DIR}/lib/win/x64/vs2017/ncnn.lib)
endif (WIN32)

if (UNIX)
	find_package(OpenMP)
	
	if (OPENMP_FOUND)
	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
	endif()

	find_package( OpenCV REQUIRED )
	SET(NCNN_LIBS ${NCNN_DIR}/lib/linux/x64/gcc5.4/libncnn.a)
endif (UNIX)

include_directories(${OPENCV_DIR}/include)
include_directories(${NCNN_DIR}/include)

add_executable(mtcnn ${SRC})
TARGET_LINK_LIBRARIES (mtcnn ${OpenCV_LIBS} ${NCNN_LIBS})
