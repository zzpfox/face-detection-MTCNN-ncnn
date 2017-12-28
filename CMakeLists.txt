SET(SRC
	main.cpp
    Mtcnn.cpp
)

SET(NCNN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/packages/ncnn_20171225)

if (WIN32)
	SET(OPENCV_DIR C:/library/opencv-3.3.1/opencv/build)
	SET(OPENCV_LIB ${OPENCV_DIR}/x64/vc14/lib/opencv_world331.lib)
	SET(NCNN_LIB ${NCNN_DIR}/lib/win/x64/vs2017/ncnn.lib)
endif (WIN32)

include_directories(${OPENCV_DIR}/include)
include_directories(${NCNN_DIR}/include)

add_executable(mtcnn ${SRC})
TARGET_LINK_LIBRARIES (mtcnn ${OPENCV_LIB} ${NCNN_LIB})
