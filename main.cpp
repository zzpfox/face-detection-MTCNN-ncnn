#include <iostream>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "Mtcnn.h"

using namespace std;
using namespace cv;

void PlotDetectionResult(const Mat& frame, const std::vector<SBoundingBox>& bbox)
{
    for (vector<SBoundingBox>::const_iterator it = bbox.begin(); it != bbox.end(); it++)
    {
        if ((*it).bExist)
        {
            rectangle(frame, Point((*it).x1, (*it).y1), Point((*it).x2, (*it).y2), Scalar(0, 0, 255), 2, 8, 0);
            for (int num = 0; num < 5; num++)
            {
                circle(frame, Point((int)*(it->ppoint + num), (int)*(it->ppoint + num + 5)), 3, Scalar(0, 255, 255), -1);
            }
        }
    }
}

int main(int argc, char** argv)
{
    VideoCapture cap(0);

    if (!cap.isOpened())
    {
        cout << "video is not open" << endl;
        return -1;
    }

    Mat frame;
    CMtcnn mtcnn;
    mtcnn.LoadModel("det1.param", "det1.bin", "det2.param", "det2.bin", "det3.param", "det3.bin");

    while (1)
    {
        cap >> frame;

        std::vector<SBoundingBox> finalBbox;
        ncnn::Mat ncnn_img = ncnn::Mat::from_pixels(frame.data, ncnn::Mat::PIXEL_BGR2RGB, frame.cols, frame.rows);

        double t1 = (double)getTickCount();
        mtcnn.Detect(ncnn_img, finalBbox);
        double t2 = (double)getTickCount();
        double t = 1000 * double(t2 - t1) / getTickFrequency();
        cout << "time = " << t << " ms, FPS = " << 1000 / t << endl;

        PlotDetectionResult(frame, finalBbox);

        imshow("frame", frame);

        if (waitKey(1) == 'q')
            break;
    }

    return 0;
}
