#include <iostream>
#include <fstream>

#include "opencv2/highgui/highgui.hpp"
#include "FileIterator.h"
#include "Mtcnn.h"

using namespace std;
using namespace cv;

#define MIN_FACE_SIZE 120
#define OUTPUT_FILE_PATH "all-result-120.txt"
#define WIDER_FACE_ROOT "D:/data/face detection/wider_face/full/WIDER_val/images"

vector<string> ParseWiderFaceDir(const std::string root)
{
    vector<string> ret;
    CFileIterator iter(root);
    while (iter.FindNext())
    {
        string name = iter.FileName();
        if (name[0] != '.')
        {
            ret.push_back(name);
        }
    }

    return move(ret);
}

vector<string> GetImgList(const std::string root)
{
    vector<string> ret;
    CFileIterator iter(root, "*.jpg");
    while (iter.FindNext())
    {
        string name = iter.FileName();
        ret.push_back(name);
    }

    return move(ret);
}

void WriteDetectionResult(const string& img_name, const vector<SMtcnnFace>& reuslt)
{
    fstream f;
    f.open(OUTPUT_FILE_PATH, ios::app);
    f << img_name << endl;
    f << reuslt.size() << endl;

    for (int i = 0; i < reuslt.size(); ++i)
    {
        int x1 = reuslt[i].boundingBox[0];
        int y1 = reuslt[i].boundingBox[1];
        int w = reuslt[i].boundingBox[2] - x1;
        int h = reuslt[i].boundingBox[3] - y1;
        f << x1 << " " << y1 << " " << w << " " << h << endl;
    }

    f.close();
}

int main(int argc, char** argv)
{
    CMtcnn mtcnn;
    double sumOfDetectionTimeInMs = 0;
    int detectImgCount = 0;

    mtcnn.LoadModel("../../../model/det1.param", "../../../model/det1.bin",
        "../../../model/det2.param", "../../../model/det2.bin",
        "../../../model/det3.param", "../../../model/det3.bin");

    string widerFaceRoot = WIDER_FACE_ROOT;
    vector<string> dirs = ParseWiderFaceDir(widerFaceRoot);

    for (int i = 0; i < dirs.size(); ++i)
    {
        string imgRoot = widerFaceRoot + "/" + dirs[i];
        vector<string> imgs = GetImgList(imgRoot);

        for (int j = 0; j < imgs.size(); ++j)
        {
            vector<SMtcnnFace> result;
            Mat frame = imread(imgRoot + "/" + imgs[j]);

            SImageFormat format(frame.cols, frame.rows, eBGR888);
            mtcnn.SetParam(format, MIN_FACE_SIZE, 0.709);
            double t1 = (double)getTickCount();
            mtcnn.Detect(frame.data, result);
            double t2 = (double)getTickCount();
            sumOfDetectionTimeInMs += 1000 * double(t2 - t1) / getTickFrequency();
            ++detectImgCount;

            WriteDetectionResult(dirs[i] + "/" + imgs[j], result);
        }
    }

    return 0;
}
