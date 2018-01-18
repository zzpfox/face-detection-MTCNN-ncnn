#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <vector>
#include <list>
#include <map>
#define NUMBER_OF_WIDER_FACE_EVENT_CATEGORY 62
#define DETECTION_THRESHOLD 0.6

#define ALTORITHM_RESULT_PATH "all-result-120.txt"
#define GROUND_TRUTH_PATH  "D:/data/face detection/wider_face/full/wider_face_split/wider_face_val_bbx_gt.txt"
#define ESTIMATION_RESULT "EstimationResult-120.csv"

using namespace std;

struct BBox
{
    int x1;
    int y1;
    int w;
    int h;
};

// list of bounding box
typedef list<BBox> BBoxList;

// image index -> BBoxList
typedef map<int, BBoxList > GroundTruthMap;

struct SAnalyticResult
{
    SAnalyticResult() : truePositive(0), falsePositive(0), falseNegative(0)
    {
    }

    int truePositive;
    int falsePositive;
    int falseNegative;
};

/**
*   Return the image key for image path
*   @imgPath Example: 12--Group/12_Group_Group_12_Group_Group_12_10.jpg
*/
int GetWiderFaceImageKeyFromPath(const string imgPath)
{
    int index = 0;
    for (int i = imgPath.size() - 1; i > 0; --i)
    {
        if (imgPath[i] == '_')
        {
            string number = imgPath.substr(i + 1, imgPath.size() - 5 - i);
            index = stoi(number);
            break;
        }
    }
    return (imgPath.size() * 200) + index;
}

/**
*   Return wider face event category index for image path
*   @imgPath Example: 12--Group/12_Group_Group_12_Group_Group_12_10.jpg
*/
int GetWiderFaceCategoryIndex(const string imgPath)
{
    int ret = 0;
    sscanf_s(imgPath.c_str(), "%d", &ret);
    return ret;
}

double IOU(const BBox& box1, const BBox& box2)
{
    int maxX1 = max(box1.x1, box2.x1);
    int maxY1 = max(box1.y1, box2.y1);
    int minX2 = min(box1.x1 + box1.w, box2.x1 + box2.w);
    int minY2 = max(box1.y1 + box1.h, box2.y1 + box2.h);
    int w = (minX2 - maxX1 + 1) > 0 ? (minX2 - maxX1 + 1) : 0;
    int h = (minY2 - maxY1 + 1) > 0 ? (minY2 - maxY1 + 1) : 0;
    int intersectionArea = w * h;

    return (double)intersectionArea / ((box1.w * box1.h) + (box2.w * box2.h) - intersectionArea);
}

void ParseGroundResult(const string& groundTruthPath, GroundTruthMap* gtList)
{
    fstream f;
    f.open(groundTruthPath, ios::in);
    string imagePath;
    while (getline(f, imagePath))
    {
        //cout << imagePath << endl;

        int category = GetWiderFaceCategoryIndex(imagePath);
        GroundTruthMap& imageIdx2BBoxList = gtList[category];
        int imageKey = GetWiderFaceImageKeyFromPath(imagePath);
        BBoxList& bboxs = imageIdx2BBoxList[imageKey];

        int bboxSize = 0;
        f >> bboxSize;
        //cout << bboxSize << endl;

        for (int i = 0; i < bboxSize; ++i)
        {
            BBox bbox;
            f >> bbox.x1 >> bbox.y1 >> bbox.w >> bbox.h;
            bboxs.push_back(bbox);
            string dummy;
            getline(f, dummy);
        }
    }
}

void MatchGroundTruth(GroundTruthMap* gtList, const string& algoOutputPath, SAnalyticResult* estimationList)
{
    fstream f;
    f.open(algoOutputPath, ios::in);
    string imagePath;
    while (getline(f, imagePath))
    {
        if (imagePath.empty())
        {
            continue;
        }

        int algoBBoxSize = 0;
        f >> algoBBoxSize;

        int category = GetWiderFaceCategoryIndex(imagePath);
        int imageKey = GetWiderFaceImageKeyFromPath(imagePath);
        auto gtIter = gtList[category].find(imageKey);

        if (gtIter == gtList[category].end())
        {
            string dummy;
            // flush size + 1 line
            for (int i = 0; i <= algoBBoxSize; ++i)
            {
                getline(f, dummy);
            }

            continue;
        }

        BBoxList& gtBBoxList = gtIter->second;
        SAnalyticResult& estimation = estimationList[category];

        int truePositive = 0;


        for (int i = 0; i < algoBBoxSize; ++i)
        {
            BBox algoBox;
            f >> algoBox.x1 >> algoBox.y1 >> algoBox.w >> algoBox.h;
            string dummy;
            getline(f, dummy);

            double maxIOU = 0;
            auto maxIOUIter = gtBBoxList.begin();

            for (auto iter = gtBBoxList.begin(); iter != gtBBoxList.end(); ++iter)
            {
                double iou = IOU(*iter, algoBox);

                if (iou > maxIOU)
                {
                    maxIOU = iou;
                    maxIOUIter = iter;
                }
            }

            if (maxIOU >= DETECTION_THRESHOLD)
            {
                ++truePositive;
                gtBBoxList.erase(maxIOUIter);
            }
        }   // end one image (algorithm result)

        estimation.truePositive += truePositive;
        estimation.falsePositive += algoBBoxSize - truePositive;
        estimation.falseNegative += gtBBoxList.size();
    }
}

void ExportResultToCsv(const string& sCsvPath, const SAnalyticResult* estimationList)
{
    fstream f;
    f.open(sCsvPath, ios::out);

    f << "Event Category, True positive, False Positive, False negative, Precision, Recall, F1-score" << endl;

    int tp = 0;
    int fp = 0;
    int fn = 0;

    for (int i = 0; i < NUMBER_OF_WIDER_FACE_EVENT_CATEGORY; ++i)
    {
        const SAnalyticResult& result = estimationList[i];
        double precision = (double)result.truePositive / (result.truePositive + result.falsePositive);
        double recall = (double)result.truePositive / (result.truePositive + result.falseNegative);
        double f1score = 1.f / ((1.f / precision) + (1.f / recall));
        f << i << ", " << result.truePositive << ", " 
            << result.falsePositive << ", " << result.falseNegative << ", " 
            << precision << ", " << recall << ", " << recall << endl;

        tp += result.truePositive;
        fp += result.falsePositive;
        fn += result.falseNegative;
    }

    double avgPrecision = (double)tp / (tp + fp);
    double avgRecall = (double)tp / (tp + fn);
    double avgF1score = 1.f / ((1.f / avgPrecision) + (1.f / avgRecall));
    f << "Avg precision, Avg Recall, Avg F1-score" << endl;
    f << avgPrecision << ", " << avgRecall << ", " << avgF1score << endl;
    f.close();
}

int main()
{
    string algoOutputPath = ALTORITHM_RESULT_PATH;
    string groundTruthPath = GROUND_TRUTH_PATH;
    SAnalyticResult output[NUMBER_OF_WIDER_FACE_EVENT_CATEGORY];
    GroundTruthMap gtList[NUMBER_OF_WIDER_FACE_EVENT_CATEGORY];

    memset(output, 0, sizeof(output));

    ParseGroundResult(groundTruthPath, gtList);
    MatchGroundTruth(gtList, algoOutputPath, output);
    ExportResultToCsv(ESTIMATION_RESULT, output);
}