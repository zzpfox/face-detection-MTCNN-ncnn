#include <algorithm>
#include <vector>

#include "net.h"

struct SBoundingBox
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool bExist;
    float ppoint[10];
    float regreCoord[4];
};

struct SOrderScore
{
    float score;
    int oriOrder;
};

class CMtcnn
{
public:
    CMtcnn();
    void Detect(ncnn::Mat& img_, std::vector<SBoundingBox>& finalBbox);

private:
    void GenerateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<SBoundingBox>& boundingBox_, std::vector<SOrderScore>& bboxScore_, float scale);
    void Nms(std::vector<SBoundingBox> &boundingBox_, std::vector<SOrderScore> &bboxScore_, const float overlap_threshold, std::string modelname = "Union");
    void RefineAndSquareBbox(std::vector<SBoundingBox> &vecBbox, const int &height, const int &width);

private:
    ncnn::Net m_Pnet;
    ncnn::Net m_Rnet;
    ncnn::Net m_Onet;
    ncnn::Mat m_img;

    const float nms_threshold[3] = { 0.5f, 0.7f, 0.7f };
    const float threshold[3] = { 0.6f, 0.6f, 0.6f };
    const float mean_vals[3] = { 127.5, 127.5, 127.5 };
    const float norm_vals[3] = { 0.0078125, 0.0078125, 0.0078125 };
    std::vector<SBoundingBox> firstBbox_;
    std::vector<SBoundingBox> secondBbox_;
    std::vector<SBoundingBox> thirdBbox_;
    std::vector<SOrderScore> firstOrderScore_, secondBboxScore_, thirdBboxScore_;
    int m_ImgWidth;
    int m_ImgHeight;
};
