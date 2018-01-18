#ifndef Mtcnn_h__
#define Mtcnn_h__

#include <algorithm>
#include <vector>
#include "net.h"

struct SMtcnnFace
{
    float score;
    int boundingBox[4];    // x1, y1, x2, y2
    int landmark[10];    // x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
};

enum imageType
{
    eBGR888,    /**< The image is stored using a 24-bit BGR format (8-8-8). */
    eRGB888     /**< The image is stored using a 24-bit RGB format (8-8-8). */
};

struct SImageFormat
{
    SImageFormat(unsigned int width_, unsigned int height_, imageType type_)
        : width(width_)
        , height(height_)
        , type(type_)
    {
    }

    unsigned int width;
    unsigned int height;
    imageType type;
};

struct SFaceProposal
{
    float score;
    int x1;
    int y1;
    int x2;
    int y2;
    float area;
    bool bExist;
    float ppoint[10];    // x1, x2, x3, x4, x5, y1, y2, y3, y4, y5
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

    /**
    *   Set model path of the MTCNN
    *   @param pNetStructPath Path of the PNet struct.
    *   @param pNetWeightPath Path of the PNet weight.
    *   @param rNetStructPath Path of the RNet struct.
    *   @param rNetWeightPath Path of the RNet weight.
    *   @param oNetStructPath Path of the ONet struct.
    *   @param oNetWeightPath Path of the ONet weight.
    */
    void LoadModel(const char* pNetStructPath, const char* pNetWeightPath
                 , const char* rNetStructPath, const char* rNetWeightPath
                 , const char* oNetStructPath, const char* oNetWeightPath);

    /**
    *   Set Parameter of the MTCNN
    *   @param imgFormat input image format(resolution, type...etc) for the Detect().
    *   @param iMinFaceSize Smallest size of face we want to detect. Larger the iMinFaceSize, faster the algorithm.
    *   @param fPyramidFactor scale decay rate between pyramid layer.
    *   @param iThreadNum set thread count for each Neural Network. Default count(-1) is system depended.
    *   @param faceScoreThreshold Threshold of the face classification in RNet, PNet and ONet. Pointer size should be 3, If pointer is NULL, default value will be applied.
    */
    void SetParam(const SImageFormat& imgFormat,
                  int iMinFaceSize = 90, float fPyramidFactor = 0.709,
                  int iThreadNum = -1, const float* faceScoreThreshold = NULL);

    /**
    *   Detect face in the image
    *   @param img input image pointer, format of the image should be consistent with the SetParam().
    *   @param result detection result.
    */
    void Detect(const unsigned char* src, std::vector<SMtcnnFace>& result);

private:
    //Resize bounding box coordinate back to the normal scale
    void ResizeFaceFromScale(ncnn::Mat score, ncnn::Mat location, std::vector<SFaceProposal>& boundingBox_, std::vector<SOrderScore>& bboxScore_, float scale);

    void Nms(std::vector<SFaceProposal> &boundingBox_, std::vector<SOrderScore> &bboxScore_, const float overlap_threshold, std::string modelname = "Union");
    void RefineAndSquareBbox(std::vector<SFaceProposal> &vecBbox, const int &height, const int &width);
    void ConvertToSMtcnnFace(const std::vector<SFaceProposal>& src, std::vector<SMtcnnFace>& dst);
    std::vector<float> GetPyramidScale(unsigned int width, unsigned int height, int iMinFaceSize, float fPyramidFactor);

    //Stage 1 in the paper
    std::vector<SFaceProposal> PNetWithPyramid(const ncnn::Mat& img, const std::vector<float> pyramidScale);

    //Stage 2 in the paper
    std::vector<SFaceProposal> RNet(const ncnn::Mat& img, const std::vector<SFaceProposal> PNetResult);

    //Stage 3 in the paper
    std::vector<SFaceProposal> ONet(const ncnn::Mat& img, const std::vector<SFaceProposal> RNetResult);

private:
    ncnn::Net m_Pnet;
    ncnn::Net m_Rnet;
    ncnn::Net m_Onet;

    std::vector<float> m_pyramidScale;
    SImageFormat m_ImgFormat;
    float m_FaceScoreThreshold[3] = { 0.6f, 0.6f, 0.6f };
    int m_ThreadNum;
};
#endif // Mtcnn_h__
