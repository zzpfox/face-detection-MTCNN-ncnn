#include "Mtcnn.h"
#include "net.h"
#include <cmath>
#include <iostream>

using namespace std;

const float nmsOverlapThreshold[3] = { 0.5f, 0.7f, 0.7f };
const float normalizeImageMean[3] = { 127.5, 127.5, 127.5 };
const float normalizeImageScale[3] = { 0.0078125, 0.0078125, 0.0078125 };

bool cmpScore(SOrderScore lsh, SOrderScore rsh)
{
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

int GetNcnnImageConvertType(imageType type)
{
    switch (type)
    {
    case eBGR888:
    default:
        return ncnn::Mat::PIXEL_BGR2RGB;

    }
}

CMtcnn::CMtcnn()
    : m_ImgFormat(0, 0, eBGR888)
{
    // [TODO] - Refine naming and refactor code
    // Re-implement from the following link
    // https://github.com/kpzhang93/MTCNN_face_detection_alignment/tree/master/code/codes/MTCNNv1
}


void CMtcnn::LoadModel(const char* pNetStructPath, const char* pNetWeightPath, const char* rNetStructPath, const char* rNetWeightPath, const char* oNetStructPath, const char* oNetWeightPath)
{
    m_Pnet.load_param(pNetStructPath);
    m_Pnet.load_model(pNetWeightPath);
    m_Rnet.load_param(rNetStructPath);
    m_Rnet.load_model(rNetWeightPath);
    m_Onet.load_param(oNetStructPath);
    m_Onet.load_model(oNetWeightPath);
}

void CMtcnn::SetParam(const SImageFormat& imgFormat, int iMinFaceSize /*= 90*/, float fPyramidFactor /*= 0.709*/, const float* faceScoreThreshold /*= NULL*/)
{
    m_ImgFormat = imgFormat;
    if (faceScoreThreshold)
    {
        for (int i = 0; i < 3; ++i)
        {
            m_FaceScoreThreshold[i] = faceScoreThreshold[i];
        }
    }

    m_pyramidScale = GetPyramidScale(imgFormat.width, imgFormat.height, iMinFaceSize, fPyramidFactor);
}

std::vector<float> CMtcnn::GetPyramidScale(unsigned int width, unsigned int height, int iMinFaceSize, float fPyramidFactor)
{
    vector<float> retScale;
    float minl = width < height ? width : height;
    float MIN_DET_SIZE = 12;

    float m = MIN_DET_SIZE / iMinFaceSize;
    minl = minl * m;

    while (minl > MIN_DET_SIZE)
    {
        if (!retScale.empty())
        {
            m = m * fPyramidFactor;
        }

        retScale.push_back(m);
        minl = minl * fPyramidFactor;
    }

    return std::move(retScale);
}

void CMtcnn::ConvertToSMtcnnFace(const std::vector<SFaceProposal>& src, vector<SMtcnnFace>& dst)
{
    SMtcnnFace tmpFace;
    for (auto it = src.begin(); it != src.end(); it++)
    {
        if (it->bExist)
        {
            tmpFace.score = it->score;
            tmpFace.boundingBox[0] = it->x1;
            tmpFace.boundingBox[1] = it->y1;
            tmpFace.boundingBox[2] = it->x2;
            tmpFace.boundingBox[3] = it->y2;

            for (int i = 0; i < 10; ++i)
                tmpFace.landmark[i] = (int)(it->ppoint[i]);

            dst.push_back(tmpFace);
        }
    }
}

std::vector<SFaceProposal> CMtcnn::PNetWithPyramid(const ncnn::Mat& img, const std::vector<float> pyramidScale)
{
    std::vector<SFaceProposal> firstBbox;
    std::vector<SOrderScore> firstOrderScore;
    SOrderScore order;

    for (size_t i = 0; i < m_pyramidScale.size(); ++i)
    {
        ncnn::Mat nnFaceScore;
        ncnn::Mat nnFaceBoundingBox;
        std::vector<SFaceProposal> faceRegions;
        std::vector<SOrderScore> faceScore;

        int hs = (int)ceil(m_ImgFormat.height * m_pyramidScale[i]);
        int ws = (int)ceil(m_ImgFormat.width * m_pyramidScale[i]);
        ncnn::Mat pyramidImg;
        resize_bilinear(img, pyramidImg, ws, hs);
        ncnn::Extractor ex = m_Pnet.create_extractor();
        ex.set_light_mode(true);
        // [TODO] - Check if need to set_num_threads
        ex.input("data", pyramidImg);
        ex.extract("prob1", nnFaceScore);
        ex.extract("conv4-2", nnFaceBoundingBox);
        ResizeFaceFromScale(nnFaceScore, nnFaceBoundingBox, faceRegions, faceScore, m_pyramidScale[i]);
        Nms(faceRegions, faceScore, nmsOverlapThreshold[0]);

        for (vector<SFaceProposal>::iterator it = faceRegions.begin(); it != faceRegions.end(); it++)
        {
            if ((*it).bExist)
            {
                firstBbox.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = firstOrderScore.size();
                firstOrderScore.push_back(order);
            }
        }
    }

    if (!firstOrderScore.empty())
    {
        Nms(firstBbox, firstOrderScore, nmsOverlapThreshold[0]);
        RefineAndSquareBbox(firstBbox, m_ImgFormat.height, m_ImgFormat.width);
    }

    return std::move(firstBbox);
}

std::vector<SFaceProposal> CMtcnn::RNet(const ncnn::Mat& img, const std::vector<SFaceProposal> PNetResult)
{
    std::vector<SFaceProposal> secondBbox;
    std::vector<SOrderScore> secondBboxScore;
    SOrderScore order;

    for (vector<SFaceProposal>::const_iterator it = PNetResult.begin(); it != PNetResult.end(); it++)
    {
        if ((*it).bExist)
        {
            ncnn::Mat tempImg;
            ncnn::Mat ncnnImg24;
            ncnn::Mat score;
            ncnn::Mat bbox;

            copy_cut_border(img, tempImg, (*it).y1, m_ImgFormat.height - (*it).y2, (*it).x1, m_ImgFormat.width - (*it).x2);
            resize_bilinear(tempImg, ncnnImg24, 24, 24);
            ncnn::Extractor ex = m_Rnet.create_extractor();
            ex.set_light_mode(true);
            // [TODO] - Check if need to set_num_threads
            ex.input("data", ncnnImg24);
            ex.extract("prob1", score);
            ex.extract("conv5-2", bbox);

            if (*(score.data + score.cstep)>m_FaceScoreThreshold[1])
            {
                SFaceProposal metadata = *it;

                for (int boxAxis = 0; boxAxis < 4; boxAxis++)
                    metadata.regreCoord[boxAxis] = bbox.channel(boxAxis)[0];    //*(bbox.data+channel*bbox.cstep);

                metadata.area = (metadata.x2 - metadata.x1) * (metadata.y2 - metadata.y1);
                metadata.score = score.channel(1)[0];   //*(score.data+score.cstep);
                secondBbox.push_back(metadata);
                order.score = it->score;
                order.oriOrder = secondBboxScore.size();
                secondBboxScore.push_back(order);
            }
        }
    }

    if (!secondBboxScore.empty())
    {
        Nms(secondBbox, secondBboxScore, nmsOverlapThreshold[1]);
        RefineAndSquareBbox(secondBbox, m_ImgFormat.height, m_ImgFormat.width);
    }

    return std::move(secondBbox);
}

std::vector<SFaceProposal> CMtcnn::ONet(const ncnn::Mat& img, const std::vector<SFaceProposal> RNetResult)
{
    std::vector<SFaceProposal> thirdBbox;
    std::vector<SOrderScore> thirdBboxScore;
    SOrderScore order;

    for (vector<SFaceProposal>::const_iterator it = RNetResult.begin(); it != RNetResult.end(); it++)
    {
        if ((*it).bExist)
        {
            ncnn::Mat tempImg;
            ncnn::Mat ncnnImg48;
            ncnn::Mat score;
            ncnn::Mat bbox;
            ncnn::Mat keyPoint;

            copy_cut_border(img, tempImg, (*it).y1, m_ImgFormat.height - (*it).y2, (*it).x1, m_ImgFormat.width - (*it).x2);
            resize_bilinear(tempImg, ncnnImg48, 48, 48);
            ncnn::Extractor ex = m_Onet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", ncnnImg48);
            ex.extract("prob1", score);
            ex.extract("conv6-2", bbox);
            ex.extract("conv6-3", keyPoint);
            if (score.channel(1)[0] > m_FaceScoreThreshold[2])
            {
                SFaceProposal metadata = *it;

                for (int channel = 0; channel < 4; channel++)
                    metadata.regreCoord[channel] = bbox.channel(channel)[0];
                metadata.area = (metadata.x2 - metadata.x1) * (metadata.y2 - metadata.y1);
                metadata.score = score.channel(1)[0];
                for (int num = 0; num < 5; num++)
                {
                    (metadata.ppoint)[num] = metadata.x1 + (metadata.x2 - metadata.x1) * keyPoint.channel(num)[0];
                    (metadata.ppoint)[num + 5] = metadata.y1 + (metadata.y2 - metadata.y1) * keyPoint.channel(num + 5)[0];
                }

                thirdBbox.push_back(metadata);
                order.score = metadata.score;
                order.oriOrder = thirdBboxScore.size();
                thirdBboxScore.push_back(order);
            }
        }
    }

    if (!thirdBboxScore.empty())
    {
        RefineAndSquareBbox(thirdBbox, m_ImgFormat.height, m_ImgFormat.width);
        Nms(thirdBbox, thirdBboxScore, nmsOverlapThreshold[2], "Min");
    }

    return std::move(thirdBbox);
}

void CMtcnn::ResizeFaceFromScale(ncnn::Mat nnFaceScore, ncnn::Mat nnFaceBoundingBox, std::vector<SFaceProposal>& faceRegions, std::vector<SOrderScore>& faceScores, float scale)
{
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = nnFaceScore.channel(1);//score.data + score.cstep;
    float *plocal = nnFaceBoundingBox.data;
    SFaceProposal faceRegion;
    SOrderScore order;
    for (int row = 0; row<nnFaceScore.h; row++)
    {
        for (int col = 0; col<nnFaceScore.w; col++)
        {
            if (*p > m_FaceScoreThreshold[0])
            {
                faceRegion.score = *p;
                order.score = *p;
                order.oriOrder = count++;
                faceRegion.x1 = round((stride*col + 1) / scale);
                faceRegion.y1 = round((stride*row + 1) / scale);
                faceRegion.x2 = round((stride*col + 1 + cellsize) / scale);
                faceRegion.y2 = round((stride*row + 1 + cellsize) / scale);
                faceRegion.bExist = true;
                faceRegion.area = (faceRegion.x2 - faceRegion.x1)*(faceRegion.y2 - faceRegion.y1);
                for (int channel = 0; channel < 4; channel++)
                    faceRegion.regreCoord[channel] = nnFaceBoundingBox.channel(channel)[0];
                faceRegions.push_back(faceRegion);
                faceScores.push_back(order);
            }
            ++p;
            ++plocal;
        }
    }
}

void CMtcnn::Nms(std::vector<SFaceProposal> &faceRegions, std::vector<SOrderScore> &faceScores, const float overlap_threshold, string modelname)
{
    if (faceRegions.empty())
    {
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(faceScores.begin(), faceScores.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while (faceScores.size()>0)
    {
        order = faceScores.back().oriOrder;
        faceScores.pop_back();
        if (order<0)continue;
        if (faceRegions.at(order).bExist == false) continue;
        heros.push_back(order);
        faceRegions.at(order).bExist = false;//delete it

        for (int num = 0; num < faceRegions.size(); num++)
        {
            if (faceRegions.at(num).bExist)
            {
                //the iou
                maxX = (faceRegions.at(num).x1>faceRegions.at(order).x1) ? faceRegions.at(num).x1 : faceRegions.at(order).x1;
                maxY = (faceRegions.at(num).y1>faceRegions.at(order).y1) ? faceRegions.at(num).y1 : faceRegions.at(order).y1;
                minX = (faceRegions.at(num).x2<faceRegions.at(order).x2) ? faceRegions.at(num).x2 : faceRegions.at(order).x2;
                minY = (faceRegions.at(num).y2<faceRegions.at(order).y2) ? faceRegions.at(num).y2 : faceRegions.at(order).y2;

                //maxX1 and maxY1 reuse 
                maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
                maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;

                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;

                if (!modelname.compare("Union"))
                    IOU = IOU / (faceRegions.at(num).area + faceRegions.at(order).area - IOU);
                else if (!modelname.compare("Min"))
                {
                    IOU = IOU / ((faceRegions.at(num).area<faceRegions.at(order).area) ? faceRegions.at(num).area : faceRegions.at(order).area);
                }

                if (IOU > overlap_threshold)
                {
                    faceRegions.at(num).bExist = false;
                    for (vector<SOrderScore>::iterator it = faceScores.begin(); it != faceScores.end(); it++)
                    {
                        if ((*it).oriOrder == num)
                        {
                            (*it).oriOrder = -1;
                            break;
                        }
                    }
                }
            }
        }
    }

    for (int i = 0; i<heros.size(); i++)
        faceRegions.at(heros.at(i)).bExist = true;
}

void CMtcnn::RefineAndSquareBbox(vector<SFaceProposal> &vecBbox, const int &height, const int &width)
{
    if (vecBbox.empty())
    {
        //cout << "Bbox is empty!!" << endl;
        return;
    }
    float bbw = 0, bbh = 0, maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (vector<SFaceProposal>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
    {
        if ((*it).bExist)
        {
            bbw = (*it).x2 - (*it).x1 + 1;
            bbh = (*it).y2 - (*it).y1 + 1;
            x1 = (*it).x1 + (*it).regreCoord[0] * bbw;
            y1 = (*it).y1 + (*it).regreCoord[1] * bbh;
            x2 = (*it).x2 + (*it).regreCoord[2] * bbw;
            y2 = (*it).y2 + (*it).regreCoord[3] * bbh;

            w = x2 - x1 + 1;
            h = y2 - y1 + 1;

            maxSide = (h>w) ? h : w;
            x1 = x1 + w*0.5 - maxSide*0.5;
            y1 = y1 + h*0.5 - maxSide*0.5;
            (*it).x2 = round(x1 + maxSide - 1);
            (*it).y2 = round(y1 + maxSide - 1);
            (*it).x1 = round(x1);
            (*it).y1 = round(y1);

            //boundary check
            if ((*it).x1<0)(*it).x1 = 0;
            if ((*it).y1<0)(*it).y1 = 0;
            if ((*it).x2>width)(*it).x2 = width - 1;
            if ((*it).y2>height)(*it).y2 = height - 1;

            it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
        }
    }
}

void CMtcnn::Detect(const unsigned char* src, std::vector<SMtcnnFace>& result)
{
    ncnn::Mat ncnnImg = ncnn::Mat::from_pixels(src, GetNcnnImageConvertType(m_ImgFormat.type), m_ImgFormat.width, m_ImgFormat.height);
    ncnnImg.substract_mean_normalize(normalizeImageMean, normalizeImageScale);

    std::vector<SFaceProposal> firstBbox = PNetWithPyramid(ncnnImg, m_pyramidScale);
    std::vector<SFaceProposal> secondBbox = RNet(ncnnImg, firstBbox);
    std::vector<SFaceProposal> thirdBbox = ONet(ncnnImg, secondBbox);

    ConvertToSMtcnnFace(thirdBbox, result);
}

