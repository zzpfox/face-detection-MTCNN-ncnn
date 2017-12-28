#include "Mtcnn.h"
#include "net.h"
#include <cmath>

using namespace std;

bool cmpScore(SOrderScore lsh, SOrderScore rsh)
{
    if (lsh.score < rsh.score)
        return true;
    else
        return false;
}

CMtcnn::CMtcnn()
{
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

void CMtcnn::GenerateBbox(ncnn::Mat score, ncnn::Mat location, std::vector<SBoundingBox>& boundingBox_, std::vector<SOrderScore>& bboxScore_, float scale)
{
    int stride = 2;
    int cellsize = 12;
    int count = 0;
    //score p
    float *p = score.channel(1);//score.data + score.cstep;
    float *plocal = location.data;
    SBoundingBox bbox;
    SOrderScore order;
    for (int row = 0; row<score.h; row++)
    {
        for (int col = 0; col<score.w; col++)
        {
            if (*p>m_threshold[0])
            {
                bbox.score = *p;
                order.score = *p;
                order.oriOrder = count;
                bbox.x1 = round((stride*col + 1) / scale);
                bbox.y1 = round((stride*row + 1) / scale);
                bbox.x2 = round((stride*col + 1 + cellsize) / scale);
                bbox.y2 = round((stride*row + 1 + cellsize) / scale);
                bbox.bExist = true;
                bbox.area = (bbox.x2 - bbox.x1)*(bbox.y2 - bbox.y1);
                for (int channel = 0; channel<4; channel++)
                    bbox.regreCoord[channel] = location.channel(channel)[0];
                boundingBox_.push_back(bbox);
                bboxScore_.push_back(order);
                count++;
            }
            p++;
            plocal++;
        }
    }
}
void CMtcnn::Nms(std::vector<SBoundingBox> &boundingBox_, std::vector<SOrderScore> &bboxScore_, const float overlap_threshold, string modelname)
{
    if (boundingBox_.empty())
    {
        return;
    }
    std::vector<int> heros;
    //sort the score
    sort(bboxScore_.begin(), bboxScore_.end(), cmpScore);

    int order = 0;
    float IOU = 0;
    float maxX = 0;
    float maxY = 0;
    float minX = 0;
    float minY = 0;
    while (bboxScore_.size()>0)
    {
        order = bboxScore_.back().oriOrder;
        bboxScore_.pop_back();
        if (order<0)continue;
        if (boundingBox_.at(order).bExist == false) continue;
        heros.push_back(order);
        boundingBox_.at(order).bExist = false;//delete it

        for (int num = 0; num<boundingBox_.size(); num++)
        {
            if (boundingBox_.at(num).bExist)
            {
                //the iou
                maxX = (boundingBox_.at(num).x1>boundingBox_.at(order).x1) ? boundingBox_.at(num).x1 : boundingBox_.at(order).x1;
                maxY = (boundingBox_.at(num).y1>boundingBox_.at(order).y1) ? boundingBox_.at(num).y1 : boundingBox_.at(order).y1;
                minX = (boundingBox_.at(num).x2<boundingBox_.at(order).x2) ? boundingBox_.at(num).x2 : boundingBox_.at(order).x2;
                minY = (boundingBox_.at(num).y2<boundingBox_.at(order).y2) ? boundingBox_.at(num).y2 : boundingBox_.at(order).y2;
                //maxX1 and maxY1 reuse 
                maxX = ((minX - maxX + 1)>0) ? (minX - maxX + 1) : 0;
                maxY = ((minY - maxY + 1)>0) ? (minY - maxY + 1) : 0;
                //IOU reuse for the area of two bbox
                IOU = maxX * maxY;
                if (!modelname.compare("Union"))
                    IOU = IOU / (boundingBox_.at(num).area + boundingBox_.at(order).area - IOU);
                else if (!modelname.compare("Min"))
                {
                    IOU = IOU / ((boundingBox_.at(num).area<boundingBox_.at(order).area) ? boundingBox_.at(num).area : boundingBox_.at(order).area);
                }
                if (IOU>overlap_threshold)
                {
                    boundingBox_.at(num).bExist = false;
                    for (vector<SOrderScore>::iterator it = bboxScore_.begin(); it != bboxScore_.end(); it++)
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
        boundingBox_.at(heros.at(i)).bExist = true;
}
void CMtcnn::RefineAndSquareBbox(vector<SBoundingBox> &vecBbox, const int &height, const int &width)
{
    if (vecBbox.empty())
    {
        //cout << "Bbox is empty!!" << endl;
        return;
    }
    float bbw = 0, bbh = 0, maxSide = 0;
    float h = 0, w = 0;
    float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
    for (vector<SBoundingBox>::iterator it = vecBbox.begin(); it != vecBbox.end(); it++)
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
void CMtcnn::Detect(ncnn::Mat& img_, std::vector<SBoundingBox>& finalBbox_)
{
    m_firstBbox_.clear();
    m_firstOrderScore_.clear();
    m_secondBbox_.clear();
    m_secondBboxScore_.clear();
    m_thirdBbox_.clear();
    m_thirdBboxScore_.clear();

    m_img = img_;
    m_ImgWidth = m_img.w;
    m_ImgHeight = m_img.h;
    m_img.substract_mean_normalize(m_mean_vals, m_norm_vals);

    float minl = m_ImgWidth<m_ImgHeight ? m_ImgWidth : m_ImgHeight;
    int MIN_DET_SIZE = 12;
    int minsize = 90;
    float m = (float)MIN_DET_SIZE / minsize;
    minl *= m;
    float factor = 0.709;
    int factor_count = 0;
    vector<float> scales_;
    while (minl>MIN_DET_SIZE)
    {
        if (factor_count>0)m = m*factor;
        scales_.push_back(m);
        minl *= factor;
        factor_count++;
    }
    SOrderScore order;
    int count = 0;

    for (size_t i = 0; i < scales_.size(); i++)
    {
        int hs = (int)ceil(m_ImgHeight*scales_[i]);
        int ws = (int)ceil(m_ImgWidth*scales_[i]);
        //ncnn::Mat in = ncnn::Mat::from_pixels_resize(image_data, ncnn::Mat::PIXEL_RGB2BGR, img_w, img_h, ws, hs);
        ncnn::Mat in;
        resize_bilinear(m_img, in, ws, hs);
        //in.substract_mean_normalize(mean_vals, norm_vals);
        ncnn::Extractor ex = m_Pnet.create_extractor();
        ex.set_light_mode(true);
        ex.input("data", in);
        ncnn::Mat score_, location_;
        ex.extract("prob1", score_);
        ex.extract("conv4-2", location_);
        std::vector<SBoundingBox> boundingBox_;
        std::vector<SOrderScore> bboxScore_;
        GenerateBbox(score_, location_, boundingBox_, bboxScore_, scales_[i]);
        Nms(boundingBox_, bboxScore_, m_nmsThreshold[0]);

        for (vector<SBoundingBox>::iterator it = boundingBox_.begin(); it != boundingBox_.end(); it++)
        {
            if ((*it).bExist)
            {
                m_firstBbox_.push_back(*it);
                order.score = (*it).score;
                order.oriOrder = count;
                m_firstOrderScore_.push_back(order);
                count++;
            }
        }
        bboxScore_.clear();
        boundingBox_.clear();
    }
    //the first stage's nms
    if (count<1)return;
    Nms(m_firstBbox_, m_firstOrderScore_, m_nmsThreshold[0]);
    RefineAndSquareBbox(m_firstBbox_, m_ImgHeight, m_ImgWidth);
    //printf("firstBbox_.size()=%d\n", firstBbox_.size());

    //second stage
    count = 0;
    for (vector<SBoundingBox>::iterator it = m_firstBbox_.begin(); it != m_firstBbox_.end(); it++)
    {
        if ((*it).bExist)
        {
            ncnn::Mat tempIm;
            copy_cut_border(m_img, tempIm, (*it).y1, m_ImgHeight - (*it).y2, (*it).x1, m_ImgWidth - (*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 24, 24);
            ncnn::Extractor ex = m_Rnet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox;
            ex.extract("prob1", score);
            ex.extract("conv5-2", bbox);
            if (*(score.data + score.cstep)>m_threshold[1])
            {
                for (int channel = 0; channel<4; channel++)
                    it->regreCoord[channel] = bbox.channel(channel)[0];//*(bbox.data+channel*bbox.cstep);
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];//*(score.data+score.cstep);
                m_secondBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                m_secondBboxScore_.push_back(order);
            }
            else
            {
                (*it).bExist = false;
            }
        }
    }
    //printf("secondBbox_.size()=%d\n", secondBbox_.size());
    if (count<1)return;
    Nms(m_secondBbox_, m_secondBboxScore_, m_nmsThreshold[1]);
    RefineAndSquareBbox(m_secondBbox_, m_ImgHeight, m_ImgWidth);

    //third stage 
    count = 0;
    for (vector<SBoundingBox>::iterator it = m_secondBbox_.begin(); it != m_secondBbox_.end(); it++)
    {
        if ((*it).bExist)
        {
            ncnn::Mat tempIm;
            copy_cut_border(m_img, tempIm, (*it).y1, m_ImgHeight - (*it).y2, (*it).x1, m_ImgWidth - (*it).x2);
            ncnn::Mat in;
            resize_bilinear(tempIm, in, 48, 48);
            ncnn::Extractor ex = m_Onet.create_extractor();
            ex.set_light_mode(true);
            ex.input("data", in);
            ncnn::Mat score, bbox, keyPoint;
            ex.extract("prob1", score);
            ex.extract("conv6-2", bbox);
            ex.extract("conv6-3", keyPoint);
            if (score.channel(1)[0]>m_threshold[2])
            {
                for (int channel = 0; channel<4; channel++)
                    it->regreCoord[channel] = bbox.channel(channel)[0];
                it->area = (it->x2 - it->x1)*(it->y2 - it->y1);
                it->score = score.channel(1)[0];
                for (int num = 0; num<5; num++)
                {
                    (it->ppoint)[num] = it->x1 + (it->x2 - it->x1)*keyPoint.channel(num)[0];
                    (it->ppoint)[num + 5] = it->y1 + (it->y2 - it->y1)*keyPoint.channel(num + 5)[0];
                }

                m_thirdBbox_.push_back(*it);
                order.score = it->score;
                order.oriOrder = count++;
                m_thirdBboxScore_.push_back(order);
            }
            else
                (*it).bExist = false;
        }
    }

    //printf("thirdBbox_.size()=%d\n", thirdBbox_.size());
    if (count<1)return;
    RefineAndSquareBbox(m_thirdBbox_, m_ImgHeight, m_ImgWidth);
    Nms(m_thirdBbox_, m_thirdBboxScore_, m_nmsThreshold[2], "Min");
    finalBbox_ = m_thirdBbox_;
}
