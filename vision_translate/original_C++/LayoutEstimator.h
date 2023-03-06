#ifndef DETECT_LAYOUTESTIMATOR_H_
#define DETECT_LAYOUTESTIMATOR_H_

#include <chrono>
#include <sstream>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>

#include "DistanceTable.h"
#include "../common/Filepath.h"
#include "../common/GlobalConfig.h"
#include "../utils/LogUtil.h"

// Set undistortion mode
static constexpr bool IS_UNDIST_MODE = false;

static constexpr int LAYOUT_W = 128;
static constexpr int LAYOUT_H = 128;
static constexpr double LAYOUT_SCALE = 1.0 / 255.0;
static constexpr double ROOM_HEIGHT_ASSUMPTION = 2.3;   // 실내 공간 높이 가정 [m]

static constexpr double MAX_DIST_MARGIN = 0.1;          // 최대 산출거리 margin [m]
static constexpr int MAX_SIZE_MARGIN = 16;              // center 여부 확인 및 최소 크기
static constexpr int VALID_COUNT_TH = 3;
static constexpr int EDGE_WIDTH = 16;                   // vertical edge width

static constexpr int ANGLE_STEP = 5;
static constexpr int WALL_PIXEL_THRESHOLD = 100;
static constexpr int PIXEL_THRESHOLD = 125;
static constexpr int DIFF_PIXEL_TH = 50;

static constexpr int VALID_TOP_DOWN_MARGIN = 10;
static constexpr int ELLIPSE_TH =125;

static constexpr double MIN_VERT_HEIGHT_RATIO =0.2;

// Update Logic
static constexpr double RESET_HEATMAP_RATIO = 0.2;
static constexpr double VALID_ANGLE_RANGE = 10.0;       // +5도, -5도

class CLayoutComparator
{
private:
    int count = 0;
    bool isChecked = false;
    RoomLayoutResult first;
    RoomLayoutResult last;

public:
    CLayoutComparator();
    void Push(const RoomLayoutResult& data);
    void Clear();
    bool IsFull();
    bool IsSimilarityData();
};

class CLayoutEstimator
{
public:
    CLayoutEstimator(std::string strWeightPath);
    void LoadPreTrainedModel();

    void Run(int imgIdx, const cv::Mat& inputImg);
    void Reset();

    RoomLayoutResult GetLayoutResult();
    bool IsRunMode();

    int mValidCount;

private:

    bool CheckResetData(const RoomLayoutResult& curResult);
    RoomLayoutResult LoadCurrentStatus();
    void SaveCurrentStatus(const cv::Mat& heatMapImg, int resultCont);

    // Update logic of layout estimation
    RoomLayoutResult AccumulateResults();
    bool UpdateFinalResult(RoomLayoutResult& result);

    RoomLayoutResult GetRoomInfo(const cv::Mat& heatMapImg);
    void DecideRoomLayout(RoomLayoutResult& estimationResult, bool isConnectedLines);

    cv::Mat TensorflowForward(cv::Mat& inputImg);
    std::vector<cv::Rect2i> GetLayoutHeights(const std::vector<cv::Mat>& heatMapImg);
    void RemoveAngleResult(std::vector<cv::Rect2i>& vVerticalRect);
    bool CheckUpperVertEdge(const std::vector<cv::Mat>& heatMapImg, const cv::Rect2i& verticalRect);
    bool CheckConnectedLines(const std::vector<cv::Mat>& planes, std::vector<cv::Rect2i>& verticalEdges);
    void DecideRoomShape(RoomLayoutResult& layoutResult) const;
    double GetDistanceFromTable(int heigtValue, double degree) const;
    double CalcCeilLength(double dist1, double dist2, cv::Point2i p1, cv::Point2i p2) const;

    // post processing
    void EnhanceCeilFloorEdge(cv::Mat& heatMap);
    void DrawConnectedLineBetweenEcllipses(cv::Mat& srcImg, const cv::RotatedRect& ellipse1,
                                           const cv::RotatedRect& ellipse2, const cv::Scalar& color);
    std::vector<cv::RotatedRect> GetEdgeEllipse(cv::Mat targetHeatMapImg, cv::Mat vertHeatMap);
    void DrawToConnectedPtr(cv::Mat& srcImg, cv::Point startPtr, cv::Point endPtr,
                            cv::Point connectedPtr, cv::Scalar color);
    void RefineOutsideAreaInEllipse(cv::Rect& rect, cv::Size2i imgSize);

    std::vector<std::vector<cv::Point>> GetContours(const cv::Mat& binaryImg);

    // For debugging
    void WriteInitLogFile();
    std::string PrintLayoutInfo(const RoomLayoutResult& result);
    void SaveDebugLayoutLogFile(int imgIdx, const RoomLayoutResult& result);
    void SaveDebugHeatMapImages(const cv::Mat& input, int imgIndex, const RoomLayoutResult& layoutResult) const;

    bool mResetFlag;
    CLayoutComparator m_CLayoutComparator;
    std::vector<RoomLayoutResult> m_vLayoutResult;
    RoomLayoutResult m_FinalResult;

    /* OpenCV::dnn variables */
    cv::Mat m_prob;
    cv::dnn::Net m_network;

    std::string m_strWeightPath;
};

#endif /* DETECT_LAYOUTESTIMATOR_H_ */
