#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdio>

#include "LayoutEstimator.h"
#include "ModuleAngleCalculator.h"

using std::vector;

// Caution! please load pre-trained model using LoadPreTrainedModel()
CLayoutEstimator::CLayoutEstimator(std::string strWeightPath)
{
    if (strWeightPath.empty())
    {
        m_strWeightPath = ROOM_LAYOUT_WEIGHTS_PATH;
    }
    else
    {
        m_strWeightPath = strWeightPath;
    }
    m_FinalResult = RoomLayoutResult();
    m_vLayoutResult = vector<RoomLayoutResult>(VALID_COUNT_TH);
    mValidCount = 0;
    mResetFlag = true;  // 전원 켜지고 class 생성 시 기존 모델 변경 여부 무조건 확인

    // Create Class using UPDATE
    m_CLayoutComparator = CLayoutComparator();

    // distortion coefficient array, used for undistorting image
    WriteInitLogFile();

    auto tempResult = LoadCurrentStatus();
    if (!tempResult.data.empty())
    {
        m_FinalResult = tempResult;
    }
}

void CLayoutEstimator::LoadPreTrainedModel()
{
    auto t1 = std::chrono::steady_clock::now();
    m_network = cv::dnn::readNetFromTensorflow(m_strWeightPath);
    m_network.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    m_network.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    printLog(LOG_D, "Network configuration time for layout estimation: %ld ms\n", diff.count());
}

void CLayoutEstimator::Reset()
{
    mValidCount = 0;
    m_vLayoutResult.clear();
    m_vLayoutResult = vector<RoomLayoutResult>(VALID_COUNT_TH);

    // remove save data
    std::remove(ROOM_LAYOUT_HEATMAP_PATH.c_str());
}

RoomLayoutResult CLayoutEstimator::GetLayoutResult()
{
    return m_FinalResult;
}

bool CLayoutEstimator::IsRunMode()
{
    return mResetFlag || mValidCount < VALID_COUNT_TH;
}

bool CLayoutEstimator::UpdateFinalResult(RoomLayoutResult& result)
{
    bool isUpdated = false;
    if (!IsRunMode())
    {
        // Skip when mValidCount is 3
        return isUpdated;
    }

    if (result.roomShape == SHAPE::OTHER)
    {
        for (const auto& prevValue : m_vLayoutResult)
        {
            // Exception: the previous shape is valid value(Rect,Square). then, Do not update!
            if (prevValue.roomShape == SHAPE::RECT || prevValue.roomShape == SHAPE::SQUARE)
            {
                // No Update
                return isUpdated;
            }
        }

        // In normal case, Update.
        m_FinalResult = result;
    }
    else
    {
        m_vLayoutResult[mValidCount] = result;
        mValidCount++;
        m_FinalResult = AccumulateResults();

        // Save Current Result
        SaveCurrentStatus(m_FinalResult.heatMap, mValidCount);
        m_CLayoutComparator.Clear();
    }
    isUpdated = true;

    return isUpdated;
}

RoomLayoutResult CLayoutEstimator::AccumulateResults()
{
    // Step1. accumulate & mean
    cv::Mat accuHeatMapImage = cv::Mat::zeros(cv::Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT), CV_32FC3);
    for (int iter = 0; iter < mValidCount; iter++)
    {
        auto& data = m_vLayoutResult[iter];
        cv::Mat dataHeatMap = data.heatMap.clone();
        dataHeatMap.convertTo(dataHeatMap, CV_32FC3);
        cv::accumulate(dataHeatMap, accuHeatMapImage);
    }
    accuHeatMapImage /= float(mValidCount);
    accuHeatMapImage.convertTo(accuHeatMapImage, CV_8UC3);

    // Step2. calculate vertical edges & decide room info
    return GetRoomInfo(accuHeatMapImage);
}

cv::Mat CLayoutEstimator::TensorflowForward(cv::Mat& inputImg)
{
    if (m_network.empty())
    {
        throw std::runtime_error("Please load pre-trained model for layout estimation");
    }

    cv::Scalar mean(0, 0, 0);
    bool swapRb = true;

    //! [Create a 4D blob from a frame]
    cv::Mat blob = cv::dnn::blobFromImage(inputImg, LAYOUT_SCALE, cv::Size2i(LAYOUT_W, LAYOUT_H), mean, swapRb, false);

    //! [Set input blob]
    m_network.setInput(blob, "input");

    //! [Make forward pass]
    if (!m_prob.empty())
    {
        m_prob.release();
    }
    m_prob = m_network.forward("output/truediv");

    //! [Get heatmap]
    vector<cv::Mat> heat;
    cv::dnn::imagesFromBlob(m_prob, heat);
    if (heat[0].empty())
    {
        return cv::Mat();
    }

    // Remove first channel: background class
    vector<cv::Mat> planes;
    cv::split(heat[0], planes);
    planes.erase(planes.begin());

    // Merge channels
    cv::Mat heatMapImage;
    cv::merge(planes, heatMapImage);
    cv::normalize(heatMapImage, heatMapImage, 0, 255, cv::NORM_MINMAX);
    heatMapImage.convertTo(heatMapImage, CV_8UC3);

    EnhanceCeilFloorEdge(heatMapImage);
    cv::resize(heatMapImage, heatMapImage, cv::Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT));

    return std::move(heatMapImage);
}

void CLayoutEstimator::DecideRoomShape(RoomLayoutResult& layoutResult) const
{
    auto layoutSize = layoutResult.data.size();
    if (layoutSize > 3)
    {
        layoutResult.roomShape = SHAPE::OTHER;
    }

    if (layoutSize == 3)
    {
        vector<RoomLayoutData> sortedByX(layoutResult.data);
        std::sort(sortedByX.begin(), sortedByX.end(), [](RoomLayoutData& data1, RoomLayoutData& data2) {
            return data1.rect.x < data2.rect.x;
        });

        cv::Point2i centerPoint = cv::Point2i(sortedByX[1].rect.x, sortedByX[1].rect.y);
        double centerDist = sortedByX[1].distance;

        double cmp1 = CalcCeilLength(sortedByX[0].distance, centerDist,
                                     cv::Point2i(sortedByX[0].rect.x, sortedByX[0].rect.y), centerPoint);

        double cmp2 = CalcCeilLength(sortedByX[2].distance, centerDist,
                                     cv::Point2i(sortedByX[2].rect.x, sortedByX[2].rect.y), centerPoint);

        layoutResult.roomShape = (fabs(cmp1 - cmp2) < MAX_DIST_MARGIN) ? SHAPE::SQUARE : SHAPE::RECT;
    }
    else
    {
        layoutResult.roomShape = SHAPE::RECT;
    }
}

double CLayoutEstimator::CalcCeilLength(double dist1, double dist2, cv::Point2i p1, cv::Point2i p2) const
{
    double powLength = ((ROOM_HEIGHT_ASSUMPTION) * (ROOM_HEIGHT_ASSUMPTION));

    double l1 = std::sqrt((dist1 * dist1) + powLength);
    double l2 = std::sqrt((dist2 * dist2) + powLength);
    double cosTheta = std::cos(((p1.x * p2.x) + (p1.y * p2.y)) / (l1 + l2));

    double result = std::sqrt((l1 * l1) + (l2 * l2) - (2 * l1 * l2 * cosTheta));

    return result;
}

void CLayoutEstimator::DecideRoomLayout(RoomLayoutResult& layoutResult, bool isConnectedLines)
{
    auto layoutSize = layoutResult.data.size();
    if (layoutSize == 0 || !isConnectedLines)
    {
        layoutResult.roomShape = SHAPE::OTHER;
        return;
    }

    // 1. Decide room shape
    DecideRoomShape(layoutResult);

    // 2. Decide location
    if (layoutSize > 1)
    {
        vector<RoomLayoutData> sortedByX(layoutResult.data);
        std::sort(sortedByX.begin(), sortedByX.end(), [](RoomLayoutData& data1, RoomLayoutData& data2) {
            return data1.rect.x < data2.rect.x;
        });

        int sideGap = sortedByX[0].rect.x - (DEFAULT_IMG_WIDTH - sortedByX[layoutSize - 1].rect.x);
        int topGap = sortedByX[0].rect.y - sortedByX[layoutSize - 1].rect.y;
        int heightGap = sortedByX[0].rect.height - sortedByX[layoutSize - 1].rect.height;
        if (abs(sideGap) < MAX_SIZE_MARGIN && abs(topGap) < MAX_SIZE_MARGIN && abs(heightGap) < MAX_SIZE_MARGIN)
        {
            layoutResult.locale = LOCATION::CENTER;
            layoutResult.direction = DIRECTION::MIDDLE;
        }
        else
        {
            layoutResult.locale = LOCATION::CORNER;
        }
    }
    else    // vertical layout's size is only 1
    {
        layoutResult.locale = LOCATION::CORNER;
    }

    // 3. Decide shape of rectangle
    if (LOCATION::CORNER == layoutResult.locale)
    {
        vector<RoomLayoutData> sortedByHeight(layoutResult.data);
        std::sort(sortedByHeight.begin(), sortedByHeight.end(), [](RoomLayoutData& data1, RoomLayoutData& data2) {
            return data1.rect.height < data2.rect.height;
        });

        const int centerX = DEFAULT_IMG_WIDTH / 2;
        layoutResult.direction = (sortedByHeight[0].rect.x > centerX) ? DIRECTION::LEFT : DIRECTION::RIGHT;
    }
}

void CLayoutEstimator::Run(int imgIdx, const cv::Mat& inputImg)
{
    printLog(LOG_D, "%d frame: Run Layout Estimation\n", imgIdx);

    // Step 1. Resize image
    cv::Mat resizedImg;
    cv::resize(inputImg, resizedImg, cv::Size(LAYOUT_W, LAYOUT_H), 0, 0, cv::INTER_AREA);

    auto t1 = std::chrono::steady_clock::now();

    // Step 2. Get Layout heatmap
    // Network Prediction
    cv::Mat heatMapImage = TensorflowForward(resizedImg);
    if (heatMapImage.empty())
    {
        return;
    }

    // Step 3. Get room shape/locale
    auto roomLayoutResult = GetRoomInfo(heatMapImage);
    m_CLayoutComparator.Push(roomLayoutResult);

    // Step 4. Reset & Update Final Result
    if (m_CLayoutComparator.IsSimilarityData())
    {
        if (mResetFlag)
        {
            if (CheckResetData(roomLayoutResult))
            {
                Reset();
                printLog(LOG_I, "%s\n", "RESET Layout!");
            }
            mResetFlag = false;
        }
        bool isUpdated = UpdateFinalResult(roomLayoutResult);
        // For Debugging: Save log & heatmap image
        if (isUpdated)
        {
            SaveDebugLayoutLogFile(imgIdx, roomLayoutResult);
            SaveDebugHeatMapImages(inputImg, imgIdx, roomLayoutResult);
        }
    }
    heatMapImage.release();

    auto t2 = std::chrono::steady_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    printLog(LOG_D, "Total elapsed time for layout estimation: %ld ms\n", diff.count());
}

RoomLayoutResult CLayoutEstimator::GetRoomInfo(const cv::Mat& heatMapImg)
{
    // Step2. calculate vertical edges & decide room info
    vector<cv::Mat> planes;
    cv::split(heatMapImg, planes);

    // Get Green Height Values
    auto estimationResult = GetLayoutHeights(planes);

    // Check Flag to whether the vertical edge is connected to another line or not
    bool isConnectedLines = CheckConnectedLines(planes, estimationResult);

    // Remove Small edge
    estimationResult.erase(std::remove_if(estimationResult.begin(), estimationResult.end(), [heatMapImg](auto& box) {
        return (box.height < static_cast<int>(double(heatMapImg.rows) * MIN_VERT_HEIGHT_RATIO));
    }), estimationResult.end());

    // Convert result type(RoomLayoutResult) & Decide room layout type
    RoomLayoutResult roomLayoutResult;
    roomLayoutResult.heatMap = heatMapImg.clone();

    for (auto& detBox : estimationResult)
    {
        RoomLayoutData layoutData;
        layoutData.angle = CModuleAngleCalculator::CalculateCropRectAngle(detBox);
        if (layoutData.angle < 0.0 || std::isnan(layoutData.angle))
        {
            continue;
        }

        layoutData.distance = GetDistanceFromTable(detBox.height, layoutData.angle);
        layoutData.rect = detBox;
        roomLayoutResult.data.push_back(layoutData);
    }
    DecideRoomLayout(roomLayoutResult, isConnectedLines);

    return std::move(roomLayoutResult);
}

bool CLayoutEstimator::CheckConnectedLines(const vector<cv::Mat>& planes, vector<cv::Rect2i>& verticalEdges)
{
    if (verticalEdges.empty())
    {
        return false;
    }

    const cv::Mat& ceilImage = planes[2];
    const cv::Mat& floorImage = planes[0];
    cv::threshold(ceilImage, ceilImage, PIXEL_THRESHOLD, 255, cv::THRESH_BINARY);
    cv::threshold(floorImage, floorImage, PIXEL_THRESHOLD, 255, cv::THRESH_BINARY);

    vector<cv::Rect2i> temp(verticalEdges.size());
    std::copy(verticalEdges.begin(), verticalEdges.end(), temp.begin());
    // remove unvalid edge
    auto iter = std::remove_if(temp.begin(), temp.end(), [ceilImage, floorImage](cv::Rect2i& edgeRect) {
        cv::Mat cropCeilImg = ceilImage(edgeRect);
        cv::Mat cropFloorImg = floorImage(edgeRect);

        bool isNotConnectCeil = cv::countNonZero(cropCeilImg) == 0;
        bool isNotConnectFloor = cv::countNonZero(cropFloorImg) == 0;

        int mergin = ceilImage.rows / VALID_TOP_DOWN_MARGIN;
        bool isNotVertBoundery = ((ceilImage.rows / 2) > edgeRect.y + edgeRect.height)
                                 ||
                                 ((edgeRect.y >= mergin) || (edgeRect.y + edgeRect.height < ceilImage.rows - mergin));

        return isNotConnectCeil && isNotConnectFloor && (isNotVertBoundery == false);
    });
    temp.erase(iter, temp.end());

    bool isConnected = !temp.empty();
    if (!isConnected || !std::equal(temp.begin(), temp.end(), verticalEdges.begin()))
    {
        verticalEdges.clear();
        verticalEdges.assign(temp.begin(), temp.end());
    }

    return isConnected;
}

vector<cv::Rect2i> CLayoutEstimator::GetLayoutHeights(const vector<cv::Mat>& heatMapImg)
{
    // pre-processing more clear
    cv::threshold(heatMapImg[1], heatMapImg[1], WALL_PIXEL_THRESHOLD, 255, CV_THRESH_BINARY);

    // Find contours
    vector<vector<cv::Point>> vvContours = GetContours(heatMapImg[1]);

    // Get Heights
    vector<cv::Rect2i> vVerticalRect;
    for (auto& contour : vvContours)
    {
        auto minmaxX = std::minmax_element(contour.begin(), contour.end(),
                                           [](const cv::Point2i& lhs, const cv::Point2i& rhs) {
                                               return lhs.x < rhs.x;
                                           });
        auto minmaxY = std::minmax_element(contour.begin(), contour.end(),
                                           [](const cv::Point2i& lhs, const cv::Point2i& rhs) {
                                               return lhs.y < rhs.y;
                                           });

        cv::Rect verticalRect;
        int centerX = ((*minmaxX.second).x + (*minmaxX.first).x) / 2;
        verticalRect.x = centerX >= (EDGE_WIDTH / 2) ? centerX - (EDGE_WIDTH / 2) : 0;   // resize 로 인한 image 번짐 감안.
        verticalRect.y = (*minmaxY.first).y;
        verticalRect.width = (verticalRect.x + EDGE_WIDTH <= DEFAULT_IMG_WIDTH)
                             ? EDGE_WIDTH : EDGE_WIDTH - ((verticalRect.x + EDGE_WIDTH) - DEFAULT_IMG_WIDTH);
        verticalRect.height = (*minmaxY.second).y - (*minmaxY.first).y;

        // remove vertical edge that placed upper location then ceilding edge
        bool isValidPos = CheckUpperVertEdge(heatMapImg, verticalRect);

        if (isValidPos && verticalRect.height > MAX_SIZE_MARGIN)
        {
            vVerticalRect.push_back(verticalRect);
        }
    }

    // Merge Vert. edges that placed similar position
    RemoveAngleResult(vVerticalRect);

    vVerticalRect.erase(std::remove_if(vVerticalRect.begin(), vVerticalRect.end(), [](auto& box) {
        return (box.width == -1);
    }), vVerticalRect.end());

    return vVerticalRect;
}

bool CLayoutEstimator::CheckUpperVertEdge(const vector<cv::Mat>& heatMapImg, const cv::Rect2i& verticalRect)
{
    bool isValidPos = true;

    const cv::Mat& ceilImage = heatMapImg[2];

    int curLowerY = verticalRect.y + verticalRect.height;
    int ceilWidthStep = ceilImage.step1();

    for (int q = curLowerY; q < ceilImage.rows; q++)
    {
        int colIdx = q * ceilWidthStep;
        for (int k = verticalRect.x; k < verticalRect.x + verticalRect.width; k++)
        {

            if (ceilImage.data[colIdx + k] > 128)
            {
                isValidPos = false;
                break;
            }
        }
    }

    return isValidPos;
}

// from CFDetector::RemoveAngleResult()
void CLayoutEstimator::RemoveAngleResult(vector<cv::Rect2i>& vVerticalRect)
{
    auto equivalent = [](const auto& box1, const auto& box2) {

        int right1 = box1.x + box1.width;
        int right2 = box2.x + box2.width;

        if ((right1 >= box2.x) && (right1 <= right2))
        {
            return true;
        }
        if ((box1.x <= right2) && (box1.x >= box2.x))
        {
            return true;
        }
        if ((box1.x <= box2.x) && (right1 >= right2))
        {
            return true;
        }
        if ((box1.x >= box2.x) && (right1 <= right2))
        {
            return true;
        }
        return false;
    };

    int numBox = vVerticalRect.size();
    for (int i = 0; i < numBox; i++)
    {
        for (int j = i + 1; j < numBox; j++)
        {
            if (equivalent(vVerticalRect[i], vVerticalRect[j]) && vVerticalRect[i].width != -1 &&
                vVerticalRect[j].width != -1)
            {
                int lowYPtr = std::max(vVerticalRect[i].y + vVerticalRect[i].height,
                                       vVerticalRect[j].y + vVerticalRect[j].height);

                vVerticalRect[i].x = (vVerticalRect[i].height > vVerticalRect[j].height)
                                     ? vVerticalRect[i].x : vVerticalRect[j].x;
                vVerticalRect[i].y = (vVerticalRect[i].y < vVerticalRect[j].y)
                                     ? vVerticalRect[i].y : vVerticalRect[j].y;
                vVerticalRect[i].height = lowYPtr - vVerticalRect[i].y;

                vVerticalRect[j].width = -1; // 앵간하면 16값으로 되어있음. merge되는 애를 -1로 설정하여 추후 제거
            }
        }
    }
}

double CLayoutEstimator::GetDistanceFromTable(int heigtValue, double degree) const
{
    int index = static_cast<int>(round(degree) / ANGLE_STEP);
    int maxValue = DIST_PIXEL_TABLE_2_3M[index][0];
    int minValue = DIST_PIXEL_TABLE_2_3M[index][MAX_DIST_RANGE - 1];

    if (maxValue >= heigtValue && minValue <= heigtValue)
    {
        for (int i = 0; i < MAX_DIST_RANGE - 1; i++)
        {
            if (heigtValue >= DIST_PIXEL_TABLE_2_3M[index][i + 1] && heigtValue < DIST_PIXEL_TABLE_2_3M[index][i])
            {
                return DISTANCE_TABLE[i];
            }
        }
        return DISTANCE_TABLE[MAX_DIST_RANGE - 1];
    }
    else if (maxValue < heigtValue)
    {
        return DISTANCE_TABLE[0];
    }
    else
    {
        return DISTANCE_TABLE[MAX_DIST_RANGE - 1];
    }
}

// For Debugging
void CLayoutEstimator::WriteInitLogFile()
{
    if (!RESULT_SAVE)
    {
        return;
    }

    std::remove(ROOM_LAYOUT_LOG_PATH.c_str());

    std::ofstream outputFile(ROOM_LAYOUT_LOG_PATH, std::fstream::out | std::fstream::app);
    if (!outputFile)
    {
        printLog(LOG_W, "%s", "layout_logs.txt File Open fail\n");
        return;
    }
    outputFile << "No\tRect(x,y,w,h)\t\tAng\tDist\tSHAPE\tLOC\tDIRECTION\n";
    outputFile << "===========================================================\n";
    outputFile.close();
}

void CLayoutEstimator::SaveDebugLayoutLogFile(int imgIdx, const RoomLayoutResult& result)
{
    if (!RESULT_SAVE)
    {
        return;
    }

    std::ofstream outputFile(ROOM_LAYOUT_LOG_PATH, std::fstream::out | std::fstream::app);
    if (!outputFile)
    {
        printLog(LOG_W, "%s", "layout_logs.txt File Open fail\n");
        return;
    }

    std::string strLayoutInfo = PrintLayoutInfo(result);
    outputFile << std::to_string(imgIdx) << "\t";

    if (result.data.empty())
    {
        outputFile << "(" << std::setw(3) << 0 << "," << std::setw(3) << 0 << ","
                   << std::setw(3) << 0 << "," << std::setw(3) << 0 << ")\t"
                   << std::fixed << std::setprecision(3) << 0.0 << "\t"
                   << std::setprecision(1) << 0.0 << "\t"
                   << strLayoutInfo << "\n";
    }

    bool isFirstLine = true;
    for (const auto& data : result.data)
    {
        if (!isFirstLine)
        {
            outputFile << "\t";
        }
        outputFile << "(" << std::setw(3) << data.rect.x << "," << std::setw(3) << data.rect.y << ","
                   << std::setw(3) << data.rect.width << "," << std::setw(3) << data.rect.height << ")\t"
                   << std::fixed << std::setprecision(3) << data.angle << "\t"
                   << std::setprecision(1) << data.distance << "\t";

        if (isFirstLine)
        {
            outputFile << strLayoutInfo;
            isFirstLine = false;
        }
        outputFile << "\n";
    }

    outputFile.close();
}

std::string CLayoutEstimator::PrintLayoutInfo(const RoomLayoutResult& result)
{
    std::string strShape;
    if (SHAPE::SQUARE == result.roomShape)
    {
        strShape = "SQUARE";
    }
    else
    {
        strShape = (SHAPE::RECT == result.roomShape) ? "RECT" : "OTHER";
    }
    std::string strLoc = (LOCATION::CENTER == result.locale) ? "CENTER" : "CORNER";
    std::string strDirection;
    if (DIRECTION::MIDDLE == result.direction)
    {
        strDirection = "MIDDLE";
    }
    else
    {
        strDirection = (DIRECTION::LEFT == result.direction) ? "LEFT" : "RIGHT";
    }

    return strShape + "\t" + strLoc + "\t" + strDirection;
}

void
CLayoutEstimator::SaveDebugHeatMapImages(const cv::Mat& input, int imgIndex, const RoomLayoutResult& layoutResult) const
{
    if (!RESULT_SAVE)
    {
        return;
    }

    cv::Mat origImg = input.clone();
    cv::resize(origImg, origImg, cv::Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT));
    cv::Mat computed;
    cv::addWeighted(origImg, 0.7, layoutResult.heatMap, 0.5, 0.0, computed);
    cv::RNG rng(12345);

    for (auto& verticalInfo : layoutResult.data)
    {
        cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
        cv::rectangle(computed, verticalInfo.rect, color, 2);
    }
    std::ostringstream ostr("");
    ostr << ROOM_LAYOUT_SAVE_DIR << "vertical_layout_rect_" << imgIndex << ".jpg";
    cv::imwrite(ostr.str(), computed);
    ostr.str("");
    ostr.clear();

    origImg.release();
    computed.release();
}

void CLayoutEstimator::SaveCurrentStatus(const cv::Mat& heatMapImg, int resultCont)
{
    // Step 1. Save layout heatmap
    if (heatMapImg.empty() && resultCont == 0)
    {
        return;
    }

    // Step 1. Encoding
    std::vector<unsigned char> binaryData;
    std::vector<int> param(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 95;//default -> 95

    cv::imencode(".jpg", heatMapImg, binaryData, param);

    // save current count
    binaryData.push_back((unsigned char) (resultCont));

    // Step 2. Save Data
    std::ofstream out(ROOM_LAYOUT_HEATMAP_PATH, std::ofstream::binary);
    out.write((char*) (&binaryData[0]), binaryData.size());
    out.close();

    printLog(LOG_I, "%s", "Saved current status!\n");
}

RoomLayoutResult CLayoutEstimator::LoadCurrentStatus()
{
    // Step 1. Load binary
    std::ifstream in(ROOM_LAYOUT_HEATMAP_PATH, std::ifstream::binary);
    std::vector<unsigned char> binayData;
    if (!in)
    {
        return RoomLayoutResult();
    }

    try
    {
        binayData = std::move(std::vector<unsigned char>(std::istreambuf_iterator<char>(in), {}));
        in.close();
    }
    catch (const std::exception& e)
    {
        printLog(LOG_E, "ERROR: %s\n", e.what());
        std::remove(ROOM_LAYOUT_HEATMAP_PATH.c_str());
        return RoomLayoutResult();
    }

    // Step 2. Decoding heatmap and buffer counter
    if(binayData.size()  == 0)
    {
        std::remove(ROOM_LAYOUT_HEATMAP_PATH.c_str());
        return RoomLayoutResult();
    }
    mValidCount = (int) (binayData.back());
    binayData.pop_back();

    cv::Mat restoreHeatmap;
    try
    {
        restoreHeatmap = imdecode(cv::Mat(binayData), CV_LOAD_IMAGE_COLOR);
    }
    catch (cv::Exception& e)
    {
        printLog(LOG_E, "ERROR: %s\n", e.what());
        std::remove(ROOM_LAYOUT_HEATMAP_PATH.c_str());
    }
    if (restoreHeatmap.empty())
    {
        return RoomLayoutResult();
    }

    // Step 3. Get Room type result
    auto roomLayoutResult = GetRoomInfo(restoreHeatmap);
    for (int iter = 0; iter < mValidCount; iter++)
    {
        m_vLayoutResult[iter] = roomLayoutResult;
    }

    SaveDebugLayoutLogFile(0, roomLayoutResult);
    return std::move(roomLayoutResult);
}

bool CLayoutEstimator::CheckResetData(const RoomLayoutResult& curResult)
{
    const cv::Mat& curHeatmap = curResult.heatMap;
    const cv::Mat& oldImage = m_FinalResult.heatMap;
    if (oldImage.empty() || curHeatmap.empty() || curHeatmap.size() != oldImage.size())
    {
        return false;
    }

    if (curResult.roomShape == SHAPE::OTHER && m_FinalResult.roomShape != SHAPE::OTHER)
    {
        printLog(LOG_I, "%s", "RoomShape is changed!\n");
        return true;
    }

    double diff = 0.0;
    int cols = curHeatmap.cols;
    for (int c = 0; c < cols; c++)
    {
        for (int r = 0; r < curHeatmap.rows; r++)
        {
            int diffBlue = abs(curHeatmap.data[cols * r + c] - oldImage.data[cols * r + c]);
            int diffGreen = abs(curHeatmap.data[cols * r + c + 1] - oldImage.data[cols * r + c + 1]);
            int diffRed = abs(curHeatmap.data[cols * r + c + 2] - oldImage.data[cols * r + c + 2]);
            if (diffBlue > DIFF_PIXEL_TH || diffGreen > DIFF_PIXEL_TH || diffRed > DIFF_PIXEL_TH)
            {
                diff += 1.0;
            }
        }
    }

    double ratio = static_cast<double>(diff / curHeatmap.total());
    printLog(LOG_D, "CheckResetData(): diff=%f\n", ratio);

    return ratio > RESET_HEATMAP_RATIO;
}

/*
 *  Inner Class : For update
 */
CLayoutComparator::CLayoutComparator()
{
    Clear();
}

void CLayoutComparator::Push(const RoomLayoutResult& data)
{
    if (IsFull())
    {
        Clear();
    }

    if (count == 0)
    {
        first = data;
        isChecked = false;
    }
    else
    {
        last = data;

        // Compare data
        size_t validCount = 0;
        size_t size = last.data.size();
        vector<int> vRectChecked(size);
        for (size_t idx = 0; idx < size; idx++)
        {
            if (vRectChecked[idx] == 1)
            {
                continue;
            }
            for (const auto& lastData : last.data)
            {
                if (fabs(first.data[idx].angle - lastData.angle) <= VALID_ANGLE_RANGE)
                {
                    vRectChecked[idx] = 1;
                    validCount++;
                    break;
                }
            }
        }
        isChecked = (size == first.data.size()) && (size == validCount);
    }
    count++;
}

void CLayoutComparator::Clear()
{
    count = 0;
    isChecked = false;
    first = RoomLayoutResult();
    last = RoomLayoutResult();
}

bool CLayoutComparator::IsFull()
{
    return count > 1;
}

bool CLayoutComparator::IsSimilarityData()
{
    return isChecked;
}

void CLayoutEstimator::RefineOutsideAreaInEllipse(cv::Rect& rect, cv::Size2i imgSize)
{
    if (rect.x < 0)
    {
        rect.width += rect.x;
        rect.x = 0;
    }
    if (rect.y < 0)
    {
        rect.height += rect.y;
        rect.y = 0;
    }

    if (rect.x + rect.width >= imgSize.width)
    {
        rect.width = imgSize.width - rect.x;
    }
    if (rect.y + rect.height >= imgSize.height)
    {
        rect.height = imgSize.height - rect.y;
    }

}

vector<vector<cv::Point>> CLayoutEstimator::GetContours(const cv::Mat& binaryImg)
{
    vector<vector<cv::Point>> vvContours;
    vector<cv::Vec4i> vHierarchy;
    cv::findContours(binaryImg, vvContours, vHierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    return std::move(vvContours);
}

vector<cv::RotatedRect> CLayoutEstimator::GetEdgeEllipse(cv::Mat targetHeatMapImg, cv::Mat vertHeatMap)
{
    vector<cv::RotatedRect> vRotatedBox;

    // Step 1. subtract vert. component for discriminating connected two ellipse.
    cv::threshold(targetHeatMapImg, targetHeatMapImg, ELLIPSE_TH, 255, cv::THRESH_BINARY);

    // Step 2. divided instance in heatmap based on vertical line
    cv::threshold(vertHeatMap, vertHeatMap, ELLIPSE_TH, 255, cv::THRESH_BINARY);
    targetHeatMapImg -= vertHeatMap;

    // Step 3. Get Contour and bounding box
    vector<vector<cv::Point>> vvContours = GetContours(targetHeatMapImg);
    for (auto& contour : vvContours)
    {
        if (contour.size() > 5)
        {
            vRotatedBox.push_back(std::move(cv::fitEllipse(contour)));
        }
    }
    // -> sort by x-axis
    std::sort(vRotatedBox.begin(), vRotatedBox.end(), [](cv::RotatedRect& data1, cv::RotatedRect& data2) {
        return data1.center.x < data2.center.x;
    });

    // Step 4. Remove outside ellipse
    vRotatedBox
            .erase(std::remove_if(vRotatedBox.begin(), vRotatedBox.end(), [size = targetHeatMapImg.size()](auto& box) {
                bool isOutBound1 = box.boundingRect().x + box.boundingRect().width <= 0 ||
                                   box.boundingRect().y + box.boundingRect().height <= 0;
                bool isOutBound2 = box.boundingRect().x >= size.width ||
                                   box.boundingRect().y >= size.height;
                bool isSize = box.boundingRect().width < 20;
                return (isOutBound1 || isOutBound2 || isSize);
            }), vRotatedBox.end());

    // Step 5. remove overlapped ellipse based on x-axis
    for (int iter = 0; iter < int(vRotatedBox.size()) - 1; iter++)
    {
        cv::Rect baseBoundRect = vRotatedBox[iter].boundingRect();
        RefineOutsideAreaInEllipse(baseBoundRect, targetHeatMapImg.size());

        if (baseBoundRect.area() <= 1)
        {
            continue;
        }
        for (int q = 0; q < int(vRotatedBox.size()); q++)
        {
            cv::Rect targetBoundRect = vRotatedBox[q].boundingRect();
            if (q == iter || targetBoundRect.area() <= 1)
            {
                continue;
            }

            RefineOutsideAreaInEllipse(targetBoundRect, targetHeatMapImg.size());

            if ((targetBoundRect.x >= baseBoundRect.x
                 && targetBoundRect.x + targetBoundRect.width <= baseBoundRect.x + baseBoundRect.width)
                && baseBoundRect.width > targetBoundRect.width)
            {
                vRotatedBox[q] = cv::RotatedRect();
            }
        }
    }

    vRotatedBox.erase(std::remove_if(vRotatedBox.begin(), vRotatedBox.end(), [](auto& box) {
        return (box.boundingRect().area() <= 1);
    }), vRotatedBox.end());

    return vRotatedBox;
}

void CLayoutEstimator::DrawToConnectedPtr(cv::Mat& srcImg, cv::Point startPtr, cv::Point endPtr, cv::Point connectedPtr,
                                          cv::Scalar color)
{
    line(srcImg, startPtr, connectedPtr, color, 3);
    line(srcImg, connectedPtr, endPtr, color, 3);

    if (connectedPtr.x > endPtr.x)
    {
        line(srcImg, endPtr, connectedPtr, color, 3);
    }
    else if (connectedPtr.x < startPtr.x)
    {
        line(srcImg, startPtr, connectedPtr, color, 3);
    }
}

void CLayoutEstimator::DrawConnectedLineBetweenEcllipses(cv::Mat& srcImg, const cv::RotatedRect& ellipse1,
                                                         const cv::RotatedRect& ellipse2, const cv::Scalar& color)
{
    // Step 1. Get line equation for ellipse1
    cv::Point2f vertices1[4];
    ellipse1.points(vertices1);

    cv::Point2f startPtr1 = (vertices1[1] + vertices1[2]) / 2;
    cv::Point2f endPtr1 = (vertices1[3] + vertices1[0]) / 2;

    double a1 = double(endPtr1.y - startPtr1.y) / double(endPtr1.x - startPtr1.x);
    double b1 = startPtr1.y - (a1 * startPtr1.x);

    // Step 2. Get line equation for ellipse2
    cv::Point2f vertices2[4];
    ellipse2.points(vertices2);

    cv::Point2f startPtr2 = (vertices2[1] + vertices2[2]) / 2;
    cv::Point2f endPtr2 = (vertices2[3] + vertices2[0]) / 2;

    double a2 = double(endPtr2.y - startPtr2.y) / double(endPtr2.x - startPtr2.x);
    double b2 = startPtr2.y - (a2 * startPtr2.x);

    cv::Point2f connectedPtr;
    connectedPtr.x = (b2 - b1) / (a1 - a2);
    connectedPtr.y = a1 * connectedPtr.x + b1;

    // draw line
    DrawToConnectedPtr(srcImg, startPtr1, endPtr1, connectedPtr, color);
    DrawToConnectedPtr(srcImg, startPtr2, endPtr2, connectedPtr, color);
}

void CLayoutEstimator::EnhanceCeilFloorEdge(cv::Mat& heatMap)
{
    //* fit ellipse
    cv::Mat calHeatMap = heatMap.clone();
    vector<cv::Mat> planes;
    cv::split(calHeatMap, planes);

    cv::Mat& ceilEdgeImg = planes[2];
    cv::Mat& vertEdgeImg = planes[1];
    cv::Mat& floorEdgeImg = planes[0];

    vector<cv::RotatedRect> vCeilEllipses = GetEdgeEllipse(ceilEdgeImg, vertEdgeImg);

    if (vCeilEllipses.size() > 1)
    {
        for (unsigned int iter = 0; iter < vCeilEllipses.size() - 1; iter++)
        {
            DrawConnectedLineBetweenEcllipses(heatMap, vCeilEllipses[iter], vCeilEllipses[iter + 1],
                                              cv::Scalar(0, 0, 255));
        }
    }

    vector<cv::RotatedRect> vFloorEllipses = GetEdgeEllipse(floorEdgeImg, vertEdgeImg);
    if (vFloorEllipses.size() > 1)
    {
        for (unsigned int iter = 0; iter < vFloorEllipses.size() - 1; iter++)
        {
            DrawConnectedLineBetweenEcllipses(heatMap, vFloorEllipses[iter], vFloorEllipses[iter + 1],
                                              cv::Scalar(255, 0, 0));
        }
    }
}