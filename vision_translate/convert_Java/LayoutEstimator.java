//add labrary
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.dnn.Dnn;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

//add function
public Mat TensorflowForward(Mat inputImg)
{
    if (m_network.empty())
    {
        throw new RuntimeException("Please load pre-trained model for layout estimation");
    }

    Scalar mean = new Scalar(0, 0, 0);
    boolean swapRb = true;

    //! [Create a 4D blob from a frame]
    Mat blob = Dnn.blobFromImage(inputImg, LAYOUT_SCALE, new Size2i(LAYOUT_W, LAYOUT_H), mean, swapRb, false);

    //! [Set input blob]
    m_network.setInput(blob, "input");

    //! [Make forward pass]
    if (!m_prob.empty())
    {
        m_prob.release();
    }
    m_prob = m_network.forward("output/truediv");

    //! [Get heatmap]
    ArrayList<Mat> heat = new ArrayList<>();
    Dnn.imagesFromBlob(m_prob, heat);
    if (heat.get(0).empty())
    {
        return Mat();
    }

    // Remove first channel: background class
    ArrayList<Mat> planes = new ArrayList<>();
    Core.split(heat.get(0), planes);
    planes.erase(planes.begin());

    // Merge channels
    Mat heatMapImage = new Mat();
    Core.merge(planes, heatMapImage);
    Core.normalize(heatMapImage, heatMapImage, 0, 255, Core.NORM_MINMAX);
    heatMapImage.convertTo(heatMapImage, CvType.CV_8UC3);

    EnhanceCeilFloorEdge(heatMapImage);
    Imgproc.resize(heatMapImage, heatMapImage, new Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT));

    return std::move(heatMapImage);
}

public Mat EnhanceCeilFloorEdge(Mat heatMapImage)
{
    Mat grayImage = new Mat();
    Imgproc.cvtColor(heatMapImage, grayImage, Imgproc.COLOR_BGR2GRAY);

    Mat binaryImage = new Mat();
    Imgproc.threshold(grayImage, binaryImage, 0, 255, Imgproc.THRESH

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

public void loadPreTrainedModel() {
    long t1 = System.currentTimeMillis();
    m_network = Dnn.readNetFromTensorflow(m_strWeightPath);
    m_network.setPreferableBackend(Dnn.DNN_BACKEND_OPENCV);
    m_network.setPreferableTarget(Dnn.DNN_TARGET_CPU);
    long t2 = System.currentTimeMillis();
    long diff = t2 - t1;
    printLog(LOG_D, "Network configuration time for layout estimation: %ld ms%n", diff);
}

public void reset() {
    mValidCount = 0;
    m_vLayoutResult.clear();
    m_vLayoutResult = new Vector<>(VALID_COUNT_TH);

    // remove save data
    new File(ROOM_LAYOUT_HEATMAP_PATH).delete();
}

public RoomLayoutResult getLayoutResult() {
    return m_FinalResult;
}

public boolean isRunMode() {
    return mResetFlag || mValidCount < VALID_COUNT_TH;
}

public boolean updateFinalResult(RoomLayoutResult result) {
    boolean isUpdated = false;
    if (!isRunMode()) {
        // Skip when mValidCount is 3
        return isUpdated;
    }

    if (result.roomShape == SHAPE.OTHER) {
        for (final var prevValue : m_vLayoutResult) {
            // Exception: the previous shape is valid value(Rect,Square). then, Do not update!
            if (prevValue.roomShape == SHAPE.RECT || prevValue.roomShape == SHAPE.SQUARE) {
                // No Update
                return isUpdated;
            }
        }

        // In normal case, Update.
        m_FinalResult = result;
    } else {
        m_vLayoutResult[mValidCount] = result;
        mValidCount++;
        m_FinalResult = accumulateResults();

        // Save Current Result
        saveCurrentStatus(m_FinalResult.heatMap, mValidCount);
        m_CLayoutComparator.clear();
    }
    isUpdated = true;

    return isUpdated;
}

public class CLayoutEstimator
{
    public RoomLayoutResult AccumulateResults()
    {
        // Step1. accumulate & mean
        Mat accuHeatMapImage = Mat.zeros(new Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT), CvType.CV_32FC3);
        for (int iter = 0; iter < mValidCount; iter++)
        {
            var data = m_vLayoutResult.get(iter);
            Mat dataHeatMap = data.heatMap.clone();
            dataHeatMap.convertTo(dataHeatMap, CvType.CV_32FC3);
            Core.accumulate(dataHeatMap, accuHeatMapImage);
        }
        accuHeatMapImage /= float(mValidCount);
        accuHeatMapImage.convertTo(accuHeatMapImage, CvType.CV_8UC3);

        // Step2. calculate vertical edges & decide room info
        return GetRoomInfo(accuHeatMapImage);
    }

    public Mat TensorflowForward(Mat inputImg)
    {
        if (m_network.empty())
        {
            throw new RuntimeException("Please load pre-trained model for layout estimation");
        }

        Scalar mean = new Scalar(0, 0, 0);
        boolean swapRb = true;

        //! [Create a 4D blob from a frame]
        Mat blob = Dnn.blobFromImage(inputImg, LAYOUT_SCALE, new Size2i(LAYOUT_W, LAYOUT_H), mean, swapRb, false);

        //! [Set input blob]
        m_network.setInput(blob, "input");

        //! [Make forward pass]
        if (!m_prob.empty())
        {
            m_prob.release();
        }
        m_prob = m_network.forward("output/truediv");

        //! [Get heatmap]
        ArrayList<Mat> heat = new ArrayList<>();
        Dnn.imagesFromBlob(m_prob, heat);
        if (heat.get(0).empty())
        {
            return Mat();
        }

        // Remove first channel: background class
        ArrayList<Mat> planes = new ArrayList<>();
        Core.split(heat.get(0), planes);
        planes.erase(planes.begin());

        // Merge channels
        Mat heatMapImage = new Mat();
        Core.merge(planes, heatMapImage);
        Core.normalize(heatMapImage, heatMapImage, 0, 255, Core.NORM_MINMAX);
        heatMapImage.convertTo(heatMapImage, CvType.CV_8UC3);

        EnhanceCeilFloorEdge(heatMapImage);
        Imgproc.resize(heatMapImage, heatMapImage, new Size2i(DEFAULT_IMG_WIDTH, DEFAULT_IMG_HEIGHT));

        return std::move(heatMapImage);
    }
}

private void DecideRoomShape(RoomLayoutResult layoutResult) {
    int layoutSize = layoutResult.data.size();
    if (layoutSize > 3) {
        layoutResult.roomShape = SHAPE.OTHER;
    }

    if (layoutSize == 3) {
        List<RoomLayoutData> sortedByX = new ArrayList<RoomLayoutData>(layoutResult.data);
        Collections.sort(sortedByX, new Comparator<RoomLayoutData>() {
            @Override
            public int compare(RoomLayoutData data1, RoomLayoutData data2) {
                return data1.rect.x < data2.rect.x ? -1 : 1;
            }
        });

        Point centerPoint = new Point(sortedByX.get(1).rect.x, sortedByX.get(1).rect.y);
        double centerDist = sortedByX.get(1).distance;

        double cmp1 = CalcCeilLength(sortedByX.get(0).distance, centerDist,
                                     new Point(sortedByX.get(0).rect.x, sortedByX.get(0).rect.y), centerPoint);

        double cmp2 = CalcCeilLength(sortedByX.get(2).distance, centerDist,
                                     new Point(sortedByX.get(2).rect.x, sortedByX.get(2).rect.y), centerPoint);

        layoutResult.roomShape = Math.abs(cmp1 - cmp2) < MAX_DIST_MARGIN ? SHAPE.SQUARE : SHAPE.RECT;
    } else {
        layoutResult.roomShape = SHAPE.RECT;
    }
}

private double CalcCeilLength(double dist1, double dist2, Point p1, Point p2) {
    double powLength = ROOM_HEIGHT_ASSUMPTION * ROOM_HEIGHT_ASSUMPTION;

    double l1 = Math.sqrt(dist1 * dist1 + powLength);
    double l2 = Math.sqrt(dist2 * dist2 + powLength);
    double cosTheta = Math.cos((p1.x * p2.x + p1.y * p2.y) / (l1 + l2));

    double result = Math.sqrt(l1 * l1 + l2 * l2 - 2 * l1 * l2 * cosTheta);

    return result;
}