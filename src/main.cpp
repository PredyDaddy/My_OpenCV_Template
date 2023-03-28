#include "opencv2/opencv.hpp"
#include <iostream>

// 初始化模型
// 这里用const修饰, config跟weight无法被修改
const std::string config = "./weights/opencv_face_detector.pbtxt";
const std::string weight =  "./weights/opencv_face_detector_uint8.pb";

// API要求: 权重路径, 配置路径 都要求String
cv::dnn::Net net = cv::dnn::readNetFromTensorflow(weight, config);

// 检测并绘制矩形框
void detectDrawRect(cv::Mat &frame) // 取地址直接对image/vedio操作
{
    // 获取图像的宽高
    int frameHeight = frame.rows;
    int frameWidth = frame.cols;

    /*
    blobFromImage 会把数据处理成Blog的数据
    预处理 resize + swapRB + mean + scale
    1.0: 缩放因子，长宽等比例缩放
    Size() 尺寸缩放到(300, 300)
    Scalar对于三个通道的分别减104.0, 177.0, 123.0, 
    这些数值来自于一个已经在大规模图像数据上进行训练的模型，在对图像进行预处理时，这些数值被认为可以帮助提高模型的性能
    false: 是否交换RB通道, 这里不交换
    false: 是否进行裁剪, 这里不裁剪
    */
    cv::Mat inputBlog = cv::dnn::blobFromImage(frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false, false);

    /*
    推理
    setInput() 把data传入进去, data在权重文件里面找到的
    detection_out也是在权重文件里面找到的
    前向推理完成后detection就变成了一个包含多个检测结果的一维数组
    每个检测结果都是(x, y, w, h) 左上角的坐标和宽高 
    将一维数组的指针detection.ptr<float>()传递给矩阵的构造函数，将其转换为二维矩阵detectionMat。
    就可以使用detectionMat对检测结果进行访问
    */
    net.setInput(inputBlog, "data");
    cv::Mat detection = net.forward("detection_out");

    /*
    获取结果
    detection.size[2]: rows
    detection.size[3]: cols
    CV_32F: 指定数据类型为float
    detectionMat是个二维矩阵
    detectionMat.at<float>(i, j)表示第i行、第j列元素的值
    第1列：序号
    第2列：class
    第3列：confidence
    第4-7列：归一化坐标
    */
    cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

    for (int i = 0; i < detectionMat.rows; i++)
    {
        float confidence = detectionMat.at<float>(i, 2);
        if (confidence > 0.2)
        {
            // 两点坐标, 把归一化坐标还原
            int l = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
            int t = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
            int r = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
            int b = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

            // 画框
            cv::rectangle(frame, cv::Point(l, t), cv::Point(r, b), cv::Scalar(0, 255, 0), 2);
        }
    }
}

// 图片测试
void imageTest()
{
    // 读取图片
    cv::Mat img = cv::imread("./media/test.jpg");
    // 推理
    detectDrawRect(img);
    // 保存
    cv::imwrite("./output/face_result1.jpg", img);
}

void videoTest()
{
    // 读取视频流, 实例化对象，传入参数为路径的字符串
    cv::VideoCapture cap("./media/video.mp4");
    // 获取视频流的宽高
    int width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);

    // 实例化写入器
    // 参数: 写入路径, 编码格式(H264), 帧率
    cv::VideoWriter writer("./output/result.mp4", cv::VideoWriter::fourcc('H', '2', '6', '4'), 25, cv::Size(width, height));
    
    if (!cap.isOpened())
    {
        std::cout << "打不开这个视频" << std::endl;
        // 退出
        exit(1);
    }
    cv::Mat frame;

    while (true)
    {   

        if (!cap.read(frame))
        {
            std::cout << "读完了" << std::endl;
            break;
        }

        // flip, 这里可以用于纠正摄像头铺货的镜像画面
        cv::flip(frame, frame, 1);

        // 推理
        detectDrawRect(frame);

        // writer写入
        // writer在这里面就是写入器，他写入的是frame， frame最后储存在result.mp4
        writer.write(frame);
    }
}

int main()
{

    imageTest();
    // videoTest();
    return 0;
}