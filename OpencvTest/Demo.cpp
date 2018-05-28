

#include "opencv2\opencv.hpp"
#include <iostream>
#include <io.h>
using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

	if (argc != 3)
	{
		cout << 400 << '\n' << "参数个数错误";
		return -1;
	}
	string file_name = argv[1];
	string file_path = argv[2];

	Mat data, labels;
	Mat pic = imread(file_name);
	if (pic.empty() || !pic.data)
	{
		cout << 400 << '\n' << "图片不存在";
		return -1;
	}

	if (_access(argv[2],0)==-1)
	{
		cout << 400 << '\n' << "输出文件夹不存在";
		return -1;
	}


	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
		{
			Vec3b point = pic.at<Vec3b>(i, j);
			Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			data.push_back(tmp);
		}

	//抠图
	kmeans(data, 2, labels, TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 10, 1.0),
		3, KMEANS_RANDOM_CENTERS);


	int n = 0;
	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
		{
			int clusterIdx = labels.at<int>(n);
			if (clusterIdx == labels.at<int>(0))
				pic.at<Vec3b>(i, j) = Vec3b(255, 255, 255);

			n++;
		}


	//移位置
	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	Mat src_gray;

	//自定义形态学元素结构  
	cv::Mat element5(9, 9, CV_8U, cv::Scalar(1));//5*5正方形，8位uchar型，全1结构元素    
	cv::Mat closed;
	Rect rect;


	Mat image = pic.clone();

	cvtColor(pic, src_gray, COLOR_BGR2GRAY);
	//使用Canny检测边缘    
	Canny(src_gray, threshold_output, 80, 126, (3, 3));
	//高级形态学闭运算函数    
	cv::morphologyEx(threshold_output, closed, cv::MORPH_CLOSE, element5);
	// 寻找外轮廓轮廓    
	findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	//转换轮廓点到最大外矩形框  
	rect = boundingRect(contours[0]);
	Mat cuted = image(rect);
	resize(cuted, cuted, rect.size() * 800 / rect.width);
	Mat result(2400, 2400, CV_8UC3, Vec3b(255, 255, 255));
	for (int i = 0; i < cuted.rows; i++)
		for (int j = 0; j < cuted.cols; j++)
			result.at<Vec3b>(i + (2400 - cuted.rows) / 2, j + 799) = cuted.at<Vec3b>(i, j);

	int minIdx = 0, maxIdx = 0;
	for (int i = 0; i < 2400; i++)
		if (result.at<Vec3b>(i, 820) != Vec3b(255, 255, 255))
		{
			minIdx = i;
			break;
		}
	for (int i = 2399; i > 0; i--)
		if (result.at<Vec3b>(i, 820) != Vec3b(255, 255, 255))
		{
			maxIdx = i;
			break;
		}
	for (int i = minIdx; i < maxIdx; i++)
		for (int j = 0; j < 800; j++)
			result.at<Vec3b>(i, j) = result.at<Vec3b>(i, j + 1600) = Vec3b(0, 0, 0);
	imwrite(file_path + "centerized_out.png", result);
	cout << 200<<'\n'<<"输出成功";


	return 0;
}