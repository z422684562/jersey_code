#include "opencv2\opencv.hpp"
#include <iostream>
using namespace std;
using namespace cv;
int main(int argc, char *argv[])
{
	if( argc != 3)
	{
		cout << 400;
		return -1;
	}
	string file_name = argv[1];
	string file_path = argv[2];
	
	Mat data, labels;
	Mat pic = imread(file_name);
	if (pic.empty()|| !pic.data)
	{
		cout << 400;
		return -1;
	}
	
	for (int i = 0; i < pic.rows; i++)
		for (int j = 0; j < pic.cols; j++)
		{
			Vec3b point = pic.at<Vec3b>(i, j);
			Mat tmp = (Mat_<float>(1, 3) << point[0], point[1], point[2]);
			data.push_back(tmp);
		}

	
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
	imwrite(file_path+"centerized_out.png", pic);
	cout << 200;

	return 0;
}