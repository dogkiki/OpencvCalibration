// CameraCalibration.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include "stdio.h"
#include "opencv/cv.h"
#include "opencv/cxcore.h"
#include "opencv/highgui.h"
#include "opencv2/opencv.hpp"

#ifdef _DEBUG
#pragma comment(lib,"opencv_core249d.lib")
#pragma comment(lib,"opencv_highgui249d.lib")
#pragma comment(lib,"opencv_imgproc249d.lib")
#else
#pragma comment(lib,"opencv_core249.lib")
#pragma comment(lib,"opencv_highgui249.lib")
#pragma comment(lib,"opencv_imgproc249.lib")
#endif

using namespace cv;
using namespace std;

int ImageStretchByHistogram(IplImage *src, IplImage *dst);

cv::Size m_ChessboardSize = cv::Size(9, 9);		///< 棋盘的大小
int board_n = 9 * 9;

#define  Min_Num 11								///< 最小需要的影像数目

int _tmain(int argc, char* argv[])
{
	if (argc != 2)	return -1;
	//////////////////////////////////////////////////////////////////////////
	//! Get file list
	FILE *fp = fopen(argv[1], "r");	
	if (!fp){ 
		printf("open image list file failed!\n"); 
		return -1; 
	}
	char names[512];
	vector<string> m_sFileList;					///< 总的影像列表
	vector<string> m_RecvIdx;					///< 接受的影像序号
	while (fscanf(fp, "%s ", names) == 1) { m_sFileList.push_back(string(names)); }
	fclose(fp);
	if (m_sFileList.size() < Min_Num)	return -1;
	/************************************************************************/

	//////////////////////////////////////////////////////////////////////////
	//! preprocess	
	printf("Start Finding chess points...\n");
	vector<vector<cv::Point2f>> m_imagePoints;	///< 影像角点坐标
	vector<vector<cv::Point3f>> m_objectPoints;	///< 棋盘对应的三维点的坐标
	CvPoint2D32f* corners = new CvPoint2D32f[board_n];
	for (auto &File_List : m_sFileList)
	{
		vector<cv::Point2f> imageCorners;
		vector<cv::Point3f> objectCorners;

		for (int i = 0; i < m_ChessboardSize.height; i++)
		{
			for (int j = 0; j < m_ChessboardSize.width; j++)
			{
				objectCorners.push_back(cv::Point3f(i, j, 0.0f));		///< 棋盘坐标系 （X,Y,Z）= (i, j, 0)
			}
		}
		IplImage *image = cvLoadImage(File_List.data());				///< 棋盘图像
		IplImage *gray_image = cvCreateImage(cvGetSize(image), 8, 1);	///< subpixel
		cvCvtColor(image, gray_image, CV_BGR2GRAY);
		ImageStretchByHistogram(image, gray_image);

		int corner_count;
		int found = cvFindChessboardCorners(							///< returning non-zero means success
			image, 														///< 8-bit single channel greyscale image.
			m_ChessboardSize, 											///< how many INTERIOR corners in each row and column of the chessboard
			corners, 													///< a pointer to an array where the corner locations can be recorded.
			&corner_count, 												///< optional, if non-NULL, its a point to an integer where the number of corners found can be recorded.
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);
		//! Get Subpixel accuracy on those corners
		cvFindCornerSubPix(gray_image, corners, corner_count, cvSize(11, 11), cvSize(-1, -1), cvTermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));
		cvDrawChessboardCorners(image, m_ChessboardSize, corners, corner_count, found);
		char strFile[512]; strcpy(strFile, File_List.data()); strcpy(strchr(strFile, '.'), "_chessb.jpg");
		cvSaveImage(strFile, image);
		for (int i = 0; i < board_n; ++i)
		{
			imageCorners.push_back(corners[i]);
		}
		if (corner_count != board_n)	continue;

		m_imagePoints.push_back(imageCorners);
		m_objectPoints.push_back(objectCorners);

		m_RecvIdx.push_back(File_List);
	}
	printf("%d images recognized\n", m_RecvIdx.size());
	if (m_RecvIdx.size() < Min_Num)	return -1;
	/************************************************************************/

	//////////////////////////////////////////////////////////////////////////
	//! calibrate		
	printf("Calibrate the images...\n");
	cv::Size imageSize = imread(m_RecvIdx.at(0).data()).size();	///< 取第一张影像的大小

	cv::Mat			m_cameraMatrix;				///< 相机参数矩阵
	cv::Mat			m_distCoeffs;				///< 畸变参数矩阵
	vector<cv::Mat> m_rotation;					///< 旋转矩阵
	vector<cv::Mat> m_transform;				///< 平移矩阵	
	double reProjError = cv::calibrateCamera(
		m_objectPoints,							///< 物方点坐标
		m_imagePoints,							///< 棋盘坐标点
		imageSize,								///< 影像的大小
		m_cameraMatrix,							///< 相机参数矩阵
		m_distCoeffs,							///< 畸变参数矩阵
		m_rotation,								///< 旋转参数矩阵
		m_transform,							///< 平移参数矩阵
		CV_CALIB_FIX_ASPECT_RATIO				///< 标定的方式
		);
	char strCam[512];
	strcpy(strCam, argv[1]); strcpy(strchr(strCam, '.'), "_cam.txt");

	fp = fopen(strCam, "w");
	for (int i = 0; i < m_cameraMatrix.rows; ++i)
	{
		for (int j = 0; j < m_cameraMatrix.cols; ++j)
		{
			fprintf(fp, "%f\t", m_cameraMatrix.at<double>(i, j));
		}
		fprintf(fp, "\n");
	}
	for (int i = 0; i < m_distCoeffs.rows; ++i)
	{
		for (int j = 0; j < m_distCoeffs.cols; ++j)
		{
			fprintf(fp, "%f\t", m_distCoeffs.at<double>(i, j));
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	/************************************************************************/

	printf("Remap and draw the chessboard and axis...\n");

	vector<cv::Point3f> axisCoordWorld = {	///< 坐标轴的世界坐标系
		cv::Point3f(0.0, 0.0, 0.0), cv::Point3f(3.0, 0.0, 0.0), cv::Point3f(0.0, 3.0, 0.0), cv::Point3f(0.0, 0.0, 3.0),
		cv::Point3f(1.0, 0.0, 0.0), cv::Point3f(1.0, 1.0, 0.0), cv::Point3f(0.0, 1.0, 0.0),
		cv::Point3f(1.0, 0.0, 1.0), cv::Point3f(1.0, 1.0, 1.0), cv::Point3f(0.0, 1.0, 1.0), cv::Point3f(0.0, 0.0, 1.0)
	};
	bool m_bNeedInitNudistort = true;			///< 是否已经标定过了
	cv::Mat m_map1, m_map2;						///< 去除图像的畸变x,y投影变换的映射方式矩阵
	//	int i = 0;

	char filepath[_MAX_PATH];		char filesave[_MAX_PATH];
	//	for (auto &Recv_Idx : m_RecvIdx)
	for (int i = 0; i < m_RecvIdx.size(); ++i)
	{
		auto Recv_Idx = m_RecvIdx[i];
		strcpy(filepath, Recv_Idx.data());
		cv::Mat distortImg = cv::imread(filepath);
		//////////////////////////////////////////////////////////////////////////
		//! undistort the image and output
		strcpy(filesave, filepath);
		strcpy(strchr(filesave, '.'), "_undist.jpg");    // output path

		/*char* pS = strrchr(filesave, '\\');
		if (!pS) return -1;
		*pS = 0;   // 这时候文件路径只剩到SourceData了

		char* pS1 = strrchr(filesave, '\\') + 1;
		*pS1 = 0;  //这时候filesave就是....result1\

		sprintf(filesave, "%s%d_undist.jpg", filesave, i);*/

		cv::Mat undistortImg;
		if (m_bNeedInitNudistort)				///< 只做一次标定初始化
		{
			cv::initUndistortRectifyMap(
				m_cameraMatrix,					///< 相机参数矩阵
				m_distCoeffs,					///< 畸变参数矩阵
				cv::Mat(),						///< 可选的rectification矩阵
				cv::Mat(),						///< 用于生成未畸变对象的相机矩阵
				distortImg.size(),				///< 原影像的大小
				CV_32FC1,						///< 输出对象的类型
				m_map1, m_map2					///< x,y坐标映射函数模型
				);
		}
		m_bNeedInitNudistort = false;

		cv::remap(distortImg, undistortImg, m_map1, m_map2, cv::INTER_LINEAR);	if (undistortImg.data == NULL)	continue;
		cv::imwrite(filesave, undistortImg);
		/************************************************************************/

		//////////////////////////////////////////////////////////////////////////
		//! output the image with axis	
		strcpy(filesave, Recv_Idx.data());			strcpy(strchr(filesave, '.'), "_coordi.jpg");
		strcpy(strchr(filepath, '.'), "_chessb.jpg");
		
		cv::Mat distortImg2 = cv::imread(filepath);

		//! 将空间坐标投影到影像平面上
		vector<cv::Point2f> axisCoordImg;		///< 坐标轴的影像坐标
		cv::projectPoints(axisCoordWorld, m_rotation[i], m_transform[i], m_cameraMatrix, m_distCoeffs, axisCoordImg); ///< 影像坐标的转换计算

		//! 将坐标在影像上面画出来
		cv::line(distortImg2, axisCoordImg.at(0), axisCoordImg.at(1), cv::Scalar(255, 0, 0), 2); ///< x
		cv::line(distortImg2, axisCoordImg.at(0), axisCoordImg.at(2), cv::Scalar(0, 255, 0), 2); ///< y
		cv::line(distortImg2, axisCoordImg.at(0), axisCoordImg.at(3), cv::Scalar(0, 0, 255), 2); ///< z

		cv::line(distortImg2, axisCoordImg.at(8), axisCoordImg.at(5), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(8), axisCoordImg.at(7), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(8), axisCoordImg.at(9), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(6), axisCoordImg.at(5), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(6), axisCoordImg.at(9), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(10), axisCoordImg.at(7), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(10), axisCoordImg.at(9), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(4), axisCoordImg.at(5), cv::Scalar(255, 99, 71), 2);
		cv::line(distortImg2, axisCoordImg.at(4), axisCoordImg.at(7), cv::Scalar(255, 99, 71), 2);

		imwrite(filepath, distortImg2); //++i;
		/************************************************************************/
	}
	return 0;
}

/*************************************************
Function:
Description:     因为图像质量差，需要根据直方图进行图像增强，
将图像灰度的域值拉伸到0-255
Calls:
Called By:
Input:           单通道灰度图像
Output:          同样大小的单通道灰度图像
*************************************************/
int ImageStretchByHistogram(IplImage *src, IplImage *dst)
{
	//p[]存放图像各个灰度级的出现概率；
	//p1[]存放各个灰度级之前的概率和，用于直方图变换；
	//num[]存放图象各个灰度级出现的次数;

	assert(src->width == dst->width);
	float p[256], p1[256], num[256];
	memset(p, 0, sizeof(p));
	memset(p1, 0, sizeof(p1));
	memset(num, 0, sizeof(num));

	int height = src->height;
	int width = src->width;
	long wMulh = height * width;

	//! statistics
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			uchar v = ((uchar*)(src->imageData + src->widthStep*y))[x];
			num[v]++;
		}
	}

	//! calculate probability
	for (int i = 0; i < 256; i++)
	{
		p[i] = num[i] / wMulh;
	}

	//! 求存放各个灰度级之前的概率和
	for (int i = 0; i < 256; i++)
	{
		for (int k = 0; k <= i; k++)
			p1[i] += p[k];
	}

	//! histogram transformation  
	for (int x = 0; x < width; x++)
	{
		for (int y = 0; y < height; y++)
		{
			uchar v = ((uchar*)(src->imageData + src->widthStep*y))[x];
			((uchar*)(dst->imageData + dst->widthStep*y))[x] = p1[v] * 255 + 0.5;
		}
	}

	return 0;
}