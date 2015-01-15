#include <opencv2\opencv.hpp>
#include <iostream>
#include <string>
#include <windows.h>
#include <direct.h>
#include <time.h>
#include <queue>
#include <vector>
#include <sstream>
#include "QPBO.h"
#undef ERROR
#include "glog/logging.h"
#undef min
#undef max
#include "opencv2/gpu/gpu.hpp"
#include <boost/thread/thread.hpp>
#include "boost/filesystem.hpp"
#include <boost/lexical_cast.hpp>
/*
 *Boost Filesystem库提供了对路径、文件和目录进行查询和操作提供了可一直的工具 所有的内容都处于名空间 boost::filesystem 中
 * boost::lexical_cast<string>(i)  把（）中的数据类型转变成< >中的数据类型
 */
namespace fs = boost::filesystem;

#define CV_CVX_WHITE    CV_RGB(0xff,0xff,0xff)
#define CV_CVX_BLACK    CV_RGB(0x00,0x00,0x00)
#define MAX_SYNOPSIS_LENGTH 400
using namespace cv;
using namespace std;

vector<bool> matToVector(Mat &input);

//运动对象结构体
struct ObjectCube{
	int start;						//开始对应的帧序号
	int end;							//结束对应的帧序号
	vector<vector<bool>> objectMask;	//运动序列对应的bool值编码
};

class VideoAbstraction{
public:
	VideoAbstraction(string inputpath, string out_path, string log_path, string config_path, string index_path, string videoname, string midname, int size);
	VideoAbstraction();
	void init();
	//~VideoAbstraction();
	int scaleSize;
	int framePerSecond;				//帧数/秒
	int maxLength;					//视频的最长长度
	int curMaxLength;					//当前处理的视频的最长长度
	int maxLengthToSpilt;				//视频长度限制阈值，大于这个长度将会被分割
	int cacheCollision[33][33][100];	//冲突计算的缓存
	int cacheShift;		
	int frameWidth,frameHeight; 
	int sumLength;					//压缩后视频的总长度
	int objectarea;					//凸包最小的面积
	
	VideoCapture videoCapture;			//视频读写
	VideoWriter videoWriter;
	VideoWriter indexWriter;           //存储对应每一帧上的像素对应的事件的序号
	
	int backgroundSubtractionMethod;	//前背景分离  1-高斯混合模型 2-ViBe
	int LEARNING_RATE;		
	//vector<ObjectCube> objectVector;	//存储多个运动序列，当存储数目达到motionToCompound 后，写入本地文件
	vector<ObjectCube> partToCompound;	//存储一个运动对象的运动序列 
	Mat backgroundImage;				//存储 混合高斯模型提取出的 背景信息------分割程序运行的第二步需要操作的部分
	Mat currentStartIndex,currentEndIndex;

	//path string inputpath, string out_path, string log_path, string config_path, string index_path, string videoname, string midname
	string Inputpath;					//处理的视频的路径、名字、中间文件的名字
	string Outpath;					//处理后的视频输出路径
	string Logpath;
	string Configpath;
	string Indexpath;
	//name
	string InputName;					//输入的视频文件的名字
	string MidName;					//配置文件/凸包 文件名
	string OutputName;

	int ObjectCubeNumber;				//包含运动序列的帧的总数量
	vector<int> frame_start;			//记录所有运动序列的开始帧序号/结束帧序号
	vector<int> frame_end;
	//vector<string> allObjectCube;		//存储所有运动序列的凸包信息 
	int loadIndex;					//load凸包运动信息的偏移参数

	string videoFormat;
	vector<Mat> compoundResult;
	vector<Mat> indexs,indexe;

	ObjectCube currentObject;			//外部传进的当前运动帧的信息
	int detectedMotion;				//检测到的运动序列的个数，每读够 motionToCompound个运动序列，写入一次文件
	int motionToCompound;				//写入文件最少的运动序列数目+合成的运动序列最少数目
	//Abstraction Model Part
	int sum,thres;					//统计有运动序列的帧的数量  &  运动提取的阈值
	int currentLength,tempLength;		//

	//***    GPU    ***//
	gpu::MOG2_GPU gpumog;				//调用混合高斯模型的类
	gpu::GpuMat gpuFrame;				//存储视频帧
	gpu::GpuMat gpuForegroundMask;		//存储视频的前景信息 
	gpu::GpuMat gpuBackgroundImg;		//存储视频的背景信息

	bool useGpu;
	bool useROI;
	bool useIndex;
	Rect rectROI;

	BackgroundSubtractorMOG2 mog;
	Mat gFrame;						//存储视频帧
	Mat gForegroundMask;				//存储视频的前景信息 
	Mat gBackgroundImg;				//存储视频的背景信息
	int noObjectCount;					//存储无运动序列的连续帧的帧数量
	bool flag;						//flag 为true表示当前帧上面有运动物体，若为false, 则当前帧上没有运动物体，是背景帧
	Mat currentMask;	

	string int2string(int _Val);
	vector<vector<Point>> stringToContors(string ss);										//将字符串-->凸包信息
	string contorsToString(vector<vector<Point>> &contors);
	void setVideoFormat(string Format);
	void postProc(Mat& frame);															//对得到的帧图像进行后处理(去噪 连通区域处理 凸包计算 膨胀腐蚀)							
	void ConnectedComponents(int frameindex, Mat &mask,int thres);										//凸包计算---连通区域计算算法的实现
	void stitch(Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end, int frameno);	//拼接图片帧序列并输出
//	void stitch(Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end,vector<vector<Point>>& re_contours, bool& flag);	//拼接图片帧序列并输出
//	void stitch(Mat &input1,Mat &input2,Mat &output,Mat &back,vector<bool> &mask,int start,int end);
	Mat join(Mat &input1,Mat &input2,int type);											// Returns: cv::Mat 合并两幅图像，Parameter: int type 1表示灰度图，3表示彩色图  
																					//将图片2水平添加到图片1的右侧部分合成同一张图片
	Mat join(Mat &input1,Mat &input2,Mat &input3,int type);								//合并3张图像
	double random(double start, double end);												//返回start到end之间的随机数
	int computeMaskCollision(Mat &input1,Mat &input2);									// 返回两幅图像中冲突的像素个数
	int computeMaskCollision(vector<bool> &input1,vector<bool> &input2);						//overloaded function

	int computeObjectCollision(ObjectCube &ob1,ObjectCube &ob2,int shift,string path="");		// Returns:  int 返回在当前时间偏移下两个事件的冲突像素点的个数  
																					//obj1：事件1 obj2:事件2 shift 事件2相对事件1的时间偏移  string path 事件存储目录
	int graphCut(vector<int> &shift,vector<ObjectCube> &ob,int step=5);						//给定运动序列，使用 graph_cut 算法计算其时间偏移
	int ComponentLable(Mat& fg_mask, vector<Rect>& vComponents_out, int area_threshold);
	bool isSimilarRects(const Rect& r1, const Rect& r2, double eps);
	double rectsOverlapAreaRate(const Rect& r1, const Rect& r2);
	static void on_mouse(int event,int x,int y,int flags,void* param);
	
	void saveObjectCube(ObjectCube &ob);													//存放运动序列到本地
	int saveRemainObject();															//保存多出来的运动序列信息并返回包含有运动序列的图像帧的个数 赋值给 ObjectCubeNumber
	void loadObjectCube(int index_start, int index_len);									//读取本地存放的运动序列
	void saveConfigInfo();																//存放配置信息
	void LoadConfigInfo();																//读取本地存放的配置信息
	void LoadConfigInfo(int frameCountUsed);												//读取本地存放的配置信息

	void Abstraction(Mat& currentFrame, int frameIndex);									//进行背景减除的函数 （当前处理的帧， 当前帧的编号）
	void compound(string savepath);														//视频合成函数

	void freeObject();																	//显式的析构功能函数
	void setGpu(bool isgpu);
	void setROI(bool isroi);
	void setIndex(bool isindex);
	void writeMask(Mat& input, Mat& output, int index);


	//文件存取格式：
	//文件名：frame_Num.txt
	//内容：每个mask占一行：“事件号  contorsToString(contors)”

	//将结果视频每一帧中的凸包保存到文件
	//函数调用有先后顺序，后来的事件有可能覆盖先来的事件
	//frame_Num  结果视频序号
	//mask  每帧结果视频由多个事件合成，mask来自其中一个事件。
	//indexOfMask   事件序号
	bool saveContorsOfResultFrameToFile(int frame_Num, cv::Mat& mask, int indexOfMask);

	//从文件中读取contors信息，合成结果视频中frame_Num帧对应的事件Mask信息
	//ISSUE: 文件中读出的事件序号很可能大于255，如何处理？
	//lookupTable: 生成的mask中每个像素有一个0-8的数值，使用数值下标在lookupTable中取出实际的事件序号（1-8有效）
	cv::Mat loadContorsOfResultFrameFromFile(int frame_Num, int width, int height, vector<int>& lookupTable);
};