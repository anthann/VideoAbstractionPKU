//****************************************
//user: PianoCoder
//Create date: 2015-01-07
//Class name: Main
//Discription: Test all the function
//Update: none
//****************************************

#include "VideoAbstraction.h"
#include "UserVideoAbstraction.h"
#include <time.h>

#define DIV_NUMBER 9
#define MAX_INDEX_COUNT 300
#define MIN_AREA 500

//declaration
extern int readFrameLog(string logname);
extern void readAreaLog(string logname, int &base_x, int &base_y);
extern Mat MultiImage(const vector<Mat>& SrcImg_V, Size ImgMax_Size);
extern void video_play(long index);
extern void create_path(string path);
static void bar_callback(int index,void* userdata);

//variable definition
string testVideoName;
string log_path;
string state;
//snap-shot 

VideoCapture capture;
Size SubPlot;
char* window_name="img";
char* window_play="video";
int bar_index;//指示视频播放进度条
vector<long> video_index;//用于记录每个快照对应视频帧数
int small_width;//每一个小图的宽度
int small_height;//每个小图的高度
//ROI-set 
//setROI=false by default 
Rect selectarea;
bool select_flag=false;
bool setROI=true;
Mat image,imageRoi,showimage,index_image;
//Video Index
int currentFrameIndex=0;
int frame_weidth;
string ori_video;
string index_video;
int EventNumber;
vector<int> event_start;
vector<int> event_end;
vector<int> event_length;
int event_count[MAX_INDEX_COUNT];

/* calling back event function */
//ROI selection
void mouseSelect(int mouseEvent,int x,int y,int flags,void* param)  
{  
	Point p1,p2;
	if(mouseEvent==CV_EVENT_LBUTTONDOWN){
		selectarea.x=x;
		selectarea.y=y;
		select_flag=true;
	}
	else if(select_flag && mouseEvent==CV_EVENT_MOUSEMOVE){
		image.copyTo(showimage);
		p1=Point(selectarea.x,selectarea.y);
		p2=Point(x,y);
		rectangle(showimage,p1,p2,Scalar(0,255,0),2);
		imshow("video",showimage);
	}
	else if(select_flag && mouseEvent==CV_EVENT_LBUTTONUP){
		selectarea.width=x-selectarea.x;
		selectarea.height=y-selectarea.y;
		select_flag=false;
	}
	return;  
} 
//Replay the seleted object's video shot
void mouseRecover(int mouseEvent,int x,int y,int flags,void* param)
{
	Point p1,p2;
	if(mouseEvent==CV_EVENT_LBUTTONDOWN){
		select_flag=true;
		selectarea.x=x;
		selectarea.y=y;
		
	}
	else if(select_flag && mouseEvent==CV_EVENT_MOUSEMOVE){
		image.copyTo(showimage);
		p1=Point(selectarea.x,selectarea.y);
		p2=Point(x,y);
		rectangle(showimage,p1,p2,Scalar(0,255,0),2);
		imshow("video",showimage);
	}
	else if(select_flag && mouseEvent==CV_EVENT_LBUTTONUP){
		select_flag=false;
		int ID=0, maxCount=0,baseIndex=0;
		uchar *p;
		p=index_image.ptr<uchar>(0);
		baseIndex=(int)p[0];
		cout<<"base Index"<<baseIndex<<endl;
		//waitKey(0);
		selectarea.width=x-selectarea.x;
		selectarea.height=y-selectarea.y;
		
		Mat destmat;
		index_image(selectarea).copyTo(destmat);
		for(int i=0; i<MAX_INDEX_COUNT; i++) event_count[i]=0;
		for(int i=0; i<destmat.rows; i++){
			p=destmat.ptr<uchar>(i);
			for(int j=0; j<destmat.cols; j++){
				if((int)p[j] > 100)	event_count[255-(int)p[j]]++;
				//if((int)p[j] > 100)	event_count[(int)p[j]]++;
			}
		}
		for(int i=0; i<MAX_INDEX_COUNT; i++){
			if(event_count[i] > maxCount){
				ID = i;
				maxCount = event_count[i];
			}
		}
		//ID=ID+baseIndex*256;
		cout<<"Info:	selected event No. is "<<ID<<endl;	
		cout<<event_start.size()<<endl;
		int start=event_start[ID];
		int end=event_end[ID];
		cout<<"Info:	frame index start from  "<<start<<"	to	"<<end<<endl;
		//replay the selected video
		VideoCapture vc_read;
		Mat cur_mat;
		vc_read.open(ori_video);
		vc_read.set(CV_CAP_PROP_POS_FRAMES, start);
		namedWindow("Video Contains the Object");
		for(int i=start; i<end; i++){
			vc_read>>cur_mat;
			imshow("Video Contains the Object",cur_mat);	
			waitKey(0);
		}
		destroyWindow("Video Contains the Object");
	}
	return;  
}
// snip-shot 
static void mouseSnipShot(int event, int x, int y, int flags, void* userdata)
{  
	switch(event)  
	{  
	case CV_EVENT_LBUTTONDOWN:
		//get the selected video's position
		int heng=y/small_height;
		int zong=x/small_width;
		long index=heng*SubPlot.width+zong;
		video_play(video_index[index]);
		break;
	}  
	return;  
} 


/** test thread **/
/*****************************************************************/
//you can set test = 1,2,3,4 for different test
//test=1: subtract the background and foreground of the input video
//test=2: compound the convex point sequence to produce the abstracted video
//test=3: you can replay the seleted object's event full process
//test=4: you can view 9 snip-shots of the original video 
/*****************************************************************/
void testmultithread(string inputpath, string videoname, string midname, string outputname, int frameCount, int CompoundCount, int scale, int stage){
//void testmultithread(const char* inputpath, const char* videoname, const char* midname, const char* outputname, int frameCount, int CompoundCount, int scale, int stage){
	time_t start_time,end_time;
	
	start_time=time(NULL);

	testVideoName=videoname;
	//set all the necessary paths
	string path=inputpath;
	string out_path=path+"OutputVideo/";
	string config_path=path+"Config/";
	string index_path=path+"indexMat/";
	log_path=path+"Log/";
	//create the path if not exist
	create_path(out_path);
	create_path(log_path);
	create_path(config_path);
	create_path(index_path);
	
	UserVideoAbstraction* user=new UserVideoAbstraction((char*)path.data(), (char*)out_path.data(), (char*)log_path.data(), (char*)config_path.data(),
														(char*)index_path.data(), (char*)videoname.data(), (char*)midname.data(), scale);
	user->UsersetGpu(true);
	user->UsersetIndex(false);
	user->UsersetMinArea(MIN_AREA/(scale*scale));
	
	int test = stage;
	ofstream ff(log_path+"TimeLog.txt", ofstream::app);
	if(test==1){
		state="Background/Foreground Subtraction";
		VideoCapture capture;
		string t1=inputpath,t2=videoname;
		string t3 = t1+t2;
		capture.open(t3);
		ofstream ff(log_path+"FrameLog.txt", ofstream::app);
		ff<<endl<<videoname<<"\t"<<capture.get(CV_CAP_PROP_FRAME_COUNT);
		int number=0;
		setROI=false;
		while (capture.read(image))
		{
			if(number==0){
				namedWindow("video");
				imshow("video",image);
				setMouseCallback("video",mouseSelect);
				waitKey(0);
				cvDestroyWindow("video");
				//user->UsersetROI(selectarea);
				ofstream ff(log_path+"AreaLog.txt", ofstream::app);
				ff<<endl<<selectarea.x<<":"<<selectarea.y;
			}
			if(setROI){
				if(number==0){
					namedWindow("video");
					imshow("video",image);
					setMouseCallback("video",mouseSelect);
					waitKey(0);
					cvDestroyWindow("video");
					//user->UsersetROI(selectarea);
					ofstream ff(log_path+"AreaLog.txt", ofstream::app);
					ff<<endl<<selectarea.x<<":"<<selectarea.y;
				}
				number++;
				image(selectarea).copyTo(imageRoi);
				user->UserAbstraction(imageRoi,number);
			}
			else{
				number++;
				user->UserAbstraction(image,number);
			}
		}
		int UsedFrameCount = user->UsersaveConfigInfo();
		frameCount=UsedFrameCount;
		user->~UserVideoAbstraction();
		ff<<"\t"<<UsedFrameCount<<"\t"<<(double)UsedFrameCount/(double)capture.get(CV_CAP_PROP_FRAME_COUNT)<<":"<<UsedFrameCount;
		ff.close();
	}
	else if(test==2){
		state="compound the result video";
		string t3 = out_path+outputname;
		if(setROI){
			int x,y;
			readAreaLog(log_path+"AreaLog.txt", x, y);
			Rect selectRoi(x,y,100,100);
			user->UsersetROI(selectRoi);
		}
		//int frCount=frameCount;
		int frCount = readFrameLog(log_path+"FrameLog.txt");
		cout<<frCount<<endl;
		//int frCount=frameCount;
		user->Usercompound(CompoundCount, (char*)t3.data(), frCount);
		user->UserfreeObject();
	}
	else if(test==3){
		state="test the index video function";
		int frCount = readFrameLog(log_path+"FrameLog.txt");
		//int frCount=frameCount;
		string t1=inputpath,t2=midname,t=videoname;
		ori_video=inputpath+t;
		string t3 = outputname;
		t3=out_path+t3;
		string temp;
		//读取中间文件获取 event_start event_end event_length 信息
		ifstream file(config_path+t2);
		cout<<t2<<endl;
		cout<<frCount<<endl;
		for(int i=0; i<frCount; i++) {		
			getline(file, temp, '#');
		}
		event_start.clear();
		event_end.clear();
		EventNumber=0;
		while(!file.eof()){
			int s,e,len;
			file>>s;
			file>>e;
			len=e-s;
			event_start.push_back(s);
			event_end.push_back(e);
			event_length.push_back(len);
			EventNumber++;
		}
		file.close();
		//index
		VideoCapture abstract_video;
		abstract_video.open(t3);
		currentFrameIndex=0;
		string filepath=index_path+t+"/";
		namedWindow("video");
		setMouseCallback("video",mouseRecover);
		abstract_video.read(image);
		imshow("video",image);
		waitKey(0);
		abstract_video.open(t3);
		while(abstract_video.read(image)){
			currentFrameIndex++;	
			string filename=boost::lexical_cast<string>(currentFrameIndex)+".bmp";
			index_image=imread(filepath+filename);
			imshow("video",image);
			//waitKey(30);
			//imshow("index", index_image);
			//setMouseCallback("video",mouseRecover);
			int key = waitKey(30); 
			if(key==27)
				waitKey(0);
		}
	}
	else if(test==4){
		state="test the snip-shot function";
		namedWindow(window_name);
		setMouseCallback(window_name,mouseSnipShot);//设置鼠标回调函数
		SubPlot=Size(3,3);//最终快照显示图像为3*3 矩阵的九张图像
		string t1=inputpath,t2=videoname;
		string t3 = t1+t2;
		capture.open(t3);
		//检测是否正常打开:成功打开时，isOpened返回ture
		if(!capture.isOpened())
		{
			cout<<"fail to open!"<<endl;
		}
		double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);//获取整个帧数
		cout<<"整个视频共"<<totalFrameNumber<<"帧"<<endl;
		bar_index=0;//进度条起始位置
		int number_per_div=(int)totalFrameNumber/DIV_NUMBER;//计算两个快照间的间隔帧数
		vector<Mat> images(DIV_NUMBER);//保存快照
		video_index=vector<long>(DIV_NUMBER);
		for (int i=0;i<DIV_NUMBER;i++)
		{
			long now_frame=number_per_div*i;//计算第i张快照 对应的帧数
			video_index[i]=now_frame;

			capture.set(CV_CAP_PROP_POS_FRAMES,now_frame);//设置读取这一帧

			Mat zhong;
			capture.read(zhong);//读取快照
			zhong.copyTo(images[i]);
			zhong.release();
		}
		Mat result_image=MultiImage(images, Size(images[0].cols,images[0].rows));
		imshow(window_name,result_image);
		waitKey(0);
		capture.release();
		result_image.release();
	}
	else{
		//check or debug 
	}
	end_time=time(NULL);
	cout<<testVideoName<<"\t"<<state<<"\t"<<"video abstraction time: "<<end_time-start_time<<" s"<<endl;
	ff<<testVideoName<<"\t"<<state<<"\t"<<"video abstraction time: "<<end_time-start_time<<endl;
	ff.close();
}


int main(){
	// boost thread control ...
	
	string testset1[] = {"20111201_170301.avi", "20111202_082713.avi", "juminxiaoqu.avi", "testvideo.avi", "xiezilou.avi", "LOD_CIF_HQ_4_2.avi",
						"road.avi", "loumenkou.avi", "damenkou.avi", "AA012507.avi", "AA013101.avi", "AA013102.avi", "AA013103.avi", "AA013106.avi", "Cam01.avi", 
						"Cam3.avi", "Cam4.avi"};

	int framecount[] = {48374, 36675, 3008, 994, 3002, 4094, 7942, 25967, 17447, 2973, 6618, 5029, 4918, 5100, 14100, 5522, 9860};
	/* Tong Hao Test !!! */
	//for(int i=0; i<testset1->size(); i++){	
	//	string result_name="result_"+testset1[i];
	//	string config_name="config_"+boost::lexical_cast<string>(i);
	//	boost::thread test1(testmultithread,"D:/summarytest1/", testset1[i], config_name, result_name, framecount[i], 8, 1, 1);
	//	test1.join();
	//	cout<<"finished..."<<endl;
	//}

	//int all=testset1->size()-1;
	//for(int i=0; i<testset1->size(); i++){	
	//	cout<<testset1[i]<<endl;
	//	string result_name="result_"+testset1[i];
	//	string config_name="config_"+boost::lexical_cast<string>(i);
	//	boost::thread test1(testmultithread,"D:/summarytest1/", testset1[i], config_name, result_name, framecount[i], 8, 1, 2);
	//	test1.join();
	//	cout<<"finished..."<<endl;
	//}

	cout<<"***********************************************************************************"<<endl;
	cout<<"\t"<<"Using Guidance "<<endl;
	cout<<"\t"<<"Please input 1 / 2 / 3 / 4"<<endl;
	cout<<"\t"<<"1:   subtract the background and foreground of the input video"<<endl;
	cout<<"\t"<<"2:   compound the convex point sequence to produce the abstracted video"<<endl;
	cout<<"\t"<<"3:   you can replay the seleted object's event full process"<<endl;
	cout<<"\t"<<"4:   you can view 9 snip-shots of the original video"<<endl;
	cout<<"\t"<<"others:   Exit !"<<endl;
	cout<<"*********************************************************************************"<<endl;
	string result_name="result_test.avi";
	string config_name="config_test";
	//boost::thread test1(testmultithread,"test/", "test.avi", config_name, result_name, 1025, 8, 2, 3);
	
	int choice;
	bool flag=true;
	while(flag){
		cout<<"Please input the choice No. : ";
		cin>>choice;
		if(choice== 1 || choice == 2 || choice == 3 || choice == 4){
			boost::thread test1(testmultithread,"D:\\workspace\\data\\videoabstract\\", "AA000805.mp4", config_name, result_name, 2199, 8, 1, choice);
			test1.join();
			cout<<"finish step "<<choice<<endl;
		}
		else
			return 0;
	}
	return 0;
}

int readFrameLog(string logname){
	ifstream fin;
	fin.open(logname);
	if(fin.is_open()) {
		fin.seekg(-1,ios_base::end);                // go to one spot before the EOF
		bool keepLooping = true;
		while(keepLooping) {
			char ch;
			fin.get(ch);                            // Get current byte's data
			if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
				fin.seekg(0);                       // The first line is the last line
				keepLooping = false;                // So stop there
			}
			else if(ch == '\n') {                   // If the data was a newline
				keepLooping = false;                // Stop at the current position.
			}
			else {                                  // If the data was neither a newline nor at the 0 byte
				fin.seekg(-2,ios_base::cur);        // Move to the front of that data, then to the front of the data before it
			}
		}
		string lastLine;            
		getline(fin,lastLine);                      // Read the current line
		cout<< lastLine<<'\n';     // Display it
		//lastLine
		
	    string token = lastLine.substr(lastLine.find(":")+1, lastLine.size());
		return atoi(token.c_str());
		fin.close();
	}
	return 0;
}

void readAreaLog(string logname, int &base_x, int &base_y){
	ifstream fin;
	fin.open(logname);
	if(fin.is_open()) {
		fin.seekg(-1,ios_base::end);                // go to one spot before the EOF
		bool keepLooping = true;
		while(keepLooping) {
			char ch;
			fin.get(ch);                            // Get current byte's data
			if((int)fin.tellg() <= 1) {             // If the data was at or before the 0th byte
				fin.seekg(0);                       // The first line is the last line
				keepLooping = false;                // So stop there
			}
			else if(ch == '\n') {                   // If the data was a newline
				keepLooping = false;                // Stop at the current position.
			}
			else {                                  // If the data was neither a newline nor at the 0 byte
				fin.seekg(-2,ios_base::cur);        // Move to the front of that data, then to the front of the data before it
			}
		}
		string lastLine;            
		getline(fin,lastLine);                      // Read the current line
		cout<< lastLine<<'\n';     // Display it
		string x = lastLine.substr(0,lastLine.find(":"));
		string y = lastLine.substr(lastLine.find(":")+1);
		base_x = atoi(x.c_str());
		base_y = atoi(y.c_str());
		fin.close();
	}
}

Mat MultiImage(const vector<Mat>& SrcImg_V, Size ImgMax_Size)
{
	/*
	*函数功能：	将多张图像拼接到一张图像上
	*SrcImg_v:	需要拼接的图像向量
	*ImgMax_Size:	最总显示的图像的大小
	*/
	Mat return_image(SrcImg_V[0].rows*SubPlot.height,SrcImg_V[0].cols*SubPlot.width,SrcImg_V[0].type());//初始化返回图像

	int heng=SubPlot.height;//最终图像显示多少行
	int zong=SubPlot.width;//最终图像显示多少列

	int width=SrcImg_V[0].cols;
	int height=SrcImg_V[0].rows;

	int num=0;//当前图像的编号
	for (int i=0;i<heng;i++)//纵向循环 每一行
	{
		for (int j=0;j<zong;j++)//横向循环 每一列
		{	
			int start_x=(num%zong)*width;//小图在大图中 左上角的横坐标
			int start_y=(num/zong)*height;//小图在大图中 左上角的纵坐标

			SrcImg_V[num].copyTo(return_image(Rect(start_x,start_y,width,height)));
			num++;//图像编号加一
		}
	}
	resize(return_image,return_image,ImgMax_Size);	
	small_width=ImgMax_Size.width/zong;//计算每一个小图的宽度
	small_height=ImgMax_Size.height/heng;//计算每个小图的高度
	return return_image;
}

static void bar_callback(int index,void* userdata)
{
	capture.set(CV_CAP_PROP_POS_FRAMES,index);
	bar_index=index;  
}

void video_play(long index)
{
	//视频播放函数
	Mat zhong;
	namedWindow(window_play);

	capture.set(CV_CAP_PROP_POS_FRAMES,index);
	double totalFrameNumber = capture.get(CV_CAP_PROP_FRAME_COUNT);//获取整个帧数
	double fps=capture.get(CV_CAP_PROP_FPS);
	if (totalFrameNumber>0)
	{
		cv::createTrackbar("position",window_play,&bar_index,totalFrameNumber,bar_callback);//创建进度条，bar_callback是回调函数
		bar_index=index;
	}

	while(capture.read(zhong))
	{
		bar_index++;
		setTrackbarPos("position",window_play,bar_index);//
		imshow(window_play,zhong);
		//waitKey(1000.0/fps);//间隔时间可以调整，这样显示会比实际的慢
		waitKey(20);//间隔时间可以调整，这样显示会比实际的慢
	}
	zhong.release();
	cv::destroyWindow(window_play);
}

void create_path(string path){
	fstream testfile;
	testfile.open(path, ios::in);
	if(!testfile){
		boost::filesystem::path dir(path);
		boost::filesystem::create_directories(dir);
	}
}
