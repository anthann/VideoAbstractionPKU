//****************************************
//user: PianoCoder
//Create date:
//Class name: VideoAbstraction
//Discription:  implement the background/foreground subtraction and the video compounding
//Update: 2014/01/07
//****************************************
#include "VideoAbstraction.h"
//
VideoAbstraction::VideoAbstraction(string inputpath, string out_path, string log_path, string config_path, string index_path, string videoname, string midname, int size){
	scaleSize=size;
	objectarea=100/(scaleSize*scaleSize);
	useGpu=true;
	Inputpath=inputpath;
	Outpath=out_path;
	Logpath=log_path;
	Configpath=config_path;
	Indexpath=index_path;
	InputName=videoname;
	MidName=midname;
	thres=1000;
	videoCapture.open(inputpath+videoname);
	frameHeight=videoCapture.get(CV_CAP_PROP_FRAME_HEIGHT)/scaleSize;
	frameWidth=videoCapture.get(CV_CAP_PROP_FRAME_WIDTH)/scaleSize;
	framePerSecond=videoCapture.get(CV_CAP_PROP_FPS);
	useROI=false;
	init();
}

VideoAbstraction::VideoAbstraction(){
	init();
}

void VideoAbstraction::init(){
	objectarea=60;
	useGpu=true;
	backgroundSubtractionMethod=1;
	LEARNING_RATE=-1;
	ObjectCubeNumber=0;
	sumLength=0;
	loadIndex=0;
	frame_start.clear();
	frame_end.clear();
	currentObject.start=-1;
	detectedMotion=0;
	cacheShift=50;
	motionToCompound=10;
	maxLength=-1;
	maxLengthToSpilt=300;
	sum=0;
	thres=1000;
	currentLength=0;
	tempLength=0;
	noObjectCount=0;
	flag=false;
	useROI=false;
}
//
void VideoAbstraction::freeObject(){
	videoCapture.~VideoCapture();
	videoWriter.~VideoWriter();
	backgroundImage.release();				
	currentStartIndex.release();
	currentEndIndex.release();
	mog.~BackgroundSubtractorMOG2();
	gFrame.release();			
	gForegroundMask.release();	
	gBackgroundImg.release();	
	currentMask.release();	
	vector<ObjectCube>().swap(partToCompound);	
	vector<Mat>().swap(compoundResult);
	vector<Mat>().swap(indexs);
	vector<Mat>().swap(indexe);
	vector<int>().swap(frame_start);
	vector<int>().swap(frame_end); 
}


string VideoAbstraction::int2string(int _Val){
	char _Buf[100];
	sprintf(_Buf, "%d", _Val);
	return (string(_Buf));
}

void VideoAbstraction::postProc(Mat& frame){
	blur(frame,frame,Size(25,25)); //用于去除噪声 平滑图像 blur（inputArray, outputArray, Size）
	threshold(frame,frame,100,255,THRESH_BINARY);	//对于数组元素进行固定阈值的操作  参数列表：(输入图像，目标图像，阈值，最大的二值value--8对应255, threshold类型)
	dilate(frame,frame,Mat());// 用于膨胀图像 参数列表：(输入图像，目标图像，用于膨胀的结构元素---若为null-则使用3*3的结构元素，膨胀的次数)
}

void VideoAbstraction::ConnectedComponents(int frameindex, Mat &mask,int thres){  
	//GaussianBlur(mask,mask,Size(5,5),0,0);
	//GaussianBlur(mask,mask,Size(5,5),0,0);
	//imshow("erode1",mask);
	Mat ele(2,4,CV_8U,Scalar(1));
	erode(mask,mask,ele);// 默认时，ele 为 cv::Mat() 形式  参数扩展（image， eroded, structure, cv::Point(-1,-1,), 3） 
	//右侧2个参数分别表示 是从矩阵的中间开始，3表示执行3次同样的腐蚀操作
	dilate(mask,mask,ele);
	//imshow("erode2",mask);
	//waitKey(0);

	vector<vector<Point>> contors,newcontors;
	vector<Point> hull;
	findContours(mask,contors,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); //找到所有的contour 闭包
	//
	vector<vector<Point>>::const_iterator itc=contors.begin();
	//过滤掉过小的闭包，其他闭包全部存放到 newcontors 中
	while(itc!=contors.end()){
		if(contourArea(*itc)<objectarea){
		//if(itc->size()<thres){
			itc=contors.erase(itc);
		}
		else{
			convexHull(*itc,hull);
			newcontors.push_back(hull);
			itc++;
		}
	}
	mask=0;
	drawContours(mask,newcontors,-1,Scalar(255),-1); // Scalar(255) 表示对应的背景是全部黑色
	vector<vector<Point>>().swap(contors);
	vector<vector<Point>>().swap(newcontors);
}

vector<bool> matToVector(Mat &input){   //############   bug place  ############
	int step=input.step,step1=input.elemSize();
	uchar* indata=input.data;
	int row=input.rows,col=input.cols;
	vector<bool> ret;
	for(int i=0;i<row;i++){
		for(int j=0;j<col;j++){
			ret.push_back(*(indata+i*step+j*step1));
		}
	}
	return ret;
}

Mat vectorToMat(vector<bool> &input,int row,int col){
	Mat re(row,col,CV_8U,Scalar::all(0));
	int step=re.step,step1=re.elemSize();
	for(int i=0;i<input.size();++i){
		*(re.data+i*step1)=(input[i]?255:0);
	}
	return re;
}

void VideoAbstraction::stitch(Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end, int frameno){
//void VideoAbstraction::stitch(Mat &input1,Mat &input2,Mat &output,Mat &back,Mat &mask,int start,int end, vector<vector<Point>>& re_contours, bool& flag){
	int step10=input1.step,step11=input1.elemSize();
	int step20=input2.step,step21=input2.elemSize();
	int step30=output.step,step31=output.elemSize();
	int stepb1=back.step,stepb2=back.elemSize();
	int stepm1=mask.step,stepm2=mask.elemSize();
	int input1sim,input2sim;
	double alpha;
	uchar* indata1,*indata2,*outdata,*mdata,*bdata;
	for(int i=0;i<input1.rows;i++){
		for(int j=0;j<input1.cols;j++){
			mdata=mask.data+i*stepm1+j*stepm2;
			if((*mdata)!=0){
				indata1=input1.data+i*step10+j*step11;
				indata2=input2.data+i*step20+j*step21;
				outdata=output.data+i*step30+j*step31;
				bdata=back.data+i*stepb1+j*stepb2;
				input1sim=abs(bdata[0]-indata1[0])+abs(bdata[1]-indata1[1])+abs(bdata[2]-indata1[2])+1;
				input2sim=abs(bdata[0]-indata2[0])+abs(bdata[1]-indata2[1])+abs(bdata[2]-indata2[2])+1;
				alpha=input1sim*1.0/(input1sim+input2sim);
				outdata[0]=int(indata1[0]*alpha+indata2[0]*(1-alpha));
				outdata[1]=int(indata1[1]*alpha+indata2[1]*(1-alpha));
				outdata[2]=int(indata1[2]*alpha+indata2[2]*(1-alpha));
			}
		}
	}
	//if(frameno%5==0){
		start = start/framePerSecond;
		end = end/framePerSecond;
		vector<vector<Point>> m_contours;
		vector<Point> info;
		findContours(mask,m_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
		vector<vector<Point>>::const_iterator itc=m_contours.begin();

		while(itc!=m_contours.end()){
			if(contourArea(*itc) < objectarea){
				itc=m_contours.erase(itc);
			}
			else{
				convexHull(*itc,info);
				Point p1 = info.at(1);
				Point p2 = info.at(info.size()/2);
				Point mid;
				mid.x = (p1.x+p2.x)/2;
				mid.y = (p1.y+p2.y)/2;
				if(useROI){
					mid.x += rectROI.x;
					mid.y += rectROI.y;
					//cout<<rectROI.x<<"***"<<rectROI.y<<endl;
				}
				//putText(output,int2string(start)+"-"+int2string(end),mid,CV_FONT_HERSHEY_COMPLEX,0.2, Scalar(0,0,255),1);
				//itc++;
				int s1,s2,s3,e1,e2,e3;
				s1=start/3600;
				s2=(start%3600)/60;
				s3=start%60;
				e1=end/3600;
				e2=(end%3600)/60;
				e3=end%60;
				//putText(output,int2string(start)+"-"+int2string(end),mid,CV_FONT_HERSHEY_COMPLEX,0.2, Scalar(0,0,255),1);
				putText(output,int2string(s1)+":"+int2string(s2)+":"+int2string(s3)+"-"+int2string(e1)+":"+int2string(e2)+":"+int2string(e3),mid,CV_FONT_HERSHEY_COMPLEX,0.4, Scalar(0,255,0),1);
				itc++;
			}
		}
	//}
}

int VideoAbstraction::ComponentLable(Mat& fg_mask, vector<Rect>& vComponents_out, int area_threshold)
{
	const double	MAXASPECTRATIO =2;
	const double	MINASPECTRATIO =0.5;
	const double	FILLRATIO	=0.4;
	const int       DISTBTWNPART =81;
	//const int       MINRECTSIZE = hog_des.winSize.height * hog_des.winSize.width;
	const int		AREATHRESHOLD = area_threshold;
	double similar_rects_eps = 0.7;

	const int xdir[8]={-1,0,0,1,1,1,-1,-1};
	const int ydir[8]={0,-1,1,0,1,-1,-1,1};
	const int HEIGHT = fg_mask.rows;
	const int WIDTH = fg_mask.cols;

#define p_lable(x,y)    (p_lable[(y)*(WIDTH)+(x)])

	int x,y,m;
	int cur_pos=0;//start from 0
	int tail_pos=0;
	int cur_x, cur_y;
	int left_pos, right_pos, up_pos, bottom_pos;
	//CvRect rect;
	int lable_count=0;
	int* p_x=NULL;
	int* p_y=NULL;
	int* p_lable=NULL;
	//	int RectSize ;
	//	double FillRatio ;
	//	double AspectRatio ;
	int rWidth;
	int rHeight;

	p_x = new int[HEIGHT *WIDTH ];
	p_y = new int[HEIGHT*WIDTH ];
	p_lable = new int[HEIGHT*WIDTH ];

	memset(p_lable,0,WIDTH*HEIGHT*sizeof(int));
	for(y=0;y<HEIGHT; y++)//y
	{
		for(x=0;x<WIDTH;x++)//x
		{
			if( p_lable(x,y)!=0 || fg_mask.at<uchar>(y,x) != 255 ) //注意只认可255为前景
				continue;
			lable_count++;	//begin a new component
			p_lable(x,y) = lable_count;
			cur_pos = 0;
			tail_pos = 0;
			p_x[tail_pos] = x;
			p_y[tail_pos] = y;
			tail_pos++;
			left_pos = x; right_pos = x;
			up_pos = y; bottom_pos = y;

			while(cur_pos!=tail_pos)
			{
				cur_x = p_x[cur_pos];
				cur_y = p_y[cur_pos];
				cur_pos++;
				for(m=0; m<8; m++)
				{
					if( (cur_y+ydir[m])>=0 && (cur_y+ydir[m])<HEIGHT &&
						(cur_x+xdir[m])>=0 && (cur_x+xdir[m])<WIDTH &&
						fg_mask.at<uchar>(cur_y+ydir[m],cur_x+xdir[m])!=0 &&
						p_lable(cur_x+xdir[m],cur_y+ydir[m])==0 )
					{
						p_x[tail_pos] = cur_x+xdir[m];
						p_y[tail_pos] = cur_y+ydir[m];
						tail_pos++;
						p_lable(cur_x+xdir[m], cur_y+ydir[m]) = lable_count;

						//更新巨型框的坐标（topLeft,bottomRight）
						if(xdir[m]==1 && cur_x+1 > right_pos )
							right_pos = cur_x+1;
						if(xdir[m]==-1 && cur_x-1 < left_pos )
							left_pos = cur_x-1;
						if(ydir[m]==1 && cur_y+1 > bottom_pos)
							bottom_pos = cur_y+1;
						if(ydir[m]==-1 && cur_y-1 < up_pos)
							up_pos = cur_y -1;							
					}
				}

			}

			rWidth=CV_IABS((right_pos-left_pos));
			rHeight=CV_IABS(bottom_pos-up_pos);
			//RectSize =  rWidth*rHeight;
			//FillRatio = (double)cur_pos/(double)RectSize;
			//AspectRatio = (double)rWidth/(double)rHeight;

			if (
				cur_pos<AREATHRESHOLD /*|| 
									  AspectRatio>MAXASPECTRATIO || 
									  AspectRatio<MINASPECTRATIO ||
									  FillRatio<FILLRATIO*/
									  )
			{
				lable_count--;
				for (int i=0;i<tail_pos;i++)
				{
					fg_mask.at<uchar>(p_y[i],p_x[i])=0;
				}
			}
			else
			{
				//Rect r(left_pos,up_pos,rWidth,rHeight);
				Rect r;
				r.x = max(left_pos-cvRound(rWidth * 0.2) ,0);
				r.width = min(cvRound(rWidth * 1.2), WIDTH);
				r.y = max(up_pos-cvRound(rHeight * 0.2), 0);
				r.height = min(cvRound(rHeight * 1.2), HEIGHT);

				if (r.width < 64)
				{
					r.width = 64; 
				}
				if (r.height < 128)
				{
					r.height = 128;
				}
				if (r.x+r.width > WIDTH)
				{
					r.x = WIDTH - r.width;
				}
				if (r.y+r.height > HEIGHT)
				{
					r.y = HEIGHT - r.height;
				}

				bool is_similar_found = false;
				for (vector<Rect>::iterator itr = vComponents_out.begin(); itr != vComponents_out.end(); ++itr)
				{
					if (isSimilarRects(r, *itr, similar_rects_eps))
					{
						is_similar_found = true;
						lable_count--;
						for (int i=0;i<tail_pos;i++)
						{
							fg_mask.at<uchar>(p_y[i],p_x[i])=0;
						}
						break;
					}
				}
				if (!is_similar_found)
				{
					vComponents_out.push_back(r);
				}
			}
		}
	}
	// TODO: 矩形框之间不能有交集，如果有，则合并

	if (p_x!=NULL)
	{
		delete []p_x;
		p_x = NULL;
	}

	if (p_y!=NULL)
	{
		delete []p_y;
		p_y = NULL;
	}

	if (p_lable!=NULL)
	{
		delete []p_lable;
		p_lable = NULL;
	}

	return 0;
}

bool VideoAbstraction::isSimilarRects(const Rect& r1, const Rect& r2, double eps)
{
	return rectsOverlapAreaRate(r1, r2) > eps;
}

double VideoAbstraction::rectsOverlapAreaRate(const Rect& r1, const Rect& r2)
{
	CvRect cr1 = cvRect(r1.x, r1.y, r1.width, r1.height);
	CvRect cr2 = cvRect(r2.x, r2.y, r2.width, r2.height);
	CvRect cr = cvMaxRect(&cr1, &cr2);//返回包含cr1 和 cr2 2个矩阵的最小矩阵信息 然后求解重叠部分的面积
	int w = cr1.width + cr2.width - cr.width;
	int h = cr1.height + cr2.height - cr.height;
	if(w<=0||h<=0)
		return 0;
	else
		return max(double(w*h)/double(cr1.height*cr1.width), double(w * h)/double(cr2.height * cr2.width));
}

double VideoAbstraction::random(double start, double end){
	return start+(end-start)*rand()/(RAND_MAX + 1.0);
}

int VideoAbstraction::computeMaskCollision(Mat &input1,Mat &input2){
	return countNonZero(input1&input2);
}

int VideoAbstraction::computeMaskCollision(vector<bool> &input1,vector<bool> &input2){
	int ret=0;
	if(input1.size()!=input2.size()){
		LOG(ERROR)<<"input vector size do not match\t"<<input1.size()<<"\t"<<input2.size()<<"\n";
		return -1;
	}
	for(int i=0;i<input1.size();++i){
		ret+=(input1[i]&&input2[i]);
	}
	return ret;
}

int VideoAbstraction::computeObjectCollision(ObjectCube &ob1,ObjectCube &ob2,int shift,string path){
	int collision=0;
	int as,bs;
	if(shift>0){
		as=ob1.start+shift;
		bs=ob2.start;
	}
	else{
		as=ob1.start;
		bs=ob2.start-shift;
	}
	Mat amask,bmask;
	if(path==""){
		for(;as<=ob1.end&&bs<=ob2.end;as++,bs++){
			collision+=computeMaskCollision(ob1.objectMask[as-ob1.start],ob2.objectMask[bs-ob2.start]);
		}
	}else{
		for(;as<=ob1.end&&bs<=ob2.end;as++,bs++){
			amask=imread(path+int2string(as)+".pgm",CV_LOAD_IMAGE_GRAYSCALE);
			bmask=imread(path+int2string(bs)+".pgm",CV_LOAD_IMAGE_GRAYSCALE);
			collision+=computeMaskCollision(amask,bmask);
		}
	}
	return collision;
}


void VideoAbstraction::Abstraction(Mat& currentFrame, int frameIndex){	  //前背景分离函数
	//Mat currentFrame;
	//if(scaleSize > 1)
	//	pyrDown(inputFrame, currentFrame, Size(frameWidth,frameHeight));
	//else
	//	inputFrame.copyTo(currentFrame);

	if(frameIndex==50)								//如果中间文件原来已经存在，则执行清空操作
		ofstream file_flush(Configpath+MidName, ios::trunc);

	if(frameIndex <= 50){							//初始化混合高斯 取前50帧图像来更新背景信息  提示：取值50仅供参考，并非必须是50
		if(useGpu){
			//gpu module
			gpuFrame.upload(currentFrame);
			gpumog(gpuFrame,gpuForegroundMask,LEARNING_RATE);
			gpumog.getBackgroundImage(gpuBackgroundImg);
			gpuBackgroundImg.download(backgroundImage);
		}
		else{
			currentFrame.copyTo(gFrame);				//复制要处理的图像帧到 gFrame 中
			mog(gFrame,gForegroundMask,LEARNING_RATE);	//更新背景模型并且返回前景信息   参数解释： （下一个视频帧， 输出的前景帧信息， 学习速率）
			mog.getBackgroundImage(gBackgroundImg);		//输出的背景信息存储在 gBackgroundImg
			gBackgroundImg.copyTo(backgroundImage);		//保存背景图片到 backgroundImage 中
		}
		imwrite(InputName+"background.jpg",backgroundImage);
	}
	else{										//50帧之后的图像需要正常处理
		if(frameIndex%2==0){						//更新前背景信息的频率，表示每5帧做一次前背景分离
			if(useGpu){
				//gpu module
				gpuFrame.upload(currentFrame);
				gpumog(gpuFrame,gpuForegroundMask,LEARNING_RATE);
				gpuForegroundMask.download(currentMask);
			}
			else{
				currentFrame.copyTo(gFrame);
				mog(gFrame,gForegroundMask,LEARNING_RATE);
				gForegroundMask.copyTo(currentMask);		//复制运动的凸包序列到 currentMask 中
			}
			ConnectedComponents(frameIndex,currentMask, objectarea);		//计算当前前景信息中的凸包信息，存储在 currentMask 面积大于objectarea的是有效的运动物体，否则过滤掉 （取值50仅供参考）
			sum=countNonZero(currentMask);			//计算凸包中非0个数
			//整个画面
			if(sum>(thres/(scaleSize*scaleSize))){							//前景包含的点的个数大于 1000 个 认为是有意义的运动序列（取值1000仅供参考）
				flag=true;
			}
		}
		if(flag){							   //判断当前的图像帧是否包含有意义的运动序列信息
			currentObject.objectMask.push_back(matToVector(currentMask));					//将当前帧添加到运动序列中
			if(currentObject.start<0) currentObject.start=frameIndex;
			if(currentObject.start>0 && frameIndex-currentObject.start>maxLengthToSpilt*10){	//当前运动序列太长，认为其实无意义的运动序列（比如一直摇动的树叶信息或者光线变化），则清空成功新开始
				currentObject.objectMask.clear();
				currentObject.start=-1;
				flag=false;
				noObjectCount=0;
			}
			if(sum<thres){				   //当前图像中无运动序列
				if(noObjectCount>=15){														//已经有连续15帧无运动序列，运动结束  存储运动序列
					currentObject.end=frameIndex-15;
					if(currentObject.end-currentObject.start>30){								//运动序列长度大于 50 才认为是有效运动，否则不认为其是运动的
						detectedMotion++;
						currentLength=currentObject.end-currentObject.start+1;
						if(currentLength>maxLengthToSpilt*10){								//运动序列的长度太长，是无意义的运动序列，直接丢弃
							detectedMotion--;
						} 
						else if(currentLength>maxLengthToSpilt*5){							//事件过长 进行切分处理
							LOG(INFO)<<"事件过长:"<<currentLength<<endl;
							int spilt=currentLength/maxLengthToSpilt+1;
							int spiltLength=currentLength/spilt;
							ObjectCube temp;
							for(int i=0;i<spilt;++i){										//保存切分后的运动序列的信息
								vector<vector<bool>>().swap(temp.objectMask);
								temp.start=currentObject.start+i*spiltLength;
								temp.end=temp.start+spiltLength-1;
								tempLength=spiltLength;
								for(int j=0;j<spiltLength;++j){
									temp.objectMask.push_back(currentObject.objectMask[i*spiltLength+j]);
								}
								saveObjectCube(temp);
								maxLength=max(tempLength,maxLength);
								LOG(INFO)<<"事件"<<detectedMotion<<"\t开始帧"<<temp.start<<"\t结束帧"<<temp.end<<"\t长度"<<(temp.end-temp.start)*1.0/framePerSecond<<"秒"<<endl;
								detectedMotion++;
							}
							vector<vector<bool>>().swap(temp.objectMask);
							detectedMotion--;
						}

						else{														//事件正常长度，直接添加到运动序列中
							maxLength=max(currentLength,maxLength);
							saveObjectCube(currentObject);
							LOG(INFO)<<"事件"<<detectedMotion<<"\t开始帧"<<currentObject.start<<"\t结束帧"<<currentObject.end<<"\t长度"<<(currentObject.end-currentObject.start)*1.0/framePerSecond<<"秒"<<endl;
						}
					}
					vector<vector<bool>>().swap(currentObject.objectMask);
					currentObject.start=-1;
					flag=false;
					noObjectCount=0;
				}
				else noObjectCount++;
			}
			else{
				noObjectCount=0;
				flag=true;
			}
		}
		curMaxLength=maxLength;
	}
}

void VideoAbstraction::saveObjectCube(ObjectCube &ob){			//保存运动的凸包序列的函数
	frame_start.push_back(ob.start);						//保存凸包的开始帧号
	frame_end.push_back(ob.end);							//保存凸包的结束帧号
	ofstream ff(Configpath+MidName, ofstream::app);
	for(int i=ob.start,j=0;i<=ob.end;++i,++j){
		Mat tmp=vectorToMat(ob.objectMask[j],frameHeight,frameWidth);
		vector<vector<Point>> contors;
		findContours(tmp,contors,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE); //提取凸包信息
		ff<<contorsToString(contors);
		ff<<'#';	
		ObjectCubeNumber++;
	}
	ff.close();
}

void VideoAbstraction::saveConfigInfo(){						//保存所有凸包运动序列的开始和结束帧信息
	ofstream ff(Configpath+MidName, ofstream::app);
	int size = frame_start.size();
	for(int i=0; i<size; i++){
		ff<<endl;
		ff<<frame_start[i]<<endl;
		ff<<frame_end[i];
	}
	ff.close();
}

void VideoAbstraction::loadObjectCube(int index_start, int index_end){ //将指定帧序列号范围内的运动帧导入 partToCompound 中
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<loadIndex; i++) {
		getline(file, temp, '#');
	}
	int length=0;
	ObjectCube ob;
	vector<vector<Point>> contors;
	for(int j=index_start; j<=index_end; j++){
		cout<<frame_start[j]<<"\t"<<frame_end[j]<<endl;
		ob.start=frame_start[j];
		ob.end=frame_end[j];
		length=frame_end[j]-frame_start[j]+1;
		for(int i=0;i<length;++i){
			vector<vector<Point>>().swap(contors);
			getline(file, temp, '#');
			contors=stringToContors(temp);
			Mat bb(frameHeight,frameWidth,CV_8U,Scalar::all(0));
			drawContours(bb,contors,-1,Scalar(255),-1);
			ob.objectMask.push_back(matToVector(bb));	
		}
		vector<vector<Point>>().swap(contors);
		curMaxLength=max(length,curMaxLength);
		partToCompound.push_back(ob);
		vector<vector<bool>>().swap(ob.objectMask);
		loadIndex+=length;
	}
	file.close();
}

void  VideoAbstraction::LoadConfigInfo(){		//不能分阶段处理 -- 读取中间文件中的运动起始信息
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<ObjectCubeNumber; i++) {		
		getline(file, temp, '#');
	}
	frame_start.clear();
	frame_end.clear();
	while(!file.eof()){
		int start,end;
		file>>start;
		file>>end;
		frame_start.push_back(start);
		frame_end.push_back(end);
	}
	file.close();
}

void  VideoAbstraction::LoadConfigInfo(int frameCountUsed){  //用于分阶段处理 ---  需要传入有效帧的帧数信息
	this->ObjectCubeNumber=frameCountUsed;
	ifstream file(Configpath+MidName);
	string temp;
	for(int i=0; i<ObjectCubeNumber; i++) {	
		getline(file, temp, '#');
	}
	frame_start.clear();
	frame_end.clear();
	while(!file.eof()){
		int start,end;
		file>>start;
		file>>end;
		frame_start.push_back(start);
		frame_end.push_back(end);
	}
	file.close();
}

string VideoAbstraction::contorsToString(vector<vector<Point>> &contors){
	string re="";
	re+=boost::lexical_cast<string>(contors.size());
	re+="\t";
	for(int i=0;i<contors.size();++i){
		re+=boost::lexical_cast<string>(contors[i].size());
		re+="\t";
		for(int j=0;j<contors[i].size();j++){
			re+=boost::lexical_cast<string>(contors[i][j].x);
			re+="\t";
			re+=boost::lexical_cast<string>(contors[i][j].y);
			re+="\t";
		}
	}
	re+="\n";
	return re;
}

vector<vector<Point>> VideoAbstraction::stringToContors(string ss){
	vector<vector<Point>> contors;
	int s=0,e=0;
	e=ss.find("\t",s);
	string tmp=ss.substr(s,e-s);
	s=e+1;
	int n=boost::lexical_cast<int>(tmp),x,y;
	for(int i=0;i<n;i++){
		vector<Point> cur;
		e=ss.find("\t",s);
		tmp=ss.substr(s,e-s);
		s=e+1;
		int nn=boost::lexical_cast<int>(tmp);
		for(int j=0;j<nn;j++){
			e=ss.find("\t",s);
			tmp=ss.substr(s,e-s);
			s=e+1;
			x=boost::lexical_cast<int>(tmp);
			e=ss.find("\t",s);
			tmp=ss.substr(s,e-s);
			s=e+1;
			y=boost::lexical_cast<int>(tmp);
			cur.push_back(Point(x,y));
		}
		contors.push_back(cur);
	}
	return contors;
}


int VideoAbstraction::graphCut(vector<int> &shift,vector<ObjectCube> &ob,int step/* =5 */){  //计算所有运动序列的最佳偏移序列组合

	int n=ob.size(),A,B,C,D,label,collision;

	QPBO<int>* q;

	q = new QPBO<int>(n, n*(n-1)/2); // max number of nodes & edges
	q->AddNode(n); // add nodes

	collision=0;
	int mcache=0;


	clock_t starttime=clock();
	for(int i=0;i<n;i++){
		for(int j=i+1;j<n;j++){
			int diff=(shift[j]-shift[i])/step+cacheShift;
			if(cacheCollision[i][j][diff]>=0){
				A=D=cacheCollision[i][j][diff];
				mcache++;
			}
			else{
				A=D=computeObjectCollision(ob[i],ob[j],shift[j]-shift[i]);
				cacheCollision[i][j][diff]=A;
			}
			if(cacheCollision[i][j][diff+1]>=0){
				B=cacheCollision[i][j][diff+1];
				mcache++;
			}
			else{
				B=computeObjectCollision(ob[i],ob[j],shift[j]-shift[i]+step);
				cacheCollision[i][j][diff+1]=B;
			}
			if(cacheCollision[i][j][diff-1]>=0){
				C=cacheCollision[i][j][diff-1];
				mcache++;
			}
			else{
				C=computeObjectCollision(ob[i],ob[j],shift[j]-shift[i]-step);
				cacheCollision[i][j][diff-1]=C;
			}
			q->AddPairwiseTerm(i, j, A, B, C, D);
			//LOG(INFO)<<i<<"\t"<<j<<"\tA:"<<A<<"\tB:"<<B<<"\tC:"<<C<<"\t"<<B+C-A-D<<endl;
			collision+=A;
		}
	}
	//LOG(INFO)<<"计算碰撞耗时"<<clock()-starttime<<"豪秒\n";
	printf("hit cache %d times\n",mcache);
	printf("current collision:%d\n",collision);
	q->Solve();
	q->ComputeWeakPersistencies();
	q->Improve();
	bool convergence=true;
	for(int i=0;i<n;i++){
		label=q->GetLabel(i);
		if(label>0){
			convergence=false;
			if(shift[i]+ob[i].end-ob[i].start+1 > curMaxLength){
				for(int j=0;j<n;++j){
					cout<<"shift"<<j<<"\t"<<shift[j]<<"\t"<<ob[j].end-ob[j].start+1<<"\t"<<curMaxLength<<endl;
				}
				//cout<<"shift"<<i<<"\t"<<ob[i].length<<"\t"<<curMaxLength<<endl;
				return -2;//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
			}

		}
	}
	if(convergence)return -1;
	for(int i=0;i<n;i++){
		label=q->GetLabel(i);
		//printf("%d: %d\n",i+1,label);
		if(label>0){
			shift[i]+=step;
		}
	}
	return collision;
}

void VideoAbstraction::compound(string path){	
	int testcount=0;
	Outpath=path;									//获取合成文件的输出路径以及完整的文件名字
	videoCapture.open(Inputpath+InputName);			//合成操作前，需要提取背景图片信息保存到backgroundImage中
	backgroundImage=imread(InputName+"background.jpg");

	videoWriter.open(Outpath, (int)videoCapture.get(CV_CAP_PROP_FOURCC), 
		(double)videoCapture.get( CV_CAP_PROP_FPS ),
		cv::Size(frameWidth*scaleSize, frameHeight*scaleSize),
		true );								  //输出视频的属性信息和输入视频的信息完全相同
	if (!videoWriter.isOpened()){
		LOG(ERROR) <<"Can't create output video file: "<<Outpath<<endl;
		return;
	}

	int ObjectCount = frame_start.size();				//获取运动序列的个数
	int AverageCount = ObjectCount/motionToCompound;		//每次合成motionToCompound个运动序列的时候，合成的循环的执行次数
	int RemainCount = ObjectCount%motionToCompound;		//多余出来的运动序列的个数

	cout<<"进入摘要视频合成..."<<endl;
	clock_t starttime = clock();
	int index=0;
	for(int ss=0; ss<AverageCount || AverageCount==0; ss++){		//视频摘要合成的主循环
		int synopsis=motionToCompound;
		cout<<"*** 第"<<ss+1<<"次 ***"<<endl;
		vector<ObjectCube>().swap(partToCompound);
		maxLength=0;
		curMaxLength=0;		
		if(AverageCount==0){									//如果运动序列小于motionToCompound个，则只需要对所有的运动序列进行一次合成操作即可！
			if(ObjectCount==0) { cout<<"没有运动序列"<<endl; return; }
			else {
				loadObjectCube(ss, ss+RemainCount-1);            //从中间的凸包文件中读取运动序列到 partToCompound 中
				synopsis=ObjectCount;
				AverageCount=-1;
			}
		}
		else if(ss==AverageCount-1){							//如果RemainCount不为0，则最后一次合成的时候，合成 motionToCompound+RemainCount 个运动序列
			loadObjectCube(ss*motionToCompound, (ss+1)*motionToCompound+RemainCount-1);
			synopsis=motionToCompound+RemainCount;
		}
		else{												//正常合成 motionToCompound 个运动序列
			loadObjectCube(ss*motionToCompound, (ss+1)*motionToCompound-1);	
		}

		vector<int> shift(synopsis,0);							//运动序列的偏移数组
		int min=INT_MAX,cur_collision=0;						
		//Mat zeroObject(frameHeight,frameWidth,CV_8U,Scalar::all(0)),zeroObject1(frameHeight,frameWidth,CV_16U,Scalar::all(0)),oneObject(frameHeight,frameWidth,CV_16U,Scalar::all(1));
		LOG(INFO)<<"开始计算shift"<<endl;
		clock_t starttime=clock();
		vector<int> tmpshift;
		int *tempptr=(int *)cacheCollision;
		int cache_size=sizeof(cacheCollision)/4;
		for(int i=0;i<cache_size;i++){
			tempptr[i]=-1;
		}		
		for(int randtime=0;randtime<1;++randtime){
			LOG(INFO)<<"生成第"<<randtime+1<<"次初始点\n";
			for(int i=0;i<synopsis;i++){					  //初始化偏移序列
				shift[i]=0;
			}
			while(1){									  //计算满足冲突比较少的所有的偏移序列
				cur_collision=graphCut(shift,partToCompound);
				LOG(INFO)<<"当前碰撞:"<<cur_collision<<endl;
				if(cur_collision<0) break;
				if(cur_collision<min){
					min=cur_collision;
					tmpshift=shift;
				}
			}
		}
		shift=tmpshift;

		LOG(INFO)<<"最小损失"<<min<<endl;
		LOG(INFO)<<"时间偏移计算耗时"<<clock()-starttime<<"豪秒\n";
		LOG(INFO)<<"开始合成"<<endl;

		//zeroobject
		//currentStartIndex=zeroObject1.clone();
		//currentEndIndex=zeroObject1.clone();

		starttime=clock();
		Mat currentFrame;
		Mat currentResultFrame;
		Mat tempFrame;
		for(int i=0;i<synopsis;i++){
			cout<<"shift "<<i+1<<"\t"<<shift[i]<<endl;
		}
		int startCompound=INT_MAX;
		for(int i=0;i<synopsis;i++){
			startCompound=std::min(shift[i],startCompound);
		}
		cout<<"start\t"<<startCompound<<endl;
		cout<<"end\t"<<curMaxLength<<endl;
		cout<<"writing to the video ..."<<endl;
		sumLength+=(curMaxLength-startCompound);	
		for(int j=startCompound;j<curMaxLength;j++)
		{
			bool haveFrame=false;
			Mat resultMask, tempMask;
			//初始化 indexMat
			Mat indexMat(Size(frameWidth*scaleSize,frameHeight*scaleSize), CV_8U);

			//bitwise_and(currentStartIndex,zeroObject1,currentStartIndex);
			//bitwise_and(currentEndIndex,zeroObject1,currentStartIndex);

			int earliest=INT_MIN,earliestIndex=-1;
			for(int i=0;i<synopsis;i++){	//寻找序列中开始时间最早的作为背景
				if(shift[i]<=j&&shift[i]+partToCompound[i].end-partToCompound[i].start+1>j){
					if(partToCompound[i].end>earliest){
						earliest=partToCompound[i].end;
						earliestIndex=i;
					}
				}
			}

			int baseIndex, remainIndex; 
			if(earliestIndex>-1){
				baseIndex=(earliestIndex+ss*motionToCompound)/256;
				remainIndex=(earliestIndex+ss*motionToCompound)%256;
				haveFrame=true;
				videoCapture.set(CV_CAP_PROP_POS_FRAMES,partToCompound[earliestIndex].start-1+j-shift[earliestIndex]);
				//resize
				//videoCapture>>currentFrame;
				videoCapture>>currentFrame;
				//if(scaleSize > 1)		
				//	pyrDown(tempFrame, currentFrame, Size(frameWidth,frameHeight));
				//else
				//	tempFrame.copyTo(currentFrame);
				//resize
				currentResultFrame=currentFrame.clone();
				resultMask=vectorToMat(partToCompound[earliestIndex].objectMask[j-shift[earliestIndex]],frameHeight,frameWidth);
				//pyrUp(tempMask, resultMask, Size(frameWidth*scaleSize,frameHeight*scaleSize));
				for(int ii=0; ii<indexMat.rows; ii++)
				{
					uchar* pi=indexMat.ptr<uchar>(ii);
					uchar* ptr_re=resultMask.ptr<uchar>(ii);
					for(int jj=0; jj<indexMat.cols;jj++){
						pi[jj]=(earliestIndex+ss*motionToCompound)%256;
						//pi[jj]=remainIndex;
						if(ptr_re[jj]==255)
							pi[jj]=255-pi[jj];
							//pi[jj]=remainIndex;
					}
				}
			}

			if(!haveFrame){
				cout<<"没有找到最早\n";
				break;
			}

			for(int i=0;i<synopsis;i++){
				if(i==earliestIndex){
					continue;
				}
				if(shift[i]<=j&&shift[i]+partToCompound[i].end-partToCompound[i].start+1>j){
					videoCapture.set(CV_CAP_PROP_POS_FRAMES,partToCompound[i].start-1+j-shift[i]); //设置背景图片
					//resize
					//videoCapture>>currentFrame;
					videoCapture>>currentFrame;
					//if(scaleSize > 1)		
					//	pyrDown(tempFrame, currentFrame, Size(frameWidth,frameHeight));
					//else
					//	tempFrame.copyTo(currentFrame);
					//resize
					currentMask=vectorToMat(partToCompound[i].objectMask[j-shift[i]],frameHeight,frameWidth);
					//pyrUp(tempMask, currentMask, Size(frameWidth*scaleSize,frameHeight*scaleSize));
					writeMask(currentMask, indexMat, (i+ss*motionToCompound)%256);
					stitch(currentFrame,currentResultFrame,currentResultFrame,backgroundImage,currentMask,partToCompound[i].start,partToCompound[i].end, j);

					//zeroobject
					//bitwise_and(currentStartIndex,zeroObject1,currentStartIndex,currentMask);
					//bitwise_and(currentEndIndex,zeroObject1,currentEndIndex,currentMask); 
					//add(currentStartIndex,partToCompound[i].start,currentStartIndex,currentMask);
					//add(currentEndIndex,partToCompound[i].end,currentEndIndex,currentMask);

					currentMask.release();
				}
			}
			if(earliestIndex>-1){
				int start = partToCompound[earliestIndex].start/framePerSecond;
				int end = partToCompound[earliestIndex].end/framePerSecond;
				vector<Point> info;
				vector<vector<Point>> re_contours;		
				findContours(resultMask,re_contours,CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
				vector<vector<Point>>::const_iterator itc_re=re_contours.begin();
				while(itc_re!=re_contours.end()){
					if(contourArea(*itc_re) < objectarea){
						itc_re=re_contours.erase(itc_re);
					}
					else{
						convexHull(*itc_re,info);
						Point p1 = info.at(1);
						Point p2 = info.at(info.size()/2);
						Point mid;
						mid.x = (p1.x+p2.x)/2;
						mid.y = (p1.y+p2.y)/2;
						if(useROI){
							mid.x += rectROI.x;
							mid.y += rectROI.y;
							//cout<<rectROI.x<<"***"<<rectROI.y<<endl;
						}
						int s1,s2,s3,e1,e2,e3;
						s1=start/3600;
						s2=(start%3600)/60;
						s3=start%60;
						e1=end/3600;
						e2=(end%3600)/60;
						e3=end%60;
						//putText(output,int2string(start)+"-"+int2string(end),mid,CV_FONT_HERSHEY_COMPLEX,0.2, Scalar(0,0,255),1);
						putText(currentResultFrame,int2string(s1)+":"+int2string(s2)+":"+int2string(s3)+"-"+int2string(e1)+":"+int2string(e2)+":"+int2string(e3),mid,CV_FONT_HERSHEY_COMPLEX,0.4, Scalar(0,255,0),1);
						itc_re++;
					}
				}
			}
			//cout<<"earlist index"<<earliestIndex<<endl;
			//cout<<"base index"<<baseIndex<<endl;
			//uchar* pi=indexMat.ptr<uchar>(0);
			//for(int ii=0; ii<5; ii++){
			//	pi[ii]=ii;
			//	cout<<(int)pi[ii]<<":";
			//}
			//vector<int> compression_params;
			//compression_params.push_back(CV_IMWRITE_JPEG_QUALITY);
			//compression_params.push_back(100);
			
			testcount++;
			string filepath=Indexpath+InputName+"/";
			fstream testfile;
			testfile.open(filepath, ios::in);
			if(!testfile){
				boost::filesystem::path dir(filepath);
				boost::filesystem::create_directories(dir);
			}
			string filename=boost::lexical_cast<string>(testcount)+".bmp";

			imwrite(filepath+filename, indexMat);
			videoWriter.write(currentResultFrame);
			//resize
			//Mat check=imread(filepath+filename);
			//pi=check.ptr<uchar>(0);
			////cout<<255-(int)pi[0]<<endl;
			//cout<<"read image"<<endl;
			//for(int ii=0; ii<5; ii++){
			//	cout<<(int)pi[ii]<<":";
			//}
		}
		currentFrame.release();
		currentResultFrame.release();

		//zeroobject
		//zeroObject.release();
		//zeroObject1.release();
		//oneObject.release();
	}
	videoWriter.release();			//  视频合成结束
	LOG(INFO)<<"合成结束\n";
	LOG(INFO)<<"合成耗时"<<clock()-starttime<<"ms\n";
	LOG(INFO)<<"总长度"<<sumLength<<endl;
}

void VideoAbstraction::setVideoFormat(string Format){	//保存视频的格式
	videoFormat = Format;
}

void VideoAbstraction::setGpu(bool isgpu){
	useGpu=isgpu;
}

void VideoAbstraction::setROI(bool isroi){
	useROI=isroi;
}

void VideoAbstraction::writeMask(Mat& input, Mat& output, int index){
	for(int ii=0; ii<input.rows; ii++)
	{
		const uchar* ptr_input=input.ptr<uchar>(ii);
		uchar* ptr_output=output.ptr<uchar>(ii);
		for(int jj=0; jj<input.cols;jj++){
			if(ptr_input[jj]==255)
				ptr_output[jj]=255-index;
				//ptr_output[jj]=index;
		}
	}
}


bool VideoAbstraction::saveContorsOfResultFrameToFile(int frame_Num, cv::Mat& mask, int indexOfMask){
	char fileName[100];
	std::sprintf(fileName, "%d.txt", frame_Num);
	std::ofstream outfile(fileName, ios::ate);
	std::vector<std::vector<cv::Point> >contours;
	cv::findContours(mask, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	outfile << indexOfMask << " " << contorsToString(contours) << std::endl;
	outfile.close();
	return true;
}


cv::Mat VideoAbstraction::loadContorsOfResultFrameFromFile(int frame_Num, int width, int height, vector<int>& lookupTable){
	lookupTable.clear();
	lookupTable.push_back(-1);//

	cv::Mat resultMask(width, height, CV_8UC1, Scalar::all(0));
	char fileName[100];
	std::sprintf(fileName, "%d.txt", frame_Num);
	std::ifstream infile(fileName);

	std::string line;
	while (std::getline(infile, line))
	{
		std::stringstream ss(line);
		std::string sContors;
		int indexOfMask = -1;
		ss << indexOfMask;//每行第一个数值为事件标号，后面为Contors定点坐标
		if (indexOfMask != lookupTable.back())
			lookupTable.push_back(indexOfMask);

		std::getline(ss, sContors);
		std::vector<std::vector<cv::Point> >contours = stringToContors(sContors);
		cv::Mat mask(width, height, CV_8UC1, Scalar::all(0));
		cv::drawContours(mask, contours, -1, Scalar(255), -1);
		restoreMaskOfFram(resultMask, mask, lookupTable.size());
	}

	infile.close();
	return resultMask;
}

bool restoreMaskOfFram(cv::Mat& FrameMask, cv::Mat& oneContors, int index){
	int nc = FrameMask.cols;
	int nl = FrameMask.rows;
	if (oneContors.cols != nc || oneContors.rows != nl)
		return false;
	if (FrameMask.isContinuous() && oneContors.isContinuous())
	{
		nc = nc * nl;
		nl = 1;
	}
	for (int j = 0; j < nl; ++j)
	{
		uchar* c_data = oneContors.ptr<uchar>(j);
		uchar* m_data = FrameMask.ptr(j);
		for (int i = 0; i < nc; ++i)
		{
			if (*c_data++ != 0)
			{
				*m_data++ = index;
			}
		}
	}

	return true;
}