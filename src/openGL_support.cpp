#include <ros/ros.h>
#include <geometry_msgs/Twist.h>
#include <sensor_msgs/Joy.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <GL/glut.h>
#include <dirent.h>
#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>
#include <opencv2/core/cuda.hpp>
#include<GL/freeglut.h>
#include <stdlib.h>
#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include<ros/callback_queue.h>

//�萔
#define LEFT_U (215+robot_trimming_point[0][0])					//���{�b�g����x���W
#define UP_V (160+robot_trimming_point[0][1])					//���{�b�g����y���W
#define RIGHT_U (415+robot_trimming_point[1][0])					//���{�b�g�E��x���W
#define DOWN_V (280+robot_trimming_point[1][1])					//���{�b�g�E��y���W
#define Imgwidth 640								//�L���v�`���摜�̉��T�C�Y
#define Imgheight 480								//�L���v�`���摜�̏c�T�C�Y
#define FRONT_WINDOW_WIDTH 1920		//�O�ʉ摜�\��window�̉��T�C�Y
#define FRONT_WINDOW_HEIGHT 1080	//�O�ʉ摜�\��window�̏c�T�C�Y
#define LEFT_WINDOW_WIDTH 960		//���ʉ摜�\��window�̉��T�C�Y
#define LEFT_WINDOW_HEIGHT 1080		//���ʉ摜�\��window�̏c�T�C�Y
#define RIGHT_WINDOW_WIDTH 960		//�E�ʉ摜�\��window�̉��T�C�Y
#define RIGHT_WINDOW_HEIGHT 1080	//�E�ʉ摜�\��window�̏c�T�C�Y
#define BACK_WINDOW_WIDTH 1024		//��ʉ摜�\��window�̉��T�C�Y
#define BACK_WINDOW_HEIGHT 768		//��ʉ摜�\��window�̏c�T�C�Y
#define WINDOW_NUM 5				//OpenGL�Ő�������window�̐�
#define line_distance_default 0
#define window5_ratio_default 1.5
#define turn_scale 1.0

//�e��֐��̃v���g�^�C�v�錾

int turn_direction_mode(void);
void save_param(void);
void reset_mode(void);
void speed_change(void);
void robot_trimming_size_change(int);
void line_distance_change(void);
void line_point_set(void);
cv::Point2d Bezier_curve(cv::Point2d,cv::Point2d,double);
void one_updown(double& ,int,int&,double);

//OpenCV
void CV_FRONT_FUNC(cv::Mat&);
void CV_CALL_FUNC(int, cv::Mat&);
void tmpImgcreate(cv::Mat&);
cv::Mat warp(int, cv::Mat);
cv::Mat Dividewarp(int, cv::Mat);
void cv_fixed_line(cv::Mat&);
void cv_mode_change(int&);
void cv_move_line(cv::Mat&);
void cv_move_elipse(double, cv::Mat&,int);
void gain(double&, cv::Mat&);
void front_scale_change(double&,cv::Mat&);
void cv_mode(cv::Mat&);
void cv_turn_mode(cv::Mat&);
void cv_draw_target(cv::Mat&);
void cv_save_img(int, cv::Mat);
void cv_turn_mat(cv::Mat&, double);
void turn_change(void);

//OpenGL
void mainLoop(void);
void GLUT_INIT1(int&, const char*);
void GLUT_INIT2(int&, const char*);
void GLUT_INIT3(int&, const char*);
void GLUT_INIT4(int&, const char*);
void GLUT_CALL_FUNC1(void);
void GLUT_CALL_FUNC2(void);
void GLUT_CALL_FUNC3(void);
void GLUT_CALL_FUNC4(void);
void GLUT_CALL_FUNC5(void);
void display1(void);
void display2(void);
void display3(void);
void display4(void);
void idle(void);
void GLUT_INIT5(int&, const char*);
void display5(void);
void resize1(int ,int);
void resize2(int ,int);
void resize3(int ,int);
void resize4(int ,int);
void resize5(int ,int);

//joystic���씻��
bool joystick_buttons_save(void);

void data_save(int64, int64);

//OpenGL�������p�֐��|�C���^
void(*GLUT_INIT_Ary[])(int&, const char*) = { GLUT_INIT1,GLUT_INIT2,GLUT_INIT3,GLUT_INIT4,GLUT_INIT5};
void(*GLUT_CALL_FUNC_Ary[])() = { GLUT_CALL_FUNC1,GLUT_CALL_FUNC2,GLUT_CALL_FUNC3,GLUT_CALL_FUNC4,GLUT_CALL_FUNC5 };

//�O���[�o���ϐ�
int WinID[WINDOW_NUM];				//��������window���Ǘ�����z��
const char *WindowName[] = { "frontImg","leftImg","rightImg","backImg","aroundImg" };		//��������window�̖��O���Ǘ�����z��A�h���X
int face[] = { cv::FONT_HERSHEY_SIMPLEX, cv::FONT_HERSHEY_PLAIN, cv::FONT_HERSHEY_DUPLEX, cv::FONT_HERSHEY_COMPLEX,			//opencv�̕����t�H���g�f�[�^�p�z��
cv::FONT_HERSHEY_TRIPLEX, cv::FONT_HERSHEY_COMPLEX_SMALL, cv::FONT_HERSHEY_SCRIPT_SIMPLEX,
cv::FONT_HERSHEY_SCRIPT_COMPLEX, cv::FONT_ITALIC };
cv::Mat srcImg;		//���摜�ۑ�Mat
cv::Mat copyImg;	//��Ɖ摜�ۑ�Mat
int mode = 0;						//���샂�[�h�ۑ��p�ϐ�
GLfloat red[] = { 1.0,0.0,0.0,1.0 };
GLfloat lightpos[] = { 0.0, 0.0, 0.0, 1.0 };//���C�g�̈ʒu
int line_mode=0;
int Buttons_data[13]={};
int Rxyz_data[3]={};
bool low_speed=0;
bool display5_view_mode=0;
double robot_trimming_point[2][2]={};
bool window_change_mode[4]={};
int window_size[4][2]={};
double line_distance=line_distance_default;
cv::Point2d left_line_points[3];
cv::Point2d right_line_points[3];
double gain_k=1,front_scale=1;
double window5_ratio=window5_ratio_default;
const cv::Point2d windowPos[WINDOW_NUM] = {	//����window�ʒu�̌���
	{1920,0},
	{970,0},
	{3840,0},
	{2364,1080},
	{0,0}
};

//�f�[�^�Ǘ��p
std::ofstream fout("/home/mouse/data/0115_data.csv");		//�f�[�^���o�͂���t�@�C���̍쐬

class joydata{
	public:
	joydata();
	double X;
	double Y;
	double Z;
	double Rx;
	double Ry;
	double Rz;
	double slider;
	int pov[4];
	int Buttons[13];
};

joydata::joydata()
:X(0),Y(0),Z(0),Rx(0),Ry(0),Rz(0),slider(0){
	for(int i=0;i<13;++ i){
		if(i<4)
			pov[i]=0;
		Buttons[i]=0;
	}
}

joydata joyinput={};

class joystick{
	public:
		double X;
		double Y;
		double Z;
		double Rx;
		double Ry;
		double Rz;
		double slider;
		int pov[4];
		int Buttons[13];
		joystick();
		
	private:
		void joyCallback(const sensor_msgs::Joy::ConstPtr& joy);
		ros::NodeHandle nh;
		int vel_linear,vel_angular,vel_viceangular;
		double l_scale_,a_scale_;
		ros::Subscriber joy_sub_;
};

joystick::joystick():vel_linear(1),vel_angular(0),l_scale_(1), a_scale_(1), vel_viceangular(2)
{
	joy_sub_=nh.subscribe<sensor_msgs::Joy>("joy",10,&joystick::joyCallback,this);
}

void joystick::joyCallback(const sensor_msgs::Joy::ConstPtr& joy)
{
	if(low_speed){
		a_scale_=0.5;
		l_scale_=0.5;
	}
	else {
		a_scale_=1;
		l_scale_=1;
	}
	joyinput.X=l_scale_*joy->axes[vel_linear];
	switch (mode)
	{
		case 1:
			joyinput.Y=a_scale_*joy->axes[vel_viceangular];
			break;
		
		case 2:
			if(joyinput.X==0)
				joyinput.Y=a_scale_*joy->axes[2];
			else joyinput.Y=a_scale_*joy->axes[0];
			break;
		
		case 0:
		default:
			joyinput.Y=a_scale_*joy->axes[vel_angular];
	}

	joyinput.Rx=joy->axes[5];
	joyinput.Ry=joy->axes[4];

	if(joy->axes[5]==-1){
		joyinput.X*=-1;
		joyinput.Y*=-1;
	}

	for(int i=0;i<13;++i)
		joyinput.Buttons[i]=joy->buttons[i];
	
	ROS_INFO_STREAM("("<<joyinput.X<<""<<joyinput.Y<<")");
}

class Omnidirectional_Img{
	public:
		Omnidirectional_Img();
		void callOne(void);
	private:
		void imageCallback(const sensor_msgs::ImageConstPtr& msg);
		ros::NodeHandle nh;
		ros::CallbackQueue queue;
		ros::Subscriber image_sub;
};

int main(int argc,char** argv)
{
	ros::init(argc,argv,"openGL_support");

	Omnidirectional_Img capture;
	joystick a;

	//opengl:GLUT�֘A�̏�����
 	glutInit(&argc,argv);
	 for (int i = 0; i < WINDOW_NUM; i++)
	{
		(*GLUT_INIT_Ary[i])(WinID[i], WindowName[i]);
		(*GLUT_CALL_FUNC_Ary[i])();
	}

	//���C�����[�v
	while (ros::ok())
	{
		int64 start = cv::getTickCount();						//���[�J���^�C���̎擾
		
		ros::spinOnce();

		capture.callOne();
		
		if(!joyinput.Buttons[6]&&!joyinput.Buttons[8]&&!joyinput.Buttons[10]&&!joyinput.Buttons[7]&&!joyinput.Buttons[9]){
			turn_change();
			speed_change();
			cv_mode_change(mode);					//���[�h�̌���
		}

		if(joyinput.Buttons[6])
			robot_trimming_size_change(1);
		if(joyinput.Buttons[8])
			robot_trimming_size_change(-1);
		if(joyinput.Buttons[6]||joyinput.Buttons[8])
			tmpImgcreate(copyImg);							//tmpImg�����
		if(joyinput.Buttons[10])
			line_distance_change();

		if(joyinput.Buttons[1])
			reset_mode();

		line_point_set();

		glutMainLoopEvent();								//OpenGL�̃C�x���g�̊J�n
		idle();												//�edisplay�֐��̌Ăяo��
		
		std::cout<<mode<<std::endl;
		int64 end = cv::getTickCount();						//���[�J���^�C���̎擾
		data_save(start, end);								//�f�[�^�̕ۑ�
		if (joystick_buttons_save())
		{
			fout.close();									//�t�@�C���̕ۑ�
			srcImg.copyTo(copyImg);
			tmpImgcreate(copyImg);							//tmpImg�����
			imwrite("/home/mouse/data/srcImg.png", srcImg);					//srcImg�̕ۑ�
			imwrite("/home/mouse/data/tmpImg.png", copyImg);					//copyImg�̕ۑ�
			save_param();
			exit(0);										//���[�v�𔲂���
		}
	}

	return 0;
}

int turn_direction_mode(void){
	if (joyinput.Rx==1)
		return 0;
	else if(joyinput.Rx==-1)
		return 1;
	else if(joyinput.Ry==-1)
		return 2;
	else if(joyinput.Ry==1)
		return 3;
}

void speed_change(void){
	if(joyinput.Buttons[2]&&!Buttons_data[2]&&!low_speed){
		++low_speed;
		++Buttons_data[2];
	}
	else if(joyinput.Buttons[2]&&Buttons_data[2]==0){
		low_speed=0;
		++Buttons_data[2];
	}
	else if(!joyinput.Buttons[2])
		Buttons_data[2]=0;
}

Omnidirectional_Img::Omnidirectional_Img(){
	ros::NodeHandle nh("~");
	// image_transport::ImageTransport it(nh);
	// image_transport::Subscriber image_sub = it.subscribe("/img_publisher/image", 10, &Omnidirectional_Img::imageCallback,this);
	nh.setCallbackQueue(&queue);
	image_sub = nh.subscribe("/cv_bridge_omnidirectional_cam/image", 10, &Omnidirectional_Img::imageCallback,this);
}
void Omnidirectional_Img::callOne(void){
	queue.callOne(ros::WallDuration(100));
}

void Omnidirectional_Img::imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try {
		srcImg = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8)->image;
		if(!srcImg.empty())
			srcImg.copyTo(copyImg);			//���摜����Ɨp�ɃR�s�[
	}
	catch (cv_bridge::Exception& e) {
		ROS_ERROR("cv_bridge exception: %s", e.what());
	}
 }

void save_param(void){
	std::ofstream ofs ("/home/mouse/data/last_param.csv");		//�f�[�^���o�͂���t�@�C���̍쐬
	ofs<<gain_k<<std::endl;
	ofs<<front_scale<<std::endl;
	for(int i=0;i<2;++i)
		for(int j=0;j<2;++j)
			ofs<<robot_trimming_point[i][j]<<std::endl;;
	ofs<<line_distance<<std::endl;
	ofs.close();
}
void reset_mode(void){
	std::ifstream ifs("/home/mouse/data/last_param.csv");
	mode=0;
	low_speed=0;
	ifs>>gain_k;
	ifs>>front_scale;
	for(int i=0;i<2;++i)
		for(int j=0;j<2;++j)
			ifs>>robot_trimming_point[i][j];
	ifs>>line_distance;
	ifs.close();
}

void robot_trimming_size_change(int up_down){
	if(joyinput.Buttons[4])
		robot_trimming_point[0][0]-=up_down;
	if(joyinput.Buttons[2])
		robot_trimming_point[0][1]-=up_down;
	if(joyinput.Buttons[5])
		robot_trimming_point[1][0]+=up_down;
	if(joyinput.Buttons[3])
		robot_trimming_point[1][1]+=up_down;
}

void line_distance_change(void){
	if(joyinput.Buttons[4])
		++line_distance;
	if(joyinput.Buttons[2])
		--line_distance;
}

void line_point_set(void){
	double left_start=UP_V,right_start=UP_V;
	left_start-=((UP_V/LEFT_U)*line_distance);
	right_start-=((UP_V/(srcImg.cols-RIGHT_U))*line_distance);
	left_line_points[0]={LEFT_U-line_distance, left_start};
	right_line_points[0]={RIGHT_U+line_distance,UP_V-((UP_V/(srcImg.cols-RIGHT_U))*line_distance)};
	left_line_points[1]={LEFT_U-line_distance, (1-joyinput.X*front_scale)*left_start};
	right_line_points[1]={RIGHT_U+line_distance, (1-joyinput.X*front_scale)*right_start};
	left_line_points[2]={(1-joyinput.Y*gain_k)*(LEFT_U-line_distance), (1-joyinput.X*front_scale)*left_start};
	right_line_points[2]={(RIGHT_U+line_distance)-(joyinput.Y*gain_k*(RIGHT_U-line_distance)), (1-joyinput.X*front_scale)*right_start};
}

cv::Point2d Bezier_curve(cv::Point2d start_point,cv::Point2d end_point,double t){
	cv::Point2d ans;
	ans.x=(1-t)*start_point.x+t*end_point.x;
	ans.y=(1-t)*start_point.y+t*end_point.y;

	return ans;
}

void one_updown(double& change,int Buttons,int& data,double up_down){
	if(Buttons&&data==0&&change==0){
		change+=up_down;
		++data;
	}
	else if(!Buttons)
		data=0;
}

//OpenCV�֐��̂܂Ƃ�
//OpenCV�ł̏����pCALL�֐�
void CV_FRONT_FUNC(cv::Mat &Img)
{
		if(line_mode==0||line_mode==1){
			if(line_mode==0)
				cv_mode(Img);						//��ʂɑ��샂�[�h��\��
			if(low_speed)
				putText(Img, cv::format("Low Speed Mode"), cv::Point(window_size[0][0]-420, 100), face[3], 1.0, CV_RGB(180, 180, 255), 4, CV_AA);
			if ((joyinput.Buttons[3] || joyinput.Buttons[5]||joyinput.Buttons[4]||joyinput.Buttons[2])&&!joyinput.Buttons[6]&&!joyinput.Buttons[8]&&!joyinput.Buttons[10]&&joyinput.Buttons[7]&&!joyinput.Buttons[9]){
				if(joyinput.Buttons[3] || joyinput.Buttons[5])
					gain(gain_k,Img);							//�\���Q�C���̕ύX
				else front_scale_change(front_scale,Img);
			}
		}
		if(line_mode!=0)
			cv_turn_mode(Img);
		if ((joyinput.Buttons[3] || joyinput.Buttons[5]||joyinput.Buttons[0])&&!joyinput.Buttons[6]&&!joyinput.Buttons[8]&&!joyinput.Buttons[10]&&!joyinput.Buttons[7]&&!joyinput.Buttons[9])
			cv_draw_target(Img);					//joystick�̃{�^���ɍ��킹�ă^�[�Q�b�g�̏�Ԃ�\��
}

void CV_CALL_FUNC(int ID, cv::Mat &Img)
{
	if (joystick_buttons_save())					//�ۑ��{�^���������ꂽ���𔻒�
		cv_save_img(ID, Img);						//�摜�̕ۑ�
	flip(Img, Img, 0);								//Opencv�̉摜��OpenGL�ɍ����悤�ɍ��W����ύX
	cvtColor(Img, Img, cv::COLOR_BGR2RGB);				//Opencv�̉摜��OpenGL�ɍ����悤�ɉ摜�F��ύX
}
//�؂���͈͂�\������tmpImg�����֐�
void tmpImgcreate(cv::Mat &tmp)
{
	rectangle(tmp, cv::Point(LEFT_U, UP_V), cv::Point(RIGHT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);		//���{�b�g���l�p�ň͂�
	line(tmp, cv::Point(0, 0), cv::Point(LEFT_U, UP_V), CV_RGB(0, 0, 255), 2, 4);						//�ȉ��ŋ�؂��������
	line(tmp, cv::Point(Imgwidth, 0), cv::Point(RIGHT_U, UP_V), CV_RGB(0, 0, 255), 2, 4);
	line(tmp, cv::Point(0, Imgheight), cv::Point(LEFT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);
	line(tmp, cv::Point(Imgwidth, Imgheight), cv::Point(RIGHT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);
}
//�����ϊ��p�֐�
cv::Mat warp(int ID, cv::Mat warpImg)
{
	int out_width=0,out_height=0;
	cv::Mat map_matrix;																								//�ϊ��s��ۑ��pMat
	cv::Mat gsrcImg;																						//src�摜�pMat
	cv::Point2f src_pnt[4], dst_pnt[4];																				//�ϊ��O����ѕϊ���̒��_�ۑ��p�ϐ�
	gsrcImg=warpImg.clone();																					//GPU�Ɍ��摜�f�[�^�𑗐M
	switch (ID)
	{
	case 0:																													//�O��
	{
		out_width=window_size[0][0];
		out_height=window_size[0][1];
		dst_pnt[0] = cv::Point2f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cv::Point2f(out_width, 0.0);																	//�o�͉E����W
		dst_pnt[2] = cv::Point2f(out_width, out_height);														//�o�͉E�����W
		dst_pnt[3] = cv::Point2f(0.0, out_height);																	//�o�͍������W
		src_pnt[0] = cv::Point2f(0,0);
		src_pnt[1] = cv::Point2f(Imgwidth,0);
		src_pnt[2] = cv::Point2f(RIGHT_U,UP_V);
		src_pnt[3] = cv::Point2f(LEFT_U,UP_V);
	}
		break;
	case 1:																													//����
	{
		out_width=window_size[1][0];
		out_height=window_size[1][1];
		dst_pnt[0] = cv::Point2f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cv::Point2f(out_width, 0.0);																	//�o�͉E����W				
		dst_pnt[2] = cv::Point2f(out_width, out_height);														//�o�͉E�����W
		dst_pnt[3] = cv::Point2f(0.0, out_height);																	//�o�͍������W
		src_pnt[0] = cv::Point2f(0,Imgheight);
		src_pnt[1] = cv::Point2f(0,0);
		src_pnt[2] = cv::Point2f(LEFT_U,UP_V);
		src_pnt[3] = cv::Point2f(LEFT_U,DOWN_V);
	}
		break;
	case 2:																													//�E��
	{
		out_width=window_size[2][0];
		out_height=window_size[2][1];
		dst_pnt[0] = cv::Point2f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cv::Point2f(out_width, 0.0);																	//�o�͉E����W
		dst_pnt[2] = cv::Point2f(out_width, out_height);														//�o�͉E�����W
		dst_pnt[3] = cv::Point2f(0.0, out_height);																	//�o�͍������W
		src_pnt[0] = cv::Point2f(Imgwidth,0);
		src_pnt[1] = cv::Point2f(Imgwidth,Imgheight);
		src_pnt[2] = cv::Point2f(RIGHT_U,DOWN_V);
		src_pnt[3] = cv::Point2f(RIGHT_U,UP_V);
	}
		break;
	case 3:																													//���
	{
		out_width=window_size[3][0];
		out_height=window_size[3][1];
		dst_pnt[0] = cv::Point2f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cv::Point2f(out_width, 0.0);																	//�o�͉E����W
		dst_pnt[2] = cv::Point2f(out_width, out_height);														//�o�͉E�����W
		dst_pnt[3] = cv::Point2f(0.0, out_height);																	//�o�͍������W
		src_pnt[0] = cv::Point2f(0,Imgheight);
		src_pnt[1] = cv::Point2f(Imgwidth,Imgheight);
		src_pnt[2] = cv::Point2f(RIGHT_U,DOWN_V);
		src_pnt[3] = cv::Point2f(LEFT_U,DOWN_V);
	}
	}
	if(ID>=0&&ID<=3){
		warpImg= cv::Mat(cv::Size(out_width, out_height), CV_8UC3);																//gminiImg�̐ݒ�
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//�ϊ��s������߂�
		cv::warpPerspective(gsrcImg, warpImg, map_matrix,warpImg.size());							//�����ϊ��̎��s
	}
	return warpImg;
}

//�摜�ɌŒ蒼���������֐�
void cv_fixed_line(cv::Mat &Img)
{
	cv::line(Img, left_line_points[0] , cv::Point(LEFT_U-line_distance, 0), CV_RGB(0, 255, 0), 2, 4);				//��
	cv::line(Img, right_line_points[0] , cv::Point(RIGHT_U+line_distance, 0), CV_RGB(0, 255, 0), 2, 4);				//�E
}

//���샂�[�h�̕ύX���s���֐�
void cv_mode_change(int& mode)
{
	if(joyinput.Buttons[4]&&Buttons_data[4]==0&&(mode==0||mode==1)){
		++mode;																							//X.Y���[�h��X.Rz���[�h��X.Y.Rz���[�h��X.Y���[�h
		++Buttons_data[4];
	}
	else if(joyinput.Buttons[4]&&Buttons_data[4]==0){
		mode=0;
		++Buttons_data[4];
	}
	else if(joyinput.Buttons[4]==0)
		Buttons_data[4]=0;
}
//���i���̐��������֐�
void cv_move_line(cv::Mat &Img)
{
	cv::line(Img, left_line_points[0], left_line_points[1] , CV_RGB(255, 0, 0), 2, 4);		//��
	cv::line(Img, right_line_points[0] , right_line_points[1] , CV_RGB(255, 0, 0), 2, 4);		//�E
}
//�J�[�u���̐��������֐�
void cv_move_elipse(double k, cv::Mat &Img,int mode)
{
	 if (abs(joyinput.Y *1000)>= 10 && joyinput.X*1000 >= 50)
	{
		cv::Point2d left_point,right_point, last_left=left_line_points[0] ,last_right=right_line_points[0];
		for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
		{
			left_point=Bezier_curve(Bezier_curve(left_line_points[0],left_line_points[1],t),Bezier_curve(left_line_points[1],left_line_points[2],t),t);
			cv::line(Img, last_left , left_point , cv::Scalar(255, 0, 0),2);
			last_left=left_point;
			right_point=Bezier_curve(Bezier_curve(right_line_points[0],right_line_points[1],t),Bezier_curve(right_line_points[1],right_line_points[2],t),t);
			cv::line(Img, last_right, right_point , cv::Scalar(255, 0, 0),2);
			last_right=right_point;
		}
	}
	else if (joyinput.Y <0 && joyinput.X == 0)	//�E���`��p
	{
		cv::line(Img, cv::Point(Img.cols / 2 - 10 - (joyinput.Y*0.001 * 20*-1000), UP_V / 2), cv::Point(Img.cols / 2 + 10 + (joyinput.Y*0.001 * 20*-1000), UP_V / 2), cv::Scalar(255, 0, 0),2);
		cv::line(Img, cv::Point(Img.cols / 2 + 10 + (joyinput.Y*0.001 * 20*-1000), UP_V / 2), cv::Point(Img.cols / 2 + 5 + (joyinput.Y*0.001 * 20*-1000), UP_V / 2-10), cv::Scalar(255, 0, 0),2);
		cv::line(Img, cv::Point(Img.cols / 2 + 10 + (joyinput.Y*0.001 * 20*-1000), UP_V / 2), cv::Point(Img.cols / 2 + 5 + (joyinput.Y*0.001 * 20*-1000), UP_V / 2+10), cv::Scalar(255, 0, 0),2);
	}
	else if (joyinput.Y >0 && joyinput.X == 0)	//�����`��p
	{
		cv::line(Img, cv::Point(Img.cols / 2 - 50 + (joyinput.Y*0.001 * 10*-1000), UP_V / 2), cv::Point(Img.cols / 2 + 50 - (joyinput.Y*0.001 * 10*-1000), UP_V / 2), cv::Scalar(255, 0, 0),2);
		cv::line(Img, cv::Point(Img.cols / 2 - 50 + (joyinput.Y*0.001 * 10*-1000), UP_V / 2), cv::Point(Img.cols / 2 - 25 + (joyinput.Y*0.001 * 10*-1000), UP_V / 2 - 10), cv::Scalar(255, 0, 0),2);
		cv::line(Img, cv::Point(Img.cols / 2 - 50 + (joyinput.Y*0.001 * 10*-1000), UP_V / 2), cv::Point(Img.cols / 2 - 25 + (joyinput.Y*0.001 * 10*-1000), UP_V / 2 + 10), cv::Scalar(255, 0, 0),2);
	}
}
//�J�[�u�̋Ȃ����𒲐�����֐�
void gain(double& k,cv::Mat &Img)
{
	std::string str;																		//������
	if (joyinput.Buttons[3] == 1) 
		k -= 0.1;											//�Q�C���̑���
	else if (joyinput.Buttons[5] == 1) 
		k += 0.1;
	if(k<=0)
		k=0;
	str = cv::format("Curvature correction=%.1lf", k);									//�Q�C���𕶎���ɒ���
	putText(Img, str, cv::Point(window_size[0][0]-520, 150), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//�`��
}

void front_scale_change(double& front_scale,cv::Mat& Img){
	std::string str;																		//������
	if (joyinput.Buttons[2] == 1) 
		front_scale -= 0.1;											//�Q�C���̑���
	else if (joyinput.Buttons[4] == 1) 
		front_scale += 0.1;
	if(front_scale<=0)
		front_scale=0;
	str = cv::format("Front sensitivity=%.1lf", front_scale);									//�Q�C���𕶎���ɒ���
	putText(Img, str, cv::Point(window_size[0][0]-470, 175), face[3], 1.0, CV_RGB(255, 200, 200), 4, CV_AA);	//�`��
}

//���[�h�̕`����s���֐�
void cv_mode(cv::Mat &Img)
{
	static std::string str[3] = { "mode_X,Y", "mode_X,Rz", "mode_X,Y,Rz" };						//�������`
	putText(Img, str[mode], cv::Point(window_size[0][0]-320, 50), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//�`��
}

void cv_turn_mode(cv::Mat &Img){
	static std::string str[3] = { "back","right" , "left" };						//�������`
	putText(Img, str[line_mode-1], cv::Point(window_size[0][0]-320, 50), face[3], 1.0, CV_RGB(0, 255, 255), 4, CV_AA);	//�`��
}

//�^�[�Q�b�g�̏�Ԃ���ʂɕ\������֐�
void cv_draw_target(cv::Mat &Img)
{
	if (joyinput.Buttons[5])
		putText(Img, "Safe", cv::Point(Img.cols/2,Img.rows/2), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//�`��
	else if (joyinput.Buttons[3])
		putText(Img, "Broken", cv::Point(Img.cols / 2, Img.rows / 2), face[3], 1.0, CV_RGB(255, 0, 0), 4, CV_AA);	//�`��
	if(joyinput.Buttons[0]){
		putText(Img, "Start/Stop", cv::Point(Img.cols / 2, Img.rows / 2), face[3], 1.0, CV_RGB(255, 0, 0), 4, CV_AA);	//�`��
	}
}
//�f�[�^��ۑ�����֐�
void cv_save_img(int ID, cv::Mat Img)
{
	switch (ID)
	{
	case 0:
		imwrite("/home/mouse/data/frontImg.png", Img);
		break;
	case 1:
		imwrite("/home/mouse/data/leftImg.png", Img);
		break;
	case 2:
		imwrite("/home/mouse/data/rightImg.png", Img);
		break;
	case 3:
		imwrite("/home/mouse/data/backImg.png", Img);
	}
}

void cv_turn_mat(cv::Mat &Img, double degree){
	cv::Point2f center = cv::Point2f(static_cast<float>(Img.cols / 2),
    static_cast<float>(Img.rows / 2));
 
     // �A�t�B���ϊ��s��
    cv::Mat affine;
    cv::getRotationMatrix2D(center, degree, turn_scale).copyTo(affine);
 
    cv::warpAffine(Img, Img, affine, Img.size(), cv::INTER_CUBIC);
}

void turn_change(void){
	line_mode=turn_direction_mode();
		switch(line_mode){
			case 1:	cv_turn_mat(copyImg, 180);
					break;
			case 2: cv_turn_mat(copyImg, 90);
					break;
			case 3: cv_turn_mat(copyImg, -90);
		}
}

//OpenGL��������
//display��������������
void GLUT_INIT1(int &ID, const char *name)							//�O��
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//�`����@�̐ݒ�
	glutInitWindowPosition(windowPos[0].x, windowPos[0].y);			//window�ʒu�̐ݒ�
	glutInitWindowSize(FRONT_WINDOW_WIDTH, FRONT_WINDOW_HEIGHT);	//window�T�C�Y�̐ݒ�
	ID = glutCreateWindow(name);									//window�ɖ��O�Ɣԍ���U�蕪��
	glEnable(GL_LIGHTING);											//�����ݒ��on�ɂ���
	glEnable(GL_LIGHT0);											//��ڂ̌���
	glEnable(GL_DEPTH_TEST);										//�A�ʏ����̐ݒ�
	glEnable(GL_BLEND);												//�������ݒ��on�ɂ���
	glEnable(GL_NORMALIZE);											//�@���x�N�g���������I�ɐ��K���i���_�̌����ɑ΂�����������肵�Đ^�����ȃ|���S���ɂȂ�̂�h���j
}
void GLUT_INIT2(int &ID, const char *name)							//����
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//�`����@�̐ݒ�
	glutInitWindowPosition(windowPos[1].x, windowPos[1].y);			//window�ʒu�̐ݒ�
	glutInitWindowSize(LEFT_WINDOW_WIDTH, LEFT_WINDOW_HEIGHT);		//window�T�C�Y�̐ݒ�
	ID = glutCreateWindow(name);									//window�ɖ��O�Ɣԍ���U�蕪��
}
void GLUT_INIT3(int &ID, const char *name)							//�E��
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//�`����@�̐ݒ�
	glutInitWindowPosition(windowPos[2].x, windowPos[2].y);			//window�ʒu�̐ݒ�
	glutInitWindowSize(RIGHT_WINDOW_WIDTH, RIGHT_WINDOW_HEIGHT);	//window�T�C�Y�̐ݒ�
	ID = glutCreateWindow(name);									//window�ɖ��O�Ɣԍ���U�蕪��
}
void GLUT_INIT4(int &ID, const char *name)							//���
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//�`����@�̐ݒ�
	glutInitWindowPosition(windowPos[3].x, windowPos[3].y);			//window�ʒu�̐ݒ�
	glutInitWindowSize(BACK_WINDOW_WIDTH, BACK_WINDOW_HEIGHT);		//window�T�C�Y�̐ݒ�
	ID = glutCreateWindow(name);									//window�ɖ��O�Ɣԍ���U�蕪��
}
//display�R�[���o�b�N�֐��ݒ肱���܂�
//�E�B���h�E�X�V�p�֐�
void idle(void)
{
	for (int i = 0; i < WINDOW_NUM; ++i) {
		glutSetWindow(WinID[i]);	//�X�V����window�̃Z�b�g
		glutPostRedisplay();		//�X�V�̎��s
	}
}

//joystick���͔���p�֐��A��������
bool joystick_buttons_save(void)
{
	if (joyinput.Buttons[11])
		return true;
	else
		return false;
}
//�`��p�֐������܂�
//display�R�[���o�b�N�֐��ݒ肱������
void GLUT_CALL_FUNC1(void)
{
	glutDisplayFunc(display1);	//window��display�����蓖��
	glutReshapeFunc(resize1);
}
void GLUT_CALL_FUNC2(void)
{
	glutDisplayFunc(display2);	//window��display�����蓖��
	glutReshapeFunc(resize2);
}
void GLUT_CALL_FUNC3(void)
{
	glutDisplayFunc(display3);	//window��display�����蓖��
	glutReshapeFunc(resize3);
}
void GLUT_CALL_FUNC4(void)
{
	glutDisplayFunc(display4);	//window��display�����蓖��
	glutReshapeFunc(resize4);
}
void GLUT_CALL_FUNC5(void)
{
	glutDisplayFunc(display5);	//window��display�����蓖��
	glutReshapeFunc(resize5);
}

//display�����������܂�
//�`��p�֐���������
void display1(void)
{	
	if(line_mode==0||line_mode==1){
		cv_fixed_line(copyImg);							//�Œ�̐���`��
		if (joyinput.X!=0||joyinput.Y!=0)
		{
			if (joyinput.Y == 0){
				if(joyinput.X>0)
					cv_move_line(copyImg);				//����
			}
			else
				cv_move_elipse( gain_k, copyImg, mode);	//�Ȑ�
		}
	}																								//�O���
	cv::Mat frontImg = warp(0, copyImg);																//�����ϊ������摜�̗p��
																									//OpenGL�ł̕`��ݒ�A��������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glViewport(0, 0, FRONT_WINDOW_WIDTH, FRONT_WINDOW_HEIGHT);										//viewport�̐ݒ�
	glMatrixMode(GL_PROJECTION);																	//���e�ϊ����[�h
	glLoadIdentity();																				//���e�ϊ��̕ϊ��s���P�ʍs��ŏ�����
	gluPerspective(30.0, (double)FRONT_WINDOW_WIDTH / (double)FRONT_WINDOW_HEIGHT, 1.0, 1000.0);	//���E�̌���
	glMatrixMode(GL_MODELVIEW);																		//���f���r���[�ϊ��s��̐ݒ�
	glLoadIdentity();																				//���f���r���[�ϊ��s���P�ʍs��ŏ�����
	gluLookAt(0.0, 0.0, 65.0, //�J�����̍��W
		8.0, 180.0, 0, // �����_�̍��W
		0.0, 0.0, 1.0); // ��ʂ̏�������w���x�N�g��
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);													//���C�g�𓖂Ă�
	CV_FRONT_FUNC(frontImg);
	CV_CALL_FUNC(0, frontImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(frontImg.cols, frontImg.rows, GL_RGB, GL_UNSIGNED_BYTE, frontImg.data);			//�`�揈��
	glClear(GL_DEPTH_BUFFER_BIT);																	//�f�v�X�o�b�t�@�̃N���A
	glutSwapBuffers();																				//��ʂ��X�V
																									//OpenGL�ł̕`��ݒ�A��������
}
void display2(void)
{																									//�����
	cv::Mat leftImg = warp(1, copyImg);																	//�����ϊ������摜�̗p��
																									//OpenGL�ł̕`��ݒ�A��������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glLoadIdentity();																				//�P�ʍs��̏�����
	CV_CALL_FUNC(1, leftImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(leftImg.cols, leftImg.rows, GL_RGB, GL_UNSIGNED_BYTE, leftImg.data);				//�`�揈��
	glutSwapBuffers();																				//��ʂ��X�V//OpenGL�ł̕`��ݒ�A�����܂�
}
void display3(void)
{																									//�E���
	cv::Mat rightImg = warp(2, copyImg);																//�����ϊ������摜�̗p��
																									//OpenGL�ł̕`��ݒ�A��������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glLoadIdentity();																				//�P�ʍs��̏�����
	CV_CALL_FUNC(2, rightImg);																	//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(rightImg.cols, rightImg.rows, GL_RGB, GL_UNSIGNED_BYTE, rightImg.data);			//�`�揈��
	glutSwapBuffers();																			//��ʂ��X�V  OpenGL�ł̕`��ݒ�A�����܂�
}
void display4(void)
{																								//����
	cv::Mat backImg = warp(3, copyImg);																//�����ϊ������摜�̗p��
																								//OpenGL�ł̕`��ݒ�A��������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glLoadIdentity();																				//�P�ʍs��̏�����
	CV_CALL_FUNC(3, backImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(backImg.cols, backImg.rows, GL_RGB, GL_UNSIGNED_BYTE, backImg.data);				//�`�揈��
	glutSwapBuffers();																				//��ʂ��X�V  OpenGL�ł̕`��ݒ�A�����܂�
}

void GLUT_INIT5(int &ID, const char *name)							
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//�`����@�̐ݒ�
	glutInitWindowPosition(windowPos[4].x, windowPos[4].y);			//window�ʒu�̐ݒ�
	glutInitWindowSize(Imgwidth*window5_ratio, Imgheight*window5_ratio);		//window�T�C�Y�̐ݒ�
	ID = glutCreateWindow(name);									//window�ɖ��O�Ɣԍ���U�蕪��
}

void display5(void)
{	
		flip(copyImg, copyImg, 0);								//Opencv�̉摜��OpenGL�ɍ����悤�ɍ��W����ύX
		cvtColor(copyImg, copyImg, cv::COLOR_BGR2RGB);				//Opencv�̉摜��OpenGL�ɍ����悤�ɉ摜�F��ύX																
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
		glLoadIdentity();																				//�P�ʍs��̏�����
		std::cout<<window5_ratio<<std::endl;
		if(window5_ratio!=1)
			cv::resize(copyImg,copyImg,cv::Size(),window5_ratio,window5_ratio);
		glDrawPixels(copyImg.cols, copyImg.rows, GL_RGB, GL_UNSIGNED_BYTE, copyImg.data);				//�`�揈��
		glutSwapBuffers();																				//��ʂ��X�V
}
void resize1(int w,int h){
	printf("resize has been called. w=%d\th=%d\n", w, h);
	window_size[0][0]=w-(w%4);
	window_size[0][1]=h;
}
void resize2(int w,int h){
	printf("resize has been called. w=%d\th=%d\n", w, h);
	window_size[1][0]=w-(w%4);
	window_size[1][1]=h;
}
void resize3(int w,int h){
	printf("resize has been called. w=%d\th=%d\n", w, h);
	window_size[2][0]=w-(w%4);
	window_size[2][1]=h;
}
void resize4(int w,int h){
	printf("resize has been called. w=%d\th=%d\n", w, h);
	window_size[3][0]=w-(w%4);
	window_size[3][1]=h;
}
void resize5(int w,int h){
	printf("resize has been called. w=%d\th=%d\n", w, h);
	int width_tmp;
		window5_ratio=(double)w/Imgwidth;
	if(window5_ratio>(double)h/Imgheight)
		window5_ratio=(double)h/Imgheight;
	width_tmp=Imgwidth*window5_ratio;
	if(width_tmp%4){
		width_tmp-=width_tmp%4;
		window5_ratio=(double)width_tmp/Imgwidth;
	}
}

//�f�[�^�ۑ��p�֐��A�g���Ă�
void data_save(int64 s, int64 e)
{
	double msec = (e - s) * 1000 / cv::getTickFrequency();	//�v���O�����̎���
	std::cout << msec << std::endl;
	fout << joyinput.X<<",";									// X���f�[�^
	fout << joyinput.Y<<",";									// Y���f�[�^
	fout << mode<<",";										// ���[�h�f�[�^
	fout << msec<<",";										// �v���O�����쓮����
	if (joyinput.Buttons[5] == 1){						//�T����Ԃ̕\��
		fout << "Safe"<<",";
	}
	else if (joyinput.Buttons[3] == 1){
		fout << "Broken"<<",";
	}
	if(joyinput.Buttons[0])
		fout<<"Start/Stop"<<",";
	else fout << "Searching"<<",";
	auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    fout << " - " <<std::ctime(&now_time);
}