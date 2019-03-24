/*�e��C���N���[�h�t�@�C��*/
#include <stdio.h>
#include <sstream>
#include <string>
//#include <windows.h>
#include <math.h>
#include <vector>
#include<image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/cudacodec.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudawarping.hpp>
//#include "DxLib.h"
#include <iostream>
#include <fstream>
#include <GL/gl.h>
#include <GL/freeglut.h>
//#include <openGL_support/erslib.h>
#include<ros/ros.h>
/*�ʐM�p�萔*/
const int rate = 57600;				//�ʐM�prate
const int com = 7;					//�ʐM�p�|�[�g
const int BUFSIZE = 4096;			//�ʐM�p�o�b�t�@
/*�萔*/
#define WARP_TOTAL_SEGMENT 10		//�����ϊ��摜�̖���
#define LEFT_U 215					//���{�b�g����x���W
#define UP_V 160					//���{�b�g����y���W
#define RIGHT_U 415					//���{�b�g�E��x���W
#define DOWN_V 280					//���{�b�g�E��y���W
#define FRONT_WINDOW_WIDTH 1920		//�O�ʉ摜�\��window�̉��T�C�Y
#define FRONT_WINDOW_HEIGHT 650		//�O�ʉ摜�\��window�̏c�T�C�Y
#define LEFT_WINDOW_WIDTH 960		//���ʉ摜�\��window�̉��T�C�Y
#define LEFT_WINDOW_HEIGHT 1080		//���ʉ摜�\��window�̏c�T�C�Y
#define RIGHT_WINDOW_WIDTH 960		//�E�ʉ摜�\��window�̉��T�C�Y
#define RIGHT_WINDOW_HEIGHT 1080	//�E�ʉ摜�\��window�̏c�T�C�Y
#define BACK_WINDOW_WIDTH 1920		//��ʉ摜�\��window�̉��T�C�Y
#define BACK_WINDOW_HEIGHT 360		//��ʉ摜�\��window�̏c�T�C�Y
#define WINDOW_NUM 4				//OpenGL�Ő�������window�̐�
#define JPEG_QUALTY 100
int WinID[WINDOW_NUM];				//��������window���Ǘ�����z��
const char *WindowName[] = { "frontImg","leftImg","rightImg","backImg" };		//��������window�̖��O���Ǘ�����z��A�h���X
/*namespace�̓o�^*/
using namespace cv;					//opencv�p
using namespace std;				//c++�W�����C�u�����p
/*�L���v�`���ݒ�*/
VideoCapture cap("http://172.16.0.254:9176");	//�L���v�`�����URL
//kVideoCapture cap("http://169.254.251.45");
const int Imgwidth = 640;								//�L���v�`���摜�̉��T�C�Y
const int Imgheight = 480;								//�L���v�`���摜�̏c�T�C�Y
/*�e��֐��̃v���g�^�C�v�錾*/
/*OpenCV*/
void CV_CALL_CAPTURE(void);
void CV_CALL_FUNC(int, Mat&);
void capture(void);
void tmpImgcreate(Mat&);
Mat warp(int, Mat);
Mat Dividewarp(int, Mat);
void filter(Mat&);
void cv_fixed_line(Mat&);
int cv_mode_change(void);
void cv_move_line(int, Mat&);
void cv_move_elipse(int,float, Mat&,int);
void gain(float&, Mat&);
void cv_mode(Mat&);
void cv_draw_target(Mat&);
void cv_save_img(int, Mat);
/*OpenGL*/
void mainLoop(void);
void GLUT_INIT1(int&, const char*);
void GLUT_INIT2(int&, const char*);
void GLUT_INIT3(int&, const char*);
void GLUT_INIT4(int&, const char*);
/*void GLUT_CALL_FUNC1(void);
void GLUT_CALL_FUNC2(void);
void GLUT_CALL_FUNC3(void);
void GLUT_CALL_FUNC4(void);
void display1(void);
void display2(void);
void display3(void);
void display4(void);*/
void idle(void);
void keyborad(unsigned char, int, int);
/*joystic���씻��*/
/*bool joystick_xyz(void);
bool joystick_buttons_filter(void);
bool joystick_buttons_coefficient(void);
bool joystick_buttons_mode(void);
bool joystick_buttons_target(void);
bool joystick_buttons_save(void);*/
/*���{�b�g����p�ʐM*/
/*void data_init(void);
void send_data(void);
void data_save(int64, int64);*/
/*�|���N�v���O����*/
//void Ground(void);
//void recv_data(float*,float*,int);
//void takeuti_AR(void);
GLfloat red[] = { 1.0,0.0,0.0,1.0 };
GLfloat lightpos[] = { 0.0, 0.0, 0.0, 1.0 };//���C�g�̈ʒu
/*OpenGL�������p�֐��|�C���^*/
void(*GLUT_INIT_Ary[])(int&, const char*) = { GLUT_INIT1,GLUT_INIT2,GLUT_INIT3,GLUT_INIT4 };
//void(*GLUT_CALL_FUNC_Ary[])() = { GLUT_CALL_FUNC1,GLUT_CALL_FUNC2,GLUT_CALL_FUNC3,GLUT_CALL_FUNC4 };
/*�O���[�o���ϐ�*/
int face[] = { FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX,			//opencv�̕����t�H���g�f�[�^�p�z��
FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX,
FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC };
Mat srcImg;		//���摜�ۑ�Mat
Mat copyImg;	//��Ɖ摜�ۑ�Mat
int mode = 0;						//���샂�[�h�ۑ��p�ϐ�
/*�f�[�^�Ǘ��p*/
//SYSTEMTIME tm;						//�O���[�o���^�C���̕ۑ��p�ϐ�
ofstream fout("0117_data10.txt");		//�f�[�^���o�͂���t�@�C���̍쐬
/*window�|�W�V���������p�\����*/
struct Pos {
	int x;
	int y;
};
struct Pos windowPos[WINDOW_NUM] = {	//����window�ʒu�̌���
	/*{1920,0},
	{970,0},
	{3840,0},
	{1920,680}*/
	{960,0},
	{0,0},
	{960+1920,0},
	{960,680}
};
/*joystic�p�ϐ�*/
//DINPUT_JOYSTATE input;
/*���C���֐�*/
int main(int argc, char *argv[]) {
	ros::init(argc, argv, "openGL_support");
	/*�J����������*/
	if (!cap.isOpened())
		return -1;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, Imgwidth);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, Imgheight);
	/*GLUT�֘A�̏�����*/
	glutInit(&argc, argv);
	for (int i = 0; i < WINDOW_NUM; i++)
	{
		(*GLUT_INIT_Ary[i])(WinID[i], WindowName[i]);
		//(*GLUT_CALL_FUNC_Ary[i])();
	}
	glutKeyboardFunc(keyborad);
	/*�f�[�^�ʐM�̏�����*/
	//data_init();
	/*���C�����[�v�֐��̌Ăяo��*/
	//glutMainLoop();
	mainLoop();
	return 0;
}
/*OpenCV�֐��̂܂Ƃ�*/
/*�L���v�`���pCALL�֐�*/
void CV_CALL_CAPTURE(void)
{
	capture();										//capture�֐��̌Ăяo��
	filter(copyImg);								//filter�֐��̌Ăяo��
}
/*OpenCV�ł̏����pCALL�֐�*/
/*void CV_CALL_FUNC(int ID, Mat &Img)
{
	if (ID == 0)									//�O�ʉ摜�ɂ�������������A��������
	{
		static float k=1;							//�\���̃Q�C���ۑ��p�ϐ�
		cv_fixed_line(Img);							//�Œ�̐���`��
		if (joystick_buttons_mode())				//���[�h�ύX�{�^���������ꂽ���𔻒�
			mode=cv_mode_change();					//���[�h�̌���
			cv_mode(Img);						//��ʂɑ��샂�[�h��\��
		if (joystick_buttons_coefficient()) {
			gain(k,Img);							//�\���Q�C���̕ύX
		}
		if (joystick_xyz())							//joystick�̓����Ԃ�\��
		{
			if (input.X == 0 && input.Rz == 0)
				cv_move_line(ID, Img);				//����
			else
				cv_move_elipse(ID, k, Img, mode);	//�Ȑ�
		}
		if (joystick_buttons_target())
		{
			cv_draw_target(Img);					//joystick�̃{�^���ɍ��킹�ă^�[�Q�b�g�̏�Ԃ�\��
		}
	}												//�O�ʉ摜�ɂ�������������A�����܂�
	if (joystick_buttons_save())					//�ۑ��{�^���������ꂽ���𔻒�
		cv_save_img(ID, Img);						//�摜�̕ۑ�
	flip(Img, Img, 0);								//Opencv�̉摜��OpenGL�ɍ����悤�ɍ��W����ύX
	cvtColor(Img, Img, COLOR_BGR2RGB);				//Opencv�̉摜��OpenGL�ɍ����悤�ɉ摜�F��ύX
}*/
/*capture�֘A*/
void capture(void)
{
	do {
		cap >> srcImg;					//�摜�擾
	} while (srcImg.empty());			//�L���v�`������܂őҋ@
		srcImg.copyTo(copyImg);			//���摜����Ɨp�ɃR�s�[
}
/*�؂���͈͂�\������tmpImg�����֐�*/
void tmpImgcreate(Mat &tmp)
{
	rectangle(tmp, Point(LEFT_U, UP_V), Point(RIGHT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);		//���{�b�g���l�p�ň͂�
	line(tmp, Point(0, 0), Point(LEFT_U, UP_V), CV_RGB(0, 0, 255), 2, 4);						//�ȉ��ŋ�؂��������
	line(tmp, Point(640, 0), Point(RIGHT_U, UP_V), CV_RGB(0, 0, 255), 2, 4);
	line(tmp, Point(0, 480), Point(LEFT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);
	line(tmp, Point(640, 480), Point(RIGHT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);
}
/*�����ϊ��p�֐�*/
/*Mat warp(int ID, Mat warpImg)
{
	int xx[4], yy[4];																							//���_���W�ۑ��p�z��
	Mat map_matrix;																								//�ϊ��s��ۑ��pMat
	cuda::GpuMat gsrcImg;																						//src�摜�pGpuMat
	cuda::GpuMat gminiImg;																						//�ϊ���摜�ۑ��pGpuMat
	Point2f src_pnt[4], dst_pnt[4];																				//�ϊ��O����ѕϊ���̒��_�ۑ��p�ϐ�
	gsrcImg.upload(warpImg);																					//GPU�Ɍ��摜�f�[�^�𑗐M
	switch (ID)
	{
	case 0:																													//�O��
	{
		int	out_width = 1920;																								//�o�͉摜�T�C�Y��
		int	out_height = 650;																								//�o�͉摜�T�C�Y�c
		Mat front_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//�O�ʉ摜�pMat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//�o�͉E����W
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//�o�͉E�����W
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//�o�͍������W
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImg�̐ݒ�
		xx[0] = 0;																											//�؂����`�̒��_���w��A��������
		yy[0] = 0;
		xx[1] = 640;
		yy[1] = 0;
		xx[2] = RIGHT_U;
		yy[2] = UP_V;
		xx[3] = LEFT_U;
		yy[3] = UP_V;																										//�؂����`�̒��_���w��A�����܂�
		for (int i = 0; i < 4; i++)																							//�w�肵�����_��src_pnt�ɑ��
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//�ϊ��s������߂�
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//�����ϊ��̎��s
		gminiImg.download(warpImg);																							//GPU���瓧���ϊ���̉摜���󂯎��
		resize(warpImg, warpImg, front_Img.size(), 0, 0, INTER_LINEAR);														//�摜��window�T�C�Y�Ƀ��T�C�Y
	}
		break;
	case 1:																													//����
	{
		int	out_width = 960;																								//�o�͉摜�T�C�Y��
		int	out_height = 1080;																								//�o�͉摜�T�C�Y�c
		Mat left_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//���ʉ摜�pMat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//�o�͉E����W				
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//�o�͉E�����W
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//�o�͍������W
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImg�̐ݒ�
		xx[0] = 0;																											//�؂����`�̒��_���w��A��������
		yy[0] = 480;
		xx[1] = 0;
		yy[1] = 0;
		xx[2] = LEFT_U;
		yy[2] = UP_V;
		xx[3] = LEFT_U;
		yy[3] = DOWN_V;																										//�؂����`�̒��_���w��A�����܂�
		for (int i = 0; i < 4; i++)																							//�w�肵�����_��src_pnt�ɑ��
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//�ϊ��s������߂�
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//�����ϊ��̎��s
		gminiImg.download(warpImg);																							//GPU���瓧���ϊ���̉摜���󂯎��
		resize(warpImg, warpImg, left_Img.size(),0,0, INTER_LINEAR);														//�摜��window�T�C�Y�Ƀ��T�C�Y
	}
		break;
	case 2:																													//�E��
	{
		int	out_width = 960;																								//�o�͉摜�T�C�Y��
		int	out_height = 1080;																								//�o�͉摜�T�C�Y�c
		Mat right_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//�E�ʉ摜�pMat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//�o�͉E����W
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//�o�͉E�����W
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//�o�͍������W
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImg�̐ݒ�
		xx[0] = 640;																										//�؂����`�̒��_���w��A��������
		yy[0] = 0;
		xx[1] = 640;
		yy[1] = 480;
		xx[2] = RIGHT_U;
		yy[2] = DOWN_V;
		xx[3] = RIGHT_U;
		yy[3] = UP_V;																										//�؂����`�̒��_���w��A�����܂�
		for (int i = 0; i < 4; i++)																							//�w�肵�����_��src_pnt�ɑ��
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//�ϊ��s������߂�
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//�����ϊ��̎��s
		gminiImg.download(warpImg);																							//GPU���瓧���ϊ���̉摜���󂯎��
		resize(warpImg, warpImg, right_Img.size(),0,0, INTER_LINEAR);														//�摜��window�T�C�Y�Ƀ��T�C�Y
	}
		break;
	case 3:																													//���
	{
		int	out_width = 1920;																								//�o�͉摜�T�C�Y��
		int	out_height = 360;																								//�o�͉摜�T�C�Y�c
		Mat back_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//��ʉ摜�pMat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//�o�͍�����W
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//�o�͉E����W
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//�o�͉E�����W
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//�o�͍������W
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImg�̐ݒ�
		xx[0] = 0;																											//�؂����`�̒��_���w��A��������
		yy[0] = 480;
		xx[1] = 640;
		yy[1] = 480;
		xx[2] = RIGHT_U;
		yy[2] = DOWN_V;
		xx[3] = LEFT_U;
		yy[3] = DOWN_V;																										//�؂����`�̒��_���w��A�����܂�
		for (int i = 0; i < 4; i++)																							//�w�肵�����_��src_pnt�ɑ��
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//�ϊ��s������߂�
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//�����ϊ��̎��s
		gminiImg.download(warpImg);																							//GPU���瓧���ϊ���̉摜���󂯎��
		resize(warpImg, warpImg, back_Img.size(),0,0, INTER_LINEAR);														//�摜��window�T�C�Y�Ƀ��T�C�Y
	}
		break;
	default:
		break;
	}
	return warpImg;
}*/
/*�����ϊ��p�֐��i�ׂ�����؂�ꍇ�j*/
/*Mat Dividewarp(int ID, Mat warpImg)
{
	int l, s, bl, bs, p = 0;
	int xx[4], yy[4];
	int	out_width = 1280;
	int	out_height = 960;
	int robo_width = RIGHT_U - LEFT_U;
	int robo_height = DOWN_V - UP_V;
	Mat tmpImg[WARP_TOTAL_SEGMENT];
	Mat map_matrix;
	Mat front_Img(warpImg.rows / warpImg.rows * 650, warpImg.cols / warpImg.cols * 1920, warpImg.type());
	Mat leftright_Img(warpImg.rows / warpImg.rows * 1080, warpImg.cols / warpImg.cols * 960, warpImg.type());
	Mat back_Img(warpImg.rows / warpImg.rows * 360, warpImg.cols / warpImg.cols * 1920, warpImg.type());
	cuda::GpuMat gsrcImg;
	cuda::GpuMat gminiImg[WARP_TOTAL_SEGMENT];
	Point2f src_pnt[4], dst_pnt[4];
	dst_pnt[0] = cvPoint2D32f(0.0, 0.0);								// ������W
	dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);					// �E����W
	dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);		// �E�����W
	dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);					// �������W
	gsrcImg.upload(warpImg);
	for(p=0;p<WARP_TOTAL_SEGMENT;p++)
	gminiImg[p].create(Size(out_width, out_height), CV_8UC3);
	p = 0;
	switch (ID)
	{
	case 0:
		for (l = Imgwidth / WARP_TOTAL_SEGMENT, s = robo_width / WARP_TOTAL_SEGMENT, bl = 0, bs = 0; l <= Imgwidth,s <= robo_width; l += Imgwidth / WARP_TOTAL_SEGMENT,s += robo_width / WARP_TOTAL_SEGMENT)
		{
		xx[0] = 0 + bl;
		yy[0] = 0;
		xx[1] = 0 + l;
		yy[1] = 0;
		xx[2] = LEFT_U + s;
		yy[2] = UP_V;
		xx[3] = LEFT_U + bs;
		yy[3] = UP_V;
		bl = l;
		bs = s;
		for (int i = 0; i < 4; i++)
		src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);
		cuda::warpPerspective(gsrcImg, gminiImg[p], map_matrix, Size(gminiImg[p].cols, gminiImg[p].rows));
		gminiImg[p].download(tmpImg[p]);
		p++;
		}		
		hconcat(tmpImg, WARP_TOTAL_SEGMENT, warpImg);
		resize(warpImg, warpImg, front_Img.size(),0,0, INTER_LINEAR);
		break;
	case 1:
		for (l = Imgheight / WARP_TOTAL_SEGMENT, s = robo_height / WARP_TOTAL_SEGMENT, bl = 0, bs = 0; l <= Imgheight, s <= robo_height; l += (Imgheight / WARP_TOTAL_SEGMENT), s += (robo_height / WARP_TOTAL_SEGMENT))
		{
		xx[0] = 0;
		yy[0] = Imgheight - bl;
		xx[1] = 0;
		yy[1] = Imgheight - l;DINPUT_JOYSTATE input;
		xx[2] = LEFT_U;
		yy[2] = DOWN_V - s;
		xx[3] = LEFT_U;
		yy[3] = DOWN_V - bs;
		bl = l;
		bs = s;
		for (int i = 0; i < 4; i++)
		src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);
		cuda::warpPerspective(gsrcImg, gminiImg[p], map_matrix, Size(gminiImg[p].cols, gminiImg[p].rows));
		gminiImg[p].download(tmpImg[p]);
		p++;
		}
		hconcat(tmpImg, WARP_TOTAL_SEGMENT, warpImg);
		resize(warpImg, warpImg, leftright_Img.size(),0,0, INTER_LINEAR);
		break;
	case 2:
		for (l = Imgheight / WARP_TOTAL_SEGMENT,  s = robo_height / WARP_TOTAL_SEGMENT, bl = 0, bs = 0; l <= Imgheight, s <= robo_height; l += (Imgheight / WARP_TOTAL_SEGMENT), s += (robo_height / WARP_TOTAL_SEGMENT))
		{
		xx[0] = Imgwidth;
		yy[0] = 0 + bl;
		xx[1] = Imgwidth;
		yy[1] = 0 + l;
		xx[2] = RIGHT_U;
		yy[2] = UP_V + s;
		xx[3] = RIGHT_U;
		yy[3] = UP_V + bs;
		bl = l;
		bs = s;
		for (int i = 0; i < 4; i++)
		src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);
		cuda::warpPerspective(gsrcImg, gminiImg[p], map_matrix, Size(gminiImg[p].cols, gminiImg[p].rows));
		gminiImg[p].download(tmpImg[p]);
		p++;
		}
		hconcat(tmpImg, WARP_TOTAL_SEGMENT, warpImg);
		resize(warpImg, warpImg, leftright_Img.size(),0,0, INTER_LINEAR);
		break;
	case 3:
		for (l = Imgwidth / WARP_TOTAL_SEGMENT, s = robo_width / WARP_TOTAL_SEGMENT, bl = 0, bs = 0; l <= Imgwidth, s <= robo_width; l += (Imgwidth / WARP_TOTAL_SEGMENT), s += (robo_width / WARP_TOTAL_SEGMENT))
		{
		xx[0] = 0 + bl;
		yy[0] = Imgheight;
		xx[1] = 0 + l;
		yy[1] = Imgheight;
		xx[2] = LEFT_U + s;
		yy[2] = DOWN_V;
		xx[3] = LEFT_U + bs;
		yy[3] = DOWN_V;
		bl = l;
		bs = s;
		for (int i = 0; i < 4; i++)
		src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);
		cuda::warpPerspective(gsrcImg, gminiImg[p], map_matrix, Size(gminiImg[p].cols, gminiImg[p].rows));
		gminiImg[p].download(tmpImg[p]);
		p++;
		}
		hconcat(tmpImg, WARP_TOTAL_SEGMENT, warpImg);
		resize(warpImg, warpImg, back_Img.size(),0,0, INTER_LINEAR);
		break;
	default:
		break;
	}
	return warpImg;
}*/
/*�摜��filter�������邽�߂̊֐�*/
/*void filter(Mat &Img)
{
	static int count;																			//�\�����Ԓ����p�ϐ�
	string str;																					//�����p�ϐ�
	static float k=0.0, bk;																		
	if (input.Buttons[4] == 128)																//�Q�C�����グ��
	{
		count = 30;
		bk = k;
		k += 0.1;
	}
	else if (input.Buttons[2] == 128)															//�Q�C����������
	{
		count = 30;
		bk = k;
		k -= 0.1;
	}
	else if (input.Buttons[1] == 128)															//�Q�C�������Z�b�g����
	{
		count = 30;
		bk = k;
		k = 0;
	}
	if (k > 5)																					//�Q�C���̍ő�l�����߂�
		k = 5;
	else if (k <0)																				//�Q�C���̍ŏ��l�����߂�
		k = 0;

	float KernelData[] = {																		//�t�B���^�p�J�[�l���̍쐬
		-k / 9.0f, -k / 9.0f,           -k / 9.0f,
		-k / 9.0f, 1 + (8 * k) / 9.0f,  -k / 9.0f,
		-k / 9.0f, -k / 9.0f,           -k / 9.0f,
	};
	cv::Mat kernel = cv::Mat(3, 3, CV_32F, KernelData);											//�J�[�l���̔z���CvMat�֕ϊ�
	filter2D(Img, Img, -1, kernel);																//�t�B���^����
	str = format("correct %.1f", k);															//�\�����镶����̍쐬
	if (bk != k&&count != 0)																	//�����̕\��
	{
		putText(Img, str, Point(60, 40), face[0], 0.5, CV_RGB(0, 0, 255), 1, CV_AA);
		count--;
		if (count < 0)
			count = 0;
	}
}*/
/*�摜�ɌŒ蒼���������֐�*/
void cv_fixed_line(Mat &Img)
{
	line(Img, Point(410, Img.rows), Point(820, 0), CV_RGB(0, 255, 0), 2, 4);				//��
	line(Img, Point(1480, Img.rows), Point(1040, 0), CV_RGB(0, 255, 0), 2, 4);				//�E
	line(Img, Point(560, 360), Point(1310, 360), CV_RGB(0, 255, 0), 1, 4);					//20cm
	line(Img, Point(680, 190), Point(1190, 190), CV_RGB(0, 255, 0), 1, 4);					//40cm
	line(Img, Point(740, 100), Point(1130, 100), CV_RGB(0, 255, 100), 1, 4);				//60cm
	line(Img, Point(780, 55), Point(1090, 55), CV_RGB(0, 255, 100), 1, 4);					//80cm
	line(Img, Point(800, 25), Point(1065, 25), CV_RGB(0, 255, 100), 1, 4);					//100cm
}
/*���샂�[�h�̕ύX���s���֐�*/
/*int cv_mode_change(void)
{
	if (input.Buttons[6] == 128)			//joystick�{�^��7�������ꂽ�Ƃ�,Y.X���[�h
		return 0;
	else if (input.Buttons[8] == 128)		//joystick�{�^��9�������ꂽ�Ƃ�,Y.Rz���[�h
		return 1;
	else if (input.Buttons[10] == 128)		//joystick�{�^��11�������ꂽ�Ƃ�,Y.X.Rz���[�h
		return 2;
}*/
/*���i���̐��������֐�*/
/*void cv_move_line(int ID ,Mat &Img)
{
	switch (ID)
	{
	default:				//���E��ʂ̎��A�������Ȃ�
		break;
	case 0:					//�O��
		if (input.Y < 0)
		{
			line(Img, Point(410, Img.rows), Point(410 + abs(input.Y*0.410), 650 - abs(input.Y*Img.rows*0.001)), CV_RGB(255, 0, 0), 3, 4);		//��
			line(Img, Point(1480, Img.rows), Point(1480 - abs(input.Y*0.440), 650 - abs(input.Y*Img.rows*0.001)), CV_RGB(255, 0, 0), 3, 4);		//�E
			break;
		}
		else break;
	}
}*/
/*�J�[�u���̐��������֐�*/
/*void cv_move_elipse(int ID,float k, Mat &Img,int mode)
{
	int ltmpx, ltmpy, rtmpx, rtmpy;
	switch (mode)
	{
	case 0:																					//Y.X���[�h�̎�
		if (input.X >= 10 && input.Y <= -50)												//�E�J�[�u
		{
			double points[6][2] = {															//�x�W�F�Ȑ��p�p�����[�^�ݒ�
				{ 410,650 },																//���n�_
				{ 820 + input.X*0.440 ,280 + input.Y*0.280 },								//������_
				{ 820 + input.X*0.880*k,430 + input.Y*0.430 },								//���I�_
				{ 1480,650 },																//�E�n�_
				{ 1040 + input.X*0.440 + 660 * exp(input.Y*0.003),500 + input.Y*0.500 },	//�E����_
				{ 1040 + input.X*0.880*k + 780 * exp(input.Y*0.003),650 + input.Y*0.650 }	//�E�I�_
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//�e�_�̑��
			for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
			{
				ltmpx = lx;
				ltmpy = ly;
				lx = (int)((1 - t)*(1 - t)*points[0][0]) + (2 * (1 - t)*t*points[1][0]) + (t*t*points[2][0]);
				ly = (int)((1 - t)*(1 - t)*points[0][1]) + (2 * (1 - t)*t*points[1][1]) + (t*t*points[2][1]);
				line(Img, Point(ltmpx, ltmpy), Point(lx, ly), Scalar(255, 0, 0),2);
				rtmpx = rx;
				rtmpy = ry;
				rx = (int)((1 - t)*(1 - t)*points[3][0]) + (2 * (1 - t)*t*points[4][0]) + (t*t*points[5][0]);
				ry = (int)((1 - t)*(1 - t)*points[3][1]) + (2 * (1 - t)*t*points[4][1]) + (t*t*points[5][1]);
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//�x�W�F�Ȑ��`�悱���܂�
			}
		}
		else if (input.X <= -10 && input.Y <= -50)											//���J�[�u
		{
			double points[6][2] = {															//�x�W�F�Ȑ��p�p�����[�^�ݒ�
				{ 410,650 },																//���n�_
				{ 820 + input.X*0.410 - 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },	//������_
				{ 820 +input.X*0.820*k -780 * exp(input.Y*0.003) ,650 + input.Y*0.650 },	//���I�_
				{ 1480,650 },																//�E�n�_
				{ 1040 + input.X*0.410,280 + input.Y*0.280 },								//�E����_
				{ 1040 + input.X*0.820*k,430 + input.Y*0.430 }								//�E�I�_
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//�e�_�̑��
			for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
			{
				ltmpx = lx;
				ltmpy = ly;
				lx = (int)((1 - t)*(1 - t)*points[0][0]) + (2 * (1 - t)*t*points[1][0]) + (t*t*points[2][0]);
				ly = (int)((1 - t)*(1 - t)*points[0][1]) + (2 * (1 - t)*t*points[1][1]) + (t*t*points[2][1]);
				line(Img, Point(ltmpx, ltmpy), Point(lx, ly), Scalar(255, 0, 0),2);
				rtmpx = rx;
				rtmpy = ry;
				rx = (int)((1 - t)*(1 - t)*points[3][0]) + (2 * (1 - t)*t*points[4][0]) + (t*t*points[5][0]);
				ry = (int)((1 - t)*(1 - t)*points[3][1]) + (2 * (1 - t)*t*points[4][1]) + (t*t*points[5][1]);
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//�x�W�F�Ȑ��`�悱���܂�
			}
		}
		else if (input.X >= 1 && input.Y == 0)	//�E���`��p
		{
			line(Img, Point(Img.cols / 2 - 100 - (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 + (input.X*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.X*0.001 * 100), Img.rows / 2-10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.X*0.001 * 100), Img.rows / 2+10), Scalar(255, 0, 0),2);
		}
		else if (input.X <= -1 && input.Y == 0)	//�����`��p
		{
			line(Img, Point(Img.cols / 2 - 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 - (input.X*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.X*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.X*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		break;
	case 1:		//Y.Rz���[�h�̎�
		if (input.Rz >= 10 && input.Y <= -50)													//�E�J�[�u
		{
			double points[6][2] = {																//�x�W�F�Ȑ��p�p�����[�^�ݒ�
				{ 410,650 },																	//���n�_
				{ 820 + input.Rz*0.440 ,280 + input.Y*0.280 },									//������_
				{ 820 + input.Rz*0.880*k,430 + input.Y*0.430 },									//���I�_
				{ 1480,650 },																	//�E�n�_
				{ 1040 + input.Rz*0.440 + 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },		//�E����_
				{ 1040 + input.Rz*0.880*k + 780 * exp(input.Y*0.003),650 + input.Y*0.650 }		//�E�I�_
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//�e�_�̑��
			for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
			{
				ltmpx = lx;
				ltmpy = ly;
				lx = (int)((1 - t)*(1 - t)*points[0][0]) + (2 * (1 - t)*t*points[1][0]) + (t*t*points[2][0]);
				ly = (int)((1 - t)*(1 - t)*points[0][1]) + (2 * (1 - t)*t*points[1][1]) + (t*t*points[2][1]);
				line(Img, Point(ltmpx, ltmpy), Point(lx, ly), Scalar(255, 0, 0),2);
				rtmpx = rx;
				rtmpy = ry;
				rx = (int)((1 - t)*(1 - t)*points[3][0]) + (2 * (1 - t)*t*points[4][0]) + (t*t*points[5][0]);
				ry = (int)((1 - t)*(1 - t)*points[3][1]) + (2 * (1 - t)*t*points[4][1]) + (t*t*points[5][1]);
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);
			}																									//�x�W�F�Ȑ��`�悱���܂�
		}
		else if (input.Rz <= -10 && input.Y <= -50)												//���J�[�u
		{
			double points[6][2] = {																//�x�W�F�Ȑ��p�p�����[�^�ݒ�
				{ 410,650 },																	//���n�_
				{ 820 + input.Rz*0.410 - 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },		//������_
				{ 820 + input.Rz*0.820*k - 780 * exp(input.Y*0.003) ,650 + input.Y*0.650 },		//���I�_
				{ 1480,650 },																	//�E�n�_
				{ 1040 + input.Rz*0.410,280 + input.Y*0.280 },									//�E����_
				{ 1040 + input.Rz*0.820*k,430 + input.Y*0.430 }									//�E�I�_
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//�e�_�̑��
			for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
			{
				ltmpx = lx;
				ltmpy = ly;
				lx = (int)((1 - t)*(1 - t)*points[0][0]) + (2 * (1 - t)*t*points[1][0]) + (t*t*points[2][0]);
				ly = (int)((1 - t)*(1 - t)*points[0][1]) + (2 * (1 - t)*t*points[1][1]) + (t*t*points[2][1]);
				line(Img, Point(ltmpx, ltmpy), Point(lx, ly), Scalar(255, 0, 0),2);
				rtmpx = rx;
				rtmpy = ry;
				rx = (int)((1 - t)*(1 - t)*points[3][0]) + (2 * (1 - t)*t*points[4][0]) + (t*t*points[5][0]);
				ry = (int)((1 - t)*(1 - t)*points[3][1]) + (2 * (1 - t)*t*points[4][1]) + (t*t*points[5][1]);
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//�x�W�F�Ȑ��`�悱���܂�
			}
		}
		else if (input.Rz >= 1 && input.Y == 0)	//�E���`��p
		{
			line(Img, Point(Img.cols / 2 - 100 - (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		else if (input.Rz <= -1 && input.Y == 0) //�����`��p
		{
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 - (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		break;
	case 2:		//Y.X.Rz���[�h�̎�
		if (input.X >= 10 && input.Y <= -50)												//�E�J�[�u
		{
			double points[6][2] = {															//�x�W�F�Ȑ��p�p�����[�^�ݒ�
				{ 410,650 },																//���n�_
				{ 820 + input.X*0.440 ,280 + input.Y*0.280 },			DINPUT_JOYSTATE input;					//������_
				{ 820 + input.X*0.880*k,430 + input.Y*0.430 },			DINPUT_JOYSTATE input;					//���I�_
				{ 1480,650 },											DINPUT_JOYSTATE input;					//�E�n�_
				{ 1040 + input.X*0.440 + 660 * exp(input.Y*0.003) ,500 DINPUT_JOYSTATE input; input.Y*0.500 },	//�E����_
				{ 1040 + input.X*0.880*k + 780 * exp(input.Y*0.003),650DINPUT_JOYSTATE input;+ input.Y*0.650 }	//�E�I�_
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//�e�_�̑��
			for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
			{
				ltmpx = lx;
				ltmpy = ly;
				lx = (int)((1 - t)*(1 - t)*points[0][0]) + (2 * (1 - t)*t*points[1][0]) + (t*t*points[2][0]);
				ly = (int)((1 - t)*(1 - t)*points[0][1]) + (2 * (1 - t)*t*points[1][1]) + (t*t*points[2][1]);
				line(Img, Point(ltmpx, ltmpy), Point(lx, ly), Scalar(255, 0, 0),2);
				rtmpx = rx;
				rtmpy = ry;
				rx = (int)((1 - t)*(1 - t)*points[3][0]) + (2 * (1 - t)*t*points[4][0]) + (t*t*points[5][0]);
				ry = (int)((1 - t)*(1 - t)*points[3][1]) + (2 * (1 - t)*t*points[4][1]) + (t*t*points[5][1]);
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//�x�W�F�Ȑ��`�悱���܂�
			}
		}
		else if (input.X <= -10 && input.Y <= -50)											//���J�[�u
		{
			double points[6][2] = {															//�x�W�F�Ȑ��p�p�����[�^�ݒ�
				{ 410,650 },																//���n�_
				{ 820 + input.X*0.410 - 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },	//������_
				{ 820 + input.X*0.820*k - 780 * exp(input.Y*0.003),650 + input.Y*0.650 },	//���I�_
				{ 1480,650 },																//�E�n�_
				{ 1040 + input.X*0.410,280 + input.Y*0.280 },								//�E����_
				{ 1040 + input.X*0.820*k,430 + input.Y*0.430 }								//�E�I�_
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//�e�_�̑��
			for (double t = 0; t <= 1; t += 0.005)																//�x�W�F�Ȑ��`�悱������
			{
				ltmpx = lx;
				ltmpy = ly;
				lx = (int)((1 - t)*(1 - t)*points[0][0]) + (2 * (1 - t)*t*points[1][0]) + (t*t*points[2][0]);
				ly = (int)((1 - t)*(1 - t)*points[0][1]) + (2 * (1 - t)*t*points[1][1]) + (t*t*points[2][1]);
				line(Img, Point(ltmpx, ltmpy), Point(lx, ly), Scalar(255, 0, 0),2);
				rtmpx = rx;
				rtmpy = ry;
				rx = (int)((1 - t)*(1 - t)*points[3][0]) + (2 * (1 - t)*t*points[4][0]) + (t*t*points[5][0]);
				ry = (int)((1 - t)*(1 - t)*points[3][1]) + (2 * (1 - t)*t*points[4][1]) + (t*t*points[5][1]);
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//�x�W�F�Ȑ��`�悱���܂�
			}																	
		}
		else if (input.Rz >= 1 && input.Y == 0)	//�E���`��p
		{
			line(Img, Point(Img.cols / 2 - 100 - (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		else if (input.Rz <= -1 && input.Y == 0) //�����`��p
		{
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 - (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		break;
	}
	
}*/
/*�J�[�u�̋Ȃ����𒲐�����֐�*/
/*void gain(float& k,Mat &Img)
{
	string str;																		//������
	if (input.Buttons[3] == 128) k -= 0.1;											//�Q�C���̑���
	else if (input.Buttons[5] == 128) k += 0.1;
	if (k < 0.8) k = 0.8;															//�Q�C���̍ő�ŏ��̒���
	else if (k > 2.0) k = 2.0;					
	str = format("Curvature correction=%.1f", k);									//�Q�C���𕶎���ɒ���
	putText(Img, str, Point(1400, 150), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//�`��
}*/
/*���[�h�̕`����s���֐�*/
void cv_mode(Mat &Img)
{
	static string str[3] = { "mode_Y,X", "mode_Y,Rz", "mode_Y,X,Rz" };						//�������`
	putText(Img, str[mode], Point(1600, 50), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//�`��
}
/*�^�[�Q�b�g�̏�Ԃ���ʂɕ\������֐�*/
/*void cv_draw_target(Mat &Img)
{
	if (input.Buttons[7] == 128)
	{
		string str = "Safe";						//�������`
		putText(Img, str, Point(Img.cols/2,Img.rows/2), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//�`��
	}
	else if (input.Buttons[9] == 128)
	{
		string str= "Broken";						//�������`
		putText(Img, str, Point(Img.cols / 2, Img.rows / 2), face[3], 1.0, CV_RGB(255, 0, 0), 4, CV_AA);	//�`��
	}
}*/
/*�f�[�^��ۑ�����֐�*/
void cv_save_img(int ID, Mat Img)
{
	vector<int> param = vector<int>(2);
	param[0] = CV_IMWRITE_JPEG_QUALITY;
	param[1] = JPEG_QUALTY;
	switch (ID)
	{
	case 0:
		imwrite("frontImg.jpg", Img,param);
		break;
	case 1:
		imwrite("leftImg.jpg", Img,param);
		break;
	case 2:
		imwrite("rightImg.jpg", Img,param);
		break;
	case 3:
		imwrite("backImg.jpg", Img,param);
		break;
	}
}
/*OpenGL��������*/
/*���C�����[�v*/
void mainLoop(void)
{
	vector<int> param = vector<int>(2);
	param[0] = CV_IMWRITE_JPEG_QUALITY;
	param[1] = JPEG_QUALTY;
	while (1)
	{
		//GetLocalTime(&tm);									//�O���[�o���^�C���̎擾
		int64 start = getTickCount();						//���[�J���^�C���̎擾
		//GetJoypadDirectInputState(DX_INPUT_PAD1, &input);	//joystick���̎擾
		CV_CALL_CAPTURE();									//�L���v�`���֐��̌Ăяo��
		glutMainLoopEvent();								//OpenGL�̃C�x���g�̊J�n
		idle();												//�edisplay�֐��̌Ăяo��
		//send_data();										//���{�b�g����p��joystick�f�[�^�𑗐M
		int64 end = getTickCount();							//���[�J���^�C���̎擾
		//data_save(start, end);								//�f�[�^�̕ۑ�
		/*switch (joystick_buttons_save())
		{
		case 0:break;
		case 1:			
			ERS_Close(com);									//�|�[�g�����
			fout.close();									//�t�@�C���̕ۑ�
			tmpImgcreate(copyImg);							//tmpImg�����
			imwrite("srcImg.jpg", srcImg,param);					//srcImg�̕ۑ�
			imwrite("tmpImg.jpg", copyImg,param);					//copyImg�̕ۑ�
			exit(0);										//���[�v�𔲂���
			break;
		default:break;
		}*/
	}

}
/*display��������������*/
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
/*display�����������܂�*/
/*�`��p�֐���������*/
/*void display1(void)
{																									//�O���
	Mat frontImg = warp(0, copyImg);												//�����ϊ������摜�̗p��
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
	CV_CALL_FUNC(0, frontImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(frontImg.cols, frontImg.rows, GL_RGB, GL_UNSIGNED_BYTE, frontImg.data);			//�`�揈��
	glClear(GL_DEPTH_BUFFER_BIT);																	//�f�v�X�o�b�t�@�̃N���A
	//Ground();
	//takeuti_AR();																					//�|���N��AR�v���O�������Ăяo��
	glutSwapBuffers();																				//��ʂ��X�V
																									//OpenGL�ł̕`��ݒ�A�����܂�
}
void display2(void)
{																									//�����
	Mat leftImg = warp(1, copyImg);																	//�����ϊ������摜�̗p��

//OpenGL�ł̕`��ݒ�A��������
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glEnable(GL_DEPTH_TEST);																		//�A�ʏ����̐ݒ�
	glLoadIdentity();																				//�P�ʍs��̏�����
	CV_CALL_FUNC(1, leftImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(leftImg.cols, leftImg.rows, GL_RGB, GL_UNSIGNED_BYTE, leftImg.data);				//�`�揈��
	glutSwapBuffers();																				//��ʂ��X�V
//OpenGL�ł̕`��ݒ�A�����܂�
}
void display3(void)
{																									//�E���
	Mat rightImg = warp(2, copyImg);																//�����ϊ������摜�̗p��
																									//OpenGL�ł̕`��ݒ�A��������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glEnable(GL_DEPTH_TEST);																		//�A�ʏ����̐ݒ�
	glLoadIdentity();																				//�P�ʍs��̏�����
	CV_CALL_FUNC(2, rightImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(rightImg.cols, rightImg.rows, GL_RGB, GL_UNSIGNED_BYTE, rightImg.data);			//�`�揈��
	glutSwapBuffers();																				//��ʂ��X�V
//OpenGL�ł̕`��ݒ�A�����܂�
}
void display4(void)
{																									//����
	Mat backImg = warp(3, copyImg);																	//�����ϊ������摜�̗p��
																									//OpenGL�ł̕`��ݒ�A��������
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//��ʂ̐F�ƃf�v�X�̃o�b�t�@���N���A
	glEnable(GL_DEPTH_TEST);																		//�A�ʏ����̐ݒ�
	glLoadIdentity();																				//�P�ʍs��̏�����
	CV_CALL_FUNC(3, backImg);																		//opencv�̊e�������s���֐��̌Ăяo��
	glDrawPixels(backImg.cols, backImg.rows, GL_RGB, GL_UNSIGNED_BYTE, backImg.data);				//�`�揈��
	glutSwapBuffers();																				//��ʂ��X�V//OpenGL�ł̕`��ݒ�A�����܂�
}*/
//�`��p�֐������܂�
//display�R�[���o�b�N�֐��ݒ肱������
/*void GLUT_CALL_FUNC1(void)
{
	glutDisplayFunc(display1);	//window��display�����蓖��
}
void GLUT_CALL_FUNC2(void)
{
	glutDisplayFunc(display2);	//window��display�����蓖��
}
void GLUT_CALL_FUNC3(void)
{
	glutDisplayFunc(display3);	//window��display�����蓖��
}
void GLUT_CALL_FUNC4(void)
{
	glutDisplayFunc(display4);	//window��display�����蓖��
}*/
/*display�R�[���o�b�N�֐��ݒ肱���܂�*/
/*�E�B���h�E�X�V�p�֐�*/
void idle(void)
{
	for (int i = 0; i < WINDOW_NUM; i++) {
		glutSetWindow(WinID[i]);	//�X�V����window�̃Z�b�g
		glutPostRedisplay();		//�X�V�̎��s
	}
}
/*�L�[�{�[�h���͏����p�֐�*/
void keyborad(unsigned char key, int x, int y)
{
	switch (key)
	{
	case 's':
		exit(0);
	case 'q':
	case '\033':
		exit(0);
	default:
		break;
	}
}
/*joystick���͔���p�֐��A��������*/
/*bool joystick_xyz(void)
{
	if ((input.X != 0) | (input.Y != 0) | (input.Rz != 0))
		return true;
	else
		return false;
}
bool joystick_buttons_filter(void)
{
	if ((input.Buttons[1] == 128) | (input.Buttons[2] == 128) | (input.Buttons[4] == 128))
		return true;
	else
		return false;
}
bool joystick_buttons_coefficient(void)
{
	if ((input.Buttons[3] == 128) | (input.Buttons[5] == 128))
		return true;
	else
		return false;
}
bool joystick_buttons_mode(void)
{
	if ((input.Buttons[6] == 128) | (input.Buttons[8] == 128) | (input.Buttons[10] == 128))
		return true;
	else
		return false;
}
bool joystick_buttons_target(void)
{
	if (input.Buttons[7] == 128|| input.Buttons[9] == 128)
		return true;
	else
		return false;
}
bool joystick_buttons_save(void)
{
	if (input.Buttons[11] == 128)
		return true;
	else
		return false;
}*/
/*joystick���͔���p�֐��A�����܂�*/
/*���{�b�g����p�ʐM��������*/
/*�ʐM�p�������֐�*/
/*void data_init(void)
{
	int s = ERS_Open(com, BUFSIZE, BUFSIZE);	//�|�[�g�I�[�v��
	if (rate == 57600)							//�|�[�g���J�������̊m�F
		int q = ERS_Config(com, ERS_57600);
	ERS_SendTimeOut(com, 10);					//����M�̃^�C���A�E�g��ݒ�
	ERS_RecvTimeOut(com, 10);
}*/
/*joystick�f�[�^�̑��M���s���֐�*/
/*void send_data(void)
{
	static int e = 0; if (input.Buttons[11] == 128) e = 1;	//�v���O�����I������p
	if (joystick_buttons_mode())
		mode = cv_mode_change();							//���[�h���ʗp
	wchar_t js_x[512] = { 0 };								//X���f�[�^�i�[�p
	wchar_t js_y[512] = { 0 };								//y���f�[�^�i�[�p
	wchar_t js_z[512] = { 0 };								//z��]���f�[�^�i�[�p
	wchar_t js_mode[512] = { 0 };							//���[�h�ؑ֗p
	wchar_t js_end[512] = { 0 };							//�I���p
	wchar_t ptr[1024];										//���������p
	_itow_s(input.Y, js_y, 512, 10);						//joygtick��Y���f�[�^�𕶎��ɕϊ�
	_itow_s(input.X, js_x, 512, 10);						//joygtick��X���f�[�^�𕶎��ɕϊ�
	_itow_s(input.Rz, js_z, 512, 10);						//joygtick��Rz���f�[�^�𕶎��ɕϊ�
	_itow_s(mode, js_mode, 512, 10);						//���샂�[�h�̃f�[�^�𕶎��ɕϊ�
	_itow_s(e, js_end, 512, 10);							//�I������p�̃f�[�^�𕶎��ɕϊ�
	wcscpy_s(ptr, 1024, js_x);								//���M�p�ɕ����������A��������
	wcscat_s(ptr, 1024, L" ,");								//,���f�[�^�̋�؂�Ƃ��Ďg�p
	wcscat_s(ptr, 1024, js_y);
	wcscat_s(ptr, 1024, L" ,");
	wcscat_s(ptr, 1024, js_z);
	wcscat_s(ptr, 1024, L",");
	wcscat_s(ptr, 1024, js_mode);
	wcscat_s(ptr, 1024, L",");
	wcscat_s(ptr, 1024, js_end);
	wcscat_s(ptr, 1024, L",");								//���M�p�ɕ����������A�����܂�
	ERS_WPuts(com, ptr);									//�f�[�^�̑��M
	ERS_ClearSend(com);										//���M�o�b�t�@�̃N���A
}*/
/*�f�[�^�ۑ��p�֐��A�g���Ă�*/
/*void data_save(int64 s, int64 e)
{
	double msec = (e - s) * 1000 / getTickFrequency();	//�v���O�����̎���
	cout << msec << endl;
	fout << input.X;									// X���f�[�^
	fout << " ";
	fout << input.Y;									// Y���f�[�^DINPUT_JOYSTATE input;
	fout << " ";	
	fout << input.Rz;									// Z���f�[�^
	fout << " ";
	fout << mode;										// ���[�h�f�[�^
	fout << " ";
	fout << msec;										// �v���O�����쓮����
	fout << " ";
	fout << tm.wHour;									//�O���[�o���^�C��
	fout << ":";
	fout << tm.wMinute;
	fout << ":";
	fout << tm.wSecond;
	fout << ".";
	fout << tm.wMilliseconds;
	if (input.Buttons[7] == 128)						//�T����Ԃ̕\��
	{
		fout << " ";
		fout << "Safe";
	}
	else if (input.Buttons[9] == 128)
	{
		fout << " ";
		fout << "Broken";
	}
	else
	{
		fout << " ";
		fout << "Searching";
	}
	fout << "\n";
}*/
/*�|���N�v���O����*/
/*void Ground(void) {
	double ground_max_x = 10000.0;
	double ground_max_y = 10000.0;
	glColor3d(0.8, 0.8, 0.8);  // ��n�̐F
	glBegin(GL_LINES);
	for (double ly = -ground_max_y; ly <= ground_max_y; ly += 20.0) {
		glVertex3d(-ground_max_x, ly, 0);DINPUT_JOYSTATE input;
		glVertex3d(ground_max_x, ly, 0);
	}
	for (double lx = -ground_max_x; lx <= ground_max_x; lx += 20.0) {
		glVertex3d(lx, ground_max_y, 0);
		glVertex3d(lx, -ground_max_y, 0);
	}
	glEnd();
	glPushMatrix();
	glMaterialfv(GL_FRONT, GL_DIFFUSE, red);
	glTranslated(0.0, 240.0, 0.0);
	glutSolidCube(10.0);
	glPopMatrix();
}
void recv_data(float *x,float *z,int n)
{
	static wchar_t fbuf[BUFSIZE] = { NULL };
	static wchar_t* token = NULL;
	static wchar_t* temp = NULL;
	static float tmpx[5] = {NULL}, tmpz[5] = { NULL };
	int para = ERS_WGets(com, fbuf, BUFSIZE);
	if (para != 0)
	{
		printf("�f�[�^��M����=%d 000000", para);
		token = wcstok_s(fbuf, L",", &temp);
		if (token != NULL)
			x[0] = _wtof(token);
		x[0] = x[0] / 20 * 7;
		tmpx[0] = x[0];
		printf("x1=%f ", x[0]);
		token = wcstok_s(NULL, L",", &temp);
		if (token != NULL)
			x[1] = _wtof(token);
		x[1] = x[1] / 20 * 7;
		tmpx[1] = x[1];
		printf("x2=%f ", x[1]);
		token = wcstok_s(NULL, L",", &temp);
		if (token != NULL)
			z[0] = _wtof(token);
		z[0] = z[0] / 2;
		tmpz[0] = z[0];
		printf("z1=%f ", z[0]);
		token = wcstok_s(NULL, L",", &temp);
		if (token != NULL)
			z[1] = _wtof(token);
		z[1] = z[1] / 2;
		tmpz[1] = z[1];
		printf("z2=%f\n", z[1]);
	}
	else
	{
		x[0] = tmpx[0];
		x[1] = tmpx[1];
		z[0] = tmpz[0];
		z[1] = tmpz[1];
		printf("x1=%f ", x[0]);
		printf("x2=%f ", x[1]);
		printf("z1=%f ", z[0]);
		printf("z2=%f\n", z[1]);
	}
	ERS_ClearRecv(com);
}
void takeuti_AR(void)
{
	const int n = 5;
	static float X[n] = { 0 }, Z[n] = { 0 };
	static int cc = 1;
	static int dd = 0;
	static int yy[2][10] = { 0 };
	static float oo[4][10];
	//�ǂ�3�������W
	recv_data(X, Z, n);
	if (cc == 1){
		dd = 0;
		for (int i = 0; i < 1; i = i + 2){
			oo[0][dd] = X[0];//���[
			oo[1][dd] = X[1];//�E�[
			oo[2][dd] = Z[0];//�n�_�̉��s
			oo[3][dd] = Z[1];//�I�_�̉��s
			dd++;
		}
		cc++;
	}
	else if (cc == 2) {
		cc = 1;
	}
	else cc++;
	for (int i = 0; i < dd; i++){
		GLfloat vertex[8][3] =
		{
			{ oo[0][i] , oo[2][i] , 0 },
			{ oo[1][i] , oo[3][i] , 0 },
			{ oo[1][i] , oo[3][i] , 20 },
			{ oo[0][i] , oo[2][i] , 20 },
		};
		glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, red);
		glPushMatrix();
		glNormal3f(0,0,-1);
		glBegin(GL_QUADS);
		glVertex3fv(vertex[0]);
		glVertex3fv(vertex[1]);
		glVertex3fv(vertex[2]);
		glVertex3fv(vertex[3]);
		glEnd();
		glPopMatrix();
	}
}*/
