/*各種インクルードファイル*/
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
/*通信用定数*/
const int rate = 57600;				//通信用rate
const int com = 7;					//通信用ポート
const int BUFSIZE = 4096;			//通信用バッファ
/*定数*/
#define WARP_TOTAL_SEGMENT 10		//透視変換画像の枚数
#define LEFT_U 215					//ロボット左上x座標
#define UP_V 160					//ロボット左上y座標
#define RIGHT_U 415					//ロボット右下x座標
#define DOWN_V 280					//ロボット右下y座標
#define FRONT_WINDOW_WIDTH 1920		//前面画像表示windowの横サイズ
#define FRONT_WINDOW_HEIGHT 650		//前面画像表示windowの縦サイズ
#define LEFT_WINDOW_WIDTH 960		//左面画像表示windowの横サイズ
#define LEFT_WINDOW_HEIGHT 1080		//左面画像表示windowの縦サイズ
#define RIGHT_WINDOW_WIDTH 960		//右面画像表示windowの横サイズ
#define RIGHT_WINDOW_HEIGHT 1080	//右面画像表示windowの縦サイズ
#define BACK_WINDOW_WIDTH 1920		//後面画像表示windowの横サイズ
#define BACK_WINDOW_HEIGHT 360		//後面画像表示windowの縦サイズ
#define WINDOW_NUM 4				//OpenGLで生成するwindowの数
#define JPEG_QUALTY 100
int WinID[WINDOW_NUM];				//生成したwindowを管理する配列
const char *WindowName[] = { "frontImg","leftImg","rightImg","backImg" };		//生成したwindowの名前を管理する配列アドレス
/*namespaceの登録*/
using namespace cv;					//opencv用
using namespace std;				//c++標準ライブラリ用
/*キャプチャ設定*/
VideoCapture cap("http://172.16.0.254:9176");	//キャプチャ先のURL
//kVideoCapture cap("http://169.254.251.45");
const int Imgwidth = 640;								//キャプチャ画像の横サイズ
const int Imgheight = 480;								//キャプチャ画像の縦サイズ
/*各種関数のプロトタイプ宣言*/
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
/*joystic動作判定*/
/*bool joystick_xyz(void);
bool joystick_buttons_filter(void);
bool joystick_buttons_coefficient(void);
bool joystick_buttons_mode(void);
bool joystick_buttons_target(void);
bool joystick_buttons_save(void);*/
/*ロボット制御用通信*/
/*void data_init(void);
void send_data(void);
void data_save(int64, int64);*/
/*竹内君プログラム*/
//void Ground(void);
//void recv_data(float*,float*,int);
//void takeuti_AR(void);
GLfloat red[] = { 1.0,0.0,0.0,1.0 };
GLfloat lightpos[] = { 0.0, 0.0, 0.0, 1.0 };//ライトの位置
/*OpenGL初期化用関数ポインタ*/
void(*GLUT_INIT_Ary[])(int&, const char*) = { GLUT_INIT1,GLUT_INIT2,GLUT_INIT3,GLUT_INIT4 };
//void(*GLUT_CALL_FUNC_Ary[])() = { GLUT_CALL_FUNC1,GLUT_CALL_FUNC2,GLUT_CALL_FUNC3,GLUT_CALL_FUNC4 };
/*グローバル変数*/
int face[] = { FONT_HERSHEY_SIMPLEX, FONT_HERSHEY_PLAIN, FONT_HERSHEY_DUPLEX, FONT_HERSHEY_COMPLEX,			//opencvの文字フォントデータ用配列
FONT_HERSHEY_TRIPLEX, FONT_HERSHEY_COMPLEX_SMALL, FONT_HERSHEY_SCRIPT_SIMPLEX,
FONT_HERSHEY_SCRIPT_COMPLEX, FONT_ITALIC };
Mat srcImg;		//元画像保存Mat
Mat copyImg;	//作業画像保存Mat
int mode = 0;						//操作モード保存用変数
/*データ管理用*/
//SYSTEMTIME tm;						//グローバルタイムの保存用変数
ofstream fout("0117_data10.txt");		//データを出力するファイルの作成
/*windowポジション調整用構造体*/
struct Pos {
	int x;
	int y;
};
struct Pos windowPos[WINDOW_NUM] = {	//初期window位置の決定
	/*{1920,0},
	{970,0},
	{3840,0},
	{1920,680}*/
	{960,0},
	{0,0},
	{960+1920,0},
	{960,680}
};
/*joystic用変数*/
//DINPUT_JOYSTATE input;
/*メイン関数*/
int main(int argc, char *argv[]) {
	ros::init(argc, argv, "openGL_support");
	/*カメラ初期化*/
	if (!cap.isOpened())
		return -1;
	cap.set(CV_CAP_PROP_FRAME_WIDTH, Imgwidth);
	cap.set(CV_CAP_PROP_FRAME_HEIGHT, Imgheight);
	/*GLUT関連の初期化*/
	glutInit(&argc, argv);
	for (int i = 0; i < WINDOW_NUM; i++)
	{
		(*GLUT_INIT_Ary[i])(WinID[i], WindowName[i]);
		//(*GLUT_CALL_FUNC_Ary[i])();
	}
	glutKeyboardFunc(keyborad);
	/*データ通信の初期化*/
	//data_init();
	/*メインループ関数の呼び出し*/
	//glutMainLoop();
	mainLoop();
	return 0;
}
/*OpenCV関数のまとめ*/
/*キャプチャ用CALL関数*/
void CV_CALL_CAPTURE(void)
{
	capture();										//capture関数の呼び出し
	filter(copyImg);								//filter関数の呼び出し
}
/*OpenCVでの処理用CALL関数*/
/*void CV_CALL_FUNC(int ID, Mat &Img)
{
	if (ID == 0)									//前面画像にだけ処理をする、ここから
	{
		static float k=1;							//表示のゲイン保存用変数
		cv_fixed_line(Img);							//固定の線を描画
		if (joystick_buttons_mode())				//モード変更ボタンが押されたかを判定
			mode=cv_mode_change();					//モードの決定
			cv_mode(Img);						//画面に操作モードを表示
		if (joystick_buttons_coefficient()) {
			gain(k,Img);							//表示ゲインの変更
		}
		if (joystick_xyz())							//joystickの動作状態を表示
		{
			if (input.X == 0 && input.Rz == 0)
				cv_move_line(ID, Img);				//直線
			else
				cv_move_elipse(ID, k, Img, mode);	//曲線
		}
		if (joystick_buttons_target())
		{
			cv_draw_target(Img);					//joystickのボタンに合わせてターゲットの状態を表示
		}
	}												//前面画像にだけ処理をする、ここまで
	if (joystick_buttons_save())					//保存ボタンが押されたかを判定
		cv_save_img(ID, Img);						//画像の保存
	flip(Img, Img, 0);								//Opencvの画像をOpenGLに合うように座標軸を変更
	cvtColor(Img, Img, COLOR_BGR2RGB);				//Opencvの画像をOpenGLに合うように画像色を変更
}*/
/*capture関連*/
void capture(void)
{
	do {
		cap >> srcImg;					//画像取得
	} while (srcImg.empty());			//キャプチャするまで待機
		srcImg.copyTo(copyImg);			//元画像を作業用にコピー
}
/*切り取り範囲を表示するtmpImgを作る関数*/
void tmpImgcreate(Mat &tmp)
{
	rectangle(tmp, Point(LEFT_U, UP_V), Point(RIGHT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);		//ロボットを四角で囲む
	line(tmp, Point(0, 0), Point(LEFT_U, UP_V), CV_RGB(0, 0, 255), 2, 4);						//以下で区切り線を引く
	line(tmp, Point(640, 0), Point(RIGHT_U, UP_V), CV_RGB(0, 0, 255), 2, 4);
	line(tmp, Point(0, 480), Point(LEFT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);
	line(tmp, Point(640, 480), Point(RIGHT_U, DOWN_V), CV_RGB(0, 0, 255), 2, 4);
}
/*透視変換用関数*/
/*Mat warp(int ID, Mat warpImg)
{
	int xx[4], yy[4];																							//頂点座標保存用配列
	Mat map_matrix;																								//変換行列保存用Mat
	cuda::GpuMat gsrcImg;																						//src画像用GpuMat
	cuda::GpuMat gminiImg;																						//変換後画像保存用GpuMat
	Point2f src_pnt[4], dst_pnt[4];																				//変換前および変換後の頂点保存用変数
	gsrcImg.upload(warpImg);																					//GPUに元画像データを送信
	switch (ID)
	{
	case 0:																													//前面
	{
		int	out_width = 1920;																								//出力画像サイズ横
		int	out_height = 650;																								//出力画像サイズ縦
		Mat front_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//前面画像用Mat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//出力左上座標
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//出力右上座標
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//出力右下座標
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//出力左下座標
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImgの設定
		xx[0] = 0;																											//切り取る台形の頂点を指定、ここから
		yy[0] = 0;
		xx[1] = 640;
		yy[1] = 0;
		xx[2] = RIGHT_U;
		yy[2] = UP_V;
		xx[3] = LEFT_U;
		yy[3] = UP_V;																										//切り取る台形の頂点を指定、ここまで
		for (int i = 0; i < 4; i++)																							//指定した頂点をsrc_pntに代入
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//変換行列を求める
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//透視変換の実行
		gminiImg.download(warpImg);																							//GPUから透視変換後の画像を受け取る
		resize(warpImg, warpImg, front_Img.size(), 0, 0, INTER_LINEAR);														//画像をwindowサイズにリサイズ
	}
		break;
	case 1:																													//左面
	{
		int	out_width = 960;																								//出力画像サイズ横
		int	out_height = 1080;																								//出力画像サイズ縦
		Mat left_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//左面画像用Mat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//出力左上座標
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//出力右上座標				
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//出力右下座標
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//出力左下座標
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImgの設定
		xx[0] = 0;																											//切り取る台形の頂点を指定、ここから
		yy[0] = 480;
		xx[1] = 0;
		yy[1] = 0;
		xx[2] = LEFT_U;
		yy[2] = UP_V;
		xx[3] = LEFT_U;
		yy[3] = DOWN_V;																										//切り取る台形の頂点を指定、ここまで
		for (int i = 0; i < 4; i++)																							//指定した頂点をsrc_pntに代入
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//変換行列を求める
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//透視変換の実行
		gminiImg.download(warpImg);																							//GPUから透視変換後の画像を受け取る
		resize(warpImg, warpImg, left_Img.size(),0,0, INTER_LINEAR);														//画像をwindowサイズにリサイズ
	}
		break;
	case 2:																													//右面
	{
		int	out_width = 960;																								//出力画像サイズ横
		int	out_height = 1080;																								//出力画像サイズ縦
		Mat right_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//右面画像用Mat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//出力左上座標
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//出力右上座標
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//出力右下座標
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//出力左下座標
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImgの設定
		xx[0] = 640;																										//切り取る台形の頂点を指定、ここから
		yy[0] = 0;
		xx[1] = 640;
		yy[1] = 480;
		xx[2] = RIGHT_U;
		yy[2] = DOWN_V;
		xx[3] = RIGHT_U;
		yy[3] = UP_V;																										//切り取る台形の頂点を指定、ここまで
		for (int i = 0; i < 4; i++)																							//指定した頂点をsrc_pntに代入
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//変換行列を求める
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//透視変換の実行
		gminiImg.download(warpImg);																							//GPUから透視変換後の画像を受け取る
		resize(warpImg, warpImg, right_Img.size(),0,0, INTER_LINEAR);														//画像をwindowサイズにリサイズ
	}
		break;
	case 3:																													//後面
	{
		int	out_width = 1920;																								//出力画像サイズ横
		int	out_height = 360;																								//出力画像サイズ縦
		Mat back_Img(warpImg.rows / warpImg.rows * out_height, warpImg.cols / warpImg.cols * out_width, warpImg.type());	//後面画像用Mat
		dst_pnt[0] = cvPoint2D32f(0.0, 0.0);																				//出力左上座標
		dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);																	//出力右上座標
		dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);														//出力右下座標
		dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);																	//出力左下座標
		gminiImg.create(Size(out_width, out_height), CV_8UC3);																//gminiImgの設定
		xx[0] = 0;																											//切り取る台形の頂点を指定、ここから
		yy[0] = 480;
		xx[1] = 640;
		yy[1] = 480;
		xx[2] = RIGHT_U;
		yy[2] = DOWN_V;
		xx[3] = LEFT_U;
		yy[3] = DOWN_V;																										//切り取る台形の頂点を指定、ここまで
		for (int i = 0; i < 4; i++)																							//指定した頂点をsrc_pntに代入
			src_pnt[i] = cvPoint2D32f(xx[i], yy[i]);
		map_matrix = getPerspectiveTransform(src_pnt, dst_pnt);																//変換行列を求める
		cuda::warpPerspective(gsrcImg, gminiImg, map_matrix, Size(gminiImg.cols, gminiImg.rows));							//透視変換の実行
		gminiImg.download(warpImg);																							//GPUから透視変換後の画像を受け取る
		resize(warpImg, warpImg, back_Img.size(),0,0, INTER_LINEAR);														//画像をwindowサイズにリサイズ
	}
		break;
	default:
		break;
	}
	return warpImg;
}*/
/*透視変換用関数（細かく区切る場合）*/
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
	dst_pnt[0] = cvPoint2D32f(0.0, 0.0);								// 左上座標
	dst_pnt[1] = cvPoint2D32f((float)out_width, 0.0);					// 右上座標
	dst_pnt[2] = cvPoint2D32f((float)out_width, (float)out_height);		// 右下座標
	dst_pnt[3] = cvPoint2D32f(0.0, (float)out_height);					// 左下座標
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
/*画像にfilterをかけるための関数*/
/*void filter(Mat &Img)
{
	static int count;																			//表示時間調整用変数
	string str;																					//文字用変数
	static float k=0.0, bk;																		
	if (input.Buttons[4] == 128)																//ゲインを上げる
	{
		count = 30;
		bk = k;
		k += 0.1;
	}
	else if (input.Buttons[2] == 128)															//ゲインを下げる
	{
		count = 30;
		bk = k;
		k -= 0.1;
	}
	else if (input.Buttons[1] == 128)															//ゲインをリセットする
	{
		count = 30;
		bk = k;
		k = 0;
	}
	if (k > 5)																					//ゲインの最大値を決める
		k = 5;
	else if (k <0)																				//ゲインの最小値を決める
		k = 0;

	float KernelData[] = {																		//フィルタ用カーネルの作成
		-k / 9.0f, -k / 9.0f,           -k / 9.0f,
		-k / 9.0f, 1 + (8 * k) / 9.0f,  -k / 9.0f,
		-k / 9.0f, -k / 9.0f,           -k / 9.0f,
	};
	cv::Mat kernel = cv::Mat(3, 3, CV_32F, KernelData);											//カーネルの配列をCvMatへ変換
	filter2D(Img, Img, -1, kernel);																//フィルタ処理
	str = format("correct %.1f", k);															//表示する文字列の作成
	if (bk != k&&count != 0)																	//文字の表示
	{
		putText(Img, str, Point(60, 40), face[0], 0.5, CV_RGB(0, 0, 255), 1, CV_AA);
		count--;
		if (count < 0)
			count = 0;
	}
}*/
/*画像に固定直線を引く関数*/
void cv_fixed_line(Mat &Img)
{
	line(Img, Point(410, Img.rows), Point(820, 0), CV_RGB(0, 255, 0), 2, 4);				//左
	line(Img, Point(1480, Img.rows), Point(1040, 0), CV_RGB(0, 255, 0), 2, 4);				//右
	line(Img, Point(560, 360), Point(1310, 360), CV_RGB(0, 255, 0), 1, 4);					//20cm
	line(Img, Point(680, 190), Point(1190, 190), CV_RGB(0, 255, 0), 1, 4);					//40cm
	line(Img, Point(740, 100), Point(1130, 100), CV_RGB(0, 255, 100), 1, 4);				//60cm
	line(Img, Point(780, 55), Point(1090, 55), CV_RGB(0, 255, 100), 1, 4);					//80cm
	line(Img, Point(800, 25), Point(1065, 25), CV_RGB(0, 255, 100), 1, 4);					//100cm
}
/*操作モードの変更を行う関数*/
/*int cv_mode_change(void)
{
	if (input.Buttons[6] == 128)			//joystickボタン7が押されたとき,Y.Xモード
		return 0;
	else if (input.Buttons[8] == 128)		//joystickボタン9が押されたとき,Y.Rzモード
		return 1;
	else if (input.Buttons[10] == 128)		//joystickボタン11が押されたとき,Y.X.Rzモード
		return 2;
}*/
/*直進時の線を引く関数*/
/*void cv_move_line(int ID ,Mat &Img)
{
	switch (ID)
	{
	default:				//左右画面の時、何もしない
		break;
	case 0:					//前面
		if (input.Y < 0)
		{
			line(Img, Point(410, Img.rows), Point(410 + abs(input.Y*0.410), 650 - abs(input.Y*Img.rows*0.001)), CV_RGB(255, 0, 0), 3, 4);		//左
			line(Img, Point(1480, Img.rows), Point(1480 - abs(input.Y*0.440), 650 - abs(input.Y*Img.rows*0.001)), CV_RGB(255, 0, 0), 3, 4);		//右
			break;
		}
		else break;
	}
}*/
/*カーブ時の線を引く関数*/
/*void cv_move_elipse(int ID,float k, Mat &Img,int mode)
{
	int ltmpx, ltmpy, rtmpx, rtmpy;
	switch (mode)
	{
	case 0:																					//Y.Xモードの時
		if (input.X >= 10 && input.Y <= -50)												//右カーブ
		{
			double points[6][2] = {															//ベジェ曲線用パラメータ設定
				{ 410,650 },																//左始点
				{ 820 + input.X*0.440 ,280 + input.Y*0.280 },								//左制御点
				{ 820 + input.X*0.880*k,430 + input.Y*0.430 },								//左終点
				{ 1480,650 },																//右始点
				{ 1040 + input.X*0.440 + 660 * exp(input.Y*0.003),500 + input.Y*0.500 },	//右制御点
				{ 1040 + input.X*0.880*k + 780 * exp(input.Y*0.003),650 + input.Y*0.650 }	//右終点
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//各点の代入
			for (double t = 0; t <= 1; t += 0.005)																//ベジェ曲線描画ここから
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
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//ベジェ曲線描画ここまで
			}
		}
		else if (input.X <= -10 && input.Y <= -50)											//左カーブ
		{
			double points[6][2] = {															//ベジェ曲線用パラメータ設定
				{ 410,650 },																//左始点
				{ 820 + input.X*0.410 - 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },	//左制御点
				{ 820 +input.X*0.820*k -780 * exp(input.Y*0.003) ,650 + input.Y*0.650 },	//左終点
				{ 1480,650 },																//右始点
				{ 1040 + input.X*0.410,280 + input.Y*0.280 },								//右制御点
				{ 1040 + input.X*0.820*k,430 + input.Y*0.430 }								//右終点
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//各点の代入
			for (double t = 0; t <= 1; t += 0.005)																//ベジェ曲線描画ここから
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
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//ベジェ曲線描画ここまで
			}
		}
		else if (input.X >= 1 && input.Y == 0)	//右矢印描画用
		{
			line(Img, Point(Img.cols / 2 - 100 - (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 + (input.X*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.X*0.001 * 100), Img.rows / 2-10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.X*0.001 * 100), Img.rows / 2+10), Scalar(255, 0, 0),2);
		}
		else if (input.X <= -1 && input.Y == 0)	//左矢印描画用
		{
			line(Img, Point(Img.cols / 2 - 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 - (input.X*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.X*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.X*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.X*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		break;
	case 1:		//Y.Rzモードの時
		if (input.Rz >= 10 && input.Y <= -50)													//右カーブ
		{
			double points[6][2] = {																//ベジェ曲線用パラメータ設定
				{ 410,650 },																	//左始点
				{ 820 + input.Rz*0.440 ,280 + input.Y*0.280 },									//左制御点
				{ 820 + input.Rz*0.880*k,430 + input.Y*0.430 },									//左終点
				{ 1480,650 },																	//右始点
				{ 1040 + input.Rz*0.440 + 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },		//右制御点
				{ 1040 + input.Rz*0.880*k + 780 * exp(input.Y*0.003),650 + input.Y*0.650 }		//右終点
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//各点の代入
			for (double t = 0; t <= 1; t += 0.005)																//ベジェ曲線描画ここから
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
			}																									//ベジェ曲線描画ここまで
		}
		else if (input.Rz <= -10 && input.Y <= -50)												//左カーブ
		{
			double points[6][2] = {																//ベジェ曲線用パラメータ設定
				{ 410,650 },																	//左始点
				{ 820 + input.Rz*0.410 - 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },		//左制御点
				{ 820 + input.Rz*0.820*k - 780 * exp(input.Y*0.003) ,650 + input.Y*0.650 },		//左終点
				{ 1480,650 },																	//右始点
				{ 1040 + input.Rz*0.410,280 + input.Y*0.280 },									//右制御点
				{ 1040 + input.Rz*0.820*k,430 + input.Y*0.430 }									//右終点
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//各点の代入
			for (double t = 0; t <= 1; t += 0.005)																//ベジェ曲線描画ここから
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
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//ベジェ曲線描画ここまで
			}
		}
		else if (input.Rz >= 1 && input.Y == 0)	//右矢印描画用
		{
			line(Img, Point(Img.cols / 2 - 100 - (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		else if (input.Rz <= -1 && input.Y == 0) //左矢印描画用
		{
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 - (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		break;
	case 2:		//Y.X.Rzモードの時
		if (input.X >= 10 && input.Y <= -50)												//右カーブ
		{
			double points[6][2] = {															//ベジェ曲線用パラメータ設定
				{ 410,650 },																//左始点
				{ 820 + input.X*0.440 ,280 + input.Y*0.280 },			DINPUT_JOYSTATE input;					//左制御点
				{ 820 + input.X*0.880*k,430 + input.Y*0.430 },			DINPUT_JOYSTATE input;					//左終点
				{ 1480,650 },											DINPUT_JOYSTATE input;					//右始点
				{ 1040 + input.X*0.440 + 660 * exp(input.Y*0.003) ,500 DINPUT_JOYSTATE input; input.Y*0.500 },	//右制御点
				{ 1040 + input.X*0.880*k + 780 * exp(input.Y*0.003),650DINPUT_JOYSTATE input;+ input.Y*0.650 }	//右終点
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//各点の代入
			for (double t = 0; t <= 1; t += 0.005)																//ベジェ曲線描画ここから
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
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//ベジェ曲線描画ここまで
			}
		}
		else if (input.X <= -10 && input.Y <= -50)											//左カーブ
		{
			double points[6][2] = {															//ベジェ曲線用パラメータ設定
				{ 410,650 },																//左始点
				{ 820 + input.X*0.410 - 660 * exp(input.Y*0.003) ,500 + input.Y*0.500 },	//左制御点
				{ 820 + input.X*0.820*k - 780 * exp(input.Y*0.003),650 + input.Y*0.650 },	//左終点
				{ 1480,650 },																//右始点
				{ 1040 + input.X*0.410,280 + input.Y*0.280 },								//右制御点
				{ 1040 + input.X*0.820*k,430 + input.Y*0.430 }								//右終点
			};
			int lx = (int)points[0][0], ly = (int)points[0][1], rx = (int)points[3][0], ry = (int)points[3][1];	//各点の代入
			for (double t = 0; t <= 1; t += 0.005)																//ベジェ曲線描画ここから
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
				line(Img, Point(rtmpx, rtmpy), Point(rx, ry), Scalar(255, 0, 0),2);								//ベジェ曲線描画ここまで
			}																	
		}
		else if (input.Rz >= 1 && input.Y == 0)	//右矢印描画用
		{
			line(Img, Point(Img.cols / 2 - 100 - (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 + 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		else if (input.Rz <= -1 && input.Y == 0) //左矢印描画用
		{
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 + 100 - (input.Rz*0.001 * 100), Img.rows / 2), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 - 10), Scalar(255, 0, 0),2);
			line(Img, Point(Img.cols / 2 - 100 + (input.Rz*0.001 * 100), Img.rows / 2), Point(Img.cols / 2 - 50 + (input.Rz*0.001 * 100), Img.rows / 2 + 10), Scalar(255, 0, 0),2);
		}
		break;
	}
	
}*/
/*カーブの曲がり具合を調整する関数*/
/*void gain(float& k,Mat &Img)
{
	string str;																		//文字列
	if (input.Buttons[3] == 128) k -= 0.1;											//ゲインの増減
	else if (input.Buttons[5] == 128) k += 0.1;
	if (k < 0.8) k = 0.8;															//ゲインの最大最小の調整
	else if (k > 2.0) k = 2.0;					
	str = format("Curvature correction=%.1f", k);									//ゲインを文字列に直す
	putText(Img, str, Point(1400, 150), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//描画
}*/
/*モードの描画を行う関数*/
void cv_mode(Mat &Img)
{
	static string str[3] = { "mode_Y,X", "mode_Y,Rz", "mode_Y,X,Rz" };						//文字列定義
	putText(Img, str[mode], Point(1600, 50), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//描画
}
/*ターゲットの状態を画面に表示する関数*/
/*void cv_draw_target(Mat &Img)
{
	if (input.Buttons[7] == 128)
	{
		string str = "Safe";						//文字列定義
		putText(Img, str, Point(Img.cols/2,Img.rows/2), face[3], 1.0, CV_RGB(0, 255, 0), 4, CV_AA);	//描画
	}
	else if (input.Buttons[9] == 128)
	{
		string str= "Broken";						//文字列定義
		putText(Img, str, Point(Img.cols / 2, Img.rows / 2), face[3], 1.0, CV_RGB(255, 0, 0), 4, CV_AA);	//描画
	}
}*/
/*データを保存する関数*/
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
/*OpenGLここから*/
/*メインループ*/
void mainLoop(void)
{
	vector<int> param = vector<int>(2);
	param[0] = CV_IMWRITE_JPEG_QUALITY;
	param[1] = JPEG_QUALTY;
	while (1)
	{
		//GetLocalTime(&tm);									//グローバルタイムの取得
		int64 start = getTickCount();						//ローカルタイムの取得
		//GetJoypadDirectInputState(DX_INPUT_PAD1, &input);	//joystick情報の取得
		CV_CALL_CAPTURE();									//キャプチャ関数の呼び出し
		glutMainLoopEvent();								//OpenGLのイベントの開始
		idle();												//各display関数の呼び出し
		//send_data();										//ロボット操作用のjoystickデータを送信
		int64 end = getTickCount();							//ローカルタイムの取得
		//data_save(start, end);								//データの保存
		/*switch (joystick_buttons_save())
		{
		case 0:break;
		case 1:			
			ERS_Close(com);									//ポートを閉じる
			fout.close();									//ファイルの保存
			tmpImgcreate(copyImg);							//tmpImgを作る
			imwrite("srcImg.jpg", srcImg,param);					//srcImgの保存
			imwrite("tmpImg.jpg", copyImg,param);					//copyImgの保存
			exit(0);										//ループを抜ける
			break;
		default:break;
		}*/
	}

}
/*display初期化ここから*/
void GLUT_INIT1(int &ID, const char *name)							//前面
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//描画方法の設定
	glutInitWindowPosition(windowPos[0].x, windowPos[0].y);			//window位置の設定
	glutInitWindowSize(FRONT_WINDOW_WIDTH, FRONT_WINDOW_HEIGHT);	//windowサイズの設定
	ID = glutCreateWindow(name);									//windowに名前と番号を振り分け
	glEnable(GL_LIGHTING);											//光源設定をonにする
	glEnable(GL_LIGHT0);											//一つ目の光源
	glEnable(GL_DEPTH_TEST);										//陰面消去の設定
	glEnable(GL_BLEND);												//半透明設定をonにする
	glEnable(GL_NORMALIZE);											//法線ベクトルを自動的に正規化（頂点の光源に対する方向を決定して真っ黒なポリゴンになるのを防ぐ）
}
void GLUT_INIT2(int &ID, const char *name)							//左面
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//描画方法の設定
	glutInitWindowPosition(windowPos[1].x, windowPos[1].y);			//window位置の設定
	glutInitWindowSize(LEFT_WINDOW_WIDTH, LEFT_WINDOW_HEIGHT);		//windowサイズの設定
	ID = glutCreateWindow(name);									//windowに名前と番号を振り分け
}
void GLUT_INIT3(int &ID, const char *name)							//右面
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//描画方法の設定
	glutInitWindowPosition(windowPos[2].x, windowPos[2].y);			//window位置の設定
	glutInitWindowSize(RIGHT_WINDOW_WIDTH, RIGHT_WINDOW_HEIGHT);	//windowサイズの設定
	ID = glutCreateWindow(name);									//windowに名前と番号を振り分け
}
void GLUT_INIT4(int &ID, const char *name)							//後面
{
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);		//描画方法の設定
	glutInitWindowPosition(windowPos[3].x, windowPos[3].y);			//window位置の設定
	glutInitWindowSize(BACK_WINDOW_WIDTH, BACK_WINDOW_HEIGHT);		//windowサイズの設定
	ID = glutCreateWindow(name);									//windowに名前と番号を振り分け
}
/*display初期化ここまで*/
/*描画用関数ここから*/
/*void display1(void)
{																									//前画面
	Mat frontImg = warp(0, copyImg);												//透視変換した画像の用意
													//OpenGLでの描画設定、ここから
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//画面の色とデプスのバッファをクリア
	glViewport(0, 0, FRONT_WINDOW_WIDTH, FRONT_WINDOW_HEIGHT);										//viewportの設定
	glMatrixMode(GL_PROJECTION);																	//投影変換モード
	glLoadIdentity();																				//投影変換の変換行列を単位行列で初期化
	gluPerspective(30.0, (double)FRONT_WINDOW_WIDTH / (double)FRONT_WINDOW_HEIGHT, 1.0, 1000.0);	//視界の決定
	glMatrixMode(GL_MODELVIEW);																		//モデルビュー変換行列の設定
	glLoadIdentity();																				//モデルビュー変換行列を単位行列で初期化
	gluLookAt(0.0, 0.0, 65.0, //カメラの座標
		8.0, 180.0, 0, // 注視点の座標
		0.0, 0.0, 1.0); // 画面の上方向を指すベクトル
	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);													//ライトを当てる
	CV_CALL_FUNC(0, frontImg);																		//opencvの各処理を行う関数の呼び出し
	glDrawPixels(frontImg.cols, frontImg.rows, GL_RGB, GL_UNSIGNED_BYTE, frontImg.data);			//描画処理
	glClear(GL_DEPTH_BUFFER_BIT);																	//デプスバッファのクリア
	//Ground();
	//takeuti_AR();																					//竹内君のARプログラムを呼び出し
	glutSwapBuffers();																				//画面を更新
																									//OpenGLでの描画設定、ここまで
}
void display2(void)
{																									//左画面
	Mat leftImg = warp(1, copyImg);																	//透視変換した画像の用意

//OpenGLでの描画設定、ここから
glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//画面の色とデプスのバッファをクリア
	glEnable(GL_DEPTH_TEST);																		//陰面消去の設定
	glLoadIdentity();																				//単位行列の初期化
	CV_CALL_FUNC(1, leftImg);																		//opencvの各処理を行う関数の呼び出し
	glDrawPixels(leftImg.cols, leftImg.rows, GL_RGB, GL_UNSIGNED_BYTE, leftImg.data);				//描画処理
	glutSwapBuffers();																				//画面を更新
//OpenGLでの描画設定、ここまで
}
void display3(void)
{																									//右画面
	Mat rightImg = warp(2, copyImg);																//透視変換した画像の用意
																									//OpenGLでの描画設定、ここから
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//画面の色とデプスのバッファをクリア
	glEnable(GL_DEPTH_TEST);																		//陰面消去の設定
	glLoadIdentity();																				//単位行列の初期化
	CV_CALL_FUNC(2, rightImg);																		//opencvの各処理を行う関数の呼び出し
	glDrawPixels(rightImg.cols, rightImg.rows, GL_RGB, GL_UNSIGNED_BYTE, rightImg.data);			//描画処理
	glutSwapBuffers();																				//画面を更新
//OpenGLでの描画設定、ここまで
}
void display4(void)
{																									//後画面
	Mat backImg = warp(3, copyImg);																	//透視変換した画像の用意
																									//OpenGLでの描画設定、ここから
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);												//画面の色とデプスのバッファをクリア
	glEnable(GL_DEPTH_TEST);																		//陰面消去の設定
	glLoadIdentity();																				//単位行列の初期化
	CV_CALL_FUNC(3, backImg);																		//opencvの各処理を行う関数の呼び出し
	glDrawPixels(backImg.cols, backImg.rows, GL_RGB, GL_UNSIGNED_BYTE, backImg.data);				//描画処理
	glutSwapBuffers();																				//画面を更新//OpenGLでの描画設定、ここまで
}*/
//描画用関数ここまで
//displayコールバック関数設定ここから
/*void GLUT_CALL_FUNC1(void)
{
	glutDisplayFunc(display1);	//windowにdisplayを割り当て
}
void GLUT_CALL_FUNC2(void)
{
	glutDisplayFunc(display2);	//windowにdisplayを割り当て
}
void GLUT_CALL_FUNC3(void)
{
	glutDisplayFunc(display3);	//windowにdisplayを割り当て
}
void GLUT_CALL_FUNC4(void)
{
	glutDisplayFunc(display4);	//windowにdisplayを割り当て
}*/
/*displayコールバック関数設定ここまで*/
/*ウィンドウ更新用関数*/
void idle(void)
{
	for (int i = 0; i < WINDOW_NUM; i++) {
		glutSetWindow(WinID[i]);	//更新するwindowのセット
		glutPostRedisplay();		//更新の実行
	}
}
/*キーボード入力処理用関数*/
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
/*joystick入力判定用関数、ここから*/
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
/*joystick入力判定用関数、ここまで*/
/*ロボット制御用通信ここから*/
/*通信用初期化関数*/
/*void data_init(void)
{
	int s = ERS_Open(com, BUFSIZE, BUFSIZE);	//ポートオープン
	if (rate == 57600)							//ポートが開いたかの確認
		int q = ERS_Config(com, ERS_57600);
	ERS_SendTimeOut(com, 10);					//送受信のタイムアウトを設定
	ERS_RecvTimeOut(com, 10);
}*/
/*joystickデータの送信を行う関数*/
/*void send_data(void)
{
	static int e = 0; if (input.Buttons[11] == 128) e = 1;	//プログラム終了判定用
	if (joystick_buttons_mode())
		mode = cv_mode_change();							//モード識別用
	wchar_t js_x[512] = { 0 };								//X軸データ格納用
	wchar_t js_y[512] = { 0 };								//y軸データ格納用
	wchar_t js_z[512] = { 0 };								//z回転軸データ格納用
	wchar_t js_mode[512] = { 0 };							//モード切替用
	wchar_t js_end[512] = { 0 };							//終了用
	wchar_t ptr[1024];										//文字結合用
	_itow_s(input.Y, js_y, 512, 10);						//joygtickのY軸データを文字に変換
	_itow_s(input.X, js_x, 512, 10);						//joygtickのX軸データを文字に変換
	_itow_s(input.Rz, js_z, 512, 10);						//joygtickのRz軸データを文字に変換
	_itow_s(mode, js_mode, 512, 10);						//操作モードのデータを文字に変換
	_itow_s(e, js_end, 512, 10);							//終了判定用のデータを文字に変換
	wcscpy_s(ptr, 1024, js_x);								//送信用に文字を結合、ここから
	wcscat_s(ptr, 1024, L" ,");								//,をデータの区切りとして使用
	wcscat_s(ptr, 1024, js_y);
	wcscat_s(ptr, 1024, L" ,");
	wcscat_s(ptr, 1024, js_z);
	wcscat_s(ptr, 1024, L",");
	wcscat_s(ptr, 1024, js_mode);
	wcscat_s(ptr, 1024, L",");
	wcscat_s(ptr, 1024, js_end);
	wcscat_s(ptr, 1024, L",");								//送信用に文字を結合、ここまで
	ERS_WPuts(com, ptr);									//データの送信
	ERS_ClearSend(com);										//送信バッファのクリア
}*/
/*データ保存用関数、使ってる*/
/*void data_save(int64 s, int64 e)
{
	double msec = (e - s) * 1000 / getTickFrequency();	//プログラムの時間
	cout << msec << endl;
	fout << input.X;									// X軸データ
	fout << " ";
	fout << input.Y;									// Y軸データDINPUT_JOYSTATE input;
	fout << " ";	
	fout << input.Rz;									// Z軸データ
	fout << " ";
	fout << mode;										// モードデータ
	fout << " ";
	fout << msec;										// プログラム作動時間
	fout << " ";
	fout << tm.wHour;									//グローバルタイム
	fout << ":";
	fout << tm.wMinute;
	fout << ":";
	fout << tm.wSecond;
	fout << ".";
	fout << tm.wMilliseconds;
	if (input.Buttons[7] == 128)						//探索状態の表示
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
/*竹内君プログラム*/
/*void Ground(void) {
	double ground_max_x = 10000.0;
	double ground_max_y = 10000.0;
	glColor3d(0.8, 0.8, 0.8);  // 大地の色
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
		printf("データ受信完了=%d 000000", para);
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
	//壁の3次元座標
	recv_data(X, Z, n);
	if (cc == 1){
		dd = 0;
		for (int i = 0; i < 1; i = i + 2){
			oo[0][dd] = X[0];//左端
			oo[1][dd] = X[1];//右端
			oo[2][dd] = Z[0];//始点の奥行
			oo[3][dd] = Z[1];//終点の奥行
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
