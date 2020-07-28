#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;

/*
  haarlike 特徴を計算してスコアを double で返す。
  (x,y) : 評価対象となっている Subwindow の左上の座標
  (hx,hy) : Subwindow の左上を (0,0) とした時の Haar like 特徴の座標
  h : Haar like 特徴の高さ
  w : Haar like 特徴の幅
  flag : Haar like 特徴の種類
         0 : 上下に分かれた特徴 上が黒 下が白
         1 : 左右に分かれた特徴 右が黒 左が白
         2 : 白黒白と縦方向の矩形が3つ横に並んだもの
	 3 : 4 分割されて斜め方向に同じ色の特徴 右上と左下が黒、残りが白
 */

double haarlike(cv::Mat &src/*入力画像*/,
		int x, int y, int hx, int hy, int h, int w,  int flag)
{
  // haar_x と haar_y の計算における最後の + 1 は OpenCV の積分画像
  // にアクセスするためのおまじない。
  int haar_x = x + hx + 1; // Haar like 特徴の左上の x 座標
  int haar_y = y + hy + 1; // Haar like 特徴の左上の y 座標
  int black = 0; //黒のスコア
  int white = 0; //白のスコア
  double w_minus_b; //最終的なスコア
  double white_size = 0.0; //白の領域の面積
  double black_size = 0.0; //黒の領域の面積

  if(flag == 0){
    // 0 の場合の処理を記述
        black = src.at<int>(haar_y + h/2,  haar_x + w)
      - src.at<int>(haar_y,  haar_x + w)
      - src.at<int>(haar_y + h/2,  haar_x)
      + src.at<int>(haar_y,  haar_x);
    
    white = src.at<int>(haar_y + h,  haar_x + w)
      - src.at<int>(haar_y + h/2,  haar_x + w)
      - src.at<int>(haar_y + h,  haar_x)
      + src.at<int>(haar_y + h/2,  haar_x);

    white_size = (h/2)*w;
    black_size = (h/2)*w;
  }           

  else if(flag == 1){
    // 1 の場合の処理を記述
        black = src.at<int>( haar_y + h, haar_x + w)
      - src.at<int>(haar_y, haar_x + w/2)
      - src.at<int>(haar_y + h, haar_x + w/2)
      + src.at<int>(haar_y, haar_x + w/2);
	
    white = src.at<int>(haar_y + h, haar_x + w/2)
      - src.at<int>(haar_y, haar_x + w/2)
      - src.at<int>(haar_y + h, haar_x)
      + src.at<int>(haar_y, haar_x); 
	
    white_size = h*(w/2);
    black_size = h*(w/2);
  }
    
  else if(flag == 2){
    // 2 の場合の処理を記述
        black = src.at<int>(haar_y + h, haar_x + 2*w/3)
      - src.at<int>(haar_y, haar_x + 2*w/3)
      - src.at<int>(haar_y + h, haar_x + w/3)
      + src.at<int>(haar_y, haar_x + w/3);

    white = src.at<int>(haar_y + h, haar_x + w)
      - src.at<int>(haar_y, haar_x + w)
      - src.at<int>(haar_y + h, haar_x)
      + src.at<int>(haar_y, haar_x)
      - black;

    white_size = h*(2*w/3);
    black_size = h*(w/3);
  }

  else if(flag == 3){
    // 3 の場合の処理を記述
        black = src.at<int>(haar_y + h/2, haar_x + w)
      - src.at<int>(haar_y, haar_x + w)
      - src.at<int>(haar_y + h/2, haar_x + w/2)
      + src.at<int>(haar_y, haar_x + w/2);

    black = black
      + src.at<int>(haar_y + h, haar_x + w/2)
      - src.at<int>(haar_y + h/2, haar_x + w/2)
      - src.at<int>(haar_y + h, haar_x)
      + src.at<int>(haar_y + h/2, haar_x);

    white = src.at<int>(haar_y + h, haar_x + w)
      - src.at<int>(haar_y, haar_x + w)
      - src.at<int>(haar_y + h, haar_x)
      + src.at<int>(haar_y, haar_x)
      - black;

    white_size = h*(w/2);
    black_size = h*(w/2);
  }

  //白および黒のスコアを面積で割って正規化した後、引き算
  w_minus_b = (white/white_size) - (black/black_size);

  return(w_minus_b);
}
