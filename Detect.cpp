#include <iostream>
#include <stdlib.h>
#include <math.h>
#include <fstream>
#include <string>
#include <list>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "haarlike.h"

#define T 100
#define SUBSIZE 100

using namespace cv;

void detect(double thrd){

  //画像の保存用変数
  cv::Mat im;
  cv::Mat im_resize;

  //スコア保存用変数
  double score[T];
  int htx_tmp[T];  
  int htx;

  //入力画像の読み込み
  im = cv::imread("lena.jpg", 0);

  int ii = 0;
  int f_hx[T];
  int f_hy[T];
  int f_w[T];
  int f_h[T];
  int f_flag[T];
  double min_thrd[T];
  double beta[T];
  int min_p[T];
  std::string str;

  std::ifstream ifs("train_result.txt");

  //学習済み重み係数の読み込み
  while (getline(ifs, str)){
    sscanf(str.data(),"%d, %d, %d, %d, %d, %lf, %lf, %d\n"
	   ,&f_hx[ii], &f_hy[ii], &f_h[ii], &f_w[ii], &f_flag[ii], &min_thrd[ii], &beta[ii], &min_p[ii]);
    ii = ii + 1;
  }

  //for(int i = 5; i > 0; i--){ 複数スケール対応用ループ
  double par = 5; // 1/2 のサイズで動くようになっているので決め打ち
  cv::Size s = cv::Size(im.cols * (par/10), im.rows * (par/10));
  cv::resize(im, im_resize, s, 0.0, 0.0);    

  cv::Mat clone = im_resize.clone();

  //積分画像の生成
  cv::Mat int_img;
  integral(im_resize, int_img);
    
  // 1ピクセル単位で処理すると遅いので、以下の単位で画像を飛ばす。
  int x_step = 4;
  int y_step = 4;
  for(int y_idx = 0; y_idx < im_resize.rows - SUBSIZE; y_idx+=y_step){
    for(int x_idx = 0; x_idx < im_resize.cols - SUBSIZE; x_idx+=x_step){
      for(int t = 0; t < T; t++){
	//haarlike 関数を完成させること
	score[t] = haarlike(int_img, x_idx, y_idx,
			    f_hx[t], f_hy[t], f_h[t],  f_w[t], f_flag[t]);
      }
      for(int t = 0; t < T; t++){
	//以下の変数に正しい値を代入すること
            if(min_p[t]*score[t] < min_p[t]*min_thrd[t]) {
                  htx_tmp[t] = 1;
            } else {
                  htx_tmp[t] = 0;
            }
      }

      double alpha[T];
	//以下の変数に正しい値を代入すること
      for (int t = 0;t < T;t++) {
            alpha[t] = log(1/(beta[t]));
      }
      double sum_htx = 0;
      for (int t = 0;t < T;t++) {
            sum_htx += alpha[t]*htx_tmp[t];
      }
      

      //検出結果が閾値を越えていれば 1 越えていなければ 0 にする。
      if(sum_htx >= thrd){
            htx = 1;
      }else{
            htx = 0;
      }

      //以下は結果の描画
      cv::Mat cloneCp = clone.clone();
		
      if( htx == 1 ){	  
	cv::rectangle(clone, cv::Point(x_idx,y_idx),			\
		      cv::Point(x_idx+SUBSIZE, y_idx+SUBSIZE), cv::Scalar::all(255), 3, 4);	  
	cloneCp = clone.clone();
	//	   std::cout << "GOOD" << std::endl;	  
      }else{
	cv::rectangle(cloneCp, cv::Point(x_idx,y_idx),		
		      cv::Point(x_idx+SUBSIZE, y_idx+SUBSIZE), cv::Scalar::all(255), 3, 4);	  
	// std::cout << "BAD" << std::endl;
      }	
      cv::imshow("detection results", cloneCp);	
      cv::waitKey(1);
    }
  }

  //} 複数スケール対応用ループの終わり
}

int main(int argc, char* argv[]){
  
  double thrd = atof(argv[1]);
  std::cout << thrd << std::endl;

  // 与えられたデータを用いると thrd=22 で顔検出可能。
  // これより下げると False Positive もたくさん出る。
  detect( thrd );

  return(EXIT_SUCCESS);
}
