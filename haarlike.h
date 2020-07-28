#ifndef __HAARLIKE_INCLUDED__
#define __HAARLIKE_INCLUDED__
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

double haarlike(cv::Mat &src/*入力画像*/,
		  int x, int y, int hx, int hy, int h, int w,  int flag);

#endif
