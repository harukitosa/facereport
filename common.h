#ifndef __COMMON_INCLUDED__
#define __COMMON_INCLUDED__

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

void loadList(const char* listname, std::vector<std::string> &list);
void loadImages(cv::Mat *images, int* labels, 
		  std::vector<std::string> poslist,  
		  std::vector<std::string> neglist, 
		  int n_pos, int n_neg);

#endif
