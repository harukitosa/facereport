#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#include <list>
#include <iostream>
#include <fstream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "common.h"
#include "adaboost.h"

#define POS 1000
#define NEG 2000
#define ALL POS+NEG
#define T 200
#define SUBSIZE 100

int main(int argc, char* argv[])
{  
  srand((unsigned) time(NULL));

  cv::Mat images[ALL];
  int labels[ALL];
    
  std::vector<std::string> poslist;
  std::vector<std::string> neglist;
  
  // load list
  loadList("posList.lst", poslist);
  loadList("negList.lst", neglist);
  
  // load images
  loadImages(images, labels, poslist, neglist, POS, NEG);
  
  // 訓練
  adaboost( images, labels, ALL );

  return (EXIT_SUCCESS);
  
}
