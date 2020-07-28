#ifndef __ADABOOST_INCLUDED__
#define __ADABOOST_INCLUDED__

#include "common.h"
#include "haarlike.h"

void adaboost(cv::Mat* images, int* labels, int n_weight);

#endif
