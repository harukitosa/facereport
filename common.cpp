#include "common.h"

void loadList(const char* listname, std::vector<std::string> &list)
{  
  std::ifstream ifs(listname);  
  std::string str;
  if (ifs.fail()){
    std::cerr << "失敗" << std::endl;
    exit(1);
  }
  while (getline(ifs, str)){
    list.push_back(str);    
  }  
}

void loadImages(cv::Mat *images, int* labels, 
		std::vector<std::string> poslist,  
		std::vector<std::string> neglist,
		int n_pos, int n_neg){  
  
  if( (int)n_pos > (int)poslist.size() || (int)n_neg > (int)neglist.size() ){
    std::cout << "too many images to be read." << std::endl;
    exit(1);
  }
  
  for(int i = 0; i < n_pos; i++){
    images[i] = cv::imread( poslist[i], 0 );
    labels[i] = 1;
  }
  for(int i = n_pos; i < n_pos+n_neg; i++){
    images[i] = cv::imread( neglist[i], 0 );
    labels[i] = 0;
  }
}
