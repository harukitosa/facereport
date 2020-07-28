#include <fstream>

#include <iomanip>

#include <iostream>

#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>

#include "haarlike.h"

#define T 100
#define ST -255
#define MIN_T -255
#define MAX_T 255
#define SUBSIZE 100
#define F_SIZE_MIN 10

void adaboost(cv::Mat* images, int* labels, int sample_num)
{
    int min_p[T];
    double min_thrd[T];
    int n_f = T; //今回は T と等しくしているが、場合によっては異なる T>=n_f
    int T_idx[T];
    double min_error[T];

    // Haar-like 特徴の情報を保持するための変数
    int f_h[n_f];
    int f_w[n_f];
    int f_hx[n_f];
    int f_hy[n_f];
    int f_flag[n_f];

    //乱数を使った Haar-like 特徴の生成
    for (int i = 0; i < n_f; i++) {
        //Haar like feature の高さ
        f_h[i] = rand() % (SUBSIZE - F_SIZE_MIN) + F_SIZE_MIN;
        //Haar like feature の幅
        f_w[i] = rand() % (SUBSIZE - F_SIZE_MIN) + F_SIZE_MIN;
        //Haar like feature 左端の x 座標
        f_hx[i] = rand() % (SUBSIZE - f_w[i]);
        //Haar like feature の右端の y 座標
        f_hy[i] = rand() % (SUBSIZE - f_h[i]);
        // 4 種類の feature から 1つ選択
        f_flag[i] = rand() % 4;
    }

    // 論文の方式とは違うけど、重みの初期化
    // ここでは単純にサンプル数で割っている
    double w[sample_num];
    for (int i = 0; i < sample_num; i++) {
        w[i] = 1.0 / sample_num;
    }

    //訓練画像から積分画像の生成
    cv::Mat int_img[sample_num];
    for (int i = 0; i < sample_num; i++) {
        cv::integral(images[i], int_img[i]);
    }

    //Adaboost の main の loop
    for (int t = 0; t < T; t++) {
        T_idx[t] = -1;
        min_error[t] = 10000;
        min_thrd[t] = 0.0;
        min_p[t] = 0.0;
        int min_h[sample_num];

        // 1. 重みの初期化
        double sum_w = 0.0;
        // Normalize th weights
        // weightを全てたしてその合計で割って正規化していく
        for (int i = 0; i < sample_num; i++) {
            sum_w += w[i];
        }
        for (int i = 0; i < sample_num; i++) {
            // 以下の重み更新を正しく修正すること
            w[i] /= sum_w;
        }

        double f[sample_num];
        int h[sample_num];

        // 2. 弱識別器の学習、つまり、Viola Jones の論文における theta と p の最適化
        for (int j = 0; j < n_f; j++) {
            //std::cout << "i:" << i << std::endl;
            for (int i = 0; i < sample_num; i++) {
                f[i] = haarlike(int_img[i], 0, 0, f_hx[j], f_hy[j], f_h[j], f_w[j], f_flag[j]);
            }

            for (int thrd = MIN_T; thrd < MAX_T; thrd++) {
                for (int p = -1; p < 3; p += 2) {
                    double sum_error = 0.0;
                    for (int i = 0; i < sample_num; i++) {
                        if (p * f[i] < p * thrd) {
                            h[i] = 1;
                        }
                        else {
                            h[i] = 0;
                        }
                        sum_error += (w[i] * abs((double)h[i] - (double)labels[i]));
                    }

                    if (min_error[t] > sum_error) {
                        min_error[t] = sum_error;
                        T_idx[t] = j;
                        min_thrd[t] = thrd;
                        min_p[t] = p;
                        for (int k = 0; k < sample_num; k++) {
                            min_h[k] = h[k];
                        }
                    }
                }
            }
        }
        // theta に関する最適化。
        // 今回は単純に thrd (theta のこと) をある範囲内で動かして探すことにする
        // このループの中で 「3. エラーを最小にする弱識別器の選択」を行う
        //

        // 4.update the weights
        if (min_error[t] == 0) {
            //エラーが 0 になってしまった場合の例外処理
            f_h[T_idx[t]] = rand() % (SUBSIZE - F_SIZE_MIN) + F_SIZE_MIN;
            f_w[T_idx[t]] = rand() % (SUBSIZE - F_SIZE_MIN) + F_SIZE_MIN;
            f_hx[T_idx[t]] = rand() % (SUBSIZE - f_w[T_idx[t]]);
            f_hy[T_idx[t]] = rand() % (SUBSIZE - f_h[T_idx[t]]);
            f_flag[T_idx[t]] = rand() % 4;
            t--;
        }
        else {
            // 論文における 「4.重みの更新」を行う
            // for (int k = 0;k < sample_num;k++) {
            // if (min_thrd[t] != 0) {
            if (labels[t] == min_h[t]) {
                w[t] = w[t]*(min_error[t]/(1-min_error[t]));
            } 
            // w[t] = w[t] * std::pow(min_error[t] / (1.0 - min_error[t]), 1.0 - min_error[t]);
            // std::cout << w[t] * std::pow(min_error[t] / (1.0 - min_error[t]), 1.0 - min_error[t]) << std::endl;
            // std::cout << "min_error[" << t << "] " << min_error[t] << std::endl;
            // std::cout << "w[" << t << "]" << w[t] << std::endl;
            f_h[T_idx[t]] = rand() % (SUBSIZE - F_SIZE_MIN) + F_SIZE_MIN;
            f_w[T_idx[t]] = rand() % (SUBSIZE - F_SIZE_MIN) + F_SIZE_MIN;
            f_hx[T_idx[t]] = rand() % (SUBSIZE - f_w[T_idx[t]]);
            f_hy[T_idx[t]] = rand() % (SUBSIZE - f_h[T_idx[t]]);
            f_flag[T_idx[t]] = rand() % 4;
        }
        std::cout << "classifier " << t << " is finished." << std::endl;

    } // main loop の終わり

    //学習結果の出力
    std::ofstream ofs("out.txt");
    for (int i = 0; i < T; i++) {
        ofs << f_hx[T_idx[i]] << ", " << f_hy[T_idx[i]] << ", " << f_h[T_idx[i]] << ", " << f_w[T_idx[i]] << ", " << f_flag[T_idx[i]] << ", " << min_thrd[i] << ", " << min_error[i] / (1 - min_error[i]) << ", " << min_p[i] << std::endl;
    }
}