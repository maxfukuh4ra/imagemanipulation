/* 
 *  Name: Masayoshi Max Fukuhara
 *  UID: 705914579
 */

#include <stdlib.h>
#include <omp.h>

#include "utils.h"
#include "parallel.h"



/*
 *  PHASE 1: compute the mean pixel value
 *  This code is buggy! Find the bug and speed it up.
 */
void mean_pixel_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, double mean[NUM_CHANNELS]) {
    int row, col, ch;
    long count = num_rows*num_cols;
    // idea: make for loop order row, col, ch
    // idea: initialize count and rid of computation
    // idea: compute count since it is a constant 
    #pragma omp parallel for private(col, ch) reduction(+:mean[:NUM_CHANNELS])
        for (row = 0; row < num_rows; row++) {
            int temp = row*num_cols;
            for (col = 0; col < num_cols; col++) {
                for (ch = 0; ch < NUM_CHANNELS; ch++) {
                    mean[ch] += img[temp+col][ch];
                }
            }
        }

    for (ch = 0; ch < NUM_CHANNELS; ch++) {
        mean[ch] /= count;
    }
}


/*
 *  PHASE 2: convert image to grayscale and record the max grayscale value along with the number of times it appears
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void grayscale_parallel(const uint8_t img[][NUM_CHANNELS], int num_rows, int num_cols, uint32_t grayscale_img[][NUM_CHANNELS], uint8_t *max_gray, uint32_t *max_count) {
    int row, col, ch, gray_ch;
    *max_gray = 0;
    *max_count = 0;

    // idea: since we cannot do reduction on pointers, make a variable to do reduction on and update it
    uint8_t max_gray2 = 0;
    uint32_t max_count2 = 0;
    // idea: make for loop order row major and no need for ch loop since it is from 1-3 and all values are same
    // idea: make temp variable computation for row*num_cols
    // idea: find the max value first and then count them in a separate loop
    #pragma omp parallel for private(col) reduction(max: max_gray2) 
    for (row = 0; row < num_rows; row++) {
        int temp = row*num_cols; // temp computation
        for (col = 0; col < num_cols; col++){
                uint32_t grayscale_val = (img[temp+col][0] + img[temp+col][1] + img[temp+col][2]) / NUM_CHANNELS; // grayscale calculation
                // set the entire subarray to that value
                grayscale_img[temp+col][0] = grayscale_val; 
                grayscale_img[temp+col][1] = grayscale_val;
                grayscale_img[temp+col][2] = grayscale_val;
                // if that value is greater than what is currently the max, update it 
                // no race case since it is reduced with max at the end
                if (grayscale_val > max_gray2) { 
                    max_gray2 = grayscale_val;
                }
            }
        }
    
    *max_gray = max_gray2; // update the max to the original given variable

    // idea: check by 3 since if first one is the max val, the next 2 are also max vals
    #pragma omp parallel for private(col)reduction(+: max_count2)
    for (row = 0; row < num_rows; row++) {
        int temp = row*num_cols;
        for (col = 0; col < num_cols; col++) {
            if (grayscale_img[temp + col][0] == max_gray2) {
                max_count2 += 3;
            }
        }
    }
    
    *max_count = max_count2; // update the count to to original given variable
    }





/*
 *  PHASE 3: perform convolution on image
 *  This code is NOT buggy, just sequential. Speed it up.
 */
void convolution_parallel(const uint8_t padded_img[][NUM_CHANNELS], int num_rows, int num_cols, const uint32_t kernel[], int kernel_size, uint32_t convolved_img[][NUM_CHANNELS]) {
    int row, col, ch, kernel_row, kernel_col;
    int kernel_norm, i;
    int conv_rows, conv_cols;

    // compute kernel normalization factor
    kernel_norm = 0;
    for(i = 0; i < kernel_size*kernel_size; i++) {
        kernel_norm += kernel[i];
    }

    // compute dimensions of convolved image
    conv_rows = num_rows - kernel_size + 1;
    conv_cols = num_cols - kernel_size + 1;

    // perform convolution
    #pragma omp parallel for private(row, col, ch, kernel_row, kernel_col) collapse(2)
    for (ch = 0; ch < NUM_CHANNELS; ch++) {
        for (row = 0; row < conv_rows;
         row++) {
            int temp1 = row*conv_cols;
            for (col = 0; col < conv_cols; col++) { 
                //convolved_img[temp1 + col][ch] = 0; 
                for (kernel_row = 0; kernel_row < kernel_size; kernel_row++) {
                    int temp2 = (row+kernel_row)*num_cols + col;
                    int temp3 = kernel_row*kernel_size;
                    for (kernel_col = 0; kernel_col < kernel_size; kernel_col++) {
                        convolved_img[temp1 + col][ch] += padded_img[temp2 + kernel_col][ch] * kernel[temp3 + kernel_col];
                    }
                }
                convolved_img[row*conv_cols + col][ch] /= kernel_norm;
            }    
        }
    }
}