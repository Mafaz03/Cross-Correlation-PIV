#include <iostream>
#include <vector>
#include <cmath>
#include <utility>


std::pair<int, int> argmax(int* matrix, int rows, int cols){

    int max_row = 0;
    int max_col = 0;
    int max_ele = matrix[0];

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            if (matrix[i * cols + j]  > max_ele){
                max_row = i;
                max_col = j;
                max_ele = matrix[i * cols + j];
            }
        }
    }
    return {max_row, max_col};
}

std::pair<int, int> shift(int rows, int cols, std::pair<int, int> argmax_val){
    return {
        argmax_val.first  - (rows/2),   
        argmax_val.second - (cols/2)    
    };
}


int main(){

    int N = 3;

    int W1[3][3] = {
        {1, 0 , 0},
        {1, 1 , 0},
        {3, 1 , 0},
    };

    int W2[3][3] = {
        {0, 1 , 3},
        {0, 1 , 1},
        {0, 0 , 1},
    };

    int R[5][5] = {0};


    int row = 0;
    int col = 0;

    for (int row = 0; row < (2*N)-1; row++){

        for (int i = N-1; i >= 0; i--){
            for (int j = N-1; j >= i; j--){
                // printf("%d x %d | ", W1[N-1][N+i-j-1], W2[N-3][N-j-1]);
                int shift = row - (N-1);
                for (int k = 0; k < N; k++){
                    int w1_row = k;
                    int w2_row = k + shift;

                    if (w2_row >= 0 && w2_row < N){
                        R[row][col] += W1[w1_row][N+i-j-1] * W2[w2_row][N-j-1];
                    }
                }
            }
            col++;
            // printf("\n");
        }

        // decreasing
        for (int i = 1; i < N; i++) {
            for (int j = i; j < N; j++){
                // printf("%d x %d | ", W1[N-1][N-j-1], W2[N-3][N-1 - (j - i)]);
                int shift = row - (N-1);
                for (int k = 0; k < N; k++){
                    int w1_row = k;
                    int w2_row = k + shift;

                    if (w2_row >= 0 && w2_row < N){
                        R[row][col] += W1[w1_row][N-j-1] * W2[w2_row][N-1 - (j - i)];
                    }
                }

            }
            col++;
            // printf("\n");
        }
        col = 0;
    }

    auto shift_cords = shift(2*N-1, 2*N-1, argmax(&R[0][0], 2*N-1, 2*N-1));

    for (int i = 0; i < (2*N)-1; i++){
        for (int j = 0; j < (2*N)-1; j++){
            printf("%d | ", R[i][j]);
        }
        printf("\n");
    }

    printf("\n(dy, dx) ( %d, %d)", shift_cords.first, shift_cords.second);


    return 0;
}