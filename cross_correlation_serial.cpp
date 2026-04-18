#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <utility>
#include <fstream>
#include <filesystem>
#include <chrono>

#include <opencv2/opencv.hpp>


std::pair<cv::Mat, cv::Mat> load_image(cv::Mat before_img, cv::Mat after_img, int size){
    // crops the video from the center

    if ((before_img.rows != after_img.rows) || (before_img.cols != after_img.cols)){
        throw std::runtime_error("Dimention mismatch!");
    }

    std::pair<int, int> center = {before_img.rows/2, before_img.cols/2};

    cv::Rect myROI(
        center.second - size / 2, 
        center.first  - size / 2, 
        size, 
        size);

    cv::Mat before_crop = before_img(myROI);
    cv::Mat after_crop = after_img(myROI);

    cv::Mat before_gray, after_gray;
    cv::cvtColor(before_crop, before_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(after_crop,  after_gray,  cv::COLOR_BGR2GRAY);

    return {before_gray, after_gray};
}



std::tuple<int, int, int> argmax(int* matrix, int rows, int cols){
    int max_row = 0;
    int max_col = 0;
    int max_ele = matrix[0];
    int failed_cases = 0;

    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            if (matrix[i * cols + j]  > max_ele){
                max_row = i;
                max_col = j;
                max_ele = matrix[i * cols + j];
            }
            else{
                failed_cases++;
            }
        }
    }
    if (max_ele == 0)
        return {max_row, max_col, 0};  // no signal
    return {max_row, max_col, 1};      // valid peak
}

std::pair<int, int> shift(int rows, int cols, std::tuple<int, int, int> argmax_val) {
    auto [r, c, val] = argmax_val;
    if (val == 0)
        return {0, 0};
    int dy = r  - (rows / 2);  // positive = down
    int dx = c - (cols / 2);   // positive = right
    return {dx, dy};
}

std::string make_output_path(std::string template_path, int frame_index){
    std::filesystem::path p(template_path);
    if (p.has_parent_path()){
        std::filesystem::create_directories(p.parent_path());
    }

    std::string parent = p.parent_path().string();
    std::string ext = p.extension().string();
    std::string stem = p.stem().string();
    std::string file_name = parent + "/" + stem + "_" + std::to_string(frame_index) + ext;

    return file_name;
}



// main
int main(int argc, char* argv[]){

    // /a.out `video_path` `save_U_pth` `save_V_pth` `save_img_pth` `skip`
    

    if (argc != 6){
        throw std::runtime_error("Please enter before and after image path");
    }
   
    std::string video_pth    = argv[1];
    std::string save_U_pth   = argv[2];
    std::string save_V_pth   = argv[3];
    std::string save_img_pth = argv[4];
    int         skip         = std::stoi(argv[5]);
    
    
    
    const int NN = 1000;
    const int N = 15;
    const int N_window_result = 2*N-1;

    auto start = std::chrono::steady_clock::now();

    // cv::VideoCapture cap(video_pth); 
    cv::VideoCapture cap(video_pth, cv::CAP_FFMPEG);
    if (!cap.isOpened()){
        throw std::runtime_error("ERROR: Something wrong with the video");
    }
    
    cv::Mat frame_after, frame_before;
    int frameCount = 0;

    cap.read(frame_before);

    
    while (cap.read(frame_after)){
        frameCount++;
        if (frameCount % skip != 0) {

            // std::swap(frame_before, frame_after);
            continue;  
        }

        // load images
        auto imgs = load_image(frame_before, frame_after, NN);
        cv::Mat before_img = imgs.first;
        cv::Mat after_img = imgs.second;


        cv::imwrite(make_output_path(save_img_pth, frameCount-1), before_img);

        // unsigned char Before[NN][NN], After[NN][NN];
        std::vector<std::vector<unsigned char>> Before(NN, std::vector<unsigned char>(NN));
        std::vector<std::vector<unsigned char>> After(NN, std::vector<unsigned char>(NN));

        for (int i = 0; i < NN; i++){
            for (int j = 0; j < NN; j++){
                Before[i][j] = before_img.at<uchar>(i, j);
                After[i][j]  = after_img.at<uchar>(i, j);
            }
        }

        std::ofstream u_outFile(make_output_path(save_U_pth, frameCount-1));
        std::ofstream v_outFile(make_output_path(save_V_pth, frameCount-1));
        
        int W1[N][N];
        int W2[N][N];

        const int step = N/2;  // 50% overlap
        for (int shift_down = 0; shift_down < NN - N; shift_down += step){
            for (int shift_right = 0; shift_right < NN - N; shift_right += step){

                int R[N_window_result][N_window_result] = {0};

                for (int i = 0; i < N; i++){
                    for (int j = 0; j < N; j++){
                        W1[i][j] = Before[shift_down + i][shift_right + j];
                        W2[i][j] = After[shift_down + i][shift_right + j];
                    }
                }

                for (int dy = -(N-1); dy <= N-1; dy++) {
                    for (int dx = -(N-1); dx <= N-1; dx++) {
                        int sum = 0;
                        for (int i = 0; i < N; i++) {
                            for (int j = 0; j < N; j++) {
                                int i2 = i + dy;
                                int j2 = j + dx;
                                if (i2 >= 0 && i2 < N && j2 >= 0 && j2 < N) {
                                    sum += W1[i][j] * W2[i2][j2];
                                }
                            }
                        }
                        R[dy + (N-1)][dx + (N-1)] = sum;
                    }
                }

                auto shift_cords = shift(2*N-1, 2*N-1, argmax(&R[0][0], 2*N-1, 2*N-1));

                for (int i = 0; i < (2*N)-1; i++){
                    for (int j = 0; j < (2*N)-1; j++){
                    }
                }

                u_outFile << shift_cords.first << " ";
                v_outFile << shift_cords.second << " ";

            }
            u_outFile << "\n";
            v_outFile << "\n";
            printf("FRAME: %d | %d/%d\n", frameCount, shift_down, NN-N);
        }

        u_outFile.close();
        v_outFile.close();
        std::swap(frame_before, frame_after);

    }

    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end - start);
    printf("\n########################\n\n\nTime: %lld seconds\n\n\n############", duration.count());
    return 0;
}