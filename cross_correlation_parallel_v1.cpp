#include <iostream>
#include <vector>
#include <cmath>
#include <tuple>
#include <utility>
#include <fstream>
#include <filesystem>

#include <opencv2/opencv.hpp>

// parallel headers
#include <mpi.h>
#include <omp.h>

namespace fs = std::filesystem;

std::string make_output_path(const std::string &template_path, int frame_index) {
    fs::path p(template_path);
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }

    std::string parent = p.parent_path().string();
    std::string ext = p.extension().string();
    std::string stem = p.stem().string();
    std::string file_name = parent + "/" + stem + "_" + std::to_string(frame_index) + ext;
    if (p.has_parent_path()) {
        fs::create_directories(p.parent_path());
    }
    return file_name;
}

bool ensure_parent_dir(const std::string &path) {
    fs::path p(path);
    if (p.has_parent_path()) {
        return fs::create_directories(p.parent_path());
    }
    return true;
}


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



std::tuple<int,int,int> argmax(std::vector<std::vector<int>>&R, int rows, int cols){
    int max_row = 0;
    int max_col = 0;
    int max_ele = R[0][0];

    // #pragma omp parallel for collapse(2) reduction(max: max_ele)
    for (int i = 0; i < rows; i++){
        for (int j = 0; j < cols; j++){
            if (R[i][j]  > max_ele){
                max_row = i;
                max_col = j;
                max_ele = R[i][j];
            }
            else{
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

void process_frames(cv::Mat frame_before, cv::Mat frame_after, 
                   std::string save_img_pth, std::string save_U_pth, 
                   std::string save_V_pth, int currentFrame, int totalFrames, 
                   int NN, int N, int N_window_result, int step,
                   int rank, int size,int num_thread, int verbose){
    // load images
    auto imgs = load_image(frame_before, frame_after, NN);
    cv::Mat before_img = imgs.first;
    cv::Mat after_img = imgs.second;

    std::string image_path = make_output_path(save_img_pth, currentFrame);
    if (!cv::imwrite(image_path, before_img)) {
        std::cerr << "ERROR: Failed to save image to " << image_path << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    // unsigned char Before[NN][NN], After[NN][NN];
    int ny = (NN - N)/step;
    int nx = (NN - N)/step;

    std::vector<std::vector<unsigned char>> Before(NN, std::vector<unsigned char>(NN));
    std::vector<std::vector<unsigned char>> After(NN, std::vector<unsigned char>(NN));

    std::vector<std::vector<int>> U(ny, std::vector<int>(nx));
    std::vector<std::vector<int>> V(ny, std::vector<int>(nx));

    #pragma omp parallel for collapse(2) num_threads(num_thread)
    for (int i = 0; i < NN; i++){
        for (int j = 0; j < NN; j++){
            Before[i][j] = before_img.at<uchar>(i, j);
            After[i][j]  = after_img.at<uchar>(i, j);
        }
    }

    std::string u_file = make_output_path(save_U_pth, currentFrame);
    std::string v_file = make_output_path(save_V_pth, currentFrame);

    std::ofstream u_outFile(u_file);
    std::ofstream v_outFile(v_file);
    if (!u_outFile.is_open() || !v_outFile.is_open()) {
        std::cerr << "ERROR: Unable to open output files: " << u_file << " or " << v_file << std::endl;
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    
    int W1[N][N];
    int W2[N][N];

    int shift_down, shift_right;

    
    #pragma omp parallel for collapse(2) schedule(dynamic) private(W1, W2, shift_down, shift_right) num_threads(num_thread)
    for (int iy = 0; iy < ny; iy++){
        for (int ix = 0; ix < nx; ix++){

            std::vector<std::vector<int>> R(N_window_result, std::vector<int>(N_window_result, 0));

            shift_down  = iy * step;
            shift_right = ix * step;

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

            auto shift_cords = shift(2*N-1, 2*N-1, argmax(R, 2*N-1, 2*N-1));

            for (int i = 0; i < (2*N)-1; i++){
                for (int j = 0; j < (2*N)-1; j++){
                }
            }

            // u_outFile << shift_cords.first << " ";
            // v_outFile << shift_cords.second << " ";

            U[iy][ix] = shift_cords.first;
            V[iy][ix] = shift_cords.second;

        }
        // u_outFile << "\n";
        // v_outFile << "\n";

        if (verbose){
            #pragma omp critical
            printf("RANK: %d/%d | FRAME: %d/%d| COLUMNS: %d/%d\n", rank, size, currentFrame, totalFrames, shift_down, NN-N);
        }
    }

    for (int i = 0; i < ny; i++){
        for (int j = 0; j < nx; j++){
            u_outFile << U[i][j] << " ";
            v_outFile << V[i][j] << " ";
        }
        u_outFile << "\n";
        v_outFile << "\n";
    }

    u_outFile.close();
    v_outFile.close();
}



// main
int main(int argc, char* argv[]){

    // ./a.out `video_path` `save_U_pth` `save_V_pth` `save_img_pth` `skip` `num_thread` `verbose`


    
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc != 8){
        throw std::runtime_error("Please enter before and after image path");
        MPI_Finalize();
        return -1;
    }
   
    std::string video_pth    = argv[1];
    std::string save_U_pth   = argv[2];
    std::string save_V_pth   = argv[3];
    std::string save_img_pth = argv[4];
    int         skip         = std::stoi(argv[5]);
    int         num_thread   = std::stoi(argv[6]);
    int         verbose      = std::stoi(argv[7]);

    if (rank == 0){
        ensure_parent_dir(save_U_pth);
        ensure_parent_dir(save_V_pth);
        ensure_parent_dir(save_img_pth);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    const int NN = 1000;
    const int N = 15;
    const int N_window_result = 2*N-1;

    const int step = N/2;  // 50% overlap

    cv::VideoCapture cap(video_pth);
    if (!cap.isOpened()){
        throw std::runtime_error("ERROR: Cannot open video");
    }

    // offset start frame based on rank
    int totalFrames = cap.get(cv::CAP_PROP_FRAME_COUNT);
    int division    = std::floor(totalFrames/size);
    int start_frame = rank * division;
    int stop_frame  = start_frame + division;

    // rank 0: 0   -> 250
    // rank 1: 250 -> 500
    // rank 2: 500 -> 750
    // rank 3: 750 -> 1000

    cap.set(cv::CAP_PROP_POS_FRAMES, start_frame);
    cv::Mat frame_after, frame_before;
    cap.read(frame_before);

    // int currentFrame = start_frame;
    double start = MPI_Wtime();

    int currentFrame = start_frame;

    /////////
    while (cap.read(frame_after)){

        if (currentFrame >= stop_frame && rank != size - 1){ // making last rank do left over work
            break;
        }

        // process frames
        if (currentFrame % skip == 0){
            process_frames(frame_before, frame_after, save_img_pth, save_U_pth, save_V_pth,
                currentFrame, totalFrames, NN, N, N_window_result, step, rank, size, num_thread, verbose);
        }

        if (currentFrame % skip != 0) {
            currentFrame++;
            std::swap(frame_before, frame_after);
            continue;  
        }

        std::swap(frame_before, frame_after);
        currentFrame++;
    }
    /////////


    MPI_Finalize();
    
    double end = MPI_Wtime();

    if (rank == 0){
        printf("\n########################\n\n\nTime: %f seconds\n\n\n########################", end - start);
    }

    return 0;
}