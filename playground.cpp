#include <iostream>
#include <filesystem>

int main(){
    std::filesystem::create_directories("wow");
    std::filesystem::path p("wow/yikes.txt");

    std::string parent = p.parent_path().string();

    std::string ext = p.extension().string();
    std::string stem = p.stem().string();

    std::cout << parent << " " << ext << " " << stem << "\n";
    return 0;
}
