#include <iostream>
#include <filesystem>

int main(){
    std::filesystem::path p("wow/meowy/ikes.txt");
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::string parent = p.parent_path().string();

    std::string ext = p.extension().string();
    std::string stem = p.stem().string();

    std::cout << parent << " " << ext << " " << stem << "\n";
    return 0;
}
