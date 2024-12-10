#include <opencv2/opencv.hpp>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkMarchingCubes.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkOBJWriter.h>
#include <filesystem>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <unistd.h>

namespace fs = std::filesystem;

// Load individual image as a 2D slice
cv::Mat load_image(const std::string& file_path) {
    return cv::imread(file_path, cv::IMREAD_GRAYSCALE);
}

void printMemoryUsage() {
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        size_t size, resident, shared, text, lib, data, dt;
        statm >> size >> resident >> shared >> text >> lib >> data >> dt;
        std::cout << "Memory (virtual): " << size * getpagesize() / (1024 * 1024) << " MB\n";
        std::cout << "Memory (resident): " << resident * getpagesize() / (1024 * 1024) << " MB\n";
    } else {
        std::cout << "Memory usage tracking not available on this system.\n";
    }
}

// Load all images from the folder into a 3D volume
std::vector<cv::Mat> load_masks_to_volume(const std::string& mask_folder, float scale=1.0, cv::Size blur_kernel=cv::Size(25, 25)) {
    std::vector<std::string> mask_files;
    for (const auto& entry : fs::directory_iterator(mask_folder)) {
        if (entry.path().extension() == ".tif") {
            mask_files.push_back(entry.path().string());
        }
    }
    std::sort(mask_files.begin(), mask_files.end());
    
    cv::Mat first_mask = load_image(mask_files[0]);
    int width = static_cast<int>(first_mask.cols * scale);
    int height = static_cast<int>(first_mask.rows * scale);
    std::vector<cv::Mat> volume(mask_files.size());

    for (size_t i = 0; i < mask_files.size(); ++i) {
        cv::Mat img = load_image(mask_files[i]);

        cv::Mat resized;
        cv::resize(img, resized, cv::Size(width, height), 0, 0, cv::INTER_LINEAR);
        if (blur_kernel.width > 0 && blur_kernel.height > 0) {
            cv::GaussianBlur(resized, resized, blur_kernel, 0);
        }

        volume[i] = resized;
    }

    return volume;
}

// Convert 3D OpenCV volume to VTK volume
vtkSmartPointer<vtkImageData> convert_volume_to_vtk(const std::vector<cv::Mat>& volume) {
    int depth = static_cast<int>(volume.size());
    int height = volume[0].rows;
    int width = volume[0].cols;

    auto vtk_volume = vtkSmartPointer<vtkImageData>::New();
    vtk_volume->SetDimensions(width, height, depth);
    vtk_volume->AllocateScalars(VTK_UNSIGNED_CHAR, 1);
    std::cout << "volume_vtk1" << std::endl;
    

    unsigned char* vtk_data = static_cast<unsigned char*>(vtk_volume->GetScalarPointer());
    
    // Debug: Ensure the total number of voxels match
    size_t slice_size = static_cast<size_t>(width * height);  // Each slice size in pixels
    size_t expected_size = slice_size * depth;                // Total size of the volume
    std::cout << "Expected total volume size: " << expected_size << " bytes" << std::endl;

    std::cout << "loop_after" << std::endl;
    

    for (int z = 0; z < depth; ++z) {
        // Check if the current slice is valid
        if (volume[z].empty()) {
            std::cerr << "Error: Empty slice at index " << z << std::endl;
            continue;  // Skip this slice
        }

        // Check dimensions and type
        if (volume[z].rows != height || volume[z].cols != width || volume[z].type() != CV_8U) {
            std::cerr << "Mismatch in volume dimensions or type at slice " << z << std::endl;
            continue;
        }

        // Copy data from OpenCV Mat to VTK
        std::memcpy(vtk_data + z * slice_size, volume[z].data, slice_size);
        std::cout << "Copied slice " << z << " to VTK volume." << std::endl;
    }

    std::cout << "loop_done" << std::endl;
    
    return vtk_volume;
}


int main() {
    std::cout << "start" << std::endl;
    std::string mask_folder = "/mnt/c/Users/Bartek/Desktop/mgr/train/kidney_3_sparse/labels";
    auto start = std::chrono::high_resolution_clock::now();

    auto volume = load_masks_to_volume(mask_folder, 1.0, cv::Size(1, 1));
    std::cout << "volume" << std::endl;
    

    auto vtk_volume = convert_volume_to_vtk(volume);
    std::cout << "volume_vtk" << std::endl;
    

    auto marchingCubes = vtkSmartPointer<vtkMarchingCubes>::New();
    marchingCubes->SetInputData(vtk_volume);
    marchingCubes->SetValue(0, 127.5);
    marchingCubes->Update();
    
    std::cout << "mc" << std::endl;
    printMemoryUsage();

    auto objWriter = vtkSmartPointer<vtkOBJWriter>::New();
    objWriter->SetFileName("marching_cubes_output.obj");
    objWriter->SetInputData(marchingCubes->GetOutput());
    objWriter->Write();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Czas wykonania: " << elapsed.count() << " sekund." << std::endl;
    
    std::cout << "Marching Cubes: OBJ file saved as 'marching_cubes_output.obj'." << std::endl;
    std::cin.get();  // Wait for user input
    return EXIT_SUCCESS;
}
