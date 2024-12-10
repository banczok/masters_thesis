#include <opencv2/opencv.hpp>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkContourFilter.h>
#include <vtkPolyDataMapper.h>
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

// Placeholder for memory usage
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
std::vector<cv::Mat> load_masks_to_volume(const std::string& mask_folder, float scale = 1.0, cv::Size blur_kernel = cv::Size(25, 25)) {
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
    unsigned char* vtk_data = static_cast<unsigned char*>(vtk_volume->GetScalarPointer());

    size_t slice_size = static_cast<size_t>(width * height);
    for (int z = 0; z < depth; ++z) {
        if (volume[z].empty()) {
            std::cerr << "Error: Empty slice at index " << z << std::endl;
            continue;
        }

        if (volume[z].rows != height || volume[z].cols != width || volume[z].type() != CV_8U) {
            std::cerr << "Mismatch in volume dimensions or type at slice " << z << std::endl;
            continue;
        }

        std::memcpy(vtk_data + z * slice_size, volume[z].data, slice_size);
    }
    return vtk_volume;
}

// Main function for Contour Filter
int main() {
    std::string mask_folder = "/mnt/c/Users/Bartek/Desktop/mgr/train/kidney_3_sparse/labels";
    auto start = std::chrono::high_resolution_clock::now();

    auto volume = load_masks_to_volume(mask_folder, 1.0, cv::Size(1, 1));
    printMemoryUsage();

    auto vtk_volume = convert_volume_to_vtk(volume);
    printMemoryUsage();

    auto contourFilter = vtkSmartPointer<vtkContourFilter>::New();
    contourFilter->SetInputData(vtk_volume);
    double isovalue = 127.5;  // Example isovalue for isosurface extraction
    contourFilter->SetValue(0, isovalue);
    contourFilter->Update();
    printMemoryUsage();

    auto objWriter = vtkSmartPointer<vtkOBJWriter>::New();
    objWriter->SetFileName("contour_filter_output.obj");
    objWriter->SetInputData(contourFilter->GetOutput());
    objWriter->Write();

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Czas wykonania: " << elapsed.count() << " sekund." << std::endl;
    std::cout << "Contour Filter: OBJ file saved as 'contour_filter_output.obj'." << std::endl;

    std::cin.get();  // Wait for user input
    return EXIT_SUCCESS;
}
