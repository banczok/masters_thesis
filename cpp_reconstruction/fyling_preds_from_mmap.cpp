#include <opencv2/opencv.hpp>
#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <vtkFlyingEdges3D.h>
#include <vtkPolyDataMapper.h>
#include <vtkOBJWriter.h>
#include <iostream>
#include <vector>
#include <chrono>
#include <fstream>
#include <cstring>

// Function to read a portion of the volume from a `.mmap` file
std::vector<cv::Mat> load_volume_chunk(const std::string& mmap_path, const cv::Size& dimensions, int start_depth, int end_depth) {
    size_t slice_size = dimensions.width * dimensions.height;
    size_t chunk_size = slice_size * (end_depth - start_depth);

    // Open the mmap file
    std::ifstream mmap_file(mmap_path, std::ios::binary);
    if (!mmap_file.is_open()) {
        throw std::runtime_error("Failed to open mmap file: " + mmap_path);
    }

    // Move the file pointer to the start of the chunk
    mmap_file.seekg(static_cast<std::streamoff>(start_depth * slice_size), std::ios::beg);

    // Allocate memory for the chunk
    std::vector<cv::Mat> volume_chunk(end_depth - start_depth, cv::Mat(dimensions, CV_8U));

    // Read data slice by slice
    for (int z = 0; z < (end_depth - start_depth); ++z) {
        mmap_file.read(reinterpret_cast<char*>(volume_chunk[z].data), slice_size);
        if (!mmap_file) {
            throw std::runtime_error("Error reading slice " + std::to_string(start_depth + z) + " from mmap file.");
        }
    }

    mmap_file.close();
    return volume_chunk;
}

// Convert a chunk to VTK volume
vtkSmartPointer<vtkImageData> convert_chunk_to_vtk(const std::vector<cv::Mat>& volume_chunk) {
    int depth = static_cast<int>(volume_chunk.size());
    int height = volume_chunk[0].rows;
    int width = volume_chunk[0].cols;

    auto vtk_volume = vtkSmartPointer<vtkImageData>::New();
    vtk_volume->SetDimensions(width, height, depth);
    vtk_volume->AllocateScalars(VTK_UNSIGNED_CHAR, 1);

    unsigned char* vtk_data = static_cast<unsigned char*>(vtk_volume->GetScalarPointer());
    size_t slice_size = static_cast<size_t>(width * height);

    for (int z = 0; z < depth; ++z) {
        if (volume_chunk[z].empty()) {
            std::cerr << "Error: Empty slice at index " << z << std::endl;
            continue;
        }

        std::memcpy(vtk_data + z * slice_size, volume_chunk[z].data, slice_size);
    }

    return vtk_volume;
}

// Process the chunk and save the result
void process_chunk(const std::string& mmap_path, const cv::Size& dimensions, int start_depth, int end_depth, const std::string& output_file) {
    // Load the chunk
    std::vector<cv::Mat> volume_chunk = load_volume_chunk(mmap_path, dimensions, start_depth, end_depth);

    // Convert the chunk to VTK format
    auto vtk_volume = convert_chunk_to_vtk(volume_chunk);

    // Apply Flying Edges
    auto flyingEdges = vtkSmartPointer<vtkFlyingEdges3D>::New();
    flyingEdges->SetInputData(vtk_volume);
    flyingEdges->SetValue(0, 0.5);
    flyingEdges->Update();

    // Save the result
    auto objWriter = vtkSmartPointer<vtkOBJWriter>::New();
    objWriter->SetFileName(output_file.c_str());
    objWriter->SetInputData(flyingEdges->GetOutput());
    objWriter->Write();

    std::cout << "Chunk [" << start_depth << ", " << end_depth << ") processed and saved to " << output_file << std::endl;
}

int main() {
    std::string mmap_path = "/mnt/c/Users/Bartek/Desktop/mgr/stl/nower/kidney_3.mmap";
    cv::Size dimensions(1706, 1510);
    int total_depth = 1035;
    int chunk_size = 514; // Number of slices per chunk

    auto start = std::chrono::high_resolution_clock::now();

    // Process the volume in chunks
    for (int start_depth = 0; start_depth < total_depth; start_depth += chunk_size) {
        int end_depth = std::min(start_depth + chunk_size, total_depth);
        std::string output_file = "chunk_" + std::to_string(start_depth) + "_" + std::to_string(end_depth) + ".obj";

        process_chunk(mmap_path, dimensions, start_depth, end_depth, output_file);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Total processing time: " << elapsed.count() << " seconds." << std::endl;

    return EXIT_SUCCESS;
}
