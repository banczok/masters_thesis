#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
#include <filesystem>
#include <unistd.h>
#include <chrono>

namespace fs = std::filesystem;

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

// Load 3D volume from segmented masks
std::vector<cv::Mat> load_volume(const std::string& folder) {
    std::vector<cv::Mat> volume;
    for (const auto& entry : fs::directory_iterator(folder)) {
        if (entry.path().extension() == ".tif") {
            volume.push_back(cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE));
        }
    }
    return volume;
}

// Convert segmented volume to VDB Grid
openvdb::FloatGrid::Ptr convert_to_vdb_grid(const std::vector<cv::Mat>& volume) {
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (int z = 0; z < volume.size(); ++z) {
        for (int y = 0; y < volume[z].rows; ++y) {
            for (int x = 0; x < volume[z].cols; ++x) {
                accessor.setValue(openvdb::Coord(x, y, z), volume[z].at<unsigned char>(y, x) / 255.0f);
            }
        }
    }
    return grid;
}

// Save the mesh as an OBJ file
void save_mesh_as_obj(const openvdb::tools::VolumeToMesh& mesher, const std::string& filename) {
    std::ofstream obj_file(filename);
    if (!obj_file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        return;
    }

    // Write vertices
    const auto& points = mesher.pointList();
    for (size_t i = 0; i < mesher.pointListSize(); ++i) {
        obj_file << "v " << points[i][0] << " " << points[i][1] << " " << points[i][2] << std::endl;
    }

    // Write faces
    const auto& polygons = mesher.polygonPoolList();
    for (size_t i = 0; i < mesher.polygonPoolListSize(); ++i) {
        const auto& polygon = polygons[i];
        for (size_t j = 0; j < polygon.numQuads(); ++j) {
            const auto& quad = polygon.quad(j);
            obj_file << "f " << quad[0] + 1 << " " << quad[1] + 1 << " " << quad[2] + 1 << " " << quad[3] + 1 << std::endl;
        }
        for (size_t j = 0; j < polygon.numTriangles(); ++j) {
            const auto& tri = polygon.triangle(j);
            obj_file << "f " << tri[0] + 1 << " " << tri[1] + 1 << " " << tri[2] + 1 << std::endl;
        }
    }

    obj_file.close();
    std::cout << "Mesh saved as: " << filename << std::endl;
}

int main() {
    openvdb::initialize();
    std::string mask_folder = "/mnt/c/Users/Bartek/Desktop/mgr/train/kidney_3_sparse/labels";
    auto start = std::chrono::high_resolution_clock::now();

    // Load the volume
    auto volume = load_volume(mask_folder);
    std::cout << "Loaded volume size: " << volume.size() << std::endl;


    // Convert to VDB Grid
    auto grid = convert_to_vdb_grid(volume);
    std::cout << "Active voxels: " << grid->activeVoxelCount() << std::endl;


    // Apply Volume to Mesh with an adjusted isovalue
    openvdb::tools::VolumeToMesh mesher(0.5);  // Adjust the isovalue as needed
    mesher(*grid);
    printMemoryUsage();

    std::cout << "Mesh extraction done, saving mesh..." << std::endl;

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Czas wykonania: " << elapsed.count() << " sekund." << std::endl;

    // Save the resulting mesh
    save_mesh_as_obj(mesher, "mc_vdb.obj");
    std::cout << "done" << std::endl;
    
    return 0;
}

