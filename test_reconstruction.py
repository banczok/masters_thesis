import numpy as np
import os
from skimage import io, transform
from skimage.measure import marching_cubes
import vtk
import trimesh
import cv2 as cv
import cumcubes
import torch
import open3d as o3d
from concurrent.futures import ThreadPoolExecutor
import pyvista as pv
import itk
import time

def process_mask_file(mask_file_path, blur_kernel):
    img = cv.imread(mask_file_path, cv.IMREAD_GRAYSCALE)
    blur = cv.GaussianBlur(img, blur_kernel, 0)
    _, thresh = cv.threshold(blur, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv.contourArea, reverse=True)
    largest_contour = contours[0]

    mask = np.zeros_like(thresh)
    cv.drawContours(mask, [largest_contour], -1, (255), -1)
    return mask

def process_image_file(image_file_path):
    img = io.imread(image_file_path)
    return img

def load_masks_to_volume(mask_folder, kidney_folder, cuda=True, blur_kernel=(25, 25)):
    mask_files = sorted([f for f in os.listdir(mask_folder) if f.endswith('.tif')])
    kid_files = sorted([f for f in os.listdir(kidney_folder) if f.endswith('.tif')])

    first_mask = io.imread(os.path.join(mask_folder, mask_files[0]))
    height, width = first_mask.shape[:2]
    volume_shape = (len(mask_files), height, width)

    volume = np.zeros(volume_shape, dtype=np.uint8)
    volume1 = np.zeros(volume_shape, dtype=np.uint8)

    with ThreadPoolExecutor() as executor:
        # Process mask files
        futures_mask = [executor.submit(process_image_file, os.path.join(mask_folder, f)) for f in mask_files]
        for i, future in enumerate(futures_mask):
            volume[i, :, :] = future.result()

        # Process kidney files
        futures_kidney = [executor.submit(process_mask_file, os.path.join(kidney_folder, f), blur_kernel) for f in kid_files]
        for i, future in enumerate(futures_kidney):
            volume1[i, :, :] = future.result()

    if cuda:
        return torch.from_numpy(volume).cuda(), torch.from_numpy(volume1).cuda()
    else:
        return volume, volume1


def create_mesh_from_volume(volume, smoothing=True, level=0.5, iterations=10, cuda=True):
    if cuda:
        verts, faces = cumcubes.marching_cubes(volume, level, verbose=True)
        verts = verts.cpu().numpy()
        faces = faces.cpu().numpy()
    else:
        verts, faces, _, _ = marching_cubes(volume, level=level)
    print("aaa")
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

    if smoothing: 
        mesh = trimesh.smoothing.filter_laplacian(mesh, iterations=iterations)

    return mesh

def apply_marching_cubes_pyvista(volume, level=0.5):
    # Convert the NumPy array to a VTK volume
    vtk_volume = pv.vtkmatrix_from_array(volume)

    # Apply the VTK marching cubes algorithm
    contour_filter = vtk.vtkMarchingCubes()
    contour_filter.SetInputData(vtk_volume)
    contour_filter.SetValue(0, level)
    contour_filter.Update()

    # Convert the result back to a PyVista mesh
    mesh = pv.wrap(contour_filter.GetOutput())
    return mesh

def apply_marching_cubes_itk(volume, level=0.5):
    volume = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.uint8)
    itk_image = itk.image_from_array(volume)
    itk_image = itk.cast_image_filter(itk_image, ttype=(type(itk_image), itk.Image[itk.UC, 3]))

    # Set up the marching cubes algorithm
    mesh_source = itk.BinaryMask3DMeshSource[type(itk_image), itk.Mesh[itk.UC, 3]].New()
    mesh_source.SetInput(itk_image)
    mesh_source.SetObjectValue(127)

    # Get the resulting mesh
    mesh = mesh_source.GetOutput()
    itk.meshwrite(mesh, "output_mesh.vtk")
    return mesh

mask_folder = 'C:\\Users\\Bartek\\Desktop\\mgr\\train\\kidney_1_dense\\labels'
kidney_contour_dir = 'C:\\Users\\Bartek\\Desktop\\mgr\\train\\kidney_1_dense\\images'

volume, _ = load_masks_to_volume(mask_folder, kidney_contour_dir, cuda=False, blur_kernel=(35,35))
print("volume")

start_time = time.time()
#mesh= create_mesh_from_volume(volume, cuda=False, smoothing=False)
#print(mesh)
print("--- %s seconds --- normal" % (time.time() - start_time))

start_time = time.time()
#mesh= apply_marching_cubes_pyvista(volume)
#print(mesh)
print("--- %s seconds --- pyvista" % (time.time() - start_time))

start_time = time.time()
mesh= apply_marching_cubes_itk(volume)
print(mesh)
print("--- %s seconds --- itk" % (time.time() - start_time))
