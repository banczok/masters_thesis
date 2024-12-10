import os
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob

# Paths for data
path = r'C:\Users\Bartek\Desktop\mgr'
output = 'mmap\\'

groups = [group.split("/")[-1] for group in glob(os.path.join(path, "train/*"))]
print(groups)
for group in sorted(groups):
    print(os.path.join(path, "train", group, "labels", "*.tif"))
    label_paths = sorted(glob(os.path.join(path, "train", group, "labels", "*.tif")))
    slice_paths = [path.replace("/labels/", "/images/").replace("kidney_3_dense", "kidney_3_sparse") for path in label_paths]
    h, w = cv2.imread(slice_paths[0], cv2.IMREAD_UNCHANGED).shape
    volume = np.memmap(os.path.join(output, f"{group}.mmap"), dtype=np.uint16, shape=(len(slice_paths), h, w), mode="w+")
    volume_mask = np.memmap(os.path.join(output, f"{group}_mask.mmap"), dtype=np.uint8, shape=(len(slice_paths), h, w), mode="w+")
    for i, (slice_path, label_path) in tqdm(enumerate(zip(slice_paths, label_paths)), total=len(slice_paths)):
        slice = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        
        volume[i] = slice 
        volume_mask[i] = label / 255.
        
    volume.flush()
    volume_mask.flush()

# merge kidney_3_dense and kidney_3_sparse
dense_label_paths = sorted(glob(os.path.join(path, "train", "kidney_3_dense", "labels", "*.tif")))
dense_ids = [s.split("/")[-1][:-4] for s in dense_label_paths]
slice_paths = sorted(glob(os.path.join(path, "train", "kidney_3_sparse", "images", "*.tif")))
label_paths = [s.replace("/images/", "/labels/") for s in slice_paths]
label_paths = [s.replace("kidney_3_sparse", "kidney_3_dense") if s.split("/")[-1][:-4] in dense_ids else s for s in label_paths]

h, w = cv2.imread(slice_paths[0], cv2.IMREAD_UNCHANGED).shape
volume = np.memmap(os.path.join(output, f"kidney_3.mmap"), dtype=np.uint16, shape=(len(slice_paths), h, w), mode="w+")
volume_mask = np.memmap(os.path.join(output, f"kidney_3_mask.mmap"), dtype=np.uint8, shape=(len(slice_paths), h, w), mode="w+")
for i, (slice_path, label_path) in tqdm(enumerate(zip(slice_paths, label_paths)), total=len(slice_paths)):
    slice = cv2.imread(slice_path, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
    volume[i] = slice 
    volume_mask[i] = label / 255.
    
volume.flush()
volume_mask.flush()