# DESC
- plots/* - PNG charts exports
- cpp_reconstruction - cpp code used to genereate 3d models
- src - Python code source (main file train_2.5d.py)

# GOAL
> Reconstruction of the three-dimensional vascular structure of the kidney based on CT imaging

# DATA
https://www.kaggle.com/competitions/blood-vessel-segmentation
![obraz](https://github.com/user-attachments/assets/ac6028b2-b223-4b33-9078-480c19776275)

# PREPROCESSING FINAL
> Alternative orientations
![obraz](https://github.com/user-attachments/assets/54523def-0849-47fc-b1b5-7f82465b1b0f)

> Albumentations
<pre>
A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Transpose(p=0.5),
    A.Affine(scale={"x":(0.7, 1.3), "y":(0.7, 1.3)}, translate_percent={"x":(0, 0.1), "y":(0, 0.1)}, rotate=(-30, 30), shear=(-20, 20), p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
    ], p=1.0),
    A.OneOf([
        A.ElasticTransform(alpha=1, sigma=50, border_mode=1, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.1, border_mode=1, p=0.5)
    ], p=0.4),

    A.Compose([
        RandomResize(height=1024, width=1024, scale_limit=0.2, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(image_size, image_size, position="random", border_mode=cv2.BORDER_REPLICATE, p=1.0),
        A.RandomCrop(image_size, image_size, p=1.0)
    ], p=0.5),

    A.GaussNoise(var_limit=0.05, p=0.2),
])
</pre>

# RESULTS
## 2D Models
![obraz](https://github.com/user-attachments/assets/4382ed48-1424-459e-9bbc-d792aea29117)

## 2.5D Models 
![obraz](https://github.com/user-attachments/assets/e0ba837d-ebe7-4dbf-8b8b-f1c1bfb0307f)


> Best model is Unet++ with MaxViT encoder, test score on kaggle private dataset is 0.758. This score would result in top 3 in kaggle competition for this dataset.

## RECONSTRUCTION
> Final reconstructions were done using Flying Edges algorithm from VTK package.
> Red color - original
> Blue color - prediction using best model

https://github.com/user-attachments/assets/56980f2a-4a62-4add-b3b3-9443036a6848


