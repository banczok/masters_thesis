import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import monai 

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

resume=True

class newDataset(Dataset):
    def __init__(self, image_files, mask_files, patch_size=768, patch_overlap=0.2, transform=None):
        self.image_files=image_files
        self.mask_files=mask_files
        self.transform=transform
        self.patch_size = patch_size
        self.patch_overlap = patch_overlap
        self.all_patches = []
        self.initialize_patches_parallel(max_workers=16)

    def __len__(self):
        return len(self.all_patches)

    def __getitem__(self, idx):
        patch_image, patch_mask = self.all_patches[idx]

        patch_image = torch.tensor(patch_image, dtype=torch.float32)
        patch_mask = torch.tensor(patch_mask, dtype=torch.float32)

        if self.transform:
            patch_image,patch_mask=self.augment_image(patch_image,patch_mask)

        return patch_image, patch_mask
    
        
    def initialize_patches_parallel(self, max_workers=8):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.process_image, img, mask) for img, mask in zip(self.image_files, self.mask_files)]
            
            for future in futures:
                image_patches = future.result()
                self.all_patches.extend(image_patches)

    def process_image(self, img_path, mask_path):
        image = cv2.imread(img_path, 0)
        mask = cv2.imread(mask_path, 0)
        image = self.image_histogram_equalization(image.astype('float32') / image.max())
        mask = mask.astype('float32') / 255.
        return self.extract_patches(image.astype('float32')/ image.max(), mask)
    

    def extract_patches(self, image, mask):
        patches = []

        def calculate_dynamic_step(img_dim, patch_dim, overlap_fraction):
            overlap = int(patch_dim * overlap_fraction)
            step = patch_dim - overlap
            num_patches = math.ceil((img_dim - overlap) / step)
            step = (img_dim - patch_dim) / (num_patches - 1) if num_patches > 1 else 0
            return step, num_patches

        step_h, num_patches_h = calculate_dynamic_step(image.shape[0], self.patch_size, self.patch_overlap)
        step_w, num_patches_w = calculate_dynamic_step(image.shape[1], self.patch_size, self.patch_overlap)

        patches = []
        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = int(i * step_h)
                x = int(j * step_w)
                patch_img = image[y:y + self.patch_size, x:x + self.patch_size]
                patch_mask = mask[y:y + self.patch_size, x:x + self.patch_size]
                patches.append((patch_img, patch_mask))

        return patches

    def reconstruct_image_from_patches(self, patches, original_shape):
        reconstructed_image = np.zeros(original_shape, dtype=patches[0][0].dtype)

        step = int(self.patch_size * (1 - self.patch_overlap))
        num_patches_h = math.ceil((original_shape[0] - self.patch_size) / step + 1)
        num_patches_w = math.ceil((original_shape[1] - self.patch_size) / step + 1)

        for i in range(num_patches_h):
            for j in range(num_patches_w):
                y = int(i * step)
                x = int(j * step)
                patch_img, _ = patches[i * num_patches_w + j] 
                reconstructed_image[y:y + self.patch_size, x:x + self.patch_size] = patch_img

        return reconstructed_image

    def image_histogram_equalization(self, image, number_bins=256):
        image_histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
        cdf = image_histogram.cumsum() 
        cdf = (number_bins-1) * cdf / cdf[-1] 
        
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
    
        return image_equalized.reshape(image.shape)
    
    def augment_image(self, image, mask):
        shape = (*image.shape, 1 )
        image_np = image.reshape(shape).numpy()
        mask_np = mask.reshape(shape).numpy()
        augmented = self.transform(image = image_np,mask = mask_np)
        augmented_image , augmented_mask = augmented['image'],augmented['mask']
        augmented_image = augmented_image.reshape(augmented_image.shape[:-1])
        augmented_mask = augmented_mask.reshape(augmented_mask.shape[:-1])
        augmented_image = torch.tensor(augmented_image, dtype=torch.float32)
        augmented_mask  = torch.tensor(augmented_mask,dtype=torch.float32)
        
        return augmented_image,augmented_mask

    def calculate_num_patches(self, image_height, image_width):
        step_w = int(self.patch_size * (1-self.patch_overlap))
        step_h = int(self.patch_size * (1-self.patch_overlap))

        num_patches_w = math.ceil((image_width - self.patch_size) / step_w + 1)
        num_patches_h = math.ceil((image_height - self.patch_size) / step_h + 1)

        return num_patches_w * num_patches_h

    
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, epochs_no_improve=0, best_loss=None):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = best_loss
        self.epochs_no_improve = epochs_no_improve
        self.should_stop = False

    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
            if self.epochs_no_improve >= self.patience:
                self.should_stop = True


def train(train_dataset, val_dataset, batch_s=64, num_w=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = monai.networks.nets.SwinUNETR(
        img_size=(512,512), 
        in_channels=1, 
        out_channels=1, 
        use_checkpoint=True, 
        spatial_dims=2,
        depths=(4,6,6,8),
        num_heads=(4,8,16,32),
        feature_size=48,
        drop_rate=0.05,            
        attn_drop_rate=0.05
    ).to(device)
    name = 'SwinUNETR'
    model = torch.nn.DataParallel(model).to(device)

    #criterion = torch.nn.BCEWithLogitsLoss()
    
    
    criterion = smp.losses.TverskyLoss(
        mode='binary',         # Since it's binary segmentation
        from_logits=True,       # Use this if your model outputs logits
        alpha=0.3,              # Penalize false positives less
        beta=0.7,               # Penalize false negatives more
        smooth=1e-6             # Small smoothness constant for numerical stability
    )
    """
    criterion = smp.losses.DiceLoss(
        mode='binary',         # Since it's binary segmentation
        from_logits=True,      # Use this if your model outputs logits
        smooth=1e-6            # Small smoothness constant for numerical stability
    )
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)

    if resume:
        checkpoint = torch.load(f'{name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    train_dataloader= DataLoader(train_dataset,batch_size=batch_s,shuffle=True, pin_memory=True, num_workers=num_w)
    val_dataloader = DataLoader(val_dataset,batch_size=batch_s,shuffle=False, pin_memory=True, num_workers=num_w)

    num_epochs = 500

    early_stopping = EarlyStopping(patience=75, min_delta=0)
    
    all_train_losses = []
    epoch_losses = []
    val_losses = []

    scaler = GradScaler()
    if resume:
        scaler.load_state_dict(checkpoint['scaler'])
        prev_epoch = checkpoint['epoch']+1
        best_loss = checkpoint['val_loss']
    else:
        best_loss = float('inf')
        prev_epoch = 1

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    for epoch in range(prev_epoch, num_epochs):
        model.train()
        epoch_train_losses = []

        for batch_num, (images, masks) in enumerate(train_dataloader):
            b, h, w = images.shape
            images = images.reshape(b, 1, h, w).to(device)
            masks = masks.reshape(b, 1, h, w).to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            train_loss = loss.item()
            all_train_losses.append(train_loss)
            epoch_train_losses.append(train_loss)
            
            if batch_num % 20 == 0:
                log_message = f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num+1}/{len(train_dataloader)}], Batch Loss: {train_loss:.8f}'
                logging.info(log_message)


        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_losses.append(avg_train_loss)

        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, masks in val_dataloader:
                b, h, w = images.shape
                images = images.reshape(b, 1, h, w).to(device)
                masks = masks.reshape(b, 1, h, w).to(device)

                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, masks)

                running_loss += loss.item()

            avg_val_loss = running_loss / len(val_dataloader)
            val_losses.append(avg_val_loss)
            
        log_message = f"Epoch {epoch+1} - Training Loss: {avg_train_loss:.8f}, Validation Loss: {avg_val_loss:.8f}"
        logging.info(log_message)

        if avg_val_loss < best_loss: 
            logging.info(f"Validation Loss Improved ({best_loss} --> {avg_val_loss})")
            best_loss = avg_val_loss 

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss' : avg_val_loss,
                'scaler' : scaler.state_dict()
            }, f"{name}.pth")
            logging.info("Model saved as best_model.pth")

        early_stopping(avg_val_loss)
        if early_stopping.should_stop:
            logging.info("Early stopping triggered")
            break

    print(f'DONE')


def run():
    base_path = "/"
    datasets = ["kidney_1_dense"]#, "kidney_1_voi", "kidney_2", "kidney_3_sparse"]

    image_files = []
    labels_files = []
    for dataset in datasets:
        print(dataset)
        images_path = os.path.join(base_path,dataset,"images")
        label_path = os.path.join(base_path,dataset,"labels")
        print(images_path)

        image_files.extend(sorted([os.path.join(images_path,f) for f in os.listdir(images_path) if f.endswith('.tif')]))
        labels_files.extend(sorted([os.path.join(label_path,f) for f in os.listdir(label_path) if f.endswith('.tif')]))

    train_image_files, val_image_files, train_mask_files, val_mask_files = train_test_split(
        image_files, labels_files, test_size=0.15, random_state=42)

    transform = A.Compose([
        A.Resize(512,512, interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        A.RandomCrop(height=512, width=512, always_apply=True),
        A.OneOf(
            [
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.7,
        ),

    ])

    val_transform = A.Compose([
        A.Resize(512,512, interpolation=cv2.INTER_NEAREST)
    ])
    train_dataset = newDataset(
        train_image_files, 
        train_mask_files,
        transform=transform)

    val_dataset = newDataset(val_image_files, 
                            val_mask_files,
                            transform=val_transform)
    
    train(train_dataset, val_dataset, 28, 48)



if __name__ == "__main__":
    print("Start")
    run()