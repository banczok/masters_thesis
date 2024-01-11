import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import segmentation_models_pytorch as smp
import cv2
import albumentations as A
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cudnn.allow_tf32 = True

class newDataset(Dataset):
    def __init__(self, image_files, mask_files, input_size=(512,512), transform=None):
        self.image_files=image_files
        self.mask_files=mask_files
        self.input_size=input_size
        self.transform=transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path=self.image_files[idx]
        mask_path=self.mask_files[idx]  

        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)  

        image = self.image_histogram_equalization(image.astype('float32')/image.max())
        mask = mask.astype('float32')/255.

        image=torch.tensor(image)
        mask=torch.tensor(mask)

        if self.transform:
            image,mask=self.augment_image(image,mask)
        
        return image, mask
    
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

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.epochs_no_improve = 0
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
    

def train(train_dataset, val_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = smp.UnetPlusPlus(
        encoder_name='timm-resnest101e',
        encoder_weights='imagenet',
        decoder_attention_type="scse",
        in_channels=1
    ).to(device)

    model = torch.nn.DataParallel(model).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

    """
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    prev_epoch = checkpoint['epoch']+1
    val_loss = checkpoint['val_loss']
    """

    train_dataloader= DataLoader(train_dataset,batch_size=64,shuffle=True, pin_memory=True, num_workers=48)
    val_dataloader = DataLoader(val_dataset,batch_size=64,shuffle=False, pin_memory=True, num_workers=48)

    num_epochs = 500

    early_stopping = EarlyStopping(patience=75, min_delta=0)
    best_loss = float('inf')
    all_train_losses = []
    epoch_losses = []
    val_losses = []

    scaler = GradScaler()
    #scaler.load_state_dict(checkpoint['scaler'])

    for epoch in range(num_epochs):
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
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_num+1}/{len(train_dataloader)}], Batch Loss: {train_loss:.8f}')


        avg_train_loss = sum(epoch_train_losses) / len(epoch_train_losses)
        epoch_losses.append(avg_train_loss)

        model.eval()
        running_loss = 0.0
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
            
        
        if avg_val_loss < best_loss: 
            print(f"Validation Loss Improved ({best_loss} --> {avg_val_loss})")
            best_loss = avg_val_loss 

            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'val_loss' : avg_val_loss,
                'scaler' : scaler.state_dict()
            }, "best_model.pth")
            print("Model saved as best_model.pth")

        early_stopping(avg_val_loss)
        if early_stopping.should_stop:
            print("Early stopping triggered")
            break

    print(f'DONE')


def run():
    base_path = "/"
    datasets = ["kidney_1_dense"]#, "kidney_1_voi", "kidney_2", "kidney_3_sparse"]

    image_files = []
    labels_files = []
    for dataset in datasets:
        images_path = os.path.join(base_path,dataset,"images")
        label_path = os.path.join(base_path,dataset,"labels")

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
    
    train(train_dataset, val_dataset)


if __name__ == "__main__":
    print("Start")
    run()