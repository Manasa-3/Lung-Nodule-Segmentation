import glob
import os
import SimpleITK as sitk
import nibabel as nib
import numpy as np                                                                                                      
from concurrent.futures import ThreadPoolExecutor                                                                       
import gzip
import numpy as np                                                                                                      
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.checkpoint import checkpoint
from torchvision.transforms import Resize
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import Lambda, Compose
from scipy.ndimage import zoom
import psutil

import os

"""#Preprocessing and saving the original files as nii.gz"""

# # Directory containing the .mhd files
# input_dir = '/content/drive/MyDrive/seg_lung_luna'
# output_dir = '/content/drive/MyDrive/Dataset'
# # Reference size and spacing for all images
# reference_size = [256, 256, 256]  # Example reference size
# reference_spacing = [1.0, 1.0, 1.0]  # Example reference spacing

# # List all .mhd files in the directory
# mhd_files = glob.glob(os.path.join(input_dir, '*.mhd'))

# def process_mhd_file(mhd_file):
#     try:
#         # Read the .mhd file
#         img = sitk.ReadImage(mhd_file)
#         # print(f"Processing {mhd_file}"
# #         # Configure the resampler to use the new transformation
#         resampler = sitk.ResampleImageFilter()
#         resampler.SetSize(reference_size)                                                                             
#         resampler.SetOutputSpacing(reference_spacing)
#         resampler.SetOutputOrigin(img.GetOrigin())                                                                    
#         resampler.SetOutputDirection(img.GetDirection())
#         resampler.SetInterpolator(sitk.sitkLinear)

#         # Execute the resampling
#         resampled_img = resampler.Execute(img)

#         # Convert the resampled image to a NumPy array
#         volume = sitk.GetArrayFromImage(resampled_img)
#         # Normalize the volume based on the given HU values
#         min_hu = -1000
#         max_hu = 400
#         volume_normalized = (volume - min_hu) / (max_hu - min_hu)

#         # Prepare the affine matrix
#         affine_matrix = np.eye(4)  # Identity matrix for 4x4

#         # Save the preprocessed volume as a compressed .nii.gz file
#         output_filename = os.path.splitext(os.path.basename(mhd_file))[0]
#         output_path = os.path.join(output_dir, output_filename)

#         nii_img = nib.Nifti1Image(volume_normalized, affine_matrix)
#         nib.save(nii_img, output_path)  # Specify compress=True for gzip $#         with open(output_path, 'rb') as f_in:
#           with gzip.open(output_path + '.gz', 'wb') as f_out:
#               f_out.writelines(f_in)

#         print(f"Processed {mhd_file} and saved as {output_path}")


#     except Exception as e:
#         print(f"Error processing {mhd_file}: {e}")

# # Use ThreadPoolExecutor for parallel processing
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(process_mhd_file, mhd_files)

"""Creating the custom dataset"""

import os
import torch
import torchvision.transforms as transforms
import nibabel as nib
from torch.utils.data import Dataset

class CTMaskedDataset(Dataset):
    def __init__(self, root_dir, transform=None, target_size=(128, 128, 30)):  # Adjusted target size to 30
        self.root_dir = root_dir
        self.target_size = target_size
        self.scan_folders = [f for f in os.listdir(self.root_dir) if f.endswith('_extracted.nii.gz') or f.endswith('_masked_extracted.nii.gz')]
        print(f"Found {len(self.scan_folders)} NIFTI files in {self.root_dir}")
        self.transform = transform

    def __len__(self):
        valid_count = 0
        for nifti_file in self.scan_folders:
            if '_masked_extracted.nii.gz' in nifti_file:
                valid_count += 1
        print("Valid count:", valid_count)
        return valid_count

    def __getitem__(self, idx):
        scan_path = self.root_dir
        all_files = os.listdir(scan_path)
        nii_files = [f for f in all_files if f.endswith('_extracted.nii.gz')]
        masked_files = [f for f in all_files if f.endswith('_masked_extracted.nii.gz')]
        file_pairs = list(zip(nii_files, masked_files))
        original_file, masked_file = file_pairs[idx]
        original_file_path = os.path.join(scan_path, original_file)
        masked_file_path = os.path.join(scan_path, masked_file)
        if not os.path.isfile(original_file_path) or not os.path.isfile(masked_file_path):
            print(f"Missing file for {original_file}, returning zeros for missing data.")
            image = torch.zeros(1, 128, 128, 30)  # Adjust dimensions as needed
            mask = torch.zeros(1, 128, 128, 30)  # Adjust dimensions as needed

        else:
            image_slices = self.load_nifti_slices(original_file_path)[:30]  # Select top 30 slices
            mask_slices = self.load_nifti_slices(masked_file_path)[:30]  # Select top 30 slices

            # Stack and unsqueeze the slices to form a batch
            image = torch.stack(image_slices, dim=0).unsqueeze(0)
            mask = torch.stack(mask_slices, dim=0).unsqueeze(0)

            # Resize both image and mask to a fixed size, keeping the third dimension intact
            resize = transforms.Resize((128, 128), interpolation=transforms.InterpolationMode.NEAREST)

            # Apply padding to the mask along the channel dimension
            mask_padded = torch.nn.functional.pad(mask, (0, 0, 0, 30 - mask.shape[2]), mode='constant', value=0)

            if self.transform:
                image = self.transform(image)
                mask_padded = self.transform(mask_padded)

        return image, mask_padded
    
    def load_nifti_slices(self, nifti_file, num_slices=30):
      nifti_data = nib.load(nifti_file)
      nifti_array = np.array(nifti_data.get_fdata())
      slices = [torch.tensor(nifti_array[:, :, i]) for i in range(nifti_array.shape[2])]
      return slices[:num_slices]


# Define your custom collate function outside the class definition
def custom_collate_fn(batch):
    images = []
    masks = []
    for item in batch:
        image, mask = item
        images.append(image)
        masks.append(mask)
    images = torch.stack(images, dim=0)
    masks = torch.stack(masks, dim=0)
    return images, masks

"""#Res2Net definition"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Res2NetBlock3D(nn.Module):
    def __init__(self, in_channels, channels, reduction=16, scale=4):
        super(Res2NetBlock3D, self).__init__()
        self.conv_shortcut = nn.Conv3d(in_channels, channels, kernel_size=1, bias=False)
        self.bn_shortcut = nn.BatchNorm3d(channels)

        self.conv_split = nn.Conv3d(in_channels, channels // reduction, kernel_size=1, bias=False)
        self.bn_split = nn.BatchNorm3d(channels // reduction)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv_reduce = nn.Conv3d(channels // reduction, channels // scale, kernel_size=3, padding=1, bias=False)
        self.bn_reduce = nn.BatchNorm3d(channels // scale)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv_expand = nn.Conv3d(channels // scale, channels, kernel_size=1, bias=False)
        self.bn_expand = nn.BatchNorm3d(channels)

    def forward(self, x):
        shortcut = self.bn_shortcut(self.conv_shortcut(x))

        split = self.bn_split(self.conv_split(x))
        reduce = self.bn_reduce(self.conv_reduce(self.relu1(split)))
        expand = self.bn_expand(self.conv_expand(self.relu2(reduce)))

        return F.relu(shortcut + expand)


class ModifiedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super(ModifiedUNet, self).__init__()
        # Encoder
        self.encoder1 = self.conv_block(in_channels, base_channels)
        self.encoder2 = Res2NetBlock3D(base_channels, base_channels * 2)
        self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)

        self.reduce_channels = nn.Conv3d(base_channels * 4, base_channels * 2, kernel_size=1)

        self.increase_channels = nn.Conv3d(base_channels * 2, base_channels * 3, kernel_size=1)
        # self.increase_channels2 = nn.Conv3d(base_channels * 2, base_channels * 3, kernel_size=1)

        # Decoder (with upsampling and concatenation)
        self.decoder2 = self.conv_block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose3d(base_channels * 2, base_channels * 2, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(base_channels * 4, base_channels)
        self.up2 = nn.ConvTranspose3d(base_channels , base_channels, kernel_size=2, stride=2)

        self.final_conv = nn.Conv3d(base_channels , out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)


        # Decoder
        x3_up = F.interpolate(x3, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x3_up = self.reduce_channels(x3_up)
        x2_add = torch.cat([x2, x3_up], dim=1)
        x2_dec = self.decoder2(x2_add)
        x2_up = F.interpolate(x2_dec, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x2_up = self.increase_channels(x2_up)
        x1_add = torch.cat([x1, x2_up], dim=1)
        x1_dec = self.decoder1(x1_add)
        # Final Convolution
        x_final = self.final_conv(x1_dec)
        return x_final

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        intersection = (y_pred * y_true).sum(dim=1)
        denominator = (y_pred.sum(dim=1) + y_true.sum(dim=1) + self.smooth)
        dice_loss = 1 - (2 * intersection + self.smooth) / denominator
        return dice_loss.mean()

import torch
from torch.utils.data import Dataset, DataLoader

"""Training loop"""

import os
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.nn import functional as F
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Compose, ToTensor, Resize
from torch.optim import AdamW
import psutil
from torch.cuda.amp import autocast, GradScaler
import gc
from tqdm import tqdm

# Set device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_np(arr, size):
    scale_factor = min(size[0] / arr.shape[0], size[1] / arr.shape[1])
    resized_arr = np.empty((size[0], size[1], arr.shape[-1]))
    for ch in range(arr.shape[-1]):
        resized_arr[:, :, ch] = zoom(arr[:, :, ch], (scale_factor, scale_factor, scale_factor), order=1)
    return resized_arr

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Lambda(lambda x: x.float())  # Convert to float32
])

# Define your dataset path
root_dir = 'ct_scan'
# Initialize your dataset
dataset = CTMaskedDataset(root_dir=root_dir, transform=transform)

# Split the dataset into training, testing, and validation sets
train_size = int(0.7 * len(dataset))
test_size = int(0.15 * len(dataset))
val_size = len(dataset) - train_size - test_size

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

# Create separate DataLoaders for each phase
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=custom_collate_fn, pin_memory=True)
# Initialize Model, Loss Function, and Optimizer
model = ModifiedUNet(in_channels=1, out_channels=1).to(device)

criterion = DiceLoss()
optimizer = AdamW(model.parameters(), lr=0.001)

num_epochs = 20  # Define the number of epochs

print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated()} bytes")

def dice_score(prediction, ground_truth):
    intersection = torch.sum(prediction * ground_truth)
    total = torch.sum(prediction) + torch.sum(ground_truth)

    # Handle division by zero (in case of empty sets)
    if total > 0:
        dice = (2.0 * intersection) / total
    else:
        dice = 1.0  # Perfect score for empty sets

    return dice


# Training loop
scaler = GradScaler()
for epoch in range(num_epochs):
    print("Epoch: ",epoch)
    model.train()  # Set the model to training mode
    running_loss = 0.0
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training", leave=False)

    train_dice = 0.0

    for batch_idx, (images, masks) in enumerate(train_loader_tqdm):
        images, masks = images.to(device).float(), masks.to(device).float()  # Ensure data is float32

        optimizer.zero_grad()
        with autocast():

            outputs = model(images)  # Autocast will handle conversion to FP16 where appropriate
            loss = criterion(outputs, masks)
            train_dice += dice_score(outputs, masks)


        scaler.scale(loss).backward()

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print("Training dice score:",train_dice, flush=True)
    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}", flush = True)
     # Evaluate on test and validation sets
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        test_loss = 0.0
        val_loss = 0.0
        test_dice = 0.0
        val_dice = 0.0
        test_loader_tqdm = tqdm(test_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Testing", leave=False)

        for images, masks in test_loader_tqdm:
            images, masks = images.to(device).float(), masks.to(device).float()  # Ensure data is float32
            with autocast():
                outputs = model(images)
                test_loss += criterion(outputs, masks).item()
                test_dice += dice_score(outputs, masks)

        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation", leave=False)
        for images, masks in val_loader_tqdm:
            images, masks = images.to(device).float(), masks.to(device).float()  # Ensure data is float32
            with autocast():
                outputs = model(images)
                val_loss += criterion(outputs, masks).item()
                val_dice += dice_score(outputs, masks)

    print(f"Test Loss: {test_loss / len(test_loader):.4f}, Validation Loss: {val_loss / len(val_loader):.4f}", flush=True)
    print(f"Test Dice Score: {test_dice / len(test_loader):.4f}, Validation Dice Score: {val_dice / len(val_loader):.4f}", flush=True)
# Save the model after each epoch
base_dir='home/nitk222is006/Lung_Segmentation/model'
model_name = f"modified_unet_resnet.pth"
model_path = os.path.join(base_dir, model_name)
torch.save(model.state_dict(), model_path)
print(f"Model saved at {model_path}")