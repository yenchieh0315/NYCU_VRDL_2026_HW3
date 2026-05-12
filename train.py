import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

import random  
import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from tqdm import tqdm
import torch.nn as nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import torchvision.models.detection.roi_heads as roi_heads
original_maskrcnn_loss = roi_heads.maskrcnn_loss

DICE_WEIGHT = 0.25  

def custom_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs):
    bce_loss = original_maskrcnn_loss(mask_logits, proposals, gt_masks, gt_labels, mask_matched_idxs)
    
    discretization_size = mask_logits.shape[-1]
    labels =[gt_label[idxs] for gt_label, idxs in zip(gt_labels, mask_matched_idxs)]
    mask_targets =[
        roi_heads.project_masks_on_boxes(m, p, i, discretization_size)
        for m, p, i in zip(gt_masks, proposals, mask_matched_idxs)
    ]
    labels = torch.cat(labels, dim=0)
    mask_targets = torch.cat(mask_targets, dim=0)
    pos_inds = labels > 0
    if pos_inds.sum() == 0: return bce_loss 

    mask_logits = mask_logits[pos_inds, labels[pos_inds]]
    mask_targets = mask_targets[pos_inds]
    inputs = torch.sigmoid(mask_logits).flatten(1)
    targets = mask_targets.flatten(1).float()
    
    intersection = (inputs * targets).sum(1)
    dice = (2. * intersection + 1.0) / (inputs.sum(1) + targets.sum(1) + 1.0)
    dice_loss = 1.0 - dice.mean()
    return bce_loss + (DICE_WEIGHT * dice_loss)

roi_heads.maskrcnn_loss = custom_maskrcnn_loss

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x

class CBAMBottleneck(nn.Module):
    def __init__(self, original_bottleneck):
        super().__init__()
        self.conv1 = original_bottleneck.conv1
        self.bn1 = original_bottleneck.bn1
        self.conv2 = original_bottleneck.conv2
        self.bn2 = original_bottleneck.bn2
        self.conv3 = original_bottleneck.conv3
        self.bn3 = original_bottleneck.bn3
        self.relu = original_bottleneck.relu
        self.downsample = original_bottleneck.downsample
        self.stride = original_bottleneck.stride
        self.cbam = CBAM(self.conv3.out_channels)
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.cbam(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

def inject_cbam_into_model(module):
    for name, child in module.named_children():
        if isinstance(child, torchvision.models.resnet.Bottleneck):
            setattr(module, name, CBAMBottleneck(child))
        else:
            inject_cbam_into_model(child) 

class CellDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dirs =[d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.color_jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05)
        
    def __len__(self):
        return len(self.image_dirs)

    def __getitem__(self, idx):
        img_dir_name = self.image_dirs[idx]
        img_dir_path = os.path.join(self.root_dir, img_dir_name)
        
        k_rot = random.choice([0, 1, 2, 3])
        h_flip = random.random() > 0.5
        v_flip = random.random() > 0.5
        
        img_path = os.path.join(img_dir_path, "image.tif")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        if k_rot > 0: image = np.rot90(image, k_rot, axes=(0, 1))
        if h_flip: image = cv2.flip(image, 1)  
        if v_flip: image = cv2.flip(image, 0)  
        
        image = np.ascontiguousarray(image)
        image = torchvision.transforms.functional.to_tensor(image) 

        if random.random() > 0.5:
            image = self.color_jitter(image)
        if random.random() > 0.5:
            noise = torch.randn_like(image) * 0.02
            image = torch.clamp(image + noise, 0.0, 1.0)

        boxes = []
        labels =[]
        masks =[]
        
        for class_id in range(1, 5):
            mask_path = os.path.join(img_dir_path, f"class{class_id}.tif")
            if not os.path.exists(mask_path): continue
                
            mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
            if mask_img is None: continue
            
            if k_rot > 0: mask_img = np.rot90(mask_img, k_rot, axes=(0, 1))
            if h_flip: mask_img = cv2.flip(mask_img, 1)
            if v_flip: mask_img = cv2.flip(mask_img, 0)
                
            mask_img = np.ascontiguousarray(mask_img)
            instance_ids = np.unique(mask_img)
            for inst_id in instance_ids:
                if inst_id == 0: continue
                
                binary_mask = (mask_img == inst_id).astype(np.uint8)
                pos = np.where(binary_mask)
                xmin, xmax = np.min(pos[1]), np.max(pos[1])
                ymin, ymax = np.min(pos[0]), np.max(pos[0])
                
                if xmax > xmin and ymax > ymin:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(class_id)
                    masks.append(binary_mask)

        target = {}
        if len(boxes) == 0:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            target["masks"] = torch.zeros((0, image.shape[1], image.shape[2]), dtype=torch.uint8)
        else:
            target["boxes"] = torch.as_tensor(boxes, dtype=torch.float32)
            target["labels"] = torch.as_tensor(labels, dtype=torch.int64)
            target["masks"] = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        
        return image, target

def collate_fn(batch): return tuple(zip(*batch))

def get_model(num_classes, is_train=True):
    weights = 'DEFAULT' if is_train else None
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=weights)
    
    inject_cbam_into_model(backbone.body.layer4)
    
    model = MaskRCNN(backbone, 
                     num_classes=num_classes,
                     min_size=800, max_size=1024)
    return model

def main():
    DATA_DIR = "./data/train"
    WEIGHTS_DIR = "./weights"
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    num_classes = 5  
    batch_size = 4                       
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    epochs = 60
    warmup_epochs = 5
    target_lr = 5e-5

    dataset = CellDataset(DATA_DIR)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, 
                            num_workers=8, collate_fn=collate_fn, pin_memory=True, 
                            persistent_workers=True, prefetch_factor=2)

    model = get_model(num_classes)
    model.to(device)

    params =[p for p in model.parameters() if p.requires_grad]
    
    optimizer = torch.optim.AdamW(params, lr=target_lr, weight_decay=1e-4)

    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=1e-6 / target_lr, 
        end_factor=1.0, 
        total_iters=warmup_epochs
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=(epochs - warmup_epochs), 
        eta_min=1e-6
    )

    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}[LR: {scheduler.get_last_lr()[0]:.1e}]")
        for images, targets in pbar:

            images = list(image.to(device) for image in images)
            targets =[{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            pbar.set_postfix({"Loss": f"{losses.item():.4f}"})
        
        scheduler.step()
        print(f"Epoch {epoch} average Loss: {epoch_loss / len(dataloader):.4f}")

        if epoch % 10 == 0 or epoch == epochs:
            save_path = os.path.join(WEIGHTS_DIR, f"ResNet_epoch{epoch}.pth")
            torch.save(model.state_dict(), save_path)

if __name__ == "__main__":
    main()