import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL" 

import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
from pycocotools import mask as mask_util
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import MaskRCNN
from tqdm import tqdm

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(x)))) + self.fc2(self.relu1(self.fc1(self.max_pool(x)))))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=3 if kernel_size == 7 else 1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return self.sigmoid(self.conv1(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1)))

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x) * x) * (self.ca(x) * x)

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
        out = self.cbam(self.bn3(self.conv3(self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x)))))))))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)

def inject_cbam_into_model(module):
    for name, child in module.named_children():
        if isinstance(child, torchvision.models.resnet.Bottleneck):
            setattr(module, name, CBAMBottleneck(child))
        else:
            inject_cbam_into_model(child) 

def get_model(num_classes, is_train=False):
    weights = 'DEFAULT' if is_train else None
    backbone = resnet_fpn_backbone(backbone_name='resnet101', weights=weights)
    inject_cbam_into_model(backbone.body.layer4) 
    model = MaskRCNN(backbone, num_classes=num_classes, min_size=800, max_size=1024)
    return model

def main():
    TEST_DIR = "./data/test_release"
    JSON_MAP_PATH = "./data/test_image_name_to_ids.json"
    
    WEIGHTS_PATH = "./weights/ResNet_epoch60.pth" 
    OUTPUT_JSON = "./test-results.json"
    
    num_classes = 5
    score_threshold = 0.05
    nms_iou_threshold = 0.5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(JSON_MAP_PATH, "r") as f:
        mapping_dict = {item['file_name']: item['id'] for item in json.load(f)}

    model = get_model(num_classes, is_train=False)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    results =[]

    with torch.no_grad():
        test_files = [f for f in os.listdir(TEST_DIR) if f.endswith(".tif")]
        
        for file_name in tqdm(test_files, desc="Inference progress:"):
            if file_name not in mapping_dict: continue
                
            image_id = mapping_dict[file_name]
            image = cv2.cvtColor(cv2.imread(os.path.join(TEST_DIR, file_name)), cv2.COLOR_BGR2RGB)
            H, W, _ = image.shape
            
            img_tensor = TF.to_tensor(image).to(device)
            all_boxes, all_scores, all_labels, all_masks = [], [], [], []

            def get_predictions(tensor, transform_type):
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = model(tensor.unsqueeze(0))[0]
                
                scores = pred['scores'].float()
                keep = scores >= score_threshold
                if keep.sum() == 0:
                    return None
                
                b = pred['boxes'][keep].float()
                s = scores[keep]
                l = pred['labels'][keep]
                m = pred['masks'][keep].float()
                
                if transform_type == 'orig':
                    pass
                elif transform_type == 'hflip':
                    b_new = b.clone()
                    b_new[:, [0, 2]] = W - b[:, [2, 0]]
                    b = b_new
                    m = m.flip(-1)
                elif transform_type == 'vflip':
                    b_new = b.clone()
                    b_new[:, [1, 3]] = H - b[:, [3, 1]]
                    b = b_new
                    m = m.flip(-2)
                elif transform_type == 'rot90':
                    b_new = b.clone()
                    b_new[:, 0], b_new[:, 1] = W - b[:, 3], b[:, 0]
                    b_new[:, 2], b_new[:, 3] = W - b[:, 1], b[:, 2]
                    b = b_new
                    m = torch.rot90(m, -1, [-2, -1])
                elif transform_type == 'rot180':
                    b_new = b.clone()
                    b_new[:, 0], b_new[:, 1] = W - b[:, 2], H - b[:, 3]
                    b_new[:, 2], b_new[:, 3] = W - b[:, 0], H - b[:, 1]
                    b = b_new
                    m = torch.rot90(m, -2, [-2, -1])
                elif transform_type == 'rot270':
                    b_new = b.clone()
                    b_new[:, 0], b_new[:, 1] = b[:, 1], H - b[:, 2]
                    b_new[:, 2], b_new[:, 3] = b[:, 3], H - b[:, 0]
                    b = b_new
                    m = torch.rot90(m, -3,[-2, -1])
                    
                return b.cpu(), s.cpu(), l.cpu(), m.cpu()

            views =[
                (img_tensor, 'orig'),
                (img_tensor.flip(-1), 'hflip'),
                (img_tensor.flip(-2), 'vflip'),
                (torch.rot90(img_tensor, 1, [-2, -1]), 'rot90'),
                (torch.rot90(img_tensor, 2, [-2, -1]), 'rot180'),
                (torch.rot90(img_tensor, 3,[-2, -1]), 'rot270')
            ]
            
            for tensor, t_type in views:
                res = get_predictions(tensor, t_type)
                if res is not None:
                    all_boxes.append(res[0])
                    all_scores.append(res[1])
                    all_labels.append(res[2])
                    all_masks.append(res[3])
                torch.cuda.empty_cache()

            if len(all_boxes) == 0:
                continue

            cat_boxes = torch.cat(all_boxes, dim=0)
            cat_scores = torch.cat(all_scores, dim=0)
            cat_labels = torch.cat(all_labels, dim=0)
            cat_masks = torch.cat(all_masks, dim=0)

            nms_keep = torchvision.ops.batched_nms(cat_boxes, cat_scores, cat_labels, nms_iou_threshold)
            final_boxes = cat_boxes[nms_keep].numpy()
            final_scores = cat_scores[nms_keep].numpy()
            final_labels = cat_labels[nms_keep].numpy()
            final_masks = cat_masks[nms_keep].numpy()

            for i in range(len(final_scores)):
                xmin, ymin, xmax, ymax = final_boxes[i]
                coco_bbox =[float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]
                
                fortran_mask = np.asfortranarray((final_masks[i, 0] >= 0.5).astype(np.uint8))
                rle = mask_util.encode(fortran_mask)
                rle['counts'] = rle['counts'].decode('utf-8')
                
                results.append({
                    "image_id": int(image_id), "bbox": coco_bbox,
                    "score": float(final_scores[i]), "category_id": int(final_labels[i]),
                    "segmentation": {"size": [int(H), int(W)], "counts": rle['counts']}
                })

    print(f"Inference complete, finding {len(results)} cell.")
    with open(OUTPUT_JSON, "w") as f: json.dump(results, f)

if __name__ == "__main__":
    main()