
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
import os
import os.path as osp
from pycocotools.coco import COCO
from PIL import Image
import json

class CocoDataset(CocoDetection):
    """ If num_classes or reprLabel should be used, the Labels should be set first.

    Arguments:       
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """
    Labels = []

    @staticmethod
    def num_classes():
        return len(CocoDataset.Labels)

    @staticmethod
    def reprLabel(idx):
        return CocoDataset.Labels[idx]

    def __init__(self, root, annFile = None, \
            # whether to apply mask , keypoint
            mask=True, keypoint=True, \
            transform=None, target_transform=None, transforms=None, \
            train=False, debug = False, num_keypoints=2, coco_style_kp=False,
            **kwargs):
        

        if annFile is None:
            annFile = osp.join(root, "annotations.json")
        # print(root, annFile)
        super().__init__(osp.join(root, "JPEGImages"), annFile, transform=transform, \
            target_transform=target_transform, transforms=transforms)
        self.root = str(root)
        self.train = train
        self.num_keypoints = num_keypoints
        self.coco_style_kp = coco_style_kp
        self._mask = mask
        self._keypoint = keypoint
        self.debug = debug
        print("[DS] Dataset root: ", self.root, "Len: ", len(self.ids))
        print("[DS] Mask: ", self._mask, " , Keypoint: ", self._keypoint)
    
    def load_annotations(self, ann_file, catNms):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        # self.cat_ids = self.coco.getCatIds(cat_names=self.CLASSES)
        self.cat_ids = self.coco.getCatIds(
            catNms=catNms
            )
        cats = self.coco.loadCats(self.cat_ids)
        nms=[cat['name'] for cat in cats]
        print('COCO categories: {}\n'.format(' '.join(nms)))

        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.getImgIds(catIds=self.cat_ids)
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            # info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos 

    def get_filename(self, index):
        with open(osp.join(self.root, "annotations.json"), "r") as f:
            content = json.load(f)["images"]
        for image in content:
            if image["id"] == index:
                return image["file_name"]
        return None

    def get_idx_by_name(self, name):
        with open(osp.join(self.root, "annotations.json"), "r") as f:
            content = json.load(f)["images"]
        idxes = []
        for image in content:
            file_name = image["file_name"]
            if name in file_name:
                idxes.append((image["id"], file_name))
        return idxes

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        # print("DS: ", ann_ids, target)

        boxes =  []
        labels =  []
        areas =  []
        iscrowds =  []
        for t in target:
            label = t["category_id"]
            labels.append(label)
            iscrowds.append(0)
            x,y,w,h=t["bbox"]
            boxes.append([x,y,x+w,y+h])
            areas.append(t["area"]) # coco mask area is slightly different from torch
            # areas.append(w*h)  
        t_target = {
            "image_id": torch.tensor([index]),
            "boxes": torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowds, dtype=torch.uint8),
        }
        
        if self._mask:       
            masks =  []
            for t in target:
                label = t["category_id"]
                masks.append(coco.annToMask(t))
                
            t_target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
            # end mask
        if self._keypoint:
            # N K 3
            keypoints = [] 
            # print("Load Anns: ", target, len(target))
            for t in target:
                kps = []
                # print(t, "========", "keypoints" in t)
                if "keypoints" not in t:
                    ## TODO detect how many keypoints there has
                    for _ in range(self.num_keypoints):
                        kps.append([0, 0, 0])
                    keypoints.append(kps)
                    continue
                kp = t["keypoints"]
                xs = kp[0::3]
                ys = kp[1::3]
                vs = kp[2::3]
                if not self.coco_style_kp :
                    for x, y, v in zip(xs, ys, vs):
                        if v >= 1:
                            kps.append([x, y, v-1])
                        else:
                            kps.append([x, y, 0])
                else:
                    for x, y, v in zip(xs, ys, vs):
                        kps.append([x, y, v])
                keypoints.append(kps)
            # print(keypoints)
            t_target["keypoints"] = torch.as_tensor(keypoints, dtype=torch.float32)
            # end keypoint
        path = coco.loadImgs(img_id)[0]['file_name']
        
        img = Image.open(osp.join(self.root, path)).convert('RGB')
        if self.transforms is not None:
            # print(t_target)
            assert t_target is not None
            img, t_target = self.transforms(img, t_target)
        # print(t_target)
        return img, t_target

import numpy as np

class MaskDataset(Dataset):
    """Only for mask prediction and voc-style dataset
    """

    Labels = None
    @staticmethod
    def num_classes():
        return len(MaskDataset.Labels)

    @staticmethod
    def reprLabel(idx):
        """idx from 1,
        0 represets background
        """
        return MaskDataset.Labels[idx]

    def __init__(self, root, transforms=None, train=True, **kwargs):
        super().__init__()
        self.root = str(root)
        self.name = kwargs.get("name")
        self.transforms = transforms
        self.train = train
        if self.train:
            imgs = []
            for img in sorted(os.listdir(osp.join(root, "JPEGImages"))):
                if (not img.endswith(".json") and not osp.isdir(osp.join(root,img))):
                    imgs.append(img)
            self.imgs = imgs
            imgs = []
            for img in sorted(os.listdir(osp.join(root, "SegmentationClassPNG"))):
                if (not img.endswith(".json") and not osp.isdir(osp.join(root,img))):
                    imgs.append(img)
            self.masks = imgs
        else:
            imgs = []
            for img in sorted(os.listdir(root)):
                if (not img.endswith(".json") and not osp.isdir(osp.join(root,img))):
                    imgs.append(img)
            self.imgs = imgs

        print("TrainDataset Size: ", len(self.imgs))
        # print("Dataset prepared")

    def getMask(self, idx):
        mask_path = osp.join(self.root, "SegmentationClassPNG", self.masks[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask)
        return mask

    def getTarget(self, idx, mask, to_tensor=True):
        """All values are numpy datatype, Not converted to Tensor
        """
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]
        masks = (mask == obj_ids[:, None, None])
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        labels = []
        iscrowd = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            label = np.max(mask[masks[i]])
            # print(obj_ids, label)
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)
            iscrowd.append(0)
        # print(boxes)

        target = {}

        # target["maskimg"] = mask
        if to_tensor:
            target["image_id"] = torch.tensor([idx])
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            target["boxes"] = boxes
            # two class currently
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
            target["masks"] = torch.as_tensor(masks, dtype=torch.uint8)
            # target["area"] = torch.tensor(target["area"], dtype=torch.uint8)
            target["area"] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            target["iscrowd"] = torch.as_tensor(iscrowd, dtype=torch.int64)
        else:
            target["boxes"] = boxes
            target["labels"] = labels
            target["masks"] = masks
            target["image_id"] = [idx]
            target["iscrowd"] = iscrowd

        return target

    def __getitem__(self, idx):
        if not self.train:
            img_path = osp.join(self.root, self.imgs[idx])   
            img = Image.open(img_path)

            img = np.array(img)
            # img = torch.Tensor(img)
            if self.transforms is not None:
                img, _ = self.transforms(img)
            else:
                img = FT.to_tensor(img)
            return img, None

        mask_path = osp.join(self.root, "SegmentationClassPNG", self.masks[idx])
        mask = Image.open(mask_path)
        mask = np.array(mask)
        target = self.getTarget(idx, mask)

        img_path = osp.join(self.root, "JPEGImages", self.imgs[idx])
        img = Image.open(img_path).convert('L')
        img = np.array(img)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = FT.to_tensor(img)

        return img, target


    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        return f'<DS {self.name}>'

