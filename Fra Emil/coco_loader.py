import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask
from PIL import Image
import pickle
from tqdm import trange
import os
from torch.utils.data import Dataset
from typing import Tuple, List, Dict, Any, Optional, Callable


class SegmentationDataset:
    scale = 0.2

    def _sync_transform(self, img, mask):
        width, height = img.size
        scaled_width = int(self.scale * width)
        scaled_height = int(self.scale * height)

        img = img.resize((scaled_width, scaled_height), resample=Image.BICUBIC)
        mask = mask.resize((scaled_width, scaled_height), resample=Image.NEAREST)

        img, mask = self._img_transform(img), self._mask_transform(mask)

        return img, mask

    @staticmethod
    def _img_transform(img):
        return np.array(img).astype('float32')

    @staticmethod
    def _mask_transform(mask):
        return np.array(mask).astype('int32')


class COCOSegmentation(Dataset, SegmentationDataset):
    CAT_LIST = [-1, 0]

    def __init__(self, root='C:\\Users\\evr\\Eltronic Group A S\\Project SMART - General\\01 Design\\01 Software\\00 Datasets\\DemoData\\Camera_Annotated', transform: Optional[Callable] = None):
        self.root = root
        label_file = os.path.join(root, 'result.json')
        ids_file = os.path.join(root, 'ids.mx')
        self.coco = COCO(label_file)
        # Check if there exists an id_file. If not,m then we create one.
        if os.path.exists(ids_file):
            with open(ids_file, 'rb') as f:
                self.ids = pickle.load(f)
        else:
            ids = list(self.coco.imgs.keys())
            self.ids = self._preprocess(ids, ids_file)
        self.transform = transform

    def __getitem__(self, index):
        img_id = self.ids[index]
        img_metadata = self.coco.loadImgs(img_id)[0]
        path = img_metadata['file_name'].replace("\\", "/")

        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
        mask = Image.fromarray(self._gen_seg_mask(coco_target, img_metadata['height'], img_metadata['width']))

        if self.transform is not None:
            img, mask = np.array(img).astype('float32'), np.array(mask).astype('int32')
            res = self.transform(image=img, mask=mask)
            img, mask = res['image'], res['mask']
        else:
            img, mask = self._sync_transform(img, mask)
            img = img.transpose((2, 0, 1))

        return img, mask

    def _preprocess(self, ids, ids_file) -> List[int]:
        """ This function takes a list of ids and creates a file that contains the id for all images where the mask
        has more than 1k labelled pixels.
        """
        print("Preprocessing masks, this will take a while." + \
              "Will only run once for each split.")
        tbar = trange(len(ids))
        new_ids = []
        for i in tbar:
            img_id = ids[i]
            coco_target = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))
            img_metadata = self.coco.loadImgs(img_id)[0]
            mask = self._gen_seg_mask(coco_target, img_metadata['height'], img_metadata['width'])

            # if more than 1k pixels are labelled then accept.
            if (mask > 0).sum() > 1000:
                new_ids.append(img_id)

            tbar.set_description('Progress: {}/{}, got {} qualified images'. \
                                 format(i, len(ids), len(new_ids)))

        print('Found number of qualified images: ', len(new_ids))
        with open(ids_file, 'wb') as f:
            pickle.dump(new_ids, f)
        return new_ids

    def _gen_seg_mask(self, target, h, w):
        mask = np.zeros((h, w), dtype=np.uint8)

        for instance in target:
            rle = coco_mask.frPyObjects(instance['segmentation'], h, w)
            m = coco_mask.decode(rle)
            cat = instance['category_id']
            if cat in self.CAT_LIST:
                c = self.CAT_LIST.index(cat)
            else:
                continue

            # Append the label
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * ((np.sum(m, axis=2) > 0) * c).astype(np.uint8)
        return mask

    @property
    def classes(self):
        return ('background', 'lane-markings')

    def __len__(self):
        return len(self.ids)
