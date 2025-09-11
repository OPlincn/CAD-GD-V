import os
import torch
from torch.utils.data import Dataset, DataLoader
from groundingdino.util.base_api import preprocess_caption
from groundingdino.util.img_read import load_image
from utils.processor import DataProcessor
import io


def collate_fn(batch):
    """Custom collate_fn that also pads and stacks density maps.

    Each item in ``batch`` contains the tensorized image and a tensor of
    density maps (one per caption).  Images in a batch may have different
    spatial sizes due to random resizing, so we pad both the images and the
    corresponding density maps to the maximum height and width before
    stacking. Density maps are then concatenated along the caption dimension
    so that the returned tensor has a shape of
    ``(sum_i num_caps_i, H_max, W_max)`` aligning with the flattened caption
    list used during training.
    """

    images, labels, shapes, img_caps, density_maps = zip(*batch)

    # Determine max height and width to pad images and density maps
    max_height = max(img.shape[1] for img in images)
    max_width = max(img.shape[2] for img in images)

    padded_images = torch.zeros(len(images), 3, max_height, max_width)
    padded_densities = []
    for i, img in enumerate(images):
        h, w = img.shape[1], img.shape[2]
        padded_images[i, :, :h, :w] = img

        dens = density_maps[i]
        pad_dens = torch.zeros(dens.shape[0], max_height, max_width)
        pad_dens[:, :h, :w] = dens
        padded_densities.append(pad_dens)

    # Flatten density maps from all images so they align with flattened captions
    density_maps = torch.cat(padded_densities, dim=0)

    labels = list(labels)
    shapes = list(shapes)
    img_caps = list(img_caps)

    return padded_images, labels, shapes, img_caps, density_maps

def get_loader(processor: DataProcessor, split, batch_size):
    
    split_set = Rec8KDataset(processor, split)
    
    shuffle = True if split == 'train' else False
    split_loader = DataLoader(split_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    

    return split_loader


class Rec8KDataset(Dataset):
    def __init__(self, processor: DataProcessor, split): 
        
        self.processor = processor
        self.split = split

        split_set_tuples = processor.get_img_ids_for_split(split) # list of (img_id, cap)
        self.density_dir = './datasets/rec-8k/density_maps'
        split_dict = {}
        for img_id, cap in split_set_tuples:
            if img_id in split_dict:
                split_dict[img_id].append(cap)
            else:
                split_dict[img_id] = [cap]

        self.img_ids = list(split_dict.keys())
        self.labels = [list(split_dict[img_id]) for img_id in self.img_ids] # list of list of caps
        # 2245 - 2250
        # self.img_ids = self.img_ids[:10]
        # self.labels = self.labels[:10]
        
        self.img_cap_tuples = []
        for i, (img_id, caps) in enumerate(zip(self.img_ids, self.labels)):
            img_cap_tuple = [(img_id, cap) for cap in caps] 
            self.img_cap_tuples.append(img_cap_tuple)
            for j, cap in enumerate(caps):
                text_prompt = processor.get_prompt_for_image((img_id, cap))[0]
                self.labels[i][j] = preprocess_caption(caption=text_prompt)
                
    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        density_dir = os.path.join(self.density_dir, self.img_ids[idx][:-4])

        img_path = self.processor.get_image_path()
        img_file = os.path.join(img_path, self.img_ids[idx])

        label = self.labels[idx] # list of caps for same image

        image_source, image, density_maps = load_image(img_file, density_dir, label)
        h, w, _ = image_source.shape
        
        img_cap_tuple = self.img_cap_tuples[idx]  # list of tuples (img_id, cap) for same image

        return image, label, (h, w), img_cap_tuple, density_maps

