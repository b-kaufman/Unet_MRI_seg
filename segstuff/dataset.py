from torch.utils.data import Dataset
import torch
import nibabel as nib

class SegmentationDataset(Dataset):
    def __init__(self, tensor_paths, transforms=None):
        self.tensor_paths = tensor_paths
        self.transforms = transforms
    def __len__(self):
        return len(self.tensor_paths)

    def __getitem__(self, idx):
        
        tensor_path = self.tensor_paths[idx]
        tens_dict = torch.load(tensor_path)
        image = torch.Tensor(tens_dict['img']).unsqueeze(0)
        mask = torch.Tensor(tens_dict['mask']).unsqueeze(0)
        if self.transforms is not None:
        # apply the transformations to image
            image = self.transforms(image)

        return (image, mask)

