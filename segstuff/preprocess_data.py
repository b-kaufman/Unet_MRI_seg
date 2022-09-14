import torch
import numpy as np
import nibabel as nib
import os

def create_numpy_tensors(image_paths, mask_paths, tensor_dir):
    for img,mask in zip(image_paths, mask_paths):
        np_img = nib.load(img).get_fdata()
        np_mask = nib.load(mask).get_fdata()
        img_id = os.path.splitext(os.path.basename(img))[0]
        write_tensors_along_idx(np_img,np_mask,2,'z',img_id,tensor_dir)


def write_tensors_along_idx(np_img,np_mask,mat_idx,idx_name,img_id,tensor_dir,ext='.pt'):
        for ii,(img_slice,mask_slice) in enumerate(zip(np.rollaxis(np_img,mat_idx),np.rollaxis(np_mask,mat_idx))):
            filename = '_'.join([img_id,idx_name,str(ii)])
            filepath = os.path.join(tensor_dir,filename + ext)
            tens_dict = {'img': img_slice, 'mask': mask_slice}
            torch.save(tens_dict,filepath)



def compute_mean_and_std(tensor_dir):
    data_mat = [torch.load(os.path.join(tensor_dir,t))['img'] for t in os.listdir(tensor_dir)]
    mean = np.mean(data_mat)
    std = np.std(data_mat)
    return mean, std

# mean: 170.25972062830252,std: 257.8855047907557


if __name__ == "__main__":
    image_dir = '/home/bmk/personal_code/kaggle/card_MRI/data/imagesTr'
    mask_dir = '/home/bmk/personal_code/kaggle/card_MRI/data/labelsTr'
    tensor_dir = '/home/bmk/personal_code/kaggle/card_MRI/data/tensorsTr'
    image_paths = [os.path.join(image_dir,x) for x in sorted(os.listdir(image_dir))]
    mask_paths = [os.path.join(mask_dir,x) for x in sorted(os.listdir(mask_dir))]
    #create_numpy_tensors(image_paths,mask_paths, tensor_dir)
    compute_mean_and_std(tensor_dir)
