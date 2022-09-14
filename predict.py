from segstuff import config
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms 
import cv2
import os

def prepare_plot(origImage, origMask, predMask,fig_name):
    # initialize our figure
    figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
    # plot the original image, its mask, and the predicted mask
    ax[0].imshow(origImage)
    ax[1].imshow(origMask)
    ax[2].imshow(predMask)
    # set the titles of the subplots
    ax[0].set_title("Image")
    ax[1].set_title("Original Mask")
    ax[2].set_title("Predicted Mask")
    # set the layout of the figure and display it
    figure.tight_layout()
    plt.savefig(fig_name)

def make_predictions(model, image_path,fig_name):

    model.eval()
    
    mean = 170.26
    std = 257.89
    norm = transforms.Normalize(mean,std)

    with torch.no_grad():
        tens  = torch.load(image_path)
        img = tens['img']
        gt_mask = tens['mask']
        orig = img.copy()
        img = norm(torch.Tensor(img).unsqueeze(0).unsqueeze(0)).to(config.DEVICE)
        predMask = model(img).squeeze()
        print(predMask.size())
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        # filter out the weak predictions and convert them to integers
        predMask = (predMask > config.THRESHOLD)
        #predMask = predMask.astype(np.uint8)
        # prepare a plot for visualization
        print(f'pixels in original mask: {np.sum(gt_mask)} pixels in pred mask: {np.sum(predMask)}')
        prepare_plot(orig, gt_mask, predMask, fig_name)


print("[INFO] loading up test image paths...")
imagePaths = open(config.TEST_PATH).read().strip().split("\n")
imagePaths = np.random.choice(imagePaths, size=10)
# load our model from disk and flash it to the current device
print("[INFO] load up model...")
unet = torch.load(config.MODEL_PATH).to(config.DEVICE)
# iterate over the randomly selected test image paths
for path in imagePaths:
    # make predictions and visualize the results
    img_tag = os.path.splitext(os.path.basename(path))[0]
    fig_path = os.path.join(config.BASE_OUTPUT,img_tag+'.png')
    make_predictions(unet, path,fig_path)
