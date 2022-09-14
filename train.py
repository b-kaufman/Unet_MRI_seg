from segstuff.dataset import SegmentationDataset
from segstuff.model import UNet
from segstuff import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

tensor_paths = [os.path.join(config.TENSOR_DIR, t) for t in os.listdir(config.TENSOR_DIR)]

train_data, test_data = train_test_split(tensor_paths, test_size = config.TEST_SPLIT)


print("[INFO] saving test paths...")
f = open(config.TEST_PATH, "w")
f.write("\n".join(test_data))
f.close()

mean = 170.26
std = 257.89
transforms = transforms.Compose([transforms.Normalize(mean,std)])

trainDS = SegmentationDataset(tensor_paths=train_data,transforms=transforms)

testDS = SegmentationDataset(tensor_paths=test_data,transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
trainLoader = DataLoader(trainDS, shuffle=True,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,num_workers=os.cpu_count())

# initialize our UNet model
unet = UNet().to(config.DEVICE)
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=config.INIT_LR)
# calculate steps per epoch for training and test set
trainSteps = len(trainDS) // config.BATCH_SIZE
testSteps = len(testDS) // config.BATCH_SIZE
# initialize a dictionary to store training history
H = {"train_loss": [], "test_loss": []}

print("[INFO] training the network...")
startTime = time.time()
for e in tqdm(range(config.NUM_EPOCHS)):
        # set the model in training mode
        unet.train()
        # initialize the total training and validation loss
        totalTrainLoss = 0
        totalTestLoss = 0
        # loop over the training set
        for (i, (x, y)) in enumerate(trainLoader):
                # send the input to the device
                (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                # perform a forward pass and calculate the training loss
                pred = unet(x)
                loss = lossFunc(pred, y)
                # first, zero out any previously accumulated gradients, then
                # perform backpropagation, and then update model parameters
                opt.zero_grad()
                loss.backward()
                opt.step()
                # add the loss to the total training loss so far
                totalTrainLoss += loss
        # switch off autograd
        with torch.no_grad():
                # set the model in evaluation mode
                unet.eval()
                # loop over the validation set
                for (x, y) in testLoader:
                        # send the input to the device
                        (x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
                        # make the predictions and calculate the validation loss
                        pred = unet(x)
                        totalTestLoss += lossFunc(pred, y)
        # calculate the average training and validation loss
        avgTrainLoss = totalTrainLoss / trainSteps
        avgTestLoss = totalTestLoss / testSteps
        # update our training history
        H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
        H["test_loss"].append(avgTestLoss.cpu().detach().numpy())

        print("[INFO] EPOCH: {}/{}".format(e + 1, config.NUM_EPOCHS))
        print("Train loss: {:.6f}, Test loss: {:.4f}".format(avgTrainLoss, avgTestLoss))

endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(endTime - startTime))

# plot the training loss
plt.style.use("ggplot")
plt.figure()
plt.plot(H["train_loss"], label="train_loss")
plt.plot(H["test_loss"], label="test_loss")
plt.title("Training Loss on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.legend(loc="lower left")
plt.savefig(config.PLOT_PATH)
torch.save(unet, config.MODEL_PATH)
