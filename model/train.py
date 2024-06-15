import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 3e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512  
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/ml/swostika/unet/data/train"
TRAIN_MASK_DIR = "/home/ml/swostika/unet/data/train_masks"

VAL_IMG_DIR = "/home/ml/swostika/unet/data/val_train"
VAL_MASK_DIR = "/home/ml/swostika/unet/data/val_train_masks"


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():

    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        step_loss=train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

       

    



if __name__ == "__main__":
    
    main()


'''loss'''






'''


import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from model import UNET
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 2e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 15
NUM_WORKERS = 2
IMAGE_HEIGHT = 512  
IMAGE_WIDTH = 512  
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/home/ml/swostika/unet/data/train"
TRAIN_MASK_DIR = "/home/ml/swostika/unet/data/train_masks"

VAL_IMG_DIR = "/home/ml/swostika/unet/data/val_train"
VAL_MASK_DIR = "/home/ml/swostika/unet/data/val_train_masks"

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    step_losses = []

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(data)
            
            # print(f"predictions:{predictions}")
            # print(f"target:{targets}")
            
            # target_max=torch.max((targets))
            # print(f"targetmax:{target_max}")
            # print(f"targetmin:{torch.min(targets)}")
            # print(f"predictmax:{torch.max(predictions)}")
            # print(f"predictmin:{torch.min(predictions)}")


    
            loss = loss_fn(predictions, targets)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        step_losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
        # exit()

    return step_losses





def val_fn(loader, model, loss_fn, scaler):
    loop = tqdm(loader)
    val_losses = []

    model.eval()  

    with torch.no_grad():  
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=DEVICE)
            targets = targets.float().unsqueeze(1).to(device=DEVICE)

            with torch.cuda.amp.autocast():
                predictions = model(data)

                loss = loss_fn(predictions, targets)

            val_losses.append(loss.item())
            loop.set_postfix(val_loss=loss.item())

    return val_losses

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

    model = UNET(in_channels=3, out_channels=1).to(DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASK_DIR,
        VAL_IMG_DIR,
        VAL_MASK_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms,
        NUM_WORKERS,
        PIN_MEMORY,
    )

    if LOAD_MODEL:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

    check_accuracy(val_loader, model, device=DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        step_losses = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        train_loss = np.mean(step_losses)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss}")

        val_step_losses = val_fn(val_loader, model, loss_fn, scaler)
        val_loss = np.mean(val_step_losses)
        val_losses.append(val_loss)
        print(f"Validation Loss: {val_loss}")

        # Calculate accuracy for the current epoch
        train_accuracy = check_accuracy(train_loader, model, device=DEVICE)
        val_accuracy = check_accuracy(val_loader, model, device=DEVICE)
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)
        print(f"Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}")

        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

    # # Plotting the training and validation loss graph
    # plt.plot(train_losses, label='Training Loss')
    # plt.plot(val_losses, label='Validation Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Over Time')
    # plt.legend()
    # plt.savefig('loss_graph1.png')
    # plt.show()

    # Plotting the training and validation accuracy graph
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(' Accuracy Over Time')
    plt.legend()
    plt.savefig('accuracy_graph(test)1.png')
    plt.show()

if __name__ == "__main__":
    main()
'''