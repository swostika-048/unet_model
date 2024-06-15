import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
from model.dataset import CarvanaDataset
# from carvana_dataset import CarvanaDataset
from model.model import UNET
# from torchsummary import summary
import numpy as np
import cv2
import os
import sys



def single_image_inference(image_pth, model_pth, device):
  
    model = UNET()
    checkpoint = torch.load(model_pth)
    state_dict = {k: v for k, v in checkpoint["state_dict"].items() if "optimizer" not in k}
    model.load_state_dict(state_dict)
    model = model.to(device)
  

# '''modelsummary'''
#     # # Redirect the output to a file
#     # sys.stdout = open('model_summary.txt', 'w')

#     # # Print model summary
#     # summary(model, (3, 512, 512))

#     # # Close the file
#     # sys.stdout.close()
#     # exit()

    model.eval()


    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()])


    img = Image.open(image_pth)
    img = transform(img).unsqueeze(0).to(device)
    print(img.shape)


    with torch.no_grad():
        pred_mask = model(img)

    pred_mask=torch.sigmoid(pred_mask)

  
    pred_mask = pred_mask.squeeze().cpu().detach().numpy()
    # print(f"pred_mask:{pred_mask}")
    # max=np.max(pred_mask)
    # print(max)
    pred_mask = (pred_mask > 0.6).astype(int)
    print(pred_mask)
    # print(pred_mask)
    # max=np.max(pred_mask)
    # print(max)

    return pred_mask

    # Save the predicted mask as a grayscale image
    # plt.imsave("saved_images/presentation_1349.jpg", pred_mask, cmap="gray")
    





# if __name__ == "__main__":
#     SINGLE_IMG_PATH = "/home/ml/swostika/unet/uNet-master/test/slice_546.jpg"
   
    
#     MODEL_PATH = "/home/ml/swostika/unet/uNet-master/model/my_checkpoint(presnetation).pth.tar"

#     device = "cuda" if torch.cuda.is_available() else "cpu"
 
#     single_image_inference(SINGLE_IMG_PATH, MODEL_PATH, device)




if __name__ == "__main__":
    IMAGE_FOLDER_PATH = "/home/ml/swostika/unet/uNet-master/test/jpg"
    MODEL_PATH = "/home/ml/swostika/unet/uNet-master/model/my_checkpoint(presnetation).pth.tar"
    OUTPUT_FOLDER_PATH = "withouterodpredicted_masks"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    
    image_files = [os.path.join(IMAGE_FOLDER_PATH, f) for f in os.listdir(IMAGE_FOLDER_PATH) if f.endswith('.jpg')]

    
    os.makedirs(OUTPUT_FOLDER_PATH, exist_ok=True)

    for img_path in image_files:
       
        result = single_image_inference(img_path, MODEL_PATH, device)
        result=result.astype(np.uint8)
        
        # print(result)
        # kernel = np.ones((15, 15), np.uint8)  
        # result = cv2.erode(result, kernel)

        # print(f"result:{result}")
        
        filename = os.path.splitext(os.path.basename(img_path))[0]
        
       
        output_path = os.path.join(OUTPUT_FOLDER_PATH, f"{filename}.jpg")
        plt.imsave(output_path, result, cmap="gray")
    print("completed!!!")
        # cv2.imwrite(output_path, result)

