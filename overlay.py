# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# image = cv2.imread("/home/ml/swostika/uNet-master/test/jpg/slice_732.jpg")

# mask = cv2.imread("/home/ml/swostika/uNet-master/model/saved_images/newslice_732.jpg")   

# if image is not None and mask is not None:

#     mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

   
#     mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
#     mask = np.all(mask_rgb == [255, 255, 255], axis=-1)


#     mask_rgb[mask] = [0, 255, 0]

   


# import cv2


# blend = 0.5  # Adjust this value as needed (0 to 1)

# # Overlay the image with cyan color
# img_cyan = cv2.addWeighted(image, blend, mask_rgb, 1 - blend, 0)
   
# plt.imsave("result_output/newoverlayed_732_sec.jpg", img_cyan)



import cv2
import numpy as np
import os

image_folder = "/home/ml/swostika/uNet-master/test/jpg/"
mask_folder = "/home/ml/swostika/uNet-master/model/predicted_masks15/"
output_folder = "/home/ml/swostika/uNet-master/model/overlayed_images15"
os.makedirs(output_folder, exist_ok=True)


for filename in os.listdir(image_folder):
  
    mask_path = os.path.join(mask_folder, filename)
    if os.path.isfile(mask_path):
       
        image = cv2.imread(os.path.join(image_folder, filename))
        mask = cv2.imread(mask_path)
        

        
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

        print(f"mask after:{mask}")

        
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        print(f"mask_binary:{mask_binary}")
        mask_rgb = cv2.cvtColor(mask_binary, cv2.COLOR_GRAY2RGB)
        
        
        mask = np.all(mask_rgb == [255, 255, 255], axis=-1)
        

       
        mask_rgb[mask] = [0, 255, 0]
        

        
        blend = 0.7 
        img_cyan = cv2.addWeighted(image, blend, mask_rgb, 1 - blend, 0)
        print(filename)

       
        output_path = os.path.join(output_folder, "overlayed_" + filename)
        print(filename)
        cv2.imwrite(output_path, img_cyan,)
    else:
        print("Mask not found for:", filename)
print("completed!!!")
