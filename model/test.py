import csv

# Provided text
text = """
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 64, 512, 512]           1,728
       BatchNorm2d-2         [-1, 64, 512, 512]             128
              ReLU-3         [-1, 64, 512, 512]               0
            Conv2d-4         [-1, 64, 512, 512]          36,864
       BatchNorm2d-5         [-1, 64, 512, 512]             128
              ReLU-6         [-1, 64, 512, 512]               0
        DoubleConv-7         [-1, 64, 512, 512]               0
         MaxPool2d-8         [-1, 64, 256, 256]               0
            Conv2d-9        [-1, 128, 256, 256]          73,728
      BatchNorm2d-10        [-1, 128, 256, 256]             256
             ReLU-11        [-1, 128, 256, 256]               0
           Conv2d-12        [-1, 128, 256, 256]         147,456
      BatchNorm2d-13        [-1, 128, 256, 256]             256
             ReLU-14        [-1, 128, 256, 256]               0
       DoubleConv-15        [-1, 128, 256, 256]               0
        MaxPool2d-16        [-1, 128, 128, 128]               0
           Conv2d-17        [-1, 256, 128, 128]         294,912
      BatchNorm2d-18        [-1, 256, 128, 128]             512
             ReLU-19        [-1, 256, 128, 128]               0
           Conv2d-20        [-1, 256, 128, 128]         589,824
      BatchNorm2d-21        [-1, 256, 128, 128]             512
             ReLU-22        [-1, 256, 128, 128]               0
       DoubleConv-23        [-1, 256, 128, 128]               0
        MaxPool2d-24          [-1, 256, 64, 64]               0
           Conv2d-25          [-1, 512, 64, 64]       1,179,648
      BatchNorm2d-26          [-1, 512, 64, 64]           1,024
             ReLU-27          [-1, 512, 64, 64]               0
           Conv2d-28          [-1, 512, 64, 64]       2,359,296
      BatchNorm2d-29          [-1, 512, 64, 64]           1,024
             ReLU-30          [-1, 512, 64, 64]               0
       DoubleConv-31          [-1, 512, 64, 64]               0
        MaxPool2d-32          [-1, 512, 32, 32]               0
           Conv2d-33         [-1, 1024, 32, 32]       4,718,592
      BatchNorm2d-34         [-1, 1024, 32, 32]           2,048
             ReLU-35         [-1, 1024, 32, 32]               0
           Conv2d-36         [-1, 1024, 32, 32]       9,437,184
      BatchNorm2d-37         [-1, 1024, 32, 32]           2,048
             ReLU-38         [-1, 1024, 32, 32]               0
       DoubleConv-39         [-1, 1024, 32, 32]               0
  ConvTranspose2d-40          [-1, 512, 64, 64]       2,097,664
           Conv2d-41          [-1, 512, 64, 64]       4,718,592
      BatchNorm2d-42          [-1, 512, 64, 64]           1,024
             ReLU-43          [-1, 512, 64, 64]               0
           Conv2d-44          [-1, 512, 64, 64]       2,359,296
      BatchNorm2d-45          [-1, 512, 64, 64]           1,024
             ReLU-46          [-1, 512, 64, 64]               0
       DoubleConv-47          [-1, 512, 64, 64]               0
  ConvTranspose2d-48        [-1, 256, 128, 128]         524,544
           Conv2d-49        [-1, 256, 128, 128]       1,179,648
      BatchNorm2d-50        [-1, 256, 128, 128]             512
             ReLU-51        [-1, 256, 128, 128]               0
           Conv2d-52        [-1, 256, 128, 128]         589,824
      BatchNorm2d-53        [-1, 256, 128, 128]             512
             ReLU-54        [-1, 256, 128, 128]               0
       DoubleConv-55        [-1, 256, 128, 128]               0
  ConvTranspose2d-56        [-1, 128, 256, 256]         131,200
           Conv2d-57        [-1, 128, 256, 256]         294,912
      BatchNorm2d-58        [-1, 128, 256, 256]             256
             ReLU-59        [-1, 128, 256, 256]               0
           Conv2d-60        [-1, 128, 256, 256]         147,456
      BatchNorm2d-61        [-1, 128, 256, 256]             256
             ReLU-62        [-1, 128, 256, 256]               0
       DoubleConv-63        [-1, 128, 256, 256]               0
  ConvTranspose2d-64         [-1, 64, 512, 512]          32,832
           Conv2d-65         [-1, 64, 512, 512]          73,728
      BatchNorm2d-66         [-1, 64, 512, 512]             128
             ReLU-67         [-1, 64, 512, 512]               0
           Conv2d-68         [-1, 64, 512, 512]          36,864
      BatchNorm2d-69         [-1, 64, 512, 512]             128
             ReLU-70         [-1, 64, 512, 512]               0
       DoubleConv-71         [-1, 64, 512, 512]               0
           Conv2d-72          [-1, 1, 512, 512]              65
================================================================
Total params: 31,037,633
Trainable params: 31,037,633
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 3.00
Forward/backward pass size (MB): 3718.00
Params size (MB): 118.40
Estimated Total Size (MB): 3839.40
----------------------------------------------------------------
"""

# Split text into lines
lines = text.strip().split('\n')

# Split each line into columns
data = [line.split() for line in lines]

# Write data into a CSV file
with open('model_summary.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(data)

print("CSV file 'model_summary.csv' has been created.")