# "mnist_train_data" is the data file which contains a 60000*45*45 matrix(data_num*fig_w*fig_w)
# "mnist_train_label" is the label file which contains a 60000*1 matrix. Each element i is a number in [0,9]. 
# The dataset is saved as binary files and should be read by Byte. Here is an example of input the dataset and save a random figure.

import numpy as np
from PIL import Image
data_num = 60000 #The number of figures
fig_w = 45       #width of each figure

data = np.fromfile("mnist_train_data",dtype=np.uint8)
label = np.fromfile("mnist_train_label",dtype=np.uint8)

print(data.shape)
print(label.shape)

#reshape the matrix
data = data.reshape(data_num,fig_w,fig_w)

print("After reshape:",data.shape)

#choose a random index
ind = np.random.randint(0,data_num)

#print the index and label
print("index: ",ind)
print("label: ",label[ind])

#save the figure
im = Image.fromarray(data[ind])
im.save("example.png")