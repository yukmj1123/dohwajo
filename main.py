import argumentation as arg
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

img = load_img('C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_data/picture.JPG')

data = img_to_array(img)

# arg.width_shift_range(data)
# arg.Random_Brightness_Augmentation(data)
# arg.Random_Argumentation(data)
# arg.Horizontal_And_Vertical_Flip(data)
# arg.Random_Rotation(data)
# arg.Random_Zoom(data)