import augmentation as aug
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

img = load_img('C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_capture_data/picture.JPG')

data = img_to_array(img)

# aug.Width_Shift_Range(data)
aug.Random_Brightness_Augmentation(data)
#aug.Random_Argumentation(data)
#aug.Horizontal_And_Vertical_Flip(data)
#aug.Random_Rotation(data)
#aug.Random_Zoom(data)