from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

# Width_Shift_Range : 왼쪽 오른쪽으로 이동
def Width_Shift_Range(data):
    samples = expand_dims(data,0)
    datagen = ImageDataGenerator(width_shift_range=[-200,200])

    it = datagen.flow(samples, batch_size=1)

    fig = plt.figure(figsize=(30,30))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

    plt.show()
