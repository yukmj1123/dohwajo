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


# Random_Brightness_Augmentation : 이미지 밝기 랜덤
def Random_Brightness_Augmentation(data):
    samples = expand_dims(data,0)
    datagen = ImageDataGenerator(brightness_range=[0.2,1.0])

    it = datagen.flow(samples, batch_size=1)

    fig = plt.figure(figsize=(20,20))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

    plt.show()


# Horizontal_And_Vertical_Flip : 위, 아래, 왼쪽, 오른쪽으로 뒤집기
def Horizontal_And_Vertical_Flip(data):
    samples = expand_dims(data,0)
    datagen = ImageDataGenerator(horizontal_flip=True,
                                 vertical_flip=True)

    it = datagen.flow(samples, batch_size=1)

    fig = plt.figure(figsize=(30,30))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

    plt.show()

# Random_Rotation : 랜덤으로 회전
def Random_Rotation(data):
    samples = expand_dims(data,0)
    datagen = ImageDataGenerator(rotation_range=90)

    it = datagen.flow(samples, batch_size=1)

    fig = plt.figure(figsize=(30,30))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

    plt.show()

# Random_Zoom : 랜덤으로 줌
def Random_Zoom(data):
    samples = expand_dims(data,0)
    datagen = ImageDataGenerator(zoom_range=[0.5,1.0])

    it = datagen.flow(samples, batch_size=1)

    fig = plt.figure(figsize=(30,30))

    for i in range(9):
        plt.subplot(3, 3, i+1)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

    plt.show()

# Random_Argumentation : 데이터 랜덤 증강
def Random_Argumentation(data):
    samples = expand_dims(data,0)
    datagen = ImageDataGenerator(
                                zoom_range=[0.5,1.0],
                                brightness_range=[0.2,1.0],
                                rotation_range=90,
                                horizontal_flip=True,
                                vertical_flip=True,
                                width_shift_range=0.5)

    it = datagen.flow(samples, batch_size=1)

    fig = plt.figure(figsize=(30,30))

    for i in range(32):
        plt.subplot(8, 4, i+1)
        batch = it.next()
        image = batch[0].astype('uint8')
        plt.imshow(image)

    plt.show()