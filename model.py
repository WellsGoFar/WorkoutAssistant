import numpy as np
import cv2
import os
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
import time
import random
import re

train_model = 0
get_predictions_from_frames = 0
test_model_from_frames = 1

IMSIZE = (224, 224)
epochs = 5

def makeDatasetInMemory(class_folders,
                        in_path,
                        mode,
                        IMSIZE = IMSIZE):
    images = []
    labels = []

    if mode == "train":
        for c in class_folders:
            class_label_indexer = int(c[5])-1 
            print("loading class", class_label_indexer)
            for f in os.listdir(in_path + c):
                im = cv2.imread(in_path + c + f, 0)
                im = cv2.resize(im, IMSIZE)
                images.append(im)
                labels.append(class_label_indexer)

        images = np.array(images)
        labels = np.array(labels)

        indices = np.arange(labels.shape[0])
        np.random.shuffle(indices)

        print(labels[1:10])
        images = images[indices]
        labels = labels[indices]
        print(labels[1:10])

    else:
        images = []
        for f in os.listdir(in_path):
            im = cv2.imread(in_path + f, 0)
            im = cv2.resize(im, IMSIZE)
            images.append(im)

        images = np.array(images)

    return labels, images


def modelInit(IMSIZE=IMSIZE):

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMSIZE[0], IMSIZE[1], 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(len(class_folders), activation='softmax'))

    # model.summary()

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def pipeline(dataset, IMSIZE=IMSIZE):
    dataset = np.array(dataset)
    dataset = dataset / 255  # Normalize
    n = len(dataset)
    dataset = dataset.reshape(n, IMSIZE[0], IMSIZE[1], 1)

    return dataset

def pipelineSingleSample(i, IMSIZE=IMSIZE):
    i = cv2.resize(i, IMSIZE)
    i = i / 255  # Normalize
    i = i.reshape(1, IMSIZE[0], IMSIZE[1], 1)

    return i

def simulateVideo(in_path, IMSIZE = IMSIZE):
    images = []

    for f in os.listdir(in_path):
        im = cv2.imread(in_path + f)
        images.append(im)

    images = np.array(images)
    return images


if train_model:

    class_folders = ["class1/", "class2/", "class3/"]
    train_labels, train_images = makeDatasetInMemory(class_folders, "final/train/", mode="train")
    print(train_images.shape)

    train_images = pipeline(train_images)


    # n = int(len(train_labels)*0.3)
    # val_images = train_images[:n]
    # val_labels = train_labels[:n]


    class_weights = class_weight.compute_sample_weight('balanced', train_labels)

    model = modelInit()
    model.fit(train_images, train_labels, epochs=epochs, class_weight = class_weights)#, validation_data=(val_images, val_labels))
    model.save('cnn_1.h5')

if get_predictions_from_frames:
    m = models.load_model("cnn_1.h5")

    _, test_images = makeDatasetInMemory("", "final/test/raw_images/", "test")

    test_images = pipeline(test_images)

    predictions = m.predict(test_images)
    print(predictions)
    np.savetxt('predictions.csv', predictions, delimiter = ',')

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

if test_model_from_frames:

    print("Loading model")
    m = models.load_model("cnn_1.h5")
    print("Model loaded")
    test_dir = "final/test/raw_images/"

    counter = 0
    state = ""
    annotation = ""
    video = cv2.VideoCapture(0)
    # for f in os.listdir(test_dir):

    #     st1 = time.time()
#     im_color = cv2.imread(test_dir + f)qaa    ZW
    while True:

        st1 = time.time()
        # _, im_color = cv2.imread(test_dir + f)
        success, im_color1 = video.read()
        im_color = im_color1.copy()
        # im_color = cv2.imread(frame)
        im = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
        im = pipelineSingleSample(im, IMSIZE)
        print(time.time() - st1, "\n\n")
        st2 = time.time()
        predictions = m.predict(im)
        print("predisction: ", predictions)
        top = predictions[:,0]
        bottom = predictions[:,1]

        thresh = 0.5

        # State logic 
        if top > thresh and bottom > thresh:
            current = ""
        elif top > thresh:
            current = "T"
            annotation = "Top of movement"
        elif bottom > thresh:
            current = "B"
            annotation = "Bottom of movement"
        else:
            current = ""
            annotation = "Transitioning"

        if (state == "B" or state == "TB") and current == "T":
            state = "T"
            counter += 1
        else:
            state += current if current not in state else ""

        # Format img
        print(time.time() - st2)
        class_pred = str(np.argmax(predictions) + 1)

        im_color = cv2.resize(im_color, (640*2, 480*2), interpolation = cv2.INTER_AREA)

        im_color = cv2.putText(im_color, "CNN Prediction: " + annotation, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 2,
                               (255, 255, 255), thickness = 10)

        im_color = cv2.putText(im_color, "CNN Prediction: " + annotation, (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 2,
                               (0, 0, 255), thickness = 3)

        im_color = cv2.putText(im_color, "Pushups completed: " + str(counter), (10, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 3,
                               (255, 255, 255), thickness=10)

        im_color = cv2.putText(im_color, "Pushups completed: " + str(counter), (10, 170),
                               cv2.FONT_HERSHEY_SIMPLEX, 3,
                               (0, 0, 255), 3)


        cv2.imshow("", im_color)
        cv2.moveWindow("", 20, 20)

        cv2.waitKey(32)
