import numpy as np
import time
from collections import deque
from time import sleep
from picamera import PiCamera
import os

from keras.applications import *
from keras.preprocessing import image
from keras.models import Model, load_model


##current = time.strftime("%c")

counter = 0

labels = ['drinking', 'hair_makeup', 'phone_left', 'phone_right', 'radio', 'reaching_behind', 'safe',
          'talking_passenger', 'texting']
verdict = {'safe': 0, 'distracted': 1}

model = load_model("savedWeightsSNK1.1.hdf5")
distracted_history = deque()
camera = PiCamera(resolution=(227, 227), framerate=1)

while (True):
    current_image = 'image' + counter + '.jpg'
    camera.capture(current_image)  # , resize=(, 240)

    img = image.load_img(current_image, target_size=(227, 227))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    #x = preprocess_input(x)

    preds = model.predict(x)
    top3_preds = preds.argsort()[-3:]
    top3_classes = [labels[i] for i in top3_preds]

    ###################### Criteria for Distracted Decision per frame #################

    if top3_classes[0] == 'safe' and top3_preds[0] > 0.70:
        distracted_history.appendleft('safe')
    else:
        distracted_history.appendleft('distracted')

    ###################### Criteria for Distracted Decision #################

    # keeping decision within the last 4 frames
    if len(list(distracted_history)) > 4:
        distracted_history.pop()

    # Alarm
    if distracted_history.count('distracted') >= 3:
        os.system('aplay beep.mp3')

    os.remove(current_image)
    counter += 1






