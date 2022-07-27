# Import opencv
import cv2 

# Import uuid
import uuid

# Import Operating System
import os

# Import time
import time

labels = ['duckfront', 'duckbud']
number_imgs = 5

IMAGES_PATH = os.path.join(os.getcwd(), 'Tensorflow', 'workspace', 'images', 'collectedimages')

# Create a unique folder for each image
if not os.path.exists(IMAGES_PATH):
    if os.name == 'posix':
        os.makedirs(IMAGES_PATH)
    if os.name == 'nt':
        # use makedirs to create directory with multiple depth
        os.makedirs(IMAGES_PATH)
for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        # create single level of directory for each label
        os.mkdir(path)

# start capturing images
cap = cv2.VideoCapture(0)
while True:
    # Process Key (ESC: end) #################################################
    key = cv2.waitKey(10)
    if key == 27:  # ESC
        break
    ret, frame = cap.read()
    if not ret:
        break
    
    # print(key)
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
        print(number)
    # when number is not -1, save image to its directory
    if 0 <= number < len(labels):
        label = labels[number]
        imgname = os.path.join(IMAGES_PATH,label,label+'.'+'{}.jpg'.format(str(uuid.uuid1())))
        print(imgname)
        cv2.imwrite(imgname, frame)
        print('Collecting image {}'.format(number))
    cv2.imshow('frame', frame)



cap.release()
cv2.destroyAllWindows()