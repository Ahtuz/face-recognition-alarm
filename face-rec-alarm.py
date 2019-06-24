import numpy as np 
import os
import math
import cv2
import shutil
from datetime import datetime
from datetime import timedelta
import playsound as ps
import time
import string
import random

# inisialisasi classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_detect_object = cv2.face.LBPHFaceRecognizer_create()

def randomString(stringLength=5):
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for i in range(stringLength))

def captureCamera():
    while True:

        validationString = randomString(5)
        validation = input("[VALIDATION] Please insert this string to continue: "+ validationString + "\n >> ")

        if validation == validationString:
            camera = cv2.VideoCapture(0)

            while True:
                _, capturedFrame = camera.read()
                cv2.imshow("PRESS SPACE TO TAKE A PHOTO", capturedFrame)
                
                k = cv2.waitKey(1)

                if k%256 == 32:
                    # SPACE pressed
                    timeNow = datetime.now()
                    timeNow = timeNow.strftime("%d-%m-%Y_%H-%M-%S")
                    test_image_path = 'dataset/test/test_cam_'+str(timeNow)+'.jpg'
                    test_image_name = 'face_alarm_'+str(timeNow)+'.jpg'
                    print("Image saved as"+test_image_name)
                    cv2.imwrite(test_image_path, capturedFrame)
                    break

            print("\nWe just taken your picture.\nClose the camera window to continue...\n")
            cv2.waitKey(0)
            del(camera)
            break
        else:
            input("Wrong validation string, press enter to re-validate...\n")

    return test_image_path, test_image_name

while True:
    time_input = str(input("Please enter the time in HHMM format: "))
    if int(time_input[0:2]) < 24 and int(time_input[0:2]) >= 0:
        if int(time_input[2:]) < 60 and int(time_input[2:]) >= 0:
            current_date = datetime.now().date()
            selected_time = datetime.strptime('%s %s'%(current_date, time_input),"%Y-%m-%d  %H%M")
            print("\nThe alarm will ring today on " + str(time_input[0:2]) + ":" + str(time_input[2:]) + ".\n")
            remaining_time = datetime.now() - timedelta(hours=int(time_input[0:2]), minutes=int(time_input[2:]))
            print("There are " + str(23 - int(remaining_time.strftime("%H"))) + " hours & " + str(60 - int(remaining_time.strftime("%M"))) + " minutes until the alarm rings.\n")
            break

stop = False
while stop == False:
    timeNow = str(datetime.now())
    if timeNow >= str(selected_time):
        ps.playsound('alarm.mp3', block=False)
        stop = True

### training ###

train_location = 'dataset/train'
person_names = os.listdir(train_location)
train_faces = []
train_labels = []

for i, name in enumerate(person_names):
    person_images = os.listdir(train_location + '/' + name)
    for image in person_images:
        face_gray = cv2.imread(train_location + '/' + name + '/' + image, 0)

        detected_faces = face_cascade.detectMultiScale(face_gray, scaleFactor = 1.2, minNeighbors = 5)

        if len(detected_faces) < 1:
            continue
        
        for face in detected_faces:
            # ambil koordinat face yang telah didetect
            x, y, w, h = face

            # crop face
            face_rect = face_gray[y:y+h , x:x+w]
            train_faces.append(face_rect)
            train_labels.append(i)

face_detect_object.train(train_faces, np.array(train_labels))

### test ###

realOwner = False

while realOwner == False:
    test_image_path, test_image_name = captureCamera()
    face_gray = cv2.imread(test_image_path, 0)
    face_bgr = cv2.imread(test_image_path)
    detected_faces = face_cascade.detectMultiScale(face_gray, scaleFactor = 1.2, minNeighbors = 5)

    if len(detected_faces) < 1:
        print("\n\n\nNo face detected...\nClose the camera result window to redo the validation...")
        cv2.imshow("Captured Image Result", face_bgr)
        cv2.waitKey(0)
        os.remove(test_image_path)
    else:
        for faces in detected_faces:
        # ambil koordinat face
            x, y, w, h = faces

            # crop face
            face_rect = face_gray[y:y+h , x:x+w]
            
            # predict test image
            result, confidence = face_detect_object.predict(face_rect)

            confidence = 100 - (math.floor(confidence * 100) / 100)

            text = person_names[result] + ' ' + str(confidence) + '%'

            cv2.rectangle(face_bgr, (x,y), (x+w,y+h), (255,0,255), 2)

            cv2.putText(face_bgr, text, (x,y-10), 1, 1, (255,255,255))

            if confidence > 70:
                print("\n\nThe prediction confidence is "+str(confidence)+'%\n')
                shutil.move(test_image_path, 'dataset/train/'+str(person_names[result]))
                print("For better performance in the future,\nwe just moved " + test_image_name + " to folder " + 'dataset/train/'+str(person_names[result]))
                print("\nClose the camera result to dismiss the alarm...")
                cv2.imshow("Captured Image Result", face_bgr)
                cv2.waitKey(0)
                print("Alarm stopped, have a good day!")
                realOwner = True
            else:
                print("\n\n\nThe prediction confidence is too low ("+str(confidence)+"%).")
                print("Close the camera result to redo the validation...")
                cv2.imshow("Captured Image Result", face_bgr)
                cv2.waitKey(0)
                os.remove(test_image_path)
                break