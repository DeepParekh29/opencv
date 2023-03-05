#importing  library tht are required
import cv2
import numpy as np
import  face_recognition as face_rec
import os
from datetime import datetime
# for resizing the images  from big to small scale
def resize(img, size):
    width = int(img.shape[1] * size)
    hight = int(img.shape[0] * size)
    dimension = (width, hight)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

path = "dataset1\img" #path for the matching images
vpath = r'D:\Pycharm\OpenCVX\dataset1\Video_clip\Testv.mp4'#path assign for the video
workerImages = []
workerNames = []
myList = os.listdir(path)
#print(myList)

#function or removing .jpg form the image name
for cl in myList:
    curImg = cv2.imread(f'{path}\{cl}') # geting workers imagies
    workerImages.append(curImg)
    workerNames.append(os.path.splitext(cl)[0])

#print(workerNames) #after removing .jpg from the workers names

#function for finding the encoding
def findEncoding(images) :
    encoding_list = []
    #for resizing
    for img in images :
        img = resize(img, 0.50)
        encoding = face_rec.face_encodings(img)[0]
        encoding_list.append(encoding)
    return encoding_list
#function for markattendence
def MarkAttendence(name):
    with open('attendance.csv', 'r+') as f:
        myDataList = f.readline()
        nameList = []
        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timestr = now.strftime('%H:%M')
            f.writelines(f'\n{name}, {timestr}')


encode_list = findEncoding(workerImages)
vid = cv2.VideoCapture(vpath) #for acccess the web cam or any cam wright " 0 " or " 1 "
print("now Comparizition started")

#heart of the porgrame
while True :
    success, frame = vid.read()
    #frame = cv2.resize(frame, (600, 600))
    frames = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    #frames = cv2.cvtColor(frames, cv2.COLOR_BGR2RGB)

    faces_in_frame = face_rec.face_locations(frames)
    encode_in_frame = face_rec.face_encodings(frames, faces_in_frame)
    #for loop for comparing

    for encodeFace, faceloc in zip(encode_in_frame, faces_in_frame) :
        matches = face_rec.compare_faces(encode_list, encodeFace)
        facedis = face_rec.face_distance(encode_list,encodeFace)
        print(facedis)
        matchIndex = np.argmin(facedis)

        if matches[matchIndex]: #if match is done then for showing the name
            name = workerNames[matchIndex]
            #for creating box around the face
            y1, x2, y2, x1 = faceloc
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 255, 0), 3)
            cv2.rectangle(frame, (x1, y2-25), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            MarkAttendence(name) # function call for the write matchname  in csv file
    if cv2.waitKey(1) & 0xff == ord('q'): #for close the the program you can also give any exti key if you want write insted od ' q '
        break
#showing the video or camera
    cv2.imshow('video', frame)


