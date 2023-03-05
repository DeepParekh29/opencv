#importing librarys
import cv2
import numpy as np
import face_recognition as face_rec
#resize
def resize(img, size):
    width = int(img.shape[1]*size)
    hight = int(img.shape[0] * size)
    dimension = (width, hight)
    return cv2.resize(img, dimension, interpolation= cv2.INTER_AREA)

#image Detection

krupal =face_rec.load_image_file("sample_images\Krupal.jpg")
krupal =cv2.cvtColor(krupal, cv2.COLOR_BGR2RGB)
krupal = resize(krupal, 0.50)
krupal_test =face_rec.load_image_file("sample_images\Krupaltest.jpg")
krupal_test =cv2.cvtColor(krupal_test, cv2.COLOR_BGR2RGB)
krupal_test = resize(krupal_test, 0.50)


#Finding face location

faceLocation_krupal = face_rec.face_locations(krupal)[0]
encode_krupal = face_rec.face_encodings(krupal)[0]
cv2.rectangle(krupal, (faceLocation_krupal[3],faceLocation_krupal[0]), (faceLocation_krupal[1],faceLocation_krupal[2]), (255,0,255),3)

#print(encode_krupal)

faceLoc_krupaltest = face_rec.face_locations(krupal_test)[0]
encode_krupaltest = face_rec.face_encodings(krupal_test)[0]
cv2.rectangle(krupal_test, (faceLoc_krupaltest[3],faceLoc_krupaltest[0]), (faceLoc_krupaltest[1],faceLoc_krupaltest[2]), (255,0,255),3)

#comapering

results = face_rec.compare_faces([encode_krupal], encode_krupaltest)
print(results)

#print(encode_krupaltest)
cv2.imshow("main_img",krupal)
cv2.imshow("test_img",krupal_test)
cv2.waitKey(0)