import cv2
import numpy as np

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_alt.xml')

skip = 0
face_data = []
dataset_path = './data/'
file_name = input('Enter your name')

while True:
    ret, img = cap.read()

    if ret == False:
        continue

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    faces = sorted(faces, key=lambda f:f[2] * f[3])     #here we sorted after creating a lamda fxm on the basis of height and width to get largest face


    for face in faces[-1:]:             #-1: to get largest face first from sorted list of faces
        x,y,w,h = face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,255), 2)

    # Extract region of interest from the img (largest face) and add a padding of 10 in its each side

        offset = 10
        img_section = img[y-offset:y+offset+h, x-offset:x+offset+w] 
        # convention is -> frame[height, width]
        img_section = cv2.resize(img_section, (100,100))

        #Store every 10th face frame
        skip += 1
        if skip % 10 == 0:
            face_data.append(img_section)
            print(len(face_data))



        cv2.imshow('Camera', img)
        cv2.imshow('Image Section', img_section)
    

    key_pressed = cv2.waitKey(1) & 0XFF
    if key_pressed == ord('q'):
        break

# Convert our face list array into a numpy arrayq
face_data = np.array(face_data)
# rows = no of faces, colmn = auto detetcted
face_data = face_data.reshape(face_data.shape[0], -1) 

print(face_data.shape)

#now save face data into fileSystem
np.save(dataset_path+file_name+'.npy',face_data)
print('Data saved successfully at '+ dataset_path+file_name+'.npy')

cap.release()
cap.destroyAllPrograms()
