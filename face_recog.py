import cv2
import numpy as np
import os


face_data = []  #for X-values
label = []      #for Y-values

class_id = 0  #id starting with 0 for first face
names = {}      #for mapping

dataset_path = './data/'




#                    KNN 
#________________________________________________________________________________
def distance(x1, x2):
    return np.sqrt(((x2-x1)**2).sum())

def knn(train, test, k=5):      
    #train-> data in the form of matrix   test-> data point

    dist = []
    x = train.shape[0]
        
    for i in range(x):
        #we will take point from matrix one by one in ix and result in iy 
        ix = train[i , :-1]
        iy = train[i , -1]

        dis = distance(ix, test)
        dist.append([dis, iy])

    #sort the dist list and take smallest k
    dist = sorted(dist)
    dist = dist[0:k]

    labels = np.array(dist)[ : , -1]

    new_list = np.unique(labels, return_counts=True)

    index = new_list[1].argmax()
    ans = new_list[0][index]

    return ans

#______________________________________________________________________________

#################  TRAINING DATA PREPARATION   ###############

for fx in os.listdir(dataset_path):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]

        # For every person enlist data item in face_data list   (X-data)
        data_item = np.load(dataset_path+fx)
        face_data.append(data_item)

        # Create labels for data_item   (Y-data)
        target = class_id*np.ones((data_item.shape[0] , ))     
        #numpy array of size = No. of faces in each data_item
        label.append(target)
        class_id += 1

## Now let us concatenate  face_data and then store it in new var (say X-matrix)
## & store into an array type structure (say Y-array)
face_dataset = np.concatenate(face_data, axis = 0)
face_label = np.concatenate(label, axis=0).reshape((-1,1))

#Now concatenate x and y to get final training dataset
trainset = np.concatenate((face_dataset, face_label), axis = 1)

# print(trainset.shape)
# print(face_dataset.shape)
# print(face_label.shape)

##################  TESTING PART ###############

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_alt.xml')

skip = 0

while True:
    ret, img = cap.read()

    if ret == False:
        continue

    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if len(faces) == 0:
        continue

    for face in faces:
        x, y, w, h = face

        offset = 10
        img_section = img[y-offset:y+h+offset, x-offset:x+w+offset]
        img_section = cv2.resize(img_section, (100,100))

        prediction = knn(trainset, img_section.flatten())

        pred_name = names[int(prediction)]

        cv2.putText(img, pred_name, (x,y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.rectangle(img, (x,y), (x+h, y+w), (0, 0, 255), 2)

    cv2.imshow("CAMERA", img)

    key_pressed = cv2.waitKey(1) & 0XFF
    if key_pressed == ord('q'):
        break

cap.release()
cap.destroyAllWindows()

    