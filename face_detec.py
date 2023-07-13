import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades  + 'haarcascade_frontalface_alt.xml')

while True:
    ret_value, img = cap.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ret_value == False:
        continue
    
    faces = face_cascade.detectMultiScale(grayImg, 1.3, 5)
    

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)

    cv2.imshow("Camera", img)
    key_pressed = cv2.waitKey(1) & 0XFF
    if key_pressed == ord('q'):
        break

cap.release()
cap.destroyAllWindows()