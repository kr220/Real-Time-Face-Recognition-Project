import cv2

cap = cv2.VideoCapture(0)

while True:
    ret_val, img = cap.read()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if ret_val == False:
        break

    cv2.imshow("Cam", img)
    cv2.imshow("Cam_Gray", grayImg)

    key_pressed = cv2.waitKey(1) & 0XFF
    if key_pressed == ord('q'):
        break

cap.release()
cap.destroyAllWindows()