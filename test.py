import cv2
from HandTrackingModule import HandDetector
from FaceDetectionModule import FaceDetector
cap=cv2.VideoCapture(0)
#create object detector
hand_detector= HandDetector(detectionCon=0.8)
face_detector = FaceDetector()
while True:
	success,img=cap.read()
	img, bboxs = face_detector.findFaces(img)
	if bboxs:
        # bboxInfo - "id","bbox","score","center"
		center = bboxs[0]["center"]
		cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)

	hands,img= hand_detector.findHands(img)#going to return img with drawing
    

	cv2.putText(img,"Hand Count "+ str(len(hands)),(20,20),cv2.FONT_HERSHEY_PLAIN,
                                2,(255, 0, 255),2)
	cv2.putText(img,"Face Count "+ str(len(bboxs)),(20,50),cv2.FONT_HERSHEY_PLAIN,
                                2,(0, 0, 255),2)
	cv2.imshow("Image",img)


	#Wait for user input - q, then you will stop the loop
	key_pressed = cv2.waitKey(1) & 0xFF #it will wait for 1 mili second bitwise and 
	if key_pressed == ord('q'): #ord tells you ascii value of that character
		break

cap.release()
cv2.destroyALlWindows()
