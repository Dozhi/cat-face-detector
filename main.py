import cv2
import argparse

"""
pre trained cat face req script
ONLY WORKS FOR PRIME JPG FORMAT IMAGES
for future debbuging reasones there is left 3(1,2,3)
images that is not detected because of jpeg opencv bug
*
*
command to start --> python3 main.py --image images/cat_00.jpg
"""

#the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-c", "--cascade",
	default="haarcascade_frontalcatface.xml",
	help="path to cat detector haar cascade")
args = vars(ap.parse_args())


#argument(image) loading and conveting to grayscaling
image = cv2.imread(args["image"])


#print(image.shape)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

'''
loading the haar cascade that's script which already trained 
to detect cat faces
'''

detector = cv2.CascadeClassifier(args["cascade"])
rects = detector.detectMultiScale(gray, scaleFactor=1.3,
	minNeighbors=10, minSize=(75, 75))

#drawing vision represantation of found cat face on image
for (i, (x, y, w, h)) in enumerate(rects):
	cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
	cv2.putText(image, "Cat #{}".format(i + 1), (x, y - 10),
		cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
 
#showing result of detection
cv2.imshow("Cat face", image)
cv2.waitKey(0)