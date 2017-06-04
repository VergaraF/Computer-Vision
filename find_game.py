import numpy as np
import cv2

# load the games image
image = cv2.imread("games.jpg")

#find the red colour game in the image
# RGB colour backwards. Upper limit = RGB(255,65,65) which is light red
#                       Lower limit = RGB(0,0,200) which is dark red (the most red RGB val).
upper = np.array([65,65,255])
lower = np.array([0,0,200])
mask = cv2.inRange(image, lower, upper)

# find the contours in the masked image and keep the largest one

(_, cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = max(cnts, key=cv2.contourArea)

# approximate the contour
peri = cv2.arcLength(c, True)
approx = cv2.approxPolyDP(c, 0.05 * peri, True)

# draw a green bounding box surrounding the red game
cv2.drawContours(image,[approx], -1, (0, 255, 0), 4)
cv2.imshow("Image", image)
cv2.waitKey(0)

