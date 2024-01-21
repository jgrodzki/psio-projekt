import numpy as np
import cv2 as cv

img = cv.imread("nut_mask.png", cv.IMREAD_GRAYSCALE)
contours, hierarchy = cv.findContours(img, 2, cv.CHAIN_APPROX_NONE)
print(contours)
out = np.zeros_like(img)
cv.drawContours(out, contours, -1, (255, 255, 255), 3)
cv.imshow("contour", out)
while True:
    if cv.waitKey(-1) == ord("q"):
        break
cv.destroyAllWindows()
