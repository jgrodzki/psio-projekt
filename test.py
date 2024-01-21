import cv2 as cv
import numpy as np
import skimage as ski


def get_mean_color(region):
    return np.mean(np.reshape(region, (-1, 3)), axis=0)


def distance(a, b):
    return np.linalg.norm(a - b, axis=2)


nut = cv.imread("zdj_do_progowania/bolt4_mask.png", cv.IMREAD_GRAYSCALE)
# nut = ski.transform.rescale(nut, 0.3)
# nut = ((nut > 0) * 255).astype(np.uint8)
nut_contours, _ = cv.findContours(nut, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
nut_contour = nut_contours[0]
a = cv.cvtColor(nut, cv.COLOR_GRAY2BGR)
cv.drawContours(a, nut_contour, -1, (0, 255, 0), 3)
cv.imshow("img", a)
while True:
    if cv.waitKey(-1) == ord("q"):
        break


img = cv.imread("box.png")
mean = get_mean_color(img[170 * 3 : 200 * 3, : 150 * 3])
img_mask = ((distance(img, mean) > 70) * 255).astype(np.uint8)
img_mask = ski.morphology.closing(img_mask, ski.morphology.square(12))
img_mask = ski.morphology.opening(img_mask, ski.morphology.square(3))
cv.imshow("img", img_mask)
while True:
    if cv.waitKey(-1) == ord("q"):
        break

labels = ski.segmentation.clear_border(ski.measure.label(img_mask))
props = ski.measure.regionprops(labels)

for prop in props:
    print("Image!")
    contours, _ = cv.findContours(
        img_mask[prop.slice], cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
    )
    for contour in contours:
        c = cv.matchShapes(nut_contour, contour, cv.CONTOURS_MATCH_I1, 0)
        print(c)
        if c < 0.1:
            print("Match!")
        a = cv.cvtColor(img_mask[prop.slice], cv.COLOR_GRAY2BGR)
        cv.drawContours(a, contour, -1, (0, 0, 255), 1)
        cv.imshow("img", a)
        while True:
            if cv.waitKey(-1) == ord("q"):
                break
while True:
    if cv.waitKey(-1) == ord("q"):
        break
cv.destroyAllWindows()
