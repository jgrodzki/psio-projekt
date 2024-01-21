import cv2 as cv
import numpy as np
import skimage as ski

BOX_COUNT = 4
CONVEYOR_BOUNDS = range(390, 1500)
TAPE_WIDTH = 120
TAPE_INITIAL_POS = 570
TAPE_COLOR_THRESHOLD = 40
SCANNER_HEAD_POS = 90
SCANNER_TAIL_POS = 870
SCANNER_WIDTH = 30
SCANNER_DEADZONE = 100
SCANNER_THRESHOLD = 200
BOX_COLOR_THRESHOLD = 40
BOX_OPENING_SIZE = 5
BOX_CLOSING_SIZE = 10
OBJECTS = [
    ("bolt", 0.2),
    ("long_bolt", 3.0),
    ("short_bolt", 1.5),
    ("nut", 0.1),
    ("dowel", 0.1),
]


def get_mean_color(region):
    return np.mean(np.reshape(region, (-1, 3)), axis=0)


def distance(a, b):
    return np.linalg.norm(a - b, axis=2)


def build_contour_base(objects):
    contour_base = {}
    for name, detection_threshold in objects:
        img = cv.imread("zdj_do_progowania/" + name + "_mask.png", cv.IMREAD_GRAYSCALE)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour_base[name] = {
            "contour": contours[0],
            "detection_threshold": detection_threshold,
        }
    return contour_base


contour_base = build_contour_base(OBJECTS)
cap = cv.VideoCapture("a.mp4")
ret, frame = cap.read()
tape_color = get_mean_color(
    frame[TAPE_INITIAL_POS : TAPE_INITIAL_POS + TAPE_WIDTH, CONVEYOR_BOUNDS]
)
background_color = get_mean_color(frame[:500, CONVEYOR_BOUNDS])
boxes_detected = 0
remaining_deadzone = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    tape_mask = ((distance(tape_color, frame) < TAPE_COLOR_THRESHOLD) * 255).astype(
        np.uint8
    )
    if remaining_deadzone == 0:
        scanner_head_detection = (
            np.mean(
                tape_mask[
                    SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
                ]
            )
            > SCANNER_THRESHOLD
        )
        scanner_tail_detection = (
            np.mean(
                tape_mask[
                    SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
                ]
            )
            > SCANNER_THRESHOLD
        )
        if scanner_head_detection and scanner_tail_detection:
            remaining_deadzone = SCANNER_DEADZONE
            boxes_detected += 1
            box = frame[
                SCANNER_HEAD_POS + TAPE_WIDTH : SCANNER_TAIL_POS, CONVEYOR_BOUNDS
            ]
            box_mask = (
                (distance(background_color, box) > BOX_COLOR_THRESHOLD) * 255
            ).astype(np.uint8)
            box_mask = ski.morphology.closing(
                box_mask, ski.morphology.square(BOX_CLOSING_SIZE)
            )
            box_mask = ski.morphology.opening(
                box_mask, ski.morphology.square(BOX_OPENING_SIZE)
            )
            print("Box number", boxes_detected, "detected.")
            matches = np.zeros((len(contour_base)))
            # remove items on edges
            box_mask = (
                (ski.segmentation.clear_border(ski.measure.label(box_mask)) != 0) * 255
            ).astype(np.uint8)
            cv.imshow("box", box_mask)
            detected_contours, _ = cv.findContours(
                box_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            for detected_contour in detected_contours:
                for i, (name, object) in enumerate(contour_base.items()):
                    c = cv.matchShapes(
                        object["contour"], detected_contour, cv.CONTOURS_MATCH_I1, 0
                    )
                    print("Check for " + name + ":", c)
                    if c < object["detection_threshold"]:
                        matches[i] += 1
                print()
            for i, name in enumerate(contour_base.keys()):
                print(name + ":", matches[i].astype(int))
            print("No matches:", (len(detected_contours) - matches.sum()).astype(int))
            # stall
            while True:
                if cv.waitKey(0) == ord("q"):
                    break
        tape_mask = cv.cvtColor(tape_mask, cv.COLOR_GRAY2BGR)
        if scanner_head_detection:
            tape_mask[
                SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
            ] = [
                0,
                255,
                0,
            ]
        else:
            tape_mask[
                SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
            ] = [
                0,
                0,
                255,
            ]
        if scanner_tail_detection:
            tape_mask[
                SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
            ] = [
                0,
                255,
                0,
            ]
        else:
            tape_mask[
                SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
            ] = [
                0,
                0,
                255,
            ]
    else:
        remaining_deadzone -= 1
        tape_mask = cv.cvtColor(tape_mask, cv.COLOR_GRAY2BGR)
        tape_mask[
            SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
        ] = [
            255,
            0,
            0,
        ]
        tape_mask[
            SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, CONVEYOR_BOUNDS
        ] = [
            255,
            0,
            0,
        ]
    cv.imshow("mask", ski.transform.rescale(tape_mask, 0.3, channel_axis=2))
    cv.imshow("frame", ski.transform.rescale(frame, 0.3, channel_axis=2))
    if cv.waitKey(34) == ord("q"):
        break
print("Detected", boxes_detected, "out of", BOX_COUNT, "boxes")
cv.destroyAllWindows()
cap.release()
