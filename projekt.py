import cv2 as cv
import numpy as np
import skimage as ski

DEBUG = False
# Excpected number of slots on a conveyor
SLOT_COUNT = 4
# A factor by which frames are shrunk during slot detection phase (for performance)
IMAGE_DIVISION_FACTOR = 3
# Upper bound of sampled background on the first frame (used for computing avg. background color)
BACKGROUND_START_POS = 0
# Lower bound of sampled background on the first frame (used for computing avg. background color)
BACKGROUND_END_POS = 180
# Leftmost boundry of the conveyor
CONVEYOR_LEFT_BOUND = 350
# Right tmost boundry of the conveyor
CONVEYOR_RIGHT_BOUND = 1630
# Width of the tape separating the slots
TAPE_WIDTH = 140
# Position of the tape on the first frame (used for computing avg. tape color)
TAPE_INITIAL_POS = 210
# Threshold value used for tape mask binarization
TAPE_COLOR_THRESHOLD = 70
# Position of the scanner detecting the tape
SCANNER_POS = 90
# Width of the detected conveyor slot
SLOT_WIDTH = 750
# Width of the scanner region
SCANNER_WIDTH = 30
# For how many frames the slot detection is stopped after detecting a slot
SCANNER_DEADZONE = 100
# Threshold value used for tape detection
SCANNER_THRESHOLD = 200
# Threshold value used for binarization of a slot
SLOT_COLOR_THRESHOLD = 80
# Size of opening operation performed on a binary slot image
SLOT_OPENING_SIZE = 10
# Size of closing operation performed on a binary slot image
SLOT_CLOSING_SIZE = 30
# Tolerance of the contour size when matching against the threshold
CONTOUR_SIZE_TOLERANCE = 0.2

OBJECTS = [
    ("bolt", 0.2, 7500.0, (0, 0, 255)),
    ("long_bolt", 4.0, 9800.0, (0, 255, 0)),
    ("short_bolt", 3.0, 1900.0, (255, 0, 0)),
    ("nut", 0.1, 10000.0, (0, 255, 255)),
    ("dowel", 0.1, 8600.0, (255, 0, 255)),
]


def get_mean_color(region):
    return np.mean(np.reshape(region, (-1, 3)), axis=0)


def distance(a, b):
    return np.linalg.norm(a - b, axis=2)


def build_contour_base(objects):
    contour_base = {}
    for name, contour_threshold, size_threshold, highlight_color in objects:
        img = cv.imread("zdj_do_progowania/" + name + "_mask.png", cv.IMREAD_GRAYSCALE)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour_base[name] = {
            "contour": contours[0],
            "contour_threshold": contour_threshold,
            "size_threshold": size_threshold,
            "highlight_color": highlight_color,
        }
    return contour_base


contour_base = build_contour_base(OBJECTS)
cap = cv.VideoCapture("b.mp4")
ret, frame = cap.read()
cv.imshow("frame", frame)
while True:
    if cv.waitKey(-1) == ord("q"):
        break
tape_color = get_mean_color(
    frame[
        TAPE_INITIAL_POS : TAPE_INITIAL_POS + TAPE_WIDTH,
        CONVEYOR_LEFT_BOUND:CONVEYOR_RIGHT_BOUND,
    ]
)
background_color = get_mean_color(
    frame[
        BACKGROUND_START_POS:BACKGROUND_END_POS,
        CONVEYOR_LEFT_BOUND:CONVEYOR_RIGHT_BOUND,
    ]
)
slots_detected = 0
remaining_deadzone = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_scaled = ski.transform.rescale(
        frame, 1.0 / IMAGE_DIVISION_FACTOR, order=0, channel_axis=2, preserve_range=True
    )
    slot = frame[
        SCANNER_POS + TAPE_WIDTH : SCANNER_POS + TAPE_WIDTH + SLOT_WIDTH,
        CONVEYOR_LEFT_BOUND:CONVEYOR_RIGHT_BOUND,
    ]
    tape_mask = (
        (distance(tape_color, frame_scaled) < TAPE_COLOR_THRESHOLD) * 255
    ).astype(np.uint8)
    if remaining_deadzone == 0:
        scanner_head_detection = (
            np.mean(
                tape_mask[
                    SCANNER_POS
                    // IMAGE_DIVISION_FACTOR : (SCANNER_POS + SCANNER_WIDTH)
                    // IMAGE_DIVISION_FACTOR,
                    CONVEYOR_LEFT_BOUND
                    // IMAGE_DIVISION_FACTOR : CONVEYOR_RIGHT_BOUND
                    // IMAGE_DIVISION_FACTOR,
                ]
            )
            > SCANNER_THRESHOLD
        )
        if scanner_head_detection:
            remaining_deadzone = SCANNER_DEADZONE
            slots_detected += 1
            slot_mask = (
                (distance(background_color, slot) > SLOT_COLOR_THRESHOLD) * 255
            ).astype(np.uint8)
            # remove items on edges
            slot_mask = (
                (ski.segmentation.clear_border(ski.measure.label(slot_mask)) != 0) * 255
            ).astype(np.uint8)
            slot_mask = ski.morphology.closing(
                slot_mask, ski.morphology.square(SLOT_CLOSING_SIZE)
            )
            slot_mask = ski.morphology.opening(
                slot_mask, ski.morphology.square(SLOT_OPENING_SIZE)
            )
            print("Slot number", slots_detected, "detected.")
            detected_contours, _ = cv.findContours(
                slot_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            if DEBUG:
                slot_mask = cv.cvtColor(slot_mask, cv.COLOR_GRAY2BGR)
            else:
                matches = np.zeros((len(contour_base)))
            for detected_contour in detected_contours:
                for i, (name, object) in enumerate(contour_base.items()):
                    c = cv.matchShapes(
                        object["contour"], detected_contour, cv.CONTOURS_MATCH_I1, 0
                    )
                    s = cv.contourArea(detected_contour)
                    if (
                        c < object["contour_threshold"]
                        and s
                        > object["size_threshold"] * (1.0 - CONTOUR_SIZE_TOLERANCE)
                    ) and s < object["size_threshold"] * (1.0 + CONTOUR_SIZE_TOLERANCE):
                        if DEBUG:
                            cv.drawContours(
                                slot_mask,
                                detected_contour,
                                -1,
                                (0, 255, 0),
                                3,
                            )
                            print("\033[32m", end="")
                        else:
                            matches[i] += 1
                            cv.drawContours(
                                slot,
                                detected_contour,
                                -1,
                                object["highlight_color"],
                                3,
                            )
                            break
                    if DEBUG:
                        print(
                            "Check for " + name + "\tcontour:",
                            c,
                            "\tcontour_threshold:",
                            object["contour_threshold"],
                            "\tsize:",
                            s,
                            "\tsize_threshold:",
                            object["size_threshold"],
                            "\033[0m",
                        )
                if DEBUG:
                    print()
            if not DEBUG:
                for i, name in enumerate(contour_base.keys()):
                    print(name + ":", matches[i].astype(int))
                print(
                    "no matches:", (len(detected_contours) - matches.sum()).astype(int)
                )
                print()
            if DEBUG:
                cv.imshow("slot", slot_mask)
            else:
                cv.imshow("frame", slot)
            # stop to display the results
            while True:
                if cv.waitKey(-1) == ord("q"):
                    break
        if DEBUG:
            tape_mask = cv.cvtColor(tape_mask, cv.COLOR_GRAY2BGR)
            if scanner_head_detection:
                tape_mask[
                    SCANNER_POS
                    // IMAGE_DIVISION_FACTOR : (SCANNER_POS + SCANNER_WIDTH)
                    // IMAGE_DIVISION_FACTOR,
                    CONVEYOR_LEFT_BOUND
                    // IMAGE_DIVISION_FACTOR : CONVEYOR_RIGHT_BOUND
                    // IMAGE_DIVISION_FACTOR,
                ] = [
                    0,
                    255,
                    0,
                ]
            else:
                tape_mask[
                    SCANNER_POS
                    // IMAGE_DIVISION_FACTOR : (SCANNER_POS + SCANNER_WIDTH)
                    // IMAGE_DIVISION_FACTOR,
                    CONVEYOR_LEFT_BOUND
                    // IMAGE_DIVISION_FACTOR : CONVEYOR_RIGHT_BOUND
                    // IMAGE_DIVISION_FACTOR,
                ] = [
                    0,
                    0,
                    255,
                ]
    else:
        remaining_deadzone -= 1
        if DEBUG:
            tape_mask = cv.cvtColor(tape_mask, cv.COLOR_GRAY2BGR)
            tape_mask[
                SCANNER_POS
                // IMAGE_DIVISION_FACTOR : (SCANNER_POS + SCANNER_WIDTH)
                // IMAGE_DIVISION_FACTOR,
                CONVEYOR_LEFT_BOUND
                // IMAGE_DIVISION_FACTOR : CONVEYOR_RIGHT_BOUND
                // IMAGE_DIVISION_FACTOR,
            ] = [
                255,
                0,
                0,
            ]
    if DEBUG:
        cv.imshow("tape_mask", tape_mask)
        cv.imshow("frame", frame_scaled)
    else:
        cv.imshow("frame", slot)
    if cv.waitKey(34) == ord("q"):
        break
print("Detected", slots_detected, "out of", SLOT_COUNT, "slots")
cv.destroyAllWindows()
cap.release()
