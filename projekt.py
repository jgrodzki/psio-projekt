import cv2 as cv
import numpy as np

BOX_COUNT = 4
TAPE_X_BOUNDS = range(130, 500)
TAPE_WIDTH = 40
TAPE_INITIAL_POS = 190
TAPE_COLOR_THRESHOLD = 40
SCANNER_HEAD_POS = 20
SCANNER_TAIL_POS = 280
SCANNER_WIDTH = 10
SCANNER_DEADZONE = 100
SCANNER_THRESHOLD = 200


def get_mean_color(region):
    return np.mean(np.reshape(region, (-1, 3)), axis=0)


def distance(a, b):
    return np.linalg.norm(a - b, axis=2)


cap = cv.VideoCapture("b.mp4")
ret, frame = cap.read()
tape_color = get_mean_color(
    frame[TAPE_INITIAL_POS : TAPE_INITIAL_POS + TAPE_WIDTH, TAPE_X_BOUNDS]
)
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
                    SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
                ]
            )
            > SCANNER_THRESHOLD
        )
        scanner_tail_detection = (
            np.mean(
                tape_mask[
                    SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
                ]
            )
            > SCANNER_THRESHOLD
        )
        if scanner_head_detection and scanner_tail_detection:
            remaining_deadzone = SCANNER_DEADZONE
            boxes_detected += 1
            box = frame[SCANNER_HEAD_POS + TAPE_WIDTH : SCANNER_TAIL_POS, TAPE_X_BOUNDS]
            cv.imshow("box", box)
            print("Box number", boxes_detected, "detected.")
        tape_mask = cv.cvtColor(tape_mask, cv.COLOR_GRAY2BGR)
        if scanner_head_detection:
            tape_mask[
                SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
            ] = [
                0,
                255,
                0,
            ]
        else:
            tape_mask[
                SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
            ] = [
                0,
                0,
                255,
            ]
        if scanner_tail_detection:
            tape_mask[
                SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
            ] = [
                0,
                255,
                0,
            ]
        else:
            tape_mask[
                SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
            ] = [
                0,
                0,
                255,
            ]
    else:
        remaining_deadzone -= 1
        tape_mask = cv.cvtColor(tape_mask, cv.COLOR_GRAY2BGR)
        tape_mask[
            SCANNER_HEAD_POS : SCANNER_HEAD_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
        ] = [
            255,
            0,
            0,
        ]
        tape_mask[
            SCANNER_TAIL_POS : SCANNER_TAIL_POS + SCANNER_WIDTH, TAPE_X_BOUNDS
        ] = [
            255,
            0,
            0,
        ]
    cv.imshow("mask", tape_mask)
    cv.imshow("frame", frame)
    # 1000/29 ms to match original framerate
    if cv.waitKey(34) == ord("q"):
        break
cv.destroyAllWindows()
cap.release()
print("Detected", boxes_detected, "out of", BOX_COUNT, "boxes")
