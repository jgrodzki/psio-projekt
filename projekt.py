import cv2 as cv
import numpy as np
import skimage as ski

# Debug mode used for calibration
DEBUG = False
# A factor by which frames are shrunk during slot detection phase (for performance)
IMAGE_DIVISION_FACTOR = 3
# Upper bound of the region sampled on the first frame (used for computing mean slot background color)
BACKGROUND_START_POS = 0
# Lower bound of the region sampled on the first frame (used for computing mean slot background color)
BACKGROUND_END_POS = 180
# Width of a slot
SLOT_WIDTH = 780
# Leftmost boundry of the conveyor
CONVEYOR_LEFT_BOUND = 350
# Rightmost boundry of the conveyor
CONVEYOR_RIGHT_BOUND = 1630
# Width of the separator between the slots
SEPARATOR_WIDTH = 140
# Position of the separator's upper bound on the first frame (used for computing mean separator color)
SEPARATOR_INIT_POS = 210
# Threshold value used for separator mask creation
SEPARATOR_MASK_THRESHOLD = 70
# Position of the separator detection region
SEPARATOR_DETECTION_POS = 90
# Width of the separator detection region
SEPARATOR_DETECTION_WIDTH = 30
# For how many frames the separator detection is stopped after detecting a separator
SEPARATOR_DETECTION_DEADZONE = 70
# Threshold value used for separator detection
SEPARATOR_DETECTION_THRESHOLD = 200
# Factor by which threshold value used in slot mask creation is scaled
SLOT_MASK_THRESHOLD_FACTOR = 0.9
# Size of opening operation performed on a binary slot image
SLOT_OPENING_SIZE = 10
# Size of closing operation performed on a binary slot image
SLOT_CLOSING_SIZE = 30
# Tolerance of a contour area when matching against a reference value
CONTOUR_AREA_TOLERANCE = 0.2
# Width of the border clearance
CLEAR_BORDER_WIDTH = 20

# Reference object data
# Object name - name of object which has a corresponding name_mask.png file inside data folder
# Contour match threshold - a threshold value below which contour is said to match the object
# Contour area - area which the contour has to occupy (within tolerance) in order to match the object
# Highlight color - BGR color used for contour highlighting in an output image
OBJECT_DATA = [
    ("bolt", 0.2, 7500.0, (0, 0, 255)),
    ("long_bolt", 1.0, 9800.0, (0, 255, 0)),
    ("short_bolt", 1.0, 2000.0, (255, 0, 0)),
    ("nut", 0.1, 10000.0, (0, 255, 255)),
    ("dowel", 0.1, 8600.0, (255, 0, 255)),
    ("hexkey", 2.0, 10000.0, (255, 255, 0)),
]

# Name of the frame source
VIDEO_NAME = "b.mp4"


# Return a mean color of a region
def get_mean_color(region):
    return np.mean(np.reshape(region, (-1, 3)), axis=0)


# Return euclidean distance between 2 colors
def distance(a, b):
    return np.linalg.norm(a - b, axis=2)


# Return dictionary of reference objects' contours along with corresponding data
def build_contour_base(objects):
    contour_base = {}
    for name, contour_match_threshold, area_threshold, highlight_color in objects:
        img = cv.imread("data/" + name + "_mask.png", cv.IMREAD_GRAYSCALE)
        contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        contour_base[name] = {
            "contour": contours[0],
            "contour_match_threshold": contour_match_threshold,
            "contour_area_reference": area_threshold,
            "highlight_color": highlight_color,
        }
    return contour_base


# Dictionary of reference objects' contours along with corresponding data
contour_base = build_contour_base(OBJECT_DATA)
# Frame source
cap = cv.VideoCapture(VIDEO_NAME)
# The first frame and whether it exists
has_frame, frame = cap.read()
# The source has no frames, exit
if not has_frame:
    print("The frame source has no frames")
    exit(1)
# Mean color of the separator
separator_color = get_mean_color(
    frame[
        SEPARATOR_INIT_POS : SEPARATOR_INIT_POS + SEPARATOR_WIDTH,
        CONVEYOR_LEFT_BOUND:CONVEYOR_RIGHT_BOUND,
    ]
)
# Mean color of the slot's background
background_color = get_mean_color(
    frame[
        BACKGROUND_START_POS:BACKGROUND_END_POS,
        CONVEYOR_LEFT_BOUND:CONVEYOR_RIGHT_BOUND,
    ]
)
# Number of slots detected so far
slots_detected = 0
# Remaining number frames for which the separator detection will be skipped
remaining_separator_detection_deadzone = 0
# Main loop
while True:
    # Read next frame
    has_frame, frame = cap.read()
    # Stop if there are no more frames
    if not has_frame:
        break
    # Shrink frame for slot detection
    frame_scaled = ski.transform.rescale(
        frame, 1.0 / IMAGE_DIVISION_FACTOR, order=0, channel_axis=2, preserve_range=True
    )
    # Area of the frame that should contain an entire slot
    slot = frame[
        SEPARATOR_DETECTION_POS
        + SEPARATOR_WIDTH : SEPARATOR_DETECTION_POS
        + SEPARATOR_WIDTH
        + SLOT_WIDTH,
        CONVEYOR_LEFT_BOUND:CONVEYOR_RIGHT_BOUND,
    ]
    # Binary mask of separators
    separator_mask = (
        (distance(separator_color, frame_scaled) < SEPARATOR_MASK_THRESHOLD) * 255
    ).astype(np.uint8)
    # Detect slot only if outside the detection deadzone
    if remaining_separator_detection_deadzone == 0:
        # Wheter a slot was detected
        separator_detection = (
            np.mean(
                separator_mask[
                    SEPARATOR_DETECTION_POS
                    // IMAGE_DIVISION_FACTOR : (
                        SEPARATOR_DETECTION_POS + SEPARATOR_DETECTION_WIDTH
                    )
                    // IMAGE_DIVISION_FACTOR,
                    CONVEYOR_LEFT_BOUND
                    // IMAGE_DIVISION_FACTOR : CONVEYOR_RIGHT_BOUND
                    // IMAGE_DIVISION_FACTOR,
                ]
            )
            > SEPARATOR_DETECTION_THRESHOLD
        )
        # If slot is detected
        if separator_detection:
            # Stop slot detection for following number of frames
            remaining_separator_detection_deadzone = SEPARATOR_DETECTION_DEADZONE
            # Increment number of detected slots
            slots_detected += 1
            # Print detected slot number
            print("Slot number", slots_detected, "detected.")
            # Convert slot image to gray
            slot_mask = cv.cvtColor(slot, cv.COLOR_BGR2GRAY)
            # Binary mask of objects inside a slot
            slot_mask = (
                (slot_mask < SLOT_MASK_THRESHOLD_FACTOR * np.mean(background_color))
                * 255
            ).astype(np.uint8)
            # Remove unwanted objects on slot mask edges
            slot_mask = (
                (
                    ski.segmentation.clear_border(
                        ski.measure.label(slot_mask), buffer_size=CLEAR_BORDER_WIDTH
                    )
                    != 0
                )
                * 255
            ).astype(np.uint8)
            # Close objects' shapes
            slot_mask = ski.morphology.closing(
                slot_mask, ski.morphology.square(SLOT_CLOSING_SIZE)
            )
            # Remove small unwanted artifacts
            slot_mask = ski.morphology.opening(
                slot_mask, ski.morphology.square(SLOT_OPENING_SIZE)
            )
            # Detect contours of objects inside slot mask
            detected_contours, _ = cv.findContours(
                slot_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
            )
            if DEBUG:
                # In debug, convert slot mask to color for contour drawing
                slot_mask = cv.cvtColor(slot_mask, cv.COLOR_GRAY2BGR)
            else:
                # Initialize vector for match counting
                matches = np.zeros((len(contour_base)))
            # For every detected contour
            for detected_contour in detected_contours:
                # For every reference object
                for i, (name, object) in enumerate(contour_base.items()):
                    # Value denoting how similar the detected object's and reference object's contours are
                    contour_similarity = cv.matchShapes(
                        object["contour"], detected_contour, cv.CONTOURS_MATCH_I3, 0
                    )
                    # Area size of the detected object's contour
                    contour_area = cv.contourArea(detected_contour)
                    # If contour similarity is below threshold and contour areas are within tolerance, then the objects match
                    if (
                        contour_similarity < object["contour_match_threshold"]
                        and contour_area
                        > object["contour_area_reference"]
                        * (1.0 - CONTOUR_AREA_TOLERANCE)
                    ) and contour_area < object["contour_area_reference"] * (
                        1.0 + CONTOUR_AREA_TOLERANCE
                    ):
                        if DEBUG:
                            # In debug, draw contours which have 1 or more matches
                            cv.drawContours(
                                slot_mask,
                                detected_contour,
                                -1,
                                (0, 255, 0),
                                3,
                            )
                            # In debug, print successful match info in green
                            print("\033[32m", end="")
                        else:
                            # Increment matches count for matching reference object
                            matches[i] += 1
                            # Draw contour on output image
                            cv.drawContours(
                                slot,
                                detected_contour,
                                -1,
                                object["highlight_color"],
                                3,
                            )
                            # After finding a match, move to the next object
                            break
                    if DEBUG:
                        # In debug, print info about current match
                        print(
                            "Check for " + name + "\tcontour:",
                            contour_similarity,
                            "\tcontour_match_threshold:",
                            object["contour_match_threshold"],
                            "\tcontour_area:",
                            contour_area,
                            "\tcontour_area_reference:",
                            object["contour_area_reference"],
                            "\033[0m",
                        )
                if DEBUG:
                    print()
            if not DEBUG:
                # Print number of matches after processing all items inside a slot
                for i, name in enumerate(contour_base.keys()):
                    print(name + ":", matches[i].astype(int))
                print(
                    "no matches:", (len(detected_contours) - matches.sum()).astype(int)
                )
                print()
            if DEBUG:
                # In debug, show slot mask with matched contours
                cv.imshow("slot", slot_mask)
            else:
                # Show output image of a slot with classified objects
                cv.imshow("frame", slot)
            # Stop to show output (press q to continue)
            while True:
                if cv.waitKey(-1) == ord("q"):
                    break
        if DEBUG:
            # In debug, convert separator mask to color for separator check region drawing
            separator_mask = cv.cvtColor(separator_mask, cv.COLOR_GRAY2BGR)
            if separator_detection:
                # In debug, if separator is detected, color the separator check region green
                separator_mask[
                    SEPARATOR_DETECTION_POS
                    // IMAGE_DIVISION_FACTOR : (
                        SEPARATOR_DETECTION_POS + SEPARATOR_DETECTION_WIDTH
                    )
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
                # In debug, if separator is not detected, color the separator check region red
                separator_mask[
                    SEPARATOR_DETECTION_POS
                    // IMAGE_DIVISION_FACTOR : (
                        SEPARATOR_DETECTION_POS + SEPARATOR_DETECTION_WIDTH
                    )
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
        # If inside detection deadzone, skip slot detection and decrement number of remaining deadzone frames
        remaining_separator_detection_deadzone -= 1
        if DEBUG:
            # In debug, convert separator mask to color for separator check region drawing
            separator_mask = cv.cvtColor(separator_mask, cv.COLOR_GRAY2BGR)
            # In debug, if inside detection deadzone, color the separator check region blue
            separator_mask[
                SEPARATOR_DETECTION_POS
                // IMAGE_DIVISION_FACTOR : (
                    SEPARATOR_DETECTION_POS + SEPARATOR_DETECTION_WIDTH
                )
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
        # In debug, show separator mask with the separator check region
        cv.imshow("separator_mask", separator_mask)
        # In debug, show current frame
        cv.imshow("frame", frame_scaled)
    else:
        # Show current frame clipped to the slot area
        cv.imshow("frame", slot)
    # Interrupt program for rendering (press q to exit)
    if cv.waitKey(34) == ord("q"):
        break
# Cleanup
cv.destroyAllWindows()
cap.release()
