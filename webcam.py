import cv2
import dlib
from imutils.imutils import face_utils
import imutils.imutils
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector()

# Load the predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# # Face landmark ids
# FACIAL_LANDMARKS_IDXS = OrderedDict([
#     ("mouth", (48, 68)),
#     ("right_eyebrow", (17, 22)),
#     ("left_eyebrow", (22, 27)),
#     ("right_eye", (36, 42)),
#     ("left_eye", (42, 48)),
#     ("nose", (27, 35)),
#     ("jaw", (0, 17))
# ])

# read the image
cap = cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    # Convert image into grayscale
    gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)

    # Use detector to find landmarks
    rects = detector(gray)

    # loop over the face detections
    for (i, rect) in enumerate(rects):
        # determine the facial landmarks for the face region, then
        # convert the landmark (x, y)-coordinates to a NumPy array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        # loop over the face parts individually
        for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
            # clone the original image so we can draw on it, then
            # display the name of the face part on the image
            clone = frame.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            # loop over the subset of facial landmarks, drawing the
            # specific face part
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

            # extract the ROI of the face region as a separate image
            (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
            roi = frame[y:y + h, x:x + w]
            roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
            # show the particular face part
            cv2.imshow("ROI", roi)
            cv2.imshow("Image", clone)
            cv2.waitKey(0)
        # visualize all facial landmarks with a transparent overlay
        output = face_utils.visualize_facial_landmarks(frame, shape)
        cv2.imshow("Image", output)
        cv2.waitKey(0)

    # # show the image
    # cv2.imshow(winname="Face", mat=frame)
    #
    # # Exit when escape is pressed
    # if cv2.waitKey(delay=1) == 27:
    #     break

# When everything done, release the video capture and video write objects
cap.release()

# Close all windows
cv2.destroyAllWindows()