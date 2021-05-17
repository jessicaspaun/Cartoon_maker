import cv2
import dlib
from imutils.imutils import face_utils
import imutils.imutils
import numpy as np

# Load the detector
detector = dlib.get_frontal_face_detector()
# Load the predictor
predictor = dlib.shape_predictor("C:/Users/srdes/PycharmProjects/shape_predictor_68_face_landmarks.dat")
#read the image
frame = cv2.imread("face.jpg")


# Scale image down
scale_percent = 25  # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(frame, dim)
gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
rects = detector(gray)

# loop over the face detections
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
        clone = img.copy()
        cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)
        # loop over the subset of facial landmarks, drawing the
        # specific face part
        for (x, y) in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # extract the ROI of the face region as a separate image
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = img[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)
        # show the particular face part
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)
    # visualize all facial landmarks with a transparent overlay
    output = face_utils.visualize_facial_landmarks(img, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)

# # Convert image into grayscale
# gray = cv2.cvtColor(src=img, code=cv2.COLOR_BGR2GRAY)
# # Use detector to find landmarks
# # Use detector to find landmarks
# faces = detector(gray)
# for face in faces:
#     x1 = face.left() # left point
#     y1 = face.top() # top point
#     x2 = face.right() # right point
#     y2 = face.bottom() # bottom point
#     # Create landmark object
#     landmarks = predictor(image=gray, box=face)
#     # Loop through all the points
#     for n in range(0, 68):
#         x = landmarks.part(n).x
#         y = landmarks.part(n).y
#         # Draw a circle
#         cv2.circle(img=img, center=(x, y), radius=3, color=(0, 255, 0), thickness=-1)

# Scale image down
scale_percent = 25  # percent of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(frame, dim)
# show the image
cv2.imshow(winname="Face", mat=img)
# Wait for a key press to exit
cv2.waitKey(delay=0)
# Close all windows
cv2.destroyAllWindows()
