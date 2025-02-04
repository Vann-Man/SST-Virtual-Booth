# library imports
from imutils import face_utils
import dlib
import cv2
import numpy as np

# load face detector and shape predictor
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# declare props
heart_sunglasses = cv2.imread("heart_sunglasses.png", cv2.IMREAD_UNCHANGED) # heart sunglasses prop

# some data from online (chatgpt)
KNOWN_EYE_DISTANCE = 6.3  # average adult eye distance in cm
KNOWN_PIXEL_DISTANCE = 120  # pixel distance where eye distance was measured
KNOWN_REAL_DISTANCE = 50  # cm distance from camera to face

# start video capture
cap = cv2.VideoCapture(1)

# filter active flag
filter_active = False

# menu function
def menu():
    global filter_active
    while True:
        print("\nMenu:")
        print("1. Activate Filter")
        print("2. Deactivate Filter")
        print("3. Exit")
        choice = input("Enter choice: ")
        
        # input validation/choices
        if choice == "1":
            filter_active = True
            print("Filter activated.")
        elif choice == "2":
            filter_active = False
            print("Filter deactivated.")
        elif choice == "3":
            print("Exiting...")
            # end program
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            print("Invalid choice. Try again.")

# allows menu to run while the main/prop code is running
import threading
menu_thread = threading.Thread(target=menu, daemon=True)
menu_thread.start()

while True:
    # read frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert to grayscale
    faces = detector(gray, 0) # detect faces

    # start prop filter code if filter is activated
    if filter_active:
        # loop through detected faces
        for face in faces:
            # get facial landmarks
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # get eye landmarks
            left_eye = shape[36]
            right_eye = shape[45]

            # calculate eye distance in pixels
            eye_distance_pixels = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

            # calculate estimated real-world distance
            estimated_distance_cm = (KNOWN_REAL_DISTANCE * KNOWN_PIXEL_DISTANCE) / eye_distance_pixels

            # calculate how much to scale the prop
            sun_width = int(eye_distance_pixels * 2.2)
            sun_height = int(heart_sunglasses.shape[0] * (sun_width / heart_sunglasses.shape[1]))

            # resize prop
            resized_sunglasses = cv2.resize(heart_sunglasses, (sun_width, sun_height), interpolation=cv2.INTER_AREA)

            # calculate rotation angle
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            rotation_matrix = cv2.getRotationMatrix2D((sun_width // 2, sun_height // 2), -angle, 1)
            rotated_sunglasses = cv2.warpAffine(resized_sunglasses, rotation_matrix, (sun_width, sun_height),
                                                flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0, 0))

            # find where to place the prop
            center_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            top_left = (center_eye[0] - sun_width // 2, center_eye[1] - sun_height // 2)

            x1, y1 = max(0, top_left[0]), max(0, top_left[1])
            x2, y2 = min(frame.shape[1], top_left[0] + sun_width), min(frame.shape[0], top_left[1] + sun_height)

            # overlay the prop
            overlay_sunglasses = rotated_sunglasses[
                max(0, -top_left[1]):min(sun_height, frame.shape[0] - top_left[1]),
                max(0, -top_left[0]):min(sun_width, frame.shape[1] - top_left[0])
            ]

            if overlay_sunglasses.shape[0] > 0 and overlay_sunglasses.shape[1] > 0:
                overlay_image = frame[y1:y2, x1:x2]
                alpha_s = overlay_sunglasses[:, :, 3] / 255.0
                alpha_l = 1.0 - alpha_s

                for c in range(0, 3):
                    overlay_image[:, :, c] = (alpha_s * overlay_sunglasses[:, :, c] + alpha_l * overlay_image[:, :, c])

                frame[y1:y2, x1:x2] = overlay_image

    #show the video
    cv2.imshow("Output", frame)

    # detect when esc key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# end program
cap.release()
cv2.destroyAllWindows()