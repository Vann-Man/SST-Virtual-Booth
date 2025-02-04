from imutils import face_utils
import dlib
import cv2
import numpy as np

p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)
heart_sunglasses = cv2.imread("heart3.png", cv2.IMREAD_UNCHANGED)

KNOWN_EYE_DISTANCE = 6.3  # In cm (average adult eye distance)
KNOWN_PIXEL_DISTANCE = 120  # Measured eye distance in pixels at a known distance
KNOWN_REAL_DISTANCE = 50  # Distance from camera in cm when eye distance was measured

cap = cv2.VideoCapture(1)

filter_active = True

def menu():
    global filter_active
    while True:
        print("\nMenu:")
        print("1. Activate Filter")
        print("2. Deactivate Filter")
        print("3. Exit")
        choice = input("Enter choice: ")
        
        if choice == "1":
            filter_active = True
            print("Filter activated.")
        elif choice == "2":
            filter_active = False
            print("Filter deactivated.")
        elif choice == "3":
            print("Exiting...")
            cap.release()
            cv2.destroyAllWindows()
            break
        else:
            print("Invalid choice. Try again.")

import threading
menu_thread = threading.Thread(target=menu, daemon=True)
menu_thread.start()

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if filter_active:
        for face in faces:
            shape = predictor(gray, face)
            shape = face_utils.shape_to_np(shape)

            # Get eye coordinates
            left_eye = shape[36]
            right_eye = shape[45]

            # Calculate eye distance in pixels
            eye_distance_pixels = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

            # Estimate real-world distance
            estimated_distance_cm = (KNOWN_REAL_DISTANCE * KNOWN_PIXEL_DISTANCE) / eye_distance_pixels

            # Calculate size of the sunglasses dynamically based on eye distance
            sun_width = int(eye_distance_pixels * 2.2)  # Scale width dynamically
            sun_height = int(heart_sunglasses.shape[0] * (sun_width / heart_sunglasses.shape[1]))

            # Resize sunglasses
            resized_sunglasses = cv2.resize(heart_sunglasses, (sun_width, sun_height), interpolation=cv2.INTER_AREA)

            # Calculate rotation angle
            angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
            rotation_matrix = cv2.getRotationMatrix2D((sun_width // 2, sun_height // 2), -angle, 1)
            rotated_sunglasses = cv2.warpAffine(resized_sunglasses, rotation_matrix, (sun_width, sun_height),
                                                flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT,
                                                borderValue=(0, 0, 0, 0))

            # Find center position to overlay sunglasses
            center_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)
            top_left = (center_eye[0] - sun_width // 2, center_eye[1] - sun_height // 2)

            # Define overlay region
            x1, y1 = max(0, top_left[0]), max(0, top_left[1])
            x2, y2 = min(frame.shape[1], top_left[0] + sun_width), min(frame.shape[0], top_left[1] + sun_height)

            # Extract region of interest
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

    cv2.imshow("Output", frame)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()