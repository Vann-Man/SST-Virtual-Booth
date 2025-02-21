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
hat = cv2.imread("hat1.png", cv2.IMREAD_UNCHANGED) # hat prop

# some data from online (chatgpt)
KNOWN_EYE_DISTANCE = 6.3  # average adult eye distance in cm
KNOWN_PIXEL_DISTANCE = 120  # pixel distance where eye distance was measured
KNOWN_REAL_DISTANCE = 50  # cm distance from camera to face

# start video capture
cap = cv2.VideoCapture(1)

# declare variables
filter_active = False # filter active flag
program_run = True # program run flag
selection = "" # prop selection

# menu function
def menu():
    global filter_active
    global program_run
    global selection
    global prop_list

    while True:
        print("\nMenu:")
        print("1. Activate Filter")
        print("2. Deactivate Filter")
        print("3. Exit")
        print("4. Select filter")
        choice = input("Enter choice: ")
        
        # input validation/choices
        if choice == "1":
            if selection == "":
                print("No prop selected.")
            else:
                filter_active = True
                print("Filter activated.")
        elif choice == "2":
            filter_active = False
            print("Filter deactivated.")
        elif choice == "3":
            print("Exiting...")
            program_run = False
            break
        elif choice == "4":
            prop_list = []
            print("\nSelect a prop [Enter numbers separated by spaces to choose multiple]:")
            print("1. Hat")
            print("2. Heart Sunglasses")
            selection = input("Enter choice: ").split(" ")
            for num in selection:
                if num == "1":
                    prop_list.append(hat)
                elif num == "2":
                    prop_list.append(heart_sunglasses)
        else:
            print("Invalid choice. Try again.")

# allows menu to run while the main/prop code is running
import threading
menu_thread = threading.Thread(target=menu, daemon=True)
menu_thread.start()

while program_run == True:
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

            # find where to place the prop
            center_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            for prop in prop_list:
                if prop.shape == hat.shape:
                    left_forehead = shape[19]
                    right_forehead = shape[24]

                    forehead_width = int(np.linalg.norm(np.array(left_forehead) - np.array(right_forehead)))

                    hat_width = int(forehead_width * 3.3)
                    hat_height = int(prop.shape[0] * (hat_width / prop.shape[1]))
                    resized_hat = cv2.resize(prop, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

                    hat_center_x = (left_forehead[0] + right_forehead[0]) // 2
                    hat_center_y = (left_forehead[1] + right_forehead[1]) // 2

                    hat_y_offset = int(hat_height * 0.65)
                    hat_top_left = (hat_center_x - hat_width // 2, hat_center_y - hat_y_offset)

                    angle = np.degrees(np.arctan2(right_forehead[1] - left_forehead[1], right_forehead[0] - left_forehead[0]))

                    x_offset_adjustment = int(np.sin(np.radians(angle)) * hat_width * 0.3)  # Multiplies by 0.3 to control offset range
                    hat_top_left = (hat_top_left[0] + x_offset_adjustment, hat_top_left[1])

                    rotation_matrix = cv2.getRotationMatrix2D((hat_width // 2, hat_height // 2), -angle, 1)
                    rotated_hat = cv2.warpAffine(resized_hat, rotation_matrix, (hat_width, hat_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                    # allow hat coords to be negative
                    hat_x1, hat_y1 = hat_top_left[0], hat_top_left[1]
                    hat_x2, hat_y2 = hat_x1 + hat_width, hat_y1 + hat_height

                    # find visible region of hat
                    hat_region_x1 = max(0, -hat_x1)
                    hat_region_y1 = max(0, -hat_y1)
                    hat_region_x2 = hat_width - max(0, hat_x2 - frame.shape[1])
                    hat_region_y2 = hat_height - max(0, hat_y2 - frame.shape[0])

                    # find visible region of frame
                    frame_x1 = max(0, hat_x1)
                    frame_y1 = max(0, hat_y1)
                    frame_x2 = min(frame.shape[1], hat_x2)
                    frame_y2 = min(frame.shape[0], hat_y2)

                    hat_region = frame[hat_y1:hat_y2, hat_x1:hat_x2]
                    hat_alpha = rotated_hat[hat_region_y1:hat_region_y2, hat_region_x1:hat_region_x2, 3] / 255.0
                    hat_rgb = rotated_hat[hat_region_y1:hat_region_y2, hat_region_x1:hat_region_x2, :3]
                    
                    if hat_region.shape[:2] == hat_alpha.shape[:2]:
                        for c in range(0, 3):  # Blend only the visible part
                            hat_region[:, :, c] = (hat_alpha * hat_rgb[:, :, c] + (1.0 - hat_alpha) * hat_region[:, :, c])


                elif prop.shape == heart_sunglasses.shape:
                    left_eye = shape[36]
                    right_eye = shape[45]
                    eye_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

                    eye_distance = np.linalg.norm(np.array(left_eye) - np.array(right_eye))

                    glasses_width = int(eye_distance * 2.0)
                    glasses_height = int(prop.shape[0] * (glasses_width / prop.shape[1]))
                    resized_glasses = cv2.resize(prop, (glasses_width, glasses_height), interpolation=cv2.INTER_AREA)

                    glasses_center_x = eye_center[0]
                    glasses_center_y = eye_center[1]

                    glasses_y_offset = int(glasses_height * 0.5)
                    glasses_top_left = (glasses_center_x - glasses_width // 2, glasses_center_y - glasses_y_offset)

                    angle = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
                    rotation_matrix = cv2.getRotationMatrix2D((glasses_width // 2, glasses_height // 2), -angle, 1)
                    rotated_glasses = cv2.warpAffine(resized_glasses, rotation_matrix, (glasses_width, glasses_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

                    glasses_x1, glasses_y1 = max(0, glasses_top_left[0]), max(0, glasses_top_left[1])
                    glasses_x2, glasses_y2 = min(frame.shape[1], glasses_x1 + glasses_width), min(frame.shape[0], glasses_y1 + glasses_height)

                    glasses_region = frame[glasses_y1:glasses_y2, glasses_x1:glasses_x2]
                    glasses_alpha = rotated_glasses[:, :, 3] / 255.0
                    glasses_rgb = rotated_glasses[:, :, :3]
                    
                    for c in range(0, 3):
                        glasses_region[:, :, c] = (glasses_alpha * glasses_rgb[:, :, c] + (1.0 - glasses_alpha) * glasses_region[:, :, c])

    #show the video
    cv2.imshow("Output", frame)

    # detect when esc key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# end program
cap.release()
cv2.destroyAllWindows()