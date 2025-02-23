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
heart_sunglasses = cv2.imread("props/heart_sunglasses.png", cv2.IMREAD_UNCHANGED) # heart sunglasses prop
hat = cv2.imread("props/hat1.png", cv2.IMREAD_UNCHANGED) # hat prop

# start video capture
cap = cv2.VideoCapture(1) # change to 0 if error is caused

# declare variables
filter_active = False # filter active flag
program_run = True # program run flag
bg_active = False # background active flag
selection = "" # selection
bg = None # selected background

# menu function
def menu():
    global filter_active
    global program_run
    global bg_active
    global selection
    global prop_list
    global bg

    while True:
        print("\nMenu:")
        print("1. Activate Filter")
        print("2. Deactivate Filter")
        print("3. Select filter\n")
        print("4. Activate Background")
        print("5. Deactivate Background")
        print("6. Select Background\n")
        print("7. Exit")
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
            prop_list = []
            print("\nSelect a prop [Enter numbers separated by spaces to choose multiple]:")
            print("1. Hat")
            print("2. Heart Sunglasses")
            selection = input("Enter choice: ").split(" ")
            for num in selection:
                if num == "1":
                    prop_list.append(hat)
                    print("Hat selected.")
                elif num == "2":
                    prop_list.append(heart_sunglasses)
                    print("Heart Sunglasses selected.")
                else:
                    print("Invalid choice. Returning to menu...")
                    selection = ""
        elif choice == "4":
            if bg is None or bg.size == 0:
                print("No background selected.")
            else:
                bg_active = True
                print("Background activated.")
        elif choice == "5":
            bg_active = False
            print("Background deactivated.")
        elif choice == "6":
            print("\nBackgrounds:")
            print("1. Zoom")
            print("2. SST")
            selection = input("Enter choice: ")
            if selection == "1":
                bg_active = True
                bg = cv2.imread("backgrounds/zoom.jpeg")
            elif selection == "2":
                bg_active = True
                bg = cv2.imread("backgrounds/SST.jpg")
            elif selection == "":
                bg = None
                print("No background selected.")
            else:
                print("Invalid choice. Returning to menu...")
                selection = ""
        elif choice == "7":
            print("Exiting...")
            program_run = False
            break
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

    # start background code if background is activated
    if bg_active:
        # resize background to fit frame
        bg_resized = cv2.resize(bg, (frame.shape[1], frame.shape[0]))

        # create mask
        mask = np.zeros_like(frame[:, :, 0])

        # loop through detected faces
        for face in faces:
            # get face coordinates
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            mask[y:y+h, x:x+w] = 255

            # apply mask to frame
            mask = cv2.GaussianBlur(mask, (21, 21), 0) # blur mask for smoother edges
            alpha = mask.astype(float) / 255 # convert mask to float
            for c in range(0, 3):
                frame[:, :, c] = (1- alpha) * bg_resized[:, :, c] + alpha * frame[:, :, c]

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

            # find where to place the prop
            center_eye = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

            # loop through selected props
            for prop in prop_list:
                # check if prop is hat
                if prop.shape == hat.shape:
                    # calculate forehead width
                    left_forehead = shape[19] # left forehead landmark
                    right_forehead = shape[24] # right forehead landmark
                    forehead_width = int(np.linalg.norm(np.array(left_forehead) - np.array(right_forehead)))

                    # calculate hat width and height
                    hat_width = int(forehead_width * 3.0)
                    hat_height = int(prop.shape[0] * (hat_width / prop.shape[1]))
                    resized_hat = cv2.resize(prop, (hat_width, hat_height), interpolation=cv2.INTER_AREA)

                    # place hat on forehead
                    hat_center_x = (left_forehead[0] + right_forehead[0]) // 2 # calculate center x-coord of forehead
                    hat_center_y = (left_forehead[1] + right_forehead[1]) // 2 # calculate center y-coord of forehead
                    hat_y_offset = int(hat_height * 0.65) # y-offset for hat
                    hat_top_left = (hat_center_x - hat_width // 2, hat_center_y - hat_y_offset)

                    # rotation for hat
                    angle = np.degrees(np.arctan2(right_forehead[1] - left_forehead[1], right_forehead[0] - left_forehead[0])) # Angle between two points
                    x_offset_adjustment = int(np.sin(np.radians(angle)) * hat_width * 0.3)  # Multiplies by 0.3 to control offset range
                    hat_top_left = (hat_top_left[0] + x_offset_adjustment, hat_top_left[1])
                    rotation_matrix = cv2.getRotationMatrix2D((hat_width // 2, hat_height // 2), -angle, 1) # Rotate hat
                    rotated_hat = cv2.warpAffine(resized_hat, rotation_matrix, (hat_width, hat_height), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)) # Apply rotation

                    # allow hat coords to be negative
                    hat_x1, hat_y1 = max(0, hat_top_left[0]), max(0, hat_top_left[1])
                    hat_x2, hat_y2 = min(frame.shape[1], hat_x1 + hat_width), min(frame.shape[0], hat_y1 + hat_height)

                    # find visible region of hat
                    hat_region = frame[hat_y1:hat_y2, hat_x1:hat_x2]
                    hat_alpha = rotated_hat[:, :, 3] / 255.0
                    hat_rgb = rotated_hat[:, :, :3]

                    try:
                        for c in range(0, 3):  # Blend only the visible part
                            hat_region[:, :, c] = (hat_alpha * hat_rgb[:, :, c] + (1.0 - hat_alpha) * hat_region[:, :, c])
                    except:
                        cv2.putText(frame, "Please stand further away from the camera.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


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
                    
                    try:
                        for c in range(0, 3):
                            glasses_region[:, :, c] = (glasses_alpha * glasses_rgb[:, :, c] + (1.0 - glasses_alpha) * glasses_region[:, :, c])
                    except:
                        cv2.putText(frame, "Please stand further away from the camera.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #show the video
    cv2.imshow("Output", frame)

    # detect when esc key is pressed
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# end program
cap.release()
cv2.destroyAllWindows()