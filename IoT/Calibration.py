import cv2
import numpy as np
import glob
import time 

# Chessboard dimensions (number of inner corners per chessboard row and column)
chessboard_dims = (9, 6)  # 9x6 grid of inner corners

# Prepare object points (3D points in real world space)
obj_points = []  # 3D points in real world space
img_points = []  # 2D points in image plane

# Prepare object points (like (0,0,0), (1,0,0), (2,0,0), ..., (8,5,0)) based on the chessboard size
objp = np.zeros((np.prod(chessboard_dims), 3), dtype=np.float32)
objp[:, :2] = np.indices(chessboard_dims).T.reshape(-1, 2)

# Open webcam feed
cap = cv2.VideoCapture(0)  # Open default camera

# Initialize image counter
image_count = 0
max_images = 50  # Set the number of images for calibration

print("Capturing images for calibration. Press 'q' to quit.")

while image_count < max_images:
    ret, img = cap.read()
    if not ret:
        break
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, chessboard_dims, None)

    if ret:
        # If corners are found, add object points and image points
        img_points.append(corners)
        obj_points.append(objp)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, chessboard_dims, corners, ret)
        cv2.imshow('Chessboard', img)
        image_count += 1  # Increment image counter
        print(f"Captured {image_count}/{max_images} images")

        # Wait for 1 second before capturing the next image
        time.sleep(1)

    # Display the captured frame
    cv2.imshow('Capture Chessboard Image', img)

    # Wait for the user to press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close windows
cap.release()
cv2.destroyAllWindows()

# Perform the calibration once enough images are captured
if len(img_points) >= max_images:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    # Calibration result: Camera matrix (intrinsic parameters) and distortion coefficients
    print("Camera matrix:", mtx)
    print("Distortion coefficients:", dist)

    # Save calibration parameters for later use
    np.savez('camera_calibration_params.npz', mtx=mtx, dist=dist)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"Not enough images for calibration. Only captured {len(img_points)} images.")
