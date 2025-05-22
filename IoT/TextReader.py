import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import pytesseract
import re
import time
from MAIN import record

# Set path to tesseract executable if not in PATH (Change this according to your system)
pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract' 


# Open webcam
cap = cv2.VideoCapture(0)

res_x = int(3840)
res_y = int(2160)

# Optional: Set resolution if you want consistency
cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)



print("Press 'q' to quit the feed.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    roi_x, roi_y, roi_w, roi_h = 915, 360, 250, 50  # Example ROI coordinates (x, y, width, height)

    # Crop the frame to the ROI
    roi_frame = frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    # Convert the ROI to grayscale (useful for OCR processing)
    gray_frame = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)

    # Invert the image (white text on black background)
    inverted_frame = cv2.bitwise_not(gray_frame)

    # Apply thresholding to improve text contrast for OCR
    _, thresholded = cv2.threshold(inverted_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Run OCR to extract text from the thresholded ROI
    #extracted_text = pytesseract.image_to_string(thresholded)
    extracted_text = pytesseract.image_to_string(inverted_frame)
    try:
        # Display detected text (optional debugging)
        print("Detected Text:", extracted_text)
    except UnicodeEncodeError:
        print("Detected Text: Unprintable")
        
    # Use regex to filter specific text patterns
    if re.search(r"REAMER\s*20", extracted_text, re.IGNORECASE):
       print("Reamer Found")
       cap.release()
       record(20000, "reamer")
       cap = cv2.VideoCapture(0)
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)

    if re.search(r"END\s*MILL\s*26\.5", extracted_text, re.IGNORECASE):
       print("Emill Found")
       cap.release()
       record(20000, "emill")
       cap = cv2.VideoCapture(0)
       cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_x)
       cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_y)

    # Optional: Show thresholded frame for debugging (you can also show the ROI for debugging)
    #cv2.imshow('Live Camera Feed', thresholded)  # Show thresholded frame for debugging
    cv2.imshow('Live Camera Feed', inverted_frame)  # Show thresholded frame for debugging

    # Exit on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()



