import cv2
import time

def record_video(duration_minutes, output_filename="output.avi", camera_index=0, frame_width=640, frame_height=480, fps=30):
    cap = cv2.VideoCapture(camera_index)  # Open camera
    cap.set(3, frame_width)  # Set width
    cap.set(4, frame_height)  # Set height

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec
    out = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()
    duration_seconds = duration_minutes * 60  # Convert minutes to seconds

    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return

    print(f"Recording started for {duration_minutes} minutes...")

    while int(time.time() - start_time) < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        out.write(frame)  # Save frame to video
        cv2.imshow("Recording...", frame)  # Display recording in progress

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop recording
            break

    print("Recording finished.")
    cap.release()
    out.release()
    cv2.destroyAllWindows()


record_video(duration_minutes=2, output_filename="logitech_video.avi")


def check_available_cameras(max_index=5):
    """
    Check available camera indices.

    :param max_index: Maximum index to check (default: 5)
    """
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Camera found at index {i}")
            cap.release()
        else:
            print(f"❌ No camera found at index {i}")

#check_available_cameras()