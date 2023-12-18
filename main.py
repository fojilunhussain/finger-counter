import cv2
from datetime import datetime

imageCaptureFolder = "captures/"
cam = cv2.VideoCapture(0)

result, image = cam.read()

if result:
    # show image
    cv2.imshow("Say Cheese!", image)

    # append date and time to capture name
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    capture_name = f"capture_{timestamp}.png"

    # save image with appended capture name
    cv2.imwrite(f"{imageCaptureFolder}{capture_name}", image)

else:
    print("Error reading image")

input("Hit Enter to exit.")