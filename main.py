import cv2

imageCaptureFolder = "captures/"
cam = cv2.VideoCapture(0)

result, image = cam.read()

if result:
    
    # show image
    cv2.imshow("Say Cheese!", image)

    fileName = input("Choose a file name for this image:")

    # save image
    cv2.imwrite(f"{imageCaptureFolder}{fileName}.png", image)

else:

    print("Error reading image")

input("Type 'Y' to exit")