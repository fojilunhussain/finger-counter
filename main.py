import cv2



cam = cv2.VideoCapture(0)

result, image = cam.read()

if result:
    
    # show image
    cv2.imshow("img", image)

    # save image
    cv2.imwrite("img.png", image)

else:

    print("Error reading image")

input("Type 'Y' to exit")