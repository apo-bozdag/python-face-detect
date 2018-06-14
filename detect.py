import cv2
import sys
import os, os.path, glob

os.chdir(r"C:\Users\Abdullah\Desktop\tripian_image_python\111")
types = ('*.png', '*.jpg', "*.jpeg")
files_grabbed = []
for files in types:
    files_grabbed.extend(glob.glob(files))

totalFiles = len(files_grabbed)
print("Found " + str(totalFiles) + " images")

for file in files_grabbed:

    imagePath = file
    cascPath = "../xml/haarcascade_frontalface_alt.xml"

    # Create the haar cascade
    faceCascade = cv2.CascadeClassifier(cascPath)

    # Read the image
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    print(imagePath)
    print("Found {0} faces!".format(len(faces)))
    if format(len(faces)) >= str(1):
        os.rename(imagePath, "../images/unusable/"+imagePath)
    else:
        os.rename(imagePath, "../images/usable/" + imagePath)

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Faces found", image)
cv2.waitKey(0)
