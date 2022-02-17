import os
import cv2
import pytesseract
from matplotlib import pyplot as plt

tesseract_path = "C:/Program Files/Tesseract-OCR/tesseract.exe"
pytesseract.pytesseract.tesseract_cmd = tesseract_path

# Read image
folder = 'images'
#file = 'ui_001.jpg'
file = 'ui_andromeda_1.jpg'
img = cv2.imread(os.path.join(folder, file))

# Convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Binarize image
_, img_th1 = cv2.threshold(img_gray, 0, 255, cv2.THRESH_OTSU)

# Find biggest contour (screen)
contours, hierarchy = cv2.findContours(img_th1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
biggest_contour = max(contours, key=cv2.contourArea)
# Get bounding box around screen
x, y, w, h = cv2.boundingRect(biggest_contour)

# Get region of interest
img_roi = img[y:y+h, x:x+w]

# Convert to gray scale
img_roi_gray = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)

# Binarize and invert image
_, img_th2 = cv2.threshold(img_roi_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

plt.figure(1)
# subplot 1
plt.subplot(1, 4, 1)
plt.title("Original")
plt.axis("off")
plt.imshow(img)

# subplot 2
plt.subplot(1, 4, 2)
plt.title("Gray")
plt.axis("off")
plt.imshow(img_gray, cmap='gray')

# subplot 3
plt.subplot(1, 4, 3)
plt.title("Thresh1 Image")
plt.axis("off")
plt.imshow(img_th1, cmap='gray')

# subplot 4
plt.subplot(1, 4, 4)
plt.title("ROI")
plt.axis("off")
plt.imshow(img_roi)
plt.show()

# ========================= Detect words =========================
# resize image
img_resized = cv2.resize(img_roi, (640, 860), interpolation=cv2.INTER_AREA)

# Convert ROI to grayscale
img_roi_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

# Threshold with otsu
_, img_th2 = cv2.threshold(img_roi_gray, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY_INV)

# Create kernel
rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))

# Applying dilation on the threshold image
img_dilated = cv2.dilate(img_th2, rect_kernel, iterations=1)

# Finding contours
contours, hierarchy = cv2.findContours(img_th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

plt.figure(2)
# subplot 1
plt.subplot(1, 4, 1)
plt.title("ROI")
plt.axis("off")
plt.imshow(img_roi)

# subplot 2
plt.subplot(1, 4, 2)
plt.title("Resized")
plt.axis("off")
plt.imshow(img_resized)

# subplot 3
plt.subplot(1, 4, 3)
plt.title("Thresh2 Image")
plt.axis("off")
plt.imshow(img_th2, cmap='gray')

# subplot 4
plt.subplot(1, 4, 4)
plt.title("Dilated image")
plt.axis("off")
plt.imshow(img_dilated, cmap='gray')
plt.show()

# Create a copy of image
img_result = img_resized.copy()

text_list = []
area_list = []

for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)

    # Drawing a rectangle on copied image
    rect = cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Cropping the text block for giving input to OCR
    cropped = img_result[y:y + h, x:x + w]

    # Apply OCR on the cropped image
    text = pytesseract.image_to_string(cropped)
    if text:
        text_list.append(text)

        # get contour area
        area = cv2.contourArea(cnt)
        area_list.append(area)

        print(text)
        print(area)

print(text_list)

# resize image
#img_resized = cv2.resize(img_result, (640, 860), interpolation=cv2.INTER_AREA)

# Create a copy of image
#img_result = img_roi.copy()
# # Get raw data from image
# raw_data = pytesseract.image_to_data(img_result)
# for count, data in enumerate(raw_data.splitlines()):
#     # Exclude first row (column names)
#     if count > 0:
#         # Parse data to list
#         data = data.split()
#         # Check if there's text
#         if len(data) == 12:
#             # Get text coordinates
#             x, y, w, h = int(data[6]), int(data[7]), int(data[8]), int(data[9])
#             # Get text
#             content = data[11]
#             cv2.rectangle(img_result, (x, y), (x+w, y+h), (0, 255, 0), 1)
#             cropped = img_result[y:y+h, x:x+w]
#             cv2.putText(img_result, content, (x, y + 20), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
#             print(content)
#             cv2.imshow("cropped", cropped)
#             cv2.waitKey(0)
#
# plt.imshow(img_result)
# plt.show()

#640 x 860 - old pics
# NUNO
#3024 x 4032 - new pics
# ANDRE
#3468 x 4624 - new pics
