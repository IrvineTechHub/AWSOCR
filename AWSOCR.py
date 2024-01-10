import numpy as np
import platform
from PIL import ImageFont, ImageDraw, Image
 
import cv2
import boto3
import matplotlib.pyplot as plt


def plt_imshow(title='image', img=None, figsize=(8 ,5)):
    plt.figure(figsize=figsize)
 
    if type(img) == list:
        if type(title) == list:
            titles = title
        else:
            titles = []
 
            for i in range(len(img)):
                titles.append(title)
 
        for i in range(len(img)):
            if len(img[i].shape) <= 2:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_GRAY2RGB)
            else:
                rgbImg = cv2.cvtColor(img[i], cv2.COLOR_BGR2RGB)
 
            plt.subplot(1, len(img), i + 1), plt.imshow(rgbImg)
            plt.title(titles[i])
            plt.xticks([]), plt.yticks([])
 
        plt.show()
    else:
        if len(img.shape) < 3:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgbImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
 
        plt.imshow(rgbImg)
        plt.title(title)
        plt.xticks([]), plt.yticks([])
        plt.show()

def put_text(image, text, x, y, color=(0, 255, 0), font_size=22):
    if type(image) == np.ndarray:
        color_coverted = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(color_coverted)
 
    if platform.system() == 'Darwin':
        font = 'AppleGothic.ttf'
    elif platform.system() == 'Windows':
        font = 'malgun.ttf'
        
    image_font = ImageFont.truetype(font, font_size)
    font = ImageFont.load_default()
    draw = ImageDraw.Draw(image)
 
    draw.text((x, y), text, font=image_font, fill=color)
    
    numpy_image = np.array(image)
    opencv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
 
    return opencv_image

ACCESS_KEY = "AKIAXYKJQVYVENQW7IBU"
SECRET_KEY = "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"
REGION = "ap-northeast-2"
 
client = boto3.client("rekognition", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION)

path = 'test.jpg'
imageData = open(path, "rb").read()

response = client.detect_text(Image={"Bytes": imageData})
detections = response["TextDetections"]

img = cv2.imread(path)
roi_img = img.copy()
    
(h, w) = img.shape[:2]
for detection in detections:
    text = detection["DetectedText"]
    textType = detection["Type"]
    poly = detection["Geometry"]["Polygon"]
    
    if "line" == textType.lower():
        tlX = int(poly[0]["X"] * w)
        tlY = int(poly[0]["Y"] * h)
        trX = int(poly[1]["X"] * w)
        trY = int(poly[1]["Y"] * h)
        brX = int(poly[2]["X"] * w)
        brY = int(poly[2]["Y"] * h)
        blX = int(poly[3]["X"] * w)
        blY = int(poly[3]["Y"] * h)
 
        pts = ((tlX, tlY), (trX, trY), (brX, brY), (blX, blY))
        topLeft = pts[0]
        topRight = pts[1]
        bottomRight = pts[2]
        bottomLeft = pts[3]
 
        cv2.line(roi_img, topLeft, topRight, (0,255,0), 2)
        cv2.line(roi_img, topRight, bottomRight, (0,255,0), 2)
        cv2.line(roi_img, bottomRight, bottomLeft, (0,255,0), 2)
        cv2.line(roi_img, bottomLeft, topLeft, (0,255,0), 2)
 
        roi_img = put_text(roi_img, text, topLeft[0], topLeft[1] - 10, font_size=30)
 
        print(text)
 
plt_imshow(["Original", "ROI"], [img, roi_img], figsize=(16, 10))

