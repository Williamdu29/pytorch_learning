import cv2
import numpy as np

'''
d:kernel_size
sigmaColor:灰度距离里的标准差
sigmaSpace:空间距离里的标准差
'''
# img = cv2.imread("/Users/duchengtai/Library/Mobile Documents/com~apple~CloudDocs/code- iCloud/opencv/day5/ed.jpeg")

img = cv2.VideoCapture(0)
if not img.isOpened():
    print("无法打开摄像头")
    exit()

while True:
    # 读取一帧
    ret, frame = img.read()
    if not ret:
        print("无法接收视频帧")
        break
    
    # 显示当前帧
    cv2.imshow("Camera", frame)

    img_after = cv2.bilateralFilter(frame,d=7,sigmaColor=200,sigmaSpace=50)

    cv2.imshow("ed",img_after)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

# 释放资源
img.release()
cv2.destroyAllWindows()