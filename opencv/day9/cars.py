import cv2
import numpy as np

'''
1. 读取视频
2. 去除背景，提取前景
'''
cap = cv2.VideoCapture("/Users/duchengtai/Library/Mobile Documents/com~apple~CloudDocs/code- iCloud/opencv/day9/video.mp4")

# 视频大小（720，1280，3）

# cv2.namedWindow("video",cv2.WINDOW_NORMAL)
# cv2.resizeWindow("video",60,60)

# 检测线的高度
line_height = 550

# 线的偏移量
off_set = 7

# 获取去除背景的对象
bgsubmog = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=False)


# 定义形态学操作的核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

# 识别车辆检测框的阈值
min_w = 90
min_h = 90


def center(x,y,w,h):
    x1 = (x+x+w)/2
    y1 = (y+y+h)/2
    return x1,y1

# 存放有效车辆的数组
cars = []

# 有效车辆数目
carno = 0

while True:
    ret,frame = cap.read()

    # print(frame.shape)
    # break

    if ret == True: #读到了视频帧

        # 灰度化
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        # 进行双边滤波去噪（比高斯滤波更保留边缘）
        blur = cv2.bilateralFilter(gray, 9, 75, 75)

        #对视频去除背景，得到掩码
        mask = bgsubmog.apply(blur)


        # 腐蚀操作
        erode = cv2.erode(mask,kernel=kernel,iterations=1)

        # 膨胀操作还原大小
        dilate = cv2.dilate(erode,kernel=kernel,iterations=3)

        # 闭操作取出内部噪声
        close = cv2.morphologyEx(dilate,cv2.MORPH_CLOSE,kernel=kernel)

        contours, hierarchy = cv2.findContours(close,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        # 画一条检测线
        cv2.line(frame,(10,line_height),(1270,line_height),(255,0,0),thickness=3)
        

        for (i,c) in enumerate(contours): # 小的轮廓并不是车辆，加一个阈值用以消除轮廓
            (x,y,w,h) = cv2.boundingRect(c)

            isValid = (w >= min_w) and (h >= min_h)

            if isValid: #当有效的时候才进行绘制
                cv2.rectangle(frame,(x,y),(x+w,y+h),color=(0,255,0),thickness=2)

                # 绘制完成后求有效车辆的中心点
                car_center = center(x,y,w,h)

                cars.append(car_center)

                for (x,y) in cars:
                    if (y > line_height-off_set) and (y < line_height+off_set): # 线的有效区域
                        carno += 1
                        cars.remove((x,y)) # 已经统计过该车辆，没有意义，移除
                        print(carno)


        cv2.putText(frame,'CAR_COUNT:'+ str(carno),(500,60),cv2.FONT_HERSHEY_PLAIN,fontScale=3,color=(0,0,255),thickness=5)
        # cv2.imshow("video",close)
        cv2.imshow("VIDEO",frame)


    if cv2.waitKey(10) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

