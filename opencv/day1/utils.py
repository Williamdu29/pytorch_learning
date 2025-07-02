import cv2
# 展示图片的代码封装
def cv_show(name,img):
    cv2.imshow(name,img)
    key = cv2.waitKey(0)
    if key == ord('q'):
        print("销毁图片...")
        cv2.destroyAllWindows()