import cv2  

# 加载预训练的Haarcascade级联分类器  
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
 
  
# 读取图像  
img = cv2.imread('person.jpg')  
  
# 转换为灰度图像  
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
  
# 在灰度图像中检测人脸  
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))  
  
# 在图像上画出人脸矩形框  
for (x, y, w, h) in faces:  
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)  
  
# 显示图像  
cv2.imshow('img', img)  
cv2.waitKey(0)  
cv2.destroyAllWindows()