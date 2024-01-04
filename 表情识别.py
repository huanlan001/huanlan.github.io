import cv2
import numpy as np
# 加载预训练的深度学习模型
model_path = 'path/to/deploy.prototxt'
weights_path = 'path/to/res10_300x300_ssd_iter_140000.caffemodel'
net = cv2.dnn.readNetFromCaffe(model_path, weights_path)
# 读取输入图像
image_path = 'path/image/hong.jpg'
image = cv2.imread(image_path)
(h, w) = image.shape[:2]
# 构建 blob，并将其传递给网络
blob = cv2.dnn.blobFromImage(cv2.resize(
    image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
net.setInput(blob)
detections = net.forward()
# 遍历检测结果并绘制边界框
for i in range(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
# 过滤掉低置信度的检测结果
if confidence > 0.5:
    # 计算边界框的坐标
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
(startX, startY, endX, endY) = box.astype("int")
# 绘制边界框及置信度
text = "{:.2f}%".format(confidence * 100)
y = startY - 10 if startY - 10 > 10 else startY + 10
cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
cv2.putText(image, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
# 显示结果图像
cv2.imshow("Face Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
