import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
fontpath = "STSONG.TTF"
font = ImageFont.truetype(fontpath, 20)
# 加载物体识别模型
f = open("training/object_names.txt", encoding='utf-8')
object_names = [r.strip() for r in f.readlines()]
f.close()
mode = cv2.dnn.readNetFromDarknet(
    "training/yolov3.cfg", "training/yolov3.weights")
# 加载人脸检测模型
face_cascade = cv2.CascadeClassifier(
    'training/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
while True:
    ret, image = cap.read()

    if not ret:
        break

    imgH, imgW = image.shape[:2]

    out_layers = mode.getUnconnectedOutLayersNames()
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    mode.setInput(blob)
    layer_results = mode.forward(out_layers)

    result_boxes = []
    result_scores = []
    result_name_id = []

    for layer in layer_results:
        for box in layer:
            probs = box[5:]
            class_id = np.argmax(probs)
            prob = probs[class_id]
            if prob > 0.5:
                box = box[:4] * np.array([imgW, imgH, imgW, imgH])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                result_boxes.append([x, y, int(width), int(height)])
                result_scores.append(float(prob))
                result_name_id.append(class_id)

    draw_boxes = cv2.dnn.NMSBoxes(result_boxes, result_scores, 0.6, 0.3)

    if len(draw_boxes) > 0:
        for i in draw_boxes.ravel():
            (x, y) = (result_boxes[i][0], result_boxes[i][1])
            (w, h) = (result_boxes[i][2], result_boxes[i][3])
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            text = object_names[result_name_id[i]] + \
                '\n{:.1%}'.format(result_scores[i])
            img_pil = Image.fromarray(image)
            draw = ImageDraw.Draw(img_pil)
            draw.text((x + 5, y), text, font=font, fill=(0, 0, 255))
            image = np.array(img_pil)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faceRects = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=3, minSize=(50, 50))

    # if len(faceRects):
    #     for (x, y, w, h) in faceRects:
    #         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Video", image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()
cap.release()
