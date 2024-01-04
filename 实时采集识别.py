import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from PIL import ImageFont, ImageDraw, Image


def close_window(event):
    if event.key == "escape":
        plt.close()


fontpath = "STSONG.TTF"
font = ImageFont.truetype(fontpath, 20)
font2 = {'family': 'STSONG', 'size': 22}
matplotlib.rc('font', **font2)
f = open("training/object_names.txt", encoding='utf-8')
object_names = [r.strip() for r in f.readlines()]
f.close()
mode = cv2.dnn.readNetFromDarknet(
    "training/yolov3.cfg", "training/yolov3.weights")
capture = cv2.VideoCapture(0)
fig = plt.figure()
fig.canvas.mpl_connect("key_press_event", close_window)
while True:
    ret, image = capture.read()
    if image is None:
        break
    imgH, imgW = image.shape[:2]
    out_layers = mode.getUnconnectedOutLayersNames()
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    mode.setInput(blob)
    layer_results = mode.forward(out_layers)
    ptime, _ = mode.getPerfProfile()
    title_text = '完成预测时间：%.2f ms' % (ptime * 1000 / cv2.getTickFrequency())
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

    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.title(title_text)
    plt.imshow(img)
    plt.axis('off')
    plt.show()
    plt.pause(0.001)
    # 添加以下代码
    if cv2.waitKey(1) == 27:  # 27对应于"escape"键的ASCII码
        break
# 添加以下代码
cv2.destroyAllWindows()
capture.release()
