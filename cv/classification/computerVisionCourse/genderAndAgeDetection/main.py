import cv2

facePrototype = "face.pbtxt"
faceModel = "face.pb"
agePrototype = "age.prototxt"
ageModel = "age.caffemodel"
genderPrototype = "gender.prototxt"
genderModel = "gender.caffemodel"
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)  # RGB三通道图像均值

faceNet = cv2.dnn.readNet(faceModel, facePrototype)
ageNet = cv2.dnn.readNet(ageModel, agePrototype)
genderNet = cv2.dnn.readNet(genderModel, genderPrototype)


def getFaceImg(net, img, conf_threshold=0.8):
    imgFace = img.copy()
    imgH = imgFace.shape[0]
    imgW = imgFace.shape[1]
    roi = cv2.dnn.blobFromImage(imgFace, 1.0, (300, 300), [104, 117, 123])
    net.setInput(roi)
    detections = net.forward()
    faceList = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * imgW)
            y1 = int(detections[0, 0, i, 4] * imgH)
            x2 = int(detections[0, 0, i, 5] * imgW)
            y2 = int(detections[0, 0, i, 6] * imgH)
            faceList.append([x1, y1, x2, y2])
            cv2.rectangle(imgFace, (x1, y1), (x2, y2), (0, 255, 0), int(round(imgH / 150)), 8)
    return imgFace, faceList


video = cv2.VideoCapture(0)
padding = 20
while cv2.waitKey(1) < 0:
    ok_, frame = video.read()
    if not ok_:
        break
    resultImg, faceBoxes = getFaceImg(faceNet, frame, 0.8)
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1)
        , max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]
        # 对获得的图片取其左右各加20像素的图像
        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES)
        genderNet.setInput(blob)
        genderPrediction = genderNet.forward()
        gender = genderList[genderPrediction[0].argmax()]
        ageNet.setInput(blob)
        agePrediction = ageNet.forward()
        age = ageList[agePrediction[0].argmax()]
        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0] + 20, faceBox[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 255), 2, cv2.LINE_AA)
        cv2.imshow("Detecting age and gender", resultImg)
        cv2.waitKey(22)
