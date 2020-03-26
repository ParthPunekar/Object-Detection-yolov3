import cv2
import numpy as np

class ObjDetect:

    def __init__(self, confThreshold, nmsThreshold, classFile, modelConf, modelWeights, windowName = 'Default Name', inpWidth = 416, inpHeight = 416):

        self.confThreshold = confThreshold
        self.nmsThreshold = nmsThreshold
        self.classFile = classFile
        self.modelConf = modelConf
        self.modelWeights = modelWeights
        self.windowName = windowName
        self.inpWidth = inpWidth
        self.inpHeight = inpHeight
        self.classes = None
        
    def setup(self, frame):

        with open(self.classFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        blob = cv2.dnn.blobFromImage(frame, 1/255, (416, 416), [0, 0, 0], 1, crop = False)
        self.net = cv2.dnn.readNetFromDarknet(self.modelConf, self.modelWeights)
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        self.net.setInput(blob)
        cv2.namedWindow(self.windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.windowName, 1000, 1000)
        return self.net.forward(self.getOutputNames(self.net))

    def getOutputNames(self, net):

        layerNames = net.getLayerNames()
        return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def drawPred(self, frame, classId, conf, left, top, right, bottom):

        cv2.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 1)
        
        label = '%.2f' % conf
        
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s'%(self.classes[classId], label)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(self.windowName, frame)

    def postProcess(self, frame, outs):

        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIDs = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]

                if confidence > self.confThreshold:
                    centerX = int(detection[0] * frameWidth)
                    centerY = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(centerX - width / 2)
                    top = int(centerY - height / 2)
                    classIDs.append(classID)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)

        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIDs[i], confidences[i], left, top, left + width, top + height)
