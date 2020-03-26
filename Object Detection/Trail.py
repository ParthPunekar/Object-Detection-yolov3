import cv2
import objDetect as obd

ob = obd.ObjDetect(0.40, 0.40, 'coco.names', 'yolov3.cfg', 'yolov3.weights', 'Image')

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    outs = ob.setup(frame)
    ob.postProcess(frame, outs)
    key = cv2.waitKey(1)
    if key == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        break


'''
frame = cv2.imread('Test-image-2.jpg')
outs = ob.setup(frame)
ob.postProcess(frame, outs)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''