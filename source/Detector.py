import tensorflow.compat.v1 as tf
import tensornets as nets
import cv2
import numpy as np
from trackedObject import trackedObject

tf.disable_v2_behavior()
# make the program run with GPU and not with CPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# setup YOLOV3
inputs = tf.placeholder(tf.float32, [None, 416, 416, 3])
model = nets.YOLOv3COCO(inputs, nets.Darknet19)
output = None
classes = {'0': 'person'}
list_of_classes = [0]
trackedObjects = []

masterID = 0
frames = 0
radiusThreshHold = 150.0
framesThreshHold = 25
skipFrames = 15
# radiusThreshHold = 10.0
# framesThreshHold = 50
# skipFrames = 30

with tf.Session() as sess:
    sess.run(model.pretrained())
    cap = cv2.VideoCapture(r"E:\IT състезания\PeopleDetection\test_input\0001-1547.mp4")  # setup the video
    #cap = cv2.VideoCapture(r"E:\Download\0001-1547.mp4")  # setup the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    skipFrames = fps
    print(fps)
    while (cap.isOpened()):  # start looping through the video frame by frame
        # resizing the frame
        ret, frame = cap.read()
        frames += 1
        for tObj in trackedObjects:
            tObj.count()
            if tObj.frames >= framesThreshHold:
                trackedObjects.remove(tObj)

        if frame is None:
            break
        img = cv2.resize(frame, (800, 800))
        imge = np.array(img).reshape(-1, 800, 800, 3)
        preds = sess.run(model.preds, {inputs: model.preprocess(imge)})

        # take bounding boxes coordinates
        boxes = model.get_boxes(preds, imge.shape[1:3])
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)

        cv2.resizeWindow('image', 1024, 768)
        boxes = np.array(boxes)
        # draw bounding boxes and count how many people are on the image
        for j in list_of_classes:
            count = 0
            if str(j) in classes:
                lab = classes[str(j)]
            if len(boxes[j]) != 0 and lab == "person":

                for i in range(len(boxes[j])):
                    box = boxes[j][i]
                    x = box[0]
                    y = box[1]

                    if box[4] >= 0.5:
                        count += 1
                        # print(str(len(boxes[j])) + ":" + str(len(trackedObjects)))
                        tempTracked = None
                        for tObj in trackedObjects:
                           # print(f"{tObj.x} : {tObj.y} : {tObj.id}")
                            if tObj.insideRadius(x, y, radiusThreshHold):
                                tObj.clear()
                                tObj.update(x, y)
                                tempTracked = tObj

                        if tempTracked is None and frames % skipFrames == 0:
                            masterID += 1
                            tempTracked = trackedObject(x, y, masterID)
                            trackedObjects.append(tempTracked)

                        if tempTracked is not None:
                            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)
                            cv2.putText(img, lab + ":" + str(tempTracked.id), (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 0, 255), lineType=cv2.LINE_AA)
                            #cv2.circle(img, (x, y), int(radiusThreshHold), (255, 255, 0), 2)

        cv2.putText(img, "persons : " + str(count) + " : " + str(masterID), (1, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("image", img)
        # saving output every frame
        if output is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            output = cv2.VideoWriter(r"E:\IT състезания\PeopleDetection\output.avi", fourcc, fps, (1024, 768), True)
        else:
            output.write(cv2.resize(img, (1024, 768)))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
output.release()
cv2.destroyAllWindows()
