from ctypes import *
import random
import os
import cv2
import time
import darknet
import argparse
from threading import Thread, enumerate
from queue import Queue
from numpy import *
import numpy as np

from MyQueue import *
from Filter import *
from Headline import *
# >>>total size
# width=416
# height=416

# myclass:
class POINT:
    x = 0.0
    y = 0.0
    size = 0.0

    def __init__(self, xIn, yIn, sizeIn):
        self.x = xIn
        self.y = yIn
        self.size = sizeIn

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection")
    parser.add_argument("--input", type=str, default=video,
                        help="video source. If empty, uses webcam 0 stream")
    parser.add_argument("--out_filename", type=str, default="",
                        help="inference video name. Not saved if empty")
    parser.add_argument("--weights", default=weights,
                        help="yolo weights path")
    parser.add_argument("--dont_show", action='store_true',
                        help="windown inference display. For headless systems")
    parser.add_argument("--ext_output", action='store_true',
                        help="display bbox coordinates of detected objects")
    parser.add_argument("--config_file", default=cfg,
                        help="path to config file")
    parser.add_argument("--data_file", default=data,
                        help="path to data file")
    parser.add_argument("--thresh", type=float, default=thresh,
                        help="remove detections with confidence below this value")
    return parser.parse_args()


def str2int(video_path):
    """
    argparse returns and string althout webcam uses int (0, 1 ...)
    Cast to int if needed
    """
    try:
        return int(video_path)
    except ValueError:
        return video_path


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise (ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise (ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise (ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if str2int(args.input) == str and not os.path.exists(args.input):
        raise (ValueError("Invalid video path {}".format(os.path.abspath(args.input))))


def set_saved_video(input_video, output_video, size):
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    fps = int(input_video.get(cv2.CAP_PROP_FPS))
    video = cv2.VideoWriter(output_video, fourcc, fps, size)
    return video


def video_capture(frame_queue, darknet_image_queue):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        img_for_detect = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(img_for_detect, frame_resized.tobytes())
        darknet_image_queue.put(img_for_detect)
    cap.release()


def inference(darknet_image_queue, detections_queue, fps_queue):
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        prev_time = time.time()
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=args.thresh)
        detections_queue.put(detections)
        fps = int(1 / (time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
        darknet.print_detections(detections, args.ext_output)
        darknet.free_image(darknet_image)
    cap.release()


def drawing(frame_queue, detections_queue, fps_queue, curruntPointList, buffer_queue):
    random.seed(3)  # deterministic bbox colors
    video = set_saved_video(cap, args.out_filename, (width, height))
    while cap.isOpened():
        frame_resized = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame_resized is not None:
            ## loop every box:
            print("\n\n\n\n\n>>>my data process begin here:")
            curruntPointList.clear()
            for detection in detections:
                point = POINT(detection[2][0], detection[2][1],
                              detection[2][2] * detection[2][3])
                curruntPointList.append(point)
                # print("\n" + str(detection[2][0]))
            ## loop end & point processing
            pointProcessing(curruntPointList, buffer_queue)
            ###########################

            image = darknet.draw_boxes(detections, frame_resized, class_colors)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            if args.out_filename is not None:
                video.write(image)
            if not args.dont_show:
                cv2.imshow('Inference', image)
            if cv2.waitKey(fps) == 27:
                break
    cap.release()
    video.release()
    cv2.destroyAllWindows()


def pointProcessing(curruntPointList, buffer_queue):
    print("hello,pointProcessing")
    num = len(curruntPointList)
    # calculate all boxes' average size:
    average = averageSize(curruntPointList, num)

    print("Detected:" + str(num))
    if num > 0:
        for point in curruntPointList:
            '''
            print("point" + str(num)
                  + ">>>"
                  + " X_" + str(point.x)
                  + ",Y_" + str(point.y)
                  + ",SIZE_" + str(point.size))
            '''
        print("arage:" + str(average))
    filter = FILTER()
    scanBoxes = filter.scan(curruntPointList)
    curruntArray = Boxes2Array(scanBoxes)
    bufferProcess(buffer_queue, curruntArray)


def Boxes2Array(scanBoxes):
    if len(scanBoxes) == 0:
        # no object detected
        curruntArray = np.zeros((15, 15))
    else:
        # object detected
        list = []
        for box in scanBoxes:
            list.append(round(box.RATIO, 1))
        curruntArray = np.array(list)
        curruntArray = curruntArray.reshape((15, 15))
    return curruntArray
    # print(curruntArray)


def bufferProcess(buffer_queue, curruntArray):
    buffer_queue.push(curruntArray)
    # check queue:
    print("check queue")
    '''
    cnt = 0
    list = buffer_queue.check()
    for array in list:
        cnt += 1
        print(str(cnt))
        print(array)
    '''
    buffer_queue.check()

'''
def averageSize(PointList, num):
    average = 0.0
    if num > 0:
        for point in PointList:
            average += point.size
        average /= num
    return average
'''

if __name__ == '__main__':
    curruntPointList = []
    buffer_queue = myQueue(buffer_queueMaxSize)

    frame_queue = Queue()
    darknet_image_queue = Queue(maxsize=1)
    detections_queue = Queue(maxsize=1)
    fps_queue = Queue(maxsize=1)

    args = parser()
    check_arguments_errors(args)
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights,
        batch_size=1
    )
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    input_path = str2int(args.input)
    cap = cv2.VideoCapture(input_path)
    Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
    Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
    Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue, curruntPointList, buffer_queue)).start()
