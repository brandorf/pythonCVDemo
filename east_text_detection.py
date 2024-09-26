import cv2
import numpy as np


def detect_text(video_path):
    net = cv2.dnn.readNet('frozen_east_text_detection.pb')
    frames_with_text = []
    frame_index = 0

    while True:
        ret, frame = video_path.read()
        if not ret:
            break

        blob = cv2.dnn.blobFromImage(frame, 1.0, (320, 320), (123.68, 116.78, 103.94), True, False)
        net.setInput(blob)
        scores, geometry = net.forward(['feature_fusion/Conv_7/Sigmoid', 'feature_fusion/concat_3'])

        if np.any(scores > 0.5):
            frames_with_text.append(frame_index)

        frame_index += 1

    return frames_with_text