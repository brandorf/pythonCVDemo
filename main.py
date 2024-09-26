import cv2
import cvui
import os
from east_text_detection import detect_text

WINDOW_NAME = 'Text Detection UI'
cvui.init(WINDOW_NAME)

video_path = ''
frames_with_text = []

def select_video():
    global video_path
    video_path = cv2.VideoCapture(
        cv2.samples.findFile(cv2.utils.selectFile('Select Video File', '', '', 'Video Files (*.mp4 *.avi)')))
    if not video_path.isOpened():
        print("Error: Could not open video.")
        video_path = ''

def process_video():
    global frames_with_text
    if video_path:
        frames_with_text = detect_text(video_path)

def display_frame(frame_index):
    if video_path:
        video_path.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = video_path.read()
        if ret:
            cv2.imshow('Frame with Detections', frame)
            cv2.waitKey(0)

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    frame = cv2.imread('background.jpg')
    if frame is None:
        frame = cv2.imread('fallback_background.jpg')  # Ensure you have a fallback image

    cvui.beginRow(frame, 50, 50, 400, 200)
    cvui.window(frame, 0, 0, 400, 200, 'Text Detection UI')

    if cvui.button(frame, 10, 30, 'Select Video'):
        select_video()

    if cvui.button(frame, 10, 70, 'Process'):
        process_video()

    if frames_with_text:
        for i, frame_index in enumerate(frames_with_text):
            if cvui.button(frame, 10, 110 + i * 40, f'Frame {frame_index}'):
                display_frame(frame_index)

    cvui.endRow()

    cvui.update()
    cv2.imshow(WINDOW_NAME, frame)

    if cv2.waitKey(20) == 27:
        break

cv2.destroyAllWindows()