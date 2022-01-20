import cv2
import numpy as np

from utils import get_rot_matrix
from obj_parser import parse_obj

objects = parse_obj("./monkey.obj")


camera_matrix = np.array([
    [800, 0, 400, 0],
    [0, 800, 400, 0],
    [0, 0, 1, 0]
])

win = "Object"
cv2.namedWindow("Object")


def do_nothing(value):
    pass


cv2.createTrackbar("roll", win, 30, 180, do_nothing)
cv2.createTrackbar("pitch", win, 30, 180, do_nothing)
cv2.createTrackbar("yaw", win, 30, 180, do_nothing)

cv2.createTrackbar("camera roll", win, 0, 180, do_nothing)
cv2.createTrackbar("camera pitch", win, 0, 180, do_nothing)
cv2.createTrackbar("camera yaw", win, 0, 180, do_nothing)
cv2.createTrackbar("distance", win, 400, 2000, do_nothing)


cv2.setTrackbarMin("roll", win, -180)
cv2.setTrackbarMin("pitch", win, -180)
cv2.setTrackbarMin("yaw", win, -180)
cv2.setTrackbarMin("camera roll", win, -180)
cv2.setTrackbarMin("camera pitch", win, -180)
cv2.setTrackbarMin("camera yaw", win, -180)
cv2.setTrackbarMin("distance", win, -2000)

while True:
    alpha = np.radians(90)
    beta = np.radians(90)
    gamma = np.radians(0)
    R_model_to_camera = get_rot_matrix(alpha, beta, gamma)
    cam_vert_rot = np.radians(0)
    cam_horiz_rot = np.radians(0)
    camera_roll = np.radians(cv2.getTrackbarPos("camera roll", "Object"))
    camera_pitch = np.radians(cv2.getTrackbarPos("camera pitch", "Object"))
    camera_yaw = np.radians(cv2.getTrackbarPos("camera yaw", "Object"))
    Rwc = get_rot_matrix(camera_roll, camera_pitch,
                         camera_yaw)  # Camera to world
    distz = cv2.getTrackbarPos("distance", "Object")
    Twc = np.vstack(
        [
            np.hstack([Rwc, np.array([[0], [0], [0]])]),
            [0, 0, 0, 1]
        ]
    )

    T_model_to_camera = np.vstack(
        [
            np.hstack([R_model_to_camera, np.array([[0], [0], [distz]])]),
            [0, 0, 0, 1]
        ]
    )

    roll = np.radians(cv2.getTrackbarPos("roll", "Object"))
    pitch = np.radians(cv2.getTrackbarPos("pitch", "Object"))
    yaw = np.radians(cv2.getTrackbarPos("yaw", "Object"))
    R_model = get_rot_matrix(roll, pitch, yaw)

    # Transformation matrix to rotate and and translate the model w.r.t it's centre point
    T_model = np.vstack(
        [
            np.hstack([R_model, np.array([[0], [0], [0]])]),
            [0, 0, 0, 1]
        ]
    ) 
    img = np.zeros((800, 800), dtype=np.uint8)
    for points in objects:
        points = np.array(points)

        points = camera_matrix @ np.linalg.inv(Twc) @ T_model_to_camera @ T_model @ points[:, :, None]
        points = points.squeeze()
        points = points / (points[:, -1, None])

        for i in range(len(points)):
            img = cv2.line(img, points[i-1, :-1].astype('int'),
                           points[i, :-1].astype('int'), (255, 255, 0), 2)

    cv2.imshow(win, img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('c'):
        cv2.imwrite("./screenshot.png", img)
