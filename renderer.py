import cv2
import numpy as np
import argparse

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("-i", "--input", type=str, default="./plane.obj")
args = arg_parser.parse_args()

input_filename = args.input

from utils import get_rotation_matrix
from obj_parser import parse_obj

objects = parse_obj(input_filename)
# from plane import objects # To future me: Keep it here for debugging purposes

camera_matrix = np.array([
    [800, 0, 400, 0],
    [0, 800, 400, 0],
    [0, 0, 1, 0]
])

win = "Object"
cv2.namedWindow("Object", cv2.WINDOW_NORMAL)

orthographic_projection = False
anaglyph = False
anaglyph_dist_between_eyes_scale = 5 # It makes 1 unit in world coordinates 20 cm


def do_nothing(value):
    pass


cv2.createTrackbar("roll", win, 30, 180, do_nothing)
cv2.createTrackbar("pitch", win, 30, 180, do_nothing)
cv2.createTrackbar("yaw", win, 30, 180, do_nothing)

cv2.createTrackbar("camera roll", win, 0, 180, do_nothing)
cv2.createTrackbar("camera pitch", win, 0, 180, do_nothing)
cv2.createTrackbar("camera yaw", win, 0, 180, do_nothing)
cv2.createTrackbar("distance", win, 20, 2000, do_nothing)


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
    R_model_to_camera = get_rotation_matrix(alpha, beta, gamma)
    cam_vert_rot = np.radians(0)
    cam_horiz_rot = np.radians(0)
    camera_roll = np.radians(cv2.getTrackbarPos("camera roll", "Object"))
    camera_pitch = np.radians(cv2.getTrackbarPos("camera pitch", "Object"))
    camera_yaw = np.radians(cv2.getTrackbarPos("camera yaw", "Object"))
    Rwc = get_rotation_matrix(camera_roll, camera_pitch,
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
    R_model = get_rotation_matrix(roll, pitch, yaw)

    # Transformation matrix to rotate and and translate the model w.r.t it's centre point
    T_model = np.vstack(
        [
            np.hstack([R_model, np.array([[0], [0], [0]])]),
            [0, 0, 0, 1]
        ]
    ) 
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    for points in objects:
        points = np.array(points)

        if not anaglyph:
            points_wrt_camera = np.linalg.inv(Twc) @ T_model_to_camera @ T_model @ points[:, :, None]
        
            if orthographic_projection:
                points_wrt_camera[:, 2] = distz
            
            points = camera_matrix @ points_wrt_camera
            points = points.squeeze()
            points = points / (points[:, -1, None])

            for i in range(len(points)):
                img = cv2.line(img, points[i-1, :-1].astype('int'),
                            points[i, :-1].astype('int'), (255, 255, 255), 2)
        else:
            orthographic_projection = False
            points_wrt_camera_left = np.linalg.inv(Twc) @ T_model_to_camera @ T_model @ points[:, :, None]            
            points_wrt_camera_right = points_wrt_camera_left + np.array([[-0.05],[0],[0],[0]]) * anaglyph_dist_between_eyes_scale
            
            points_left = camera_matrix @ points_wrt_camera_left
            points_left = points_left.squeeze()
            points_left = points_left / (points_left[:, -1, None])

            for i in range(len(points_left)):
                img = cv2.line(img, points_left[i-1, :-1].astype('int'),
                            points_left[i, :-1].astype('int'), (0, 0, 255), 2)
            
            points_right = camera_matrix @ points_wrt_camera_right
            points_right = points_right.squeeze()
            points_right = points_right / (points_right[:, -1, None])

            for i in range(len(points_right)):
                img = cv2.line(img, points_right[i-1, :-1].astype('int'),
                            points_right[i, :-1].astype('int'), (255, 255, 0), 2)
    # Show some status
    if orthographic_projection:
        cv2.putText(img, 'Orthographic', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)
    else:
        cv2.putText(img, 'Perspective', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)
    
    cv2.imshow(win, img)
    key = cv2.waitKey(1) & 0xFF
    if key in [27, ord('q')]:
        break
    if key == ord('p'):
        orthographic_projection = not orthographic_projection
    if key == ord('c'):
        cv2.imwrite("./screenshot.png", img)
    if key == ord('3'):
        anaglyph = not anaglyph
