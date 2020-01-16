import cv2
import dlib
from math import hypot

cap = cv2.VideoCapture(0)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("dat_file/shape_predictor_68_face_landmarks.dat")


def apply_filter(mask_loc, landmarks, key_points, mask_w_mul, mask_h_mul,top_left_w_adj,top_left_h_adj,
                 frame):
    input_img = cv2.imread(mask_loc, -1)
    orig_mask = input_img[:, :, 3]
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    input_img = input_img[:, :, 0:3]

    facial_pts = list()
    for pt in key_points:
        x_y_pts = (landmarks.part(pt).x, landmarks.part(pt).y)
        facial_pts.append(x_y_pts)

    if len(key_points) == 3:
        mask_w = int((hypot(facial_pts[1][0] - facial_pts[2][0],
                            facial_pts[1][1] - facial_pts[2][1])) * mask_w_mul)
        mask_h = int(mask_w * mask_h_mul)
    else:
        mask_w = int((hypot(facial_pts[0][0] - facial_pts[1][0],
                            facial_pts[0][1] - facial_pts[1][1])) * mask_w_mul)
        mask_h = int(mask_w * mask_h_mul)

    resized_img = cv2.resize(input_img, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (mask_w, mask_h), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (mask_w, mask_h), interpolation=cv2.INTER_AREA)

    top_left = (int(facial_pts[0][0] - mask_w/top_left_w_adj),
                int(facial_pts[0][1] - mask_h/top_left_h_adj))

    area = frame[top_left[1]:top_left[1] + mask_h,
           top_left[0]:top_left[0] + mask_w]

    roi_bg = cv2.bitwise_and(area, area, mask=mask_inv)
    roi_fg = cv2.bitwise_and(resized_img, resized_img, mask=mask)

    dst = cv2.add(roi_bg, roi_fg)

    frame[top_left[1]:top_left[1] + mask_h, top_left[0]:top_left[0] + mask_w] = dst

    return frame


def main(img):
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        for face in faces:
            landmarks = predictor(gray, face)

            try:
                if img == 'camera':
                    continue
                elif img == 'cat':
                    frame = apply_filter('filters/cat.png', landmarks, [17,26], 1.2, 1.1,
                                         9, 2, frame)
                elif img == 'dog':
                    frame = apply_filter('filters/dog.png', landmarks, [18,25], 1.4, 1.1,
                                         8, 2, frame)
                elif img == 'moustache':
                    frame = apply_filter('filters/moustache.png', landmarks, [30,3,13], 1, 0.5,
                                         2, 5, frame)
                elif img == 'specs':
                    frame = apply_filter('filters/specs_1.png', landmarks, [36,45], 1.7, 0.5,
                                         5, 2, frame)
                elif img == 'pinkMask':
                    frame = apply_filter('filters/pinkMask.png', landmarks, [17,26], 1.8, 1,
                                         4, 1.9, frame)
                elif img == 'pigNose':
                    frame = apply_filter('filters/pig_nose.png', landmarks, [30,31,35], 1.7, 1,
                                         2, 2, frame)
                elif img =='whiteMask':
                    frame = apply_filter('filters/whiteMask.png', landmarks, [17, 26], 1.3, 1.2,
                                         9, 3.5, frame)


            except Exception as e:
                print(e)

        cv2.imshow("Frame", frame)

        if cv2.waitKey(10) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

