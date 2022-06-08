import mediapipe as mp
import cv2
import os
import numpy as np

def file_count(path):
    file_list = os.listdir(path)
    file_cnt = len(file_list)
    return file_cnt


mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
path = 'C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_capture_data/sick'

actions=['sick','heart']

for idx, action in enumerate(actions):
    if idx == 1:
        break
    data = []
    for i in range(file_count(path)):
        IMAGE_FILES = ['C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_capture_data/'+ action +'/frame'+str(i)+'.jpg']
        print(IMAGE_FILES)
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5) as hands:
          for file in IMAGE_FILES:
            # 이미지를 읽어 들이고, 보기 편하게 이미지를 좌우 반전합니다.
            image = cv2.flip(cv2.imread(file), 1)
            # 작업 전에 BGR 이미지를 RGB로 변환합니다.
            results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            # 손으로 프린트하고 이미지에 손 랜드마크를 그립니다.
            print('Handedness:', results.multi_handedness)
            if not results.multi_hand_landmarks:
              continue
            image_height, image_width, _ = image.shape
            annotated_image = image.copy()
            for hand_landmarks in results.multi_hand_landmarks:
                joint = np.zeros((21, 4))
                for j, lm in enumerate(hand_landmarks.landmark):
                    joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]
                v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]
                v = v2 - v1

                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                angle = np.arccos(np.einsum('nt,nt->n',
                                            v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                            v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))
                angle = np.degrees(angle)

                angle_label = np.array([angle], dtype=np.float32)
                angle_label = np.append(angle_label, idx)

                d = np.concatenate([joint.flatten(), angle_label])

                data.append(d)

    data = np.array(data)
    print(data)
    print(action, data.shape)
    np.save(os.path.join('C:/Users/kimsungwook/Desktop/dohwajo/dataset/keypoint_data', f'raw_{action}'), data)

    seq_length = 30
    full_seq_data = []
    for seq in range(len(data) - seq_length):
        full_seq_data.append(data[seq:seq + seq_length])

    full_seq_data = np.array(full_seq_data)
    print(action, full_seq_data.shape)
    np.save(os.path.join('C:/Users/kimsungwook/Desktop/dohwajo/dataset/keypoint_data', f'seq_{action}'), full_seq_data)




            #     print('hand_landmarks:', hand_landmarks)
            #     print(
            #         f'Index finger tip coordinates: (',
            #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
            #         f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
            #     )
            #     mp_drawing.draw_landmarks(
            #         annotated_image,
            #         hand_landmarks,
            #         mp_hands.HAND_CONNECTIONS,
            #         mp_drawing_styles.get_default_hand_landmarks_style(),
            #         mp_drawing_styles.get_default_hand_connections_style())
            # cv2.imwrite('C:/Users/kimsungwook/Desktop/dohwajo/dataset/mediapipe_data/sick/' + str(i) + '.jpg', cv2.flip(annotated_image, 1))



### mediapipe 출력
# for i in range(file_count(path)):
#     IMAGE_FILES = ['C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_capture_data/'+ action +'/frame'+str(i)+'.jpg']
#     print(IMAGE_FILES)
#     with mp_hands.Hands(
#         static_image_mode=True,
#         max_num_hands=2,
#         min_detection_confidence=0.5) as hands:
#       for file in IMAGE_FILES:
#         # 이미지를 읽어 들이고, 보기 편하게 이미지를 좌우 반전합니다.
#         image = cv2.flip(cv2.imread(file), 1)
#         # 작업 전에 BGR 이미지를 RGB로 변환합니다.
#         results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#         # 손으로 프린트하고 이미지에 손 랜드마크를 그립니다.
#         print('Handedness:', results.multi_handedness)
#         if not results.multi_hand_landmarks:
#           continue
#         image_height, image_width, _ = image.shape
#         annotated_image = image.copy()
#         for hand_landmarks in results.multi_hand_landmarks:
#             print('hand_landmarks:', hand_landmarks)
#             print(
#                 f'Index finger tip coordinates: (',
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width}, '
#                 f'{hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height})'
#             )
#             mp_drawing.draw_landmarks(
#                 annotated_image,
#                 hand_landmarks,
#                 mp_hands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style())
#         cv2.imwrite('C:/Users/kimsungwook/Desktop/dohwajo/dataset/mediapipe_data/sick/' + str(i) + '.jpg', cv2.flip(annotated_image, 1))

















# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     max_num_hands=2,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
#
# cap = cv2.imread('C:/Users/kimsungwook/Desktop/dohwajo/dataset/origin_capture_data/sick/frame0.jpg')
# img = hands.process(cap)
#
#
#
# cv2.imshow('color', cap)
# cv2.waitKey(0)
# cv2.destroyAllWindows()




