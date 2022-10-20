from django.shortcuts import render
from .gvoicerec import recvoice
from django.http import StreamingHttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators import gzip
import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
from gtts import gTTS
import playsound
import os

def speak(text):
    tts = gTTS(text = text, lang = 'ko')
    filename = 'C:/Users/rnvld/myenv/Scripts/siteprac/voice.mp3'
    tts.save(filename)
    playsound.playsound(filename)
    if os.path.exists(filename):
        os.remove(filename)
    else:
        return 0


part = 'EMPTY'
simptom = 'EMPTY'

def get_frame(img):
        _, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes()

def mediavideo(page):
    global part
    global simptom

    if page == '1page':
        actions = ['head', 'chest', 'stomach']
        model = load_model('C:/Users/rnvld/myenv/Scripts/siteprac/voicerec/model1.h5')
    else:
        actions = ['sick', 'stuffy', 'strange']
        model = load_model('C:/Users/rnvld/myenv/Scripts/siteprac/voicerec/model2.h5')
    
    #actions = ['sick', 'stuffy', 'strange']
    seq_length = 30

    #model = load_model('C:/Users/rnvld/myenv/Scripts/siteprac/voicerec/model2.h5')
    #model = load_model('C:/Users/rnvld/myenv/Scripts/siteprac/voicerec/model1.h5')

    # MediaPipe hands model
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(0)

    # w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    # out = cv2.VideoWriter('input.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))
    # out2 = cv2.VideoWriter('output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (w, h))

    seq = []
    action_seq = []
    time = 0
    while cap.isOpened():
        time += 1
        print(time)
        if time == 75: #종료 시간 설정 
            cap.release()
            break
        ret, img = cap.read()

        img = cv2.flip(img, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks is not None:
            joint = np.zeros((21, 4))
            joint1 = np.zeros((21, 4))
            for i, res in enumerate(result.multi_hand_landmarks):
                if len(result.multi_hand_landmarks) == 1:
                    for j, lm in enumerate(res.landmark):
                        joint[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle_label = np.degrees(angle)  # Convert radian to degree

                    v1 = joint1[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                        :3]  # Parent joint
                    v2 = joint1[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle1 = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                :]))  # [15,]

                    angle1 = np.degrees(angle1)  # Convert radian to degree

                    angle_label1 = np.nan_to_num(angle1, copy=False)

                    d = np.concatenate([joint.flatten(), angle_label, joint1.flatten(), angle_label1])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                elif len(result.multi_hand_landmarks) == 2:
                    if i == 0:
                        for j, lm in enumerate(res.landmark):
                            joint[j] = [lm.x, lm.y, lm.z, lm.visibility]
                            mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)
                        continue
                    elif i == 1:
                        for j, lm in enumerate(res.landmark):
                            joint1[j] = [lm.x, lm.y, lm.z, lm.visibility]

                    # Compute angles between joints
                    v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :3]  # Parent joint
                    v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

                    angle_label = np.degrees(angle)  # Convert radian to degree

                    v1 = joint1[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19],
                        :3]  # Parent joint
                    v2 = joint1[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                        :3]  # Child joint
                    v = v2 - v1  # [20, 3]
                    # Normalize v
                    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                    # Get angle using arcos of dot product
                    angle1 = np.arccos(np.einsum('nt,nt->n',
                                                v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                                v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19],
                                                :]))  # [15,]

                    angle_label1 = np.degrees(angle1)  # Convert radian to degree

                    d = np.concatenate([joint.flatten(), angle_label, joint1.flatten(), angle_label1])

                    seq.append(d)

                    mp_drawing.draw_landmarks(img, res, mp_hands.HAND_CONNECTIONS)

                if len(seq) < seq_length:
                    continue

                input_data = np.expand_dims(np.array(seq[-seq_length:], dtype=np.float32), axis=0)

                y_pred = model.predict(input_data).squeeze()

                i_pred = int(np.argmax(y_pred))
                conf = y_pred[i_pred]

                if conf < 0.8:
                    continue

                action = actions[i_pred]
                action_seq.append(action)

                if len(action_seq) < 4:
                    continue

                this_action = '?'
                if action_seq[-1] == action_seq[-2] == action_seq[-3] == action_seq[-4]:
                    this_action = action
                
                if this_action != '?':
                    if page == '1page':
                        part = this_action
                    else:
                        simptom = this_action

        #cv2.putText(img, f'{this_action.upper()}', org=(int(res.landmark[0].x * img.shape[1]), int(res.landmark[0].y * img.shape[0] + 20)), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(255, 255, 255), thickness=2)
        if page == '1page':
            cv2.putText(img, part,org =(0,40),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        else :
            cv2.putText(img, simptom,org =(0,40),fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 255), thickness=2)
        #cv2.putText(img, f'{action.upper()}', org=(0, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,color=(0, 0, 0), thickness=2)

        # out.write(img0)
        # out2.write(img)
        #cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break
        frame = get_frame(img)
        # frame단위로 이미지를 계속 반환한다. (yield)
        yield(b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def check():
    global part, simptom
    c = ''
    if part == 'head' and simptom == 'sick': c = '머리가 아프다.'
    elif part == 'head' and simptom == 'stuffy': c = '머리가 답답하다.'
    elif part == 'head' and  simptom== 'strange': c = '머리가 이상하다.'
    elif part == 'chest' and simptom == 'sick': c = '가슴이 아프다.'
    elif part == 'chest' and simptom == 'stuffy': c = '가슴이 답답하다.'
    elif part == 'chest' and simptom == 'strange': c = '가슴이 이상하다.'
    elif part == 'stomach' and simptom == 'sick': c = '배가 아프다.'
    elif part == 'stomach' and simptom == 'stuffy': c = '배가 답답하다.'
    elif part == 'stomach' and simptom == 'strange': c = '배가 이상하다.'
    else: c = '오류'
    return c


def video_first(request):
    return render(request,'video_first.html')

def video_second(request):
    return render(request,'video_second.html')

def print_result(request):
    result = check()
    speak(result)
    return render(request, 'result.html', {'result': result})

def voice_record(request):
    return render(request, 'voice_record.html')

@csrf_exempt
def recandprint(request):
    recvoic = recvoice()
    return render(request,'voice_record.html', {'recvoice': recvoic})
# Create your views here.

@gzip.gzip_page
def opvideo(request):
   m = mediavideo('1page')
   return StreamingHttpResponse(m, content_type="multipart/x-mixed-replace;boundary=frame")


@gzip.gzip_page
def op2video(request):
    m = mediavideo('2page')
    return StreamingHttpResponse(m, content_type="multipart/x-mixed-replace;boundary=frame")
