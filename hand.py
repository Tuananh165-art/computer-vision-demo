import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_hand = mp.solutions.hands
hands = mp_hand.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, img = cap.read()
    if not success:
        break

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    #ve ban tay
    if result.multi_hand_landmarks:
        myHand = []
        count = 0
        for idx, hand in enumerate(result.multi_hand_landmarks):
            mp_drawing.draw_landmarks(img, hand, mp_hand.HAND_CONNECTIONS)
            for id, lm in enumerate(hand.landmark):
                h, w, _ = img.shape
                myHand.append([int(lm.x * w), int(lm.y * h)]) # x = 0, y = 1
            if myHand[8][1] < myHand[5][1]:
                count = count + 1
            if myHand[12][1] < myHand[9][1]:
                count = count + 1
            if myHand[16][1] < myHand[13][1]:
                count = count + 1
            if myHand[20][1] < myHand[17][1]:
                count = count + 1
            if myHand[4][0] < myHand[2][0]:
                count = count + 1

    cv2.putText(img, str(count), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    # hien thi
    cv2.imshow("Detect hand", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
