import cv2
import mediapipe as mp


def main():
    capture = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    mp_draw = mp.solutions.drawing_utils
    stop = False

    while not stop: 
        img = capture.read()
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_img)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks: 
                for id, lm in enumerate(hand_landmarks.landmark):
                    height, width, channel = img.shape
                    cx, cy = int(lm.x * width), int(lm.y * height)
        else:
            hand_landmarks = None
        
        mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Output", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()
