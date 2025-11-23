import cv2
import mediapipe as mp
import numpy as np

class HandTracker:
    def __init__(self, max_num_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_drawing = mp.solutions.drawing_utils

    def process_frame(self, frame, depth_frame=None, draw=True):
        """
        Procesa un frame BGR y devuelve matriz np con 21 puntos (x_px, y_px, z_mm).
        Si no se pasa depth_frame, z será el valor relativo de MediaPipe.
        """
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        landmarks_array = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]

            puntos = []
            for lm in hand_landmarks.landmark:
                x_px, y_px = int(lm.x * w), int(lm.y * h)
                try:
                    if depth_frame is not None:
                        # Z en milímetros reales usando RealSense
                        z_m = depth_frame.get_distance(x_px, y_px)
                        z_mm = z_m * 1000  # convertimos a mm
                    else:
                        # Si no hay depth_frame, usamos z relativa de MediaPipe
                        z_mm = lm.z
                    x_px = int(x_px)
                    y_px = int(y_px)
                    z_mm = int(z_mm)
                    puntos.append([x_px, y_px, z_mm])
                except RuntimeError as e:
                    continue

            # Guardar en np.array (21,3)
            landmarks_array = np.array(puntos)

            # Dibujar landmarks en la imagen si se pide
            if draw:
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )

        return frame, landmarks_array