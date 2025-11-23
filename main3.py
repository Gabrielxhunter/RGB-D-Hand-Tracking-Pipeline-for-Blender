import cv2
import json
import socket
import Preproceso1 as pre
from hand_tracking import HandTracker

def main():
    # Inicializa cámara RealSense + filtros
    pipeline, align, spatial, temporal, imagen_bgr = pre.inicializar_pipeline()

    # Inicializa hand tracker
    tracker = HandTracker(max_num_hands=1)

    # Configurar socket cliente UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    try:
        while True:
            # Obtiene frames preprocesados con realsense
            filtered_color, depth_colormap, depth_frame = pre.procesar_frame(
                pipeline, align, spatial, temporal, imagen_bgr
            )
            if filtered_color is None:
                continue

            # Procesa la imagen con HandTracker para tener un trabajo más limpio
            frame_landmarks, landmarks_array = tracker.process_frame(
                filtered_color, depth_frame=depth_frame, draw=True
            )

            if landmarks_array is not None:
                # Convertir a formato que Blender espera
                landmarks_formatted = []
                for i, (x, y, z) in enumerate(landmarks_array):
                    # Convertir valores NumPy a tipos nativos de Python (CORRECCIÓN IMPORTANTE)
                    landmarks_formatted.append({
                        "id": int(i),
                        "x": int(x),
                        "y": int(y),
                        "z": int(z)
                    })

                # Serializar a JSON
                data_json = json.dumps(landmarks_formatted)

                # Enviar por socket UDP
                sock.sendto(data_json.encode("utf-8"), ("127.0.0.1", 5005))

                # Mostrar en consola lo que se envía
                print("Enviado al servidor:\n", data_json)

            # Mostrar ventanas principales
            cv2.imshow("RGB filtrado + HandTracking", frame_landmarks)
            cv2.imshow("Mapa de profundidad", depth_colormap)

            # Salida con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()