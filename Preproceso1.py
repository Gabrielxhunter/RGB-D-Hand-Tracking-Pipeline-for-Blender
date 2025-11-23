import pyrealsense2 as rs
import numpy as np
import cv2

# Constantes de rango de profundidad "Con esto definimos el espacio virtuial"
MAX_DIST = 2000
MIN_DIST = 400

def inicializar_pipeline(ruta_imagen='C:/Users/altig/Desktop/visual_studio/CapturadeMovimiento3D/UADY.jpg'):
    """
    Inicializa la cámara RealSense y devuelve pipeline, alineación, filtros y la imagen de referencia.
    """
    # Inicializa pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30) 

    # Imagen de referencia
    imagen = cv2.imread(ruta_imagen)
    if imagen is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {ruta_imagen}")
    imagen_bgr = cv2.resize(imagen, (640, 480))

    # Crear objeto de alineación (profundidad → BGR)
    align = rs.align(rs.stream.color)

    # Filtros para suabizar los bordes
    spatial = rs.spatial_filter()
    temporal = rs.temporal_filter()

    pipeline.start(config)

    return pipeline, align, spatial, temporal, imagen_bgr

def procesar_frame(pipeline, align, spatial, temporal, imagen_bgr):
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None

    depth_frame = spatial.process(depth_frame)
    depth_frame = temporal.process(depth_frame)

    depth_frame_filtered = temporal.process(spatial.process(depth_frame))
    depth_frame_filtered = rs.frame.as_depth_frame(depth_frame_filtered)

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    mask_valid = (depth_image > MIN_DIST) & (depth_image <= MAX_DIST) #Ignoramos pixeles

    filtered_color = imagen_bgr.copy()
    filtered_color[mask_valid] = color_image[mask_valid]

    depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )

    return filtered_color, depth_colormap, depth_frame_filtered
