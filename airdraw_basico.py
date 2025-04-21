import cv2
import mediapipe as mp
import numpy as np

# Configuración de MediaPipe para las manos
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Variables
drawing_canvas = None
previous_pos = None
painting = False
current_color = (0, 255, 0)  # verde predeterminado (formato BGR)
brush_thickness = 10
mode = 'draw'  # 'draw' o 'erase'

# Paleta de colores con colores vivos (opacidad completa, sin canal alfa)
color_buttons = [
    ((0, 255, 0),     (620, 40)),   # Verde brillante
    ((255, 50, 50),   (620, 100)),  # Azul eléctrico (parece rosado en BGR)
    ((50, 50, 255),   (620, 160)),  # Rojo brillante
    ((0, 255, 255),   (620, 220)),  # Amarillo brillante
    ((255, 0, 255),   (620, 280)),  # Fucsia
]
eraser_button = ((200, 200, 200), (610, 340), (700, 390))  # botón gris

# Cámara
cap = cv2.VideoCapture(0)

# Inicializar ventana
cv2.namedWindow("Air Draw - Full Tools", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Air Draw - Full Tools", 1280, 720)

# Control deslizante para el tamaño del pincel: rango de 0 a 100
slider_x, slider_y, slider_w, slider_h = 20, 20, 20, 400  # posición y tamaño del control deslizante
slider_value = 50  # valor inicial del control deslizante (50% del tamaño máximo)

# Fondo del control deslizante
slider_background = np.zeros((slider_h, slider_w, 3), dtype=np.uint8)
cv2.rectangle(slider_background, (0, 0), (slider_w, slider_h), (255, 255, 255), -1)  # Fondo blanco
cv2.rectangle(slider_background, (0, 0), (slider_w, slider_h), (0, 0, 0), 2)  # Borde

# Bucle de la cámara
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Inicializar lienzo con fondo negro (totalmente opaco)
    if drawing_canvas is None:
        drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)  # Fondo negro

    # Dibujar botones de color (relleno sólido sin transparencia)
    for color, (x, y) in color_buttons:
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), color, -1)  # Botón de color relleno
        cv2.rectangle(frame, (x, y), (x + 50, y + 50), (0, 0, 0), 2)  # Borde

    # Dibujar botón de borrador
    ex1, ey1, ex2, ey2 = eraser_button[1][0], eraser_button[1][1], eraser_button[2][0], eraser_button[2][1]
    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), eraser_button[0], -1)
    cv2.rectangle(frame, (ex1, ey1), (ex2, ey2), (0, 0, 0), 2)
    cv2.putText(frame, 'Borrar', (ex1 + 10, ey2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

    # Dibujar el control deslizante del tamaño del pincel
    frame[slider_y:slider_y + slider_h, slider_x:slider_x + slider_w] = slider_background

    # Dibujar indicador del control deslizante según el tamaño del pincel
    indicator_y = slider_y + int((100 - slider_value) * slider_h / 100)  # Valor invertido
    cv2.rectangle(frame, (slider_x - 10, indicator_y - 5), (slider_x + slider_w + 10, indicator_y + 5), (0, 255, 0), -1)

    # Procesar puntos de referencia de la mano
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Coordenadas de la punta del dedo índice y la punta del pulgar
        ix, iy = int(hand_landmarks.landmark[8].x * w), int(hand_landmarks.landmark[8].y * h)
        tx, ty = int(hand_landmarks.landmark[4].x * w), int(hand_landmarks.landmark[4].y * h)

        # Calcular la distancia entre el pulgar y el índice para controlar el tamaño del pincel
        dist = int(np.hypot(ix - tx, iy - ty))

        # Comprobar selección de botón de color
        for color, (x, y) in color_buttons:
            if x <= ix <= x + 50 and y <= iy <= y + 50:
                current_color = color
                mode = 'draw'
                cv2.rectangle(frame, (x - 3, y - 3), (x + 53, y + 53), (255, 255, 255), 2)

        # Comprobar el botón de borrador
        if ex1 <= ix <= ex2 and ey1 <= iy <= ey2:
            mode = 'erase'
            cv2.rectangle(frame, (ex1 - 2, ey1 - 2), (ex2 + 2, ey2 + 2), (0, 0, 0), 3)

        # Comprobar si estamos interactuando con el control deslizante
        if slider_x <= ix <= slider_x + slider_w and slider_y <= iy <= slider_y + slider_h:
            slider_value = max(0, min(100, int((slider_y + slider_h - iy) * 100 / slider_h)))  # Lógica del control deslizante inversa

        # Pintar si el pulgar y el índice están cerca
        if dist < 40:
            if previous_pos:
                color_to_use = (0, 0, 0) if mode == 'erase' else current_color
                cv2.line(drawing_canvas, previous_pos, (ix, iy), color_to_use, slider_value)
            previous_pos = (ix, iy)
            painting = True
        else:
            previous_pos = None
            painting = False

    # Superponer el dibujo sobre el video (esto debe mostrar el dibujo sobre el video)
    output = cv2.addWeighted(frame, 1, drawing_canvas, 1, 0)

    # Indicador de dibujo
    if painting:
        cv2.circle(output, previous_pos, 10, current_color, -1)
        cv2.putText(output, f"Modo: {'Borrar' if mode == 'erase' else 'Dibujar'}", (120, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, current_color, 2)
        cv2.putText(output, f"Grosor: {slider_value}", (120, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, current_color, 2)

    # Mostrar el resultado
    cv2.imshow("Air Draw - Full Tools", output)
    
    # Salir presionando 'Esc'
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
