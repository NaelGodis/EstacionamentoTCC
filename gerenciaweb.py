import cv2
import numpy as np
import time
from ultralytics import YOLO

# --- Importações para o Servidor Web ---
from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import threading

# --- Configurações do Sistema de Gerenciamento de Estacionamento ---

YOLO_MODEL_PATH = 'yolov8n.pt'
VIDEO_SOURCE = 'video_cortado.mp4'

# --- COLOQUE A SAÍDA DA SUA FERRAMENTA 'marcar_vagas.py' AQUI ---
PARKING_POLYGONS_CONFIG = [
    np.array([[120, 611], [241, 622], [316, 474], [199, 466]], np.int32), # Vaga 1
    np.array([[242, 622], [316, 478], [438, 469], [396, 622]], np.int32), # Vaga 2
    np.array([[965, 576], [821, 598], [785, 470], [898, 436]], np.int32), # Vaga 3
]
# Geração dos IDs das vagas baseada na lista acima
PARKING_ZONE_IDS = [f"Vaga {i+1}" for i in range(len(PARKING_POLYGONS_CONFIG))]


# --- Configurações Visuais ---
REGION_FREE_COLOR = (0, 255, 0)
REGION_OCCUPIED_COLOR = (0, 0, 255)
REGION_LINE_THICKNESS = 2

VEHICLE_POINT_COLOR = (0, 0, 255)
VEHICLE_POINT_RADIUS = 5
VEHICLE_POINT_THICKNESS = -1

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_OFFSET_Y = 10

# --- Configurações do Servidor Web ---
app = Flask(__name__)
app.config['SECRET_KEY'] = 'uma_chave_secreta_aqui'
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins="*")

# --- Variáveis Compartilhadas ---
# Usaremos uma fila para frames, ou apenas uma variável para o frame mais recente
# current_parking_status será atualizado pela thread de vídeo e lido pelo SocketIO
current_parking_status = {}
global_processed_frame = None # Para o feed de vídeo na web, se quisermos mais tarde
frame_lock = threading.Lock() # Para proteger o acesso a global_processed_frame

# --- Funções Auxiliares (mantidas) ---
def is_vehicle_in_zone(vehicle_box, zone_polygon):
    x1, y1, x2, y2 = map(int, vehicle_box[:4])
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    return cv2.pointPolygonTest(zone_polygon, (center_x, center_y), False) >= 0

# --- Funções do Servidor Web ---
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Cliente WebSocket conectado.')
    emit('parking_update', current_parking_status) # Envia o status atual ao novo cliente

@socketio.on('disconnect')
def test_disconnect():
    print('Cliente WebSocket desconectado.')

# --- Lógica de Processamento de Vídeo (agora em uma thread separada) ---
def video_processing_thread():
    """
    Thread responsável por processar o vídeo, atualizar o status das vagas
    e enviar atualizações via WebSocket. Também exibe o frame em uma janela OpenCV.
    """
    global current_parking_status, global_processed_frame

    print(f"[THREAD VÍDEO] Carregando modelo YOLO: {YOLO_MODEL_PATH}")
    print(f"[THREAD VÍDEO] Usando fonte de vídeo: {VIDEO_SOURCE}")
    print(f"[THREAD VÍDEO] Número de vagas configuradas: {len(PARKING_POLYGONS_CONFIG)}")

    try:
        model = YOLO(YOLO_MODEL_PATH)
        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            print(f"[THREAD VÍDEO] ERRO: Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")
            # Certifique-se de que a thread principal saiba que algo deu errado, ou apenas sai
            return

        for i, _ in enumerate(PARKING_POLYGONS_CONFIG):
            current_parking_status[PARKING_ZONE_IDS[i]] = 'free'

        while True:
            ret, frame = cap.read()
            if not ret:
                if VIDEO_SOURCE != '0':
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if not ret:
                        print("[THREAD VÍDEO] AVISO: Não foi possível reiniciar o vídeo ou ler o primeiro frame.")
                        break
                else:
                    break

            processed_frame = frame.copy()

            results = model.predict(source=frame, verbose=False, conf=0.25, iou=0.7)
            
            new_parking_status = {id_vaga: 'free' for id_vaga in PARKING_ZONE_IDS}

            if results and results[0].boxes.data.numel():
                detections = results[0].boxes.data

                for *xyxy, conf, cls in detections:
                    x1, y1, x2, y2 = map(int, xyxy)
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    
                    cv2.circle(processed_frame, (center_x, center_y), VEHICLE_POINT_RADIUS, 
                               VEHICLE_POINT_COLOR, VEHICLE_POINT_THICKNESS)

                    for i, polygon in enumerate(PARKING_POLYGONS_CONFIG):
                        if is_vehicle_in_zone(xyxy, polygon):
                            new_parking_status[PARKING_ZONE_IDS[i]] = 'occupied'
                            break 
            
            # Emitir atualização apenas se houver mudança
            if new_parking_status != current_parking_status:
                current_parking_status = new_parking_status
                # Usar socketio.emit para emitir da thread secundaria
                # É importante usar `socketio.emit` do contexto do Flask-SocketIO para threads
                with app.test_request_context(): # Garante um contexto de app para emit
                     socketio.emit('parking_update', current_parking_status)
                # print("[THREAD VÍDEO] Emitindo atualização:", current_parking_status)


            for i, polygon in enumerate(PARKING_POLYGONS_CONFIG):
                current_color = REGION_OCCUPIED_COLOR if current_parking_status[PARKING_ZONE_IDS[i]] == 'occupied' else REGION_FREE_COLOR
                cv2.polylines(processed_frame, [polygon], True, current_color, REGION_LINE_THICKNESS)
                
                x_text, y_text, w_text, h_text = cv2.boundingRect(polygon)
                text_position = (x_text, y_text - TEXT_OFFSET_Y)
                
                status_label = "Ocupada" if current_parking_status[PARKING_ZONE_IDS[i]] == 'occupied' else "Livre"
                display_text = f"{PARKING_ZONE_IDS[i]} ({status_label})"

                cv2.putText(processed_frame, display_text, text_position, 
                            TEXT_FONT, TEXT_SCALE, current_color, TEXT_THICKNESS, cv2.LINE_AA)
            
            # Atualiza o frame global para a janela do OpenCV
            with frame_lock:
                global_processed_frame = processed_frame

            # Controle de FPS para a thread de vídeo
            # time.sleep(0.01) # Descomente para limitar FPS se o sistema estiver sobrecarregado

        cap.release()
        print("[THREAD VÍDEO] Processamento de vídeo finalizado.")

    except FileNotFoundError as e:
        print(f"[THREAD VÍDEO] ERRO: Arquivo não encontrado - {e}.")
    except Exception as e:
        print(f"[THREAD VÍDEO] ERRO INESPERADO: {e}")

# --- Função para exibir o feed de vídeo localmente (OpenCV) ---
def display_opencv_feed():
    print("[THREAD OpenCV] Iniciando exibição local do OpenCV. Pressione 'q' para fechar.")
    while True:
        with frame_lock:
            if global_processed_frame is not None:
                cv2.imshow("Gerenciamento de Estacionamento (Local OpenCV)", global_processed_frame)
        
        # 1ms de espera é o mínimo para que o OpenCV possa processar eventos de janela
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time.sleep(0.01) # Pequeno atraso para economizar CPU enquanto espera por 'q'
    
    cv2.destroyAllWindows()
    print("[THREAD OpenCV] Janela local do OpenCV fechada.")


# --- Execução Principal (Servidor Flask-SocketIO) ---
if __name__ == '__main__':
    # Inicializa as variáveis de status globais
    for i, _ in enumerate(PARKING_POLYGONS_CONFIG):
        current_parking_status[PARKING_ZONE_IDS[i]] = 'free'

    # Inicia a thread de processamento de vídeo (e que também fará as emissões de SocketIO)
    video_thread = threading.Thread(target=video_processing_thread)
    video_thread.daemon = True
    video_thread.start()

    # Inicia a thread para a exibição da janela do OpenCV (separada para não bloquear o Flask)
    opencv_display_thread = threading.Thread(target=display_opencv_feed)
    opencv_display_thread.daemon = True
    opencv_display_thread.start()

    print("\nIniciando servidor web Flask-SocketIO em http://127.0.0.1:5000")
    print("O processamento de vídeo e a janela do OpenCV rodam em threads separadas.")
    print("Para parar tudo, pressione 'q' na janela do OpenCV e depois finalize o terminal.")
    
    # Inicia o servidor Flask-SocketIO na thread principal
    socketio.run(app, debug=False, port=5000, allow_unsafe_werkzeug=True, use_reloader=False)