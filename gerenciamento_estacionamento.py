import cv2
import numpy as np # numpy é essencial para trabalhar com os arrays de pontos
from ultralytics import YOLO

# --- Configurações do Sistema de Gerenciamento de Estacionamento ---

# Caminho para o modelo YOLO. Ex: 'yolov8n.pt' para versão nano.
YOLO_MODEL_PATH = 'yolov8n.pt'

# Fonte de vídeo. Pode ser '0' para webcam, 'caminho/do/video.mp4', ou URL de stream.
VIDEO_SOURCE = 'video_cortado.mp4'

# --- COLOQUE A SAÍDA DA SUA FERRAMENTA 'marcar_vagas.py' AQUI ---
#
# Exemplo (substitua pelo que sua ferramenta gerar):
PARKING_POLYGONS_CONFIG = [
    np.array([[50, 50], [150, 50], [150, 100], [50, 100]], np.int32), # Exemplo de Vaga 1
    np.array([[170, 50], [270, 50], [270, 100], [170, 100]], np.int32), # Exemplo de Vaga 2
    np.array([[50, 150], [150, 150], [150, 200], [50, 200]], np.int32), # Exemplo de Vaga 3
    # ADICIONE MAIS VAGAS AQUI, COPIANDO E COLANDO A SAÍDA DO 'marcar_vagas.py'
]

# Geração dos IDs das vagas baseada na lista acima
PARKING_ZONE_IDS = [f"Vaga {i+1}" for i in range(len(PARKING_POLYGONS_CONFIG))]


# --- Configurações Visuais (Cores em formato BGR) ---
REGION_FREE_COLOR = (0, 255, 0)     # Verde para vaga livre
REGION_OCCUPIED_COLOR = (0, 0, 255) # Vermelho para vaga ocupada
REGION_LINE_THICKNESS = 2           # Espessura da borda das vagas

VEHICLE_POINT_COLOR = (0, 0, 255)   # Vermelho para o ponto central do veículo
VEHICLE_POINT_RADIUS = 5            # Raio do ponto do veículo
VEHICLE_POINT_THICKNESS = -1        # Preenchimento total do ponto (-1)

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICKNESS = 2
TEXT_OFFSET_Y = 10 # Offset vertical para o texto da vaga

# --- Funções Auxiliares ---

def is_vehicle_in_zone(vehicle_box, zone_polygon):
    """
    Verifica se o centro de um veículo detectado está dentro de uma zona de estacionamento.
    vehicle_box: Tupla ou lista [x1, y1, x2, y2] das coordenadas da caixa delimitadora do veículo.
    zone_polygon: Array NumPy de pontos do polígono da vaga.
    """
    x1, y1, x2, y2 = map(int, vehicle_box[:4])
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    
    # cv2.pointPolygonTest retorna > 0 se o ponto está dentro, 0 se na borda, < 0 se fora.
    return cv2.pointPolygonTest(zone_polygon, (center_x, center_y), False) >= 0

# --- Lógica Principal ---

def start_parking_management_with_visual_feedback():
    """
    Inicializa e executa o sistema de gerenciamento de estacionamento.
    Exibe pontos vermelhos nos veículos detectados e o status das vagas.
    """
    print(f"Carregando modelo YOLO: {YOLO_MODEL_PATH}")
    print(f"Usando fonte de vídeo: {VIDEO_SOURCE}")
    print(f"Número de vagas configuradas: {len(PARKING_POLYGONS_CONFIG)}")

    try:
        model = YOLO(YOLO_MODEL_PATH)
        
        # As zonas de estacionamento já estão prontas (PARKING_POLYGONS_CONFIG)
        # e seus IDs (PARKING_ZONE_IDS).

        cap = cv2.VideoCapture(VIDEO_SOURCE)
        if not cap.isOpened():
            raise IOError(f"Não foi possível abrir a fonte de vídeo: {VIDEO_SOURCE}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break # Fim do vídeo ou erro na leitura

            # Realiza a inferência no frame atual
            results = model.predict(source=frame, verbose=False, conf=0.25, iou=0.7)
            
            # Pula o processamento se não houver detecções válidas
            if not results or not results[0].boxes.data.numel(): # numel() verifica se há elementos no tensor
                cv2.imshow("Gerenciamento de Estacionamento", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            detections = results[0].boxes.data # Extrai dados das caixas: [x1, y1, x2, y2, conf, cls]

            # Inicializa o status de todas as vagas como 'free'
            parking_status = {i: 'free' for i in range(len(PARKING_POLYGONS_CONFIG))}

            # Processa cada veículo detectado
            for *xyxy, conf, cls in detections:
                x1, y1, x2, y2 = map(int, xyxy)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                
                # Desenha o ponto vermelho no centro do veículo detectado
                cv2.circle(frame, (center_x, center_y), VEHICLE_POINT_RADIUS, 
                           VEHICLE_POINT_COLOR, VEHICLE_POINT_THICKNESS)

                # Verifica em qual vaga o veículo se encontra
                for i, polygon in enumerate(PARKING_POLYGONS_CONFIG):
                    if is_vehicle_in_zone(xyxy, polygon):
                        parking_status[i] = 'occupied'
                        break # Assume que um veículo ocupa apenas uma vaga e passa para o próximo veículo

            # Desenha as vagas e exibe seu status no frame
            for i, polygon in enumerate(PARKING_POLYGONS_CONFIG):
                current_color = REGION_OCCUPIED_COLOR if parking_status[i] == 'occupied' else REGION_FREE_COLOR
                cv2.polylines(frame, [polygon], True, current_color, REGION_LINE_THICKNESS)
                
                # Prepara e exibe o texto do status da vaga
                x_text, y_text, w_text, h_text = cv2.boundingRect(polygon)
                text_position = (x_text, y_text - TEXT_OFFSET_Y)
                
                status_label = "Ocupada" if parking_status[i] == 'occupied' else "Livre"
                display_text = f"{PARKING_ZONE_IDS[i]} ({status_label})"

                cv2.putText(frame, display_text, text_position, 
                            TEXT_FONT, TEXT_SCALE, current_color, TEXT_THICKNESS, cv2.LINE_AA)
            
            # Exibe o frame processado
            cv2.imshow("Gerenciamento de Estacionamento", frame)

            # Aguarda 1ms por uma tecla e sai se 'q' for pressionado
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    except FileNotFoundError as e:
        print(f"Erro: Arquivo não encontrado - {e}. "
              f"Verifique os caminhos: '{YOLO_MODEL_PATH}' e '{VIDEO_SOURCE}'.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

if __name__ == '__main__':
    start_parking_management_with_visual_feedback()