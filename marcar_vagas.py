import cv2
import numpy as np
import argparse

# Variáveis globais para armazenar os pontos
pontos_atuais = []
todas_vagas = []

def mouse_callback(event, x, y, flags, param):
    global pontos_atuais

    # Quando clicar com o botão esquerdo do mouse
    if event == cv2.EVENT_LBUTTONDOWN:
        pontos_atuais.append((x, y))
        print(f"Ponto capturado: {x}, {y}")

        # Se já tiver 4 pontos, temos uma vaga completa
        if len(pontos_atuais) == 4:
            todas_vagas.append(np.array(pontos_atuais, np.int32))
            print(f"--- Vaga {len(todas_vagas)} registrada! ---")
            pontos_atuais = [] # Reseta para a próxima vaga

def main():
    parser = argparse.ArgumentParser(description="Ferramenta para marcar vagas de estacionamento.")
    parser.add_argument("video_path", type=str, help="Caminho do vídeo.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video_path)
    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao ler o vídeo.")
        return

    # Redimensionar se o vídeo for gigante (opcional, ajusta conforme sua tela)
    # frame = cv2.resize(frame, (1280, 720))

    cv2.namedWindow("Marcar Vagas")
    cv2.setMouseCallback("Marcar Vagas", mouse_callback)

    print("\n=== INSTRUÇÕES ===")
    print("1. Clique nos 4 cantos de uma vaga (sentido horário ou anti-horário).")
    print("2. Assim que clicar o 4º ponto, a vaga é salva automaticamente.")
    print("3. Repita para todas as vagas.")
    print("4. Aperte 'q' para finalizar e GERAR O CÓDIGO.")
    print("==================\n")

    while True:
        temp_frame = frame.copy()

        # Desenhar vagas já salvas (Verde)
        for vaga in todas_vagas:
            cv2.polylines(temp_frame, [vaga], True, (0, 255, 0), 2)

        # Desenhar pontos que você está clicando agora (Vermelho)
        for ponto in pontos_atuais:
            cv2.circle(temp_frame, ponto, 5, (0, 0, 255), -1)
        
        # Desenhar linhas conectando os pontos atuais para facilitar
        if len(pontos_atuais) > 1:
            cv2.polylines(temp_frame, [np.array(pontos_atuais)], False, (0, 0, 255), 1)

        cv2.imshow("Marcar Vagas", temp_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- GERAR CÓDIGO FINAL ---
    print("\n\n=== COPIE E COLE ISSO NO SEU CÓDIGO PRINCIPAL (estacionamento.py) ===")
    print("vagas_config = [")
    for i, vaga in enumerate(todas_vagas):
    
        coords = ", ".join([f"[{p[0]}, {p[1]}]" for p in vaga])
        print(f"    np.array([{coords}], np.int32), # Vaga {i+1}")
    print("]")
    print("=====================================================================")

if __name__ == "__main__":
    main()