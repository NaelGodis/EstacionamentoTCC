import cv2
import numpy as np
import os

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
    # --- CONFIGURAÇÃO DO ARQUIVO ---
    nome_do_arquivo = "video_cortado.mp4"
    
    # Verifica se o arquivo existe antes de tentar abrir
    if not os.path.exists(nome_do_arquivo):
        print(f"ERRO CRÍTICO: O arquivo '{nome_do_arquivo}' não foi encontrado na pasta atual.")
        return

    cap = cv2.VideoCapture(nome_do_arquivo)
    
    # Verifica se o vídeo abriu corretamente
    if not cap.isOpened():
        print("Erro ao abrir o vídeo. O formato pode estar corrompido.")
        return

    ret, frame = cap.read()
    
    if not ret:
        print("Erro ao ler o primeiro frame do vídeo.")
        return

    # Se o vídeo for muito grande (4k ou 1080p), descomente a linha abaixo para reduzir
    # frame = cv2.resize(frame, (1280, 720))

    cv2.namedWindow("Marcar Vagas")
    cv2.setMouseCallback("Marcar Vagas", mouse_callback)

    print("\n=== INSTRUÇÕES ===")
    print("1. Clique nos 4 cantos de uma vaga (ordem: canto superior esquerdo -> horário).")
    print("2. Assim que clicar o 4º ponto, a vaga é salva automaticamente.")
    print("3. Repita para todas as vagas.")
    print("4. Aperte 'q' para finalizar e GERAR O CÓDIGO.")
    print(f"--> Lendo arquivo: {nome_do_arquivo}")
    print("==================\n")

    while True:
        temp_frame = frame.copy()

        # Desenhar vagas já salvas (Verde)
        for i, vaga in enumerate(todas_vagas):
            cv2.polylines(temp_frame, [vaga], True, (0, 255, 0), 2)
            # Opcional: Escrever o número da vaga
            cv2.putText(temp_frame, str(i+1), (vaga[0][0], vaga[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Desenhar pontos que você está clicando agora (Vermelho)
        for ponto in pontos_atuais:
            cv2.circle(temp_frame, ponto, 5, (0, 0, 255), -1)
        
        # Desenhar linhas conectando os pontos atuais para facilitar visualização
        if len(pontos_atuais) > 0:
            pts = np.array(pontos_atuais, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(temp_frame, [pts], False, (0, 0, 255), 1)

        cv2.imshow("Marcar Vagas", temp_frame)

        key = cv2.waitKey(1)
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # --- GERAR CÓDIGO FINAL ---
    if len(todas_vagas) > 0:
        print("\n\n=== COPIE E COLE ISSO NO SEU CÓDIGO PRINCIPAL (estacionamento.py) ===")
        print("vagas_config = [")
        for i, vaga in enumerate(todas_vagas):
            coords = ", ".join([f"[{p[0]}, {p[1]}]" for p in vaga])
            print(f"    np.array([{coords}], np.int32), # Vaga {i+1}")
        print("]")
        print("=====================================================================")
    else:
        print("\nNenhuma vaga foi marcada.")

if __name__ == "__main__":
    main()