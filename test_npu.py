import os
import time
import urllib.request
import openvino as ov
import numpy as np

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx"
MODEL_PATH = "mnist-12.onnx"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[+] A descarregar modelo leve de teste (MNIST ONNX)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Download concluído.")

def run_npu_demo():
    # 1. Inicializar o Core do OpenVINO
    core = ov.Core()
    devices = core.available_devices
    print(f"\n[+] Dispositivos de IA detetados no Linux: {devices}")

    # Definir o dispositivo-alvo (Pode alterar para 'CPU' para comparar a velocidade)
    target_device = "NPU"

    if target_device not in devices:
        print(f"[-] ERRO: O dispositivo '{target_device}' não foi encontrado.")
        print("    Verifique se os drivers (ex: intel_vpu) estão carregados no seu kernel.")
        return

    # 2. Carregar o modelo descarregado
    download_model()
    print(f"[+] A ler o modelo {MODEL_PATH}...")
    model = core.read_model(model=MODEL_PATH)

    # 3. Compilar o modelo para a NPU
    print(f"[+] A compilar o modelo especificamente para a {target_device} (isto pode demorar alguns segundos)...")
    start_compile = time.time()
    compiled_model = core.compile_model(model=model, device_name=target_device)
    print(f"[+] Modelo compilado com sucesso em {time.time() - start_compile:.2f}s")

    # 4. Criar dados de teste aleatórios (Formato esperado pelo MNIST: 1 canal, 28x28 píxeis)
    input_layer = compiled_model.input(0)
    output_layer = compiled_model.output(0)
    dummy_input = np.random.randn(*input_layer.shape).astype(np.float32)

    # 5. Executar Inferências (Benchmark de aquecimento + loop)
    print(f"[+] A executar inferências na {target_device}...")
    infer_request = compiled_model.create_infer_request()
    
    # Execução de aquecimento (Warm-up)
    infer_request.infer({input_layer.any_name: dummy_input})

    # Teste de stress rápido
    iterations = 500
    start_time = time.time()
    for _ in range(iterations):
        infer_request.infer({input_layer.any_name: dummy_input})
    end_time = time.time()

    # 6. Resultados
    total_time = end_time - start_time
    avg_time = (total_time / iterations) * 1000
    print("\n================ RESULTADOS ================")
    print(f"Dispositivo Utilizado:      {target_device}")
    print(f"Total de Inferências:      {iterations}")
    print(f"Tempo Total do Loop:       {total_time:.4f} segundos")
    print(f"Tempo Médio por Imagem:    {avg_time:.2f} ms")
    print("============================================")

if __name__ == "__main__":
    run_npu_demo()
