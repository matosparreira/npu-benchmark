import os
import time
import urllib.request
import openvino as ov
import numpy as np

# Modelo "ainda maior" (VGG-19) para forçar a NPU mais do que o VGG-16.
# Mesma entrada (1x3x224x224), mas ~19.6 GFLOPs (~1.26x o VGG-16): a versão
# de 19 camadas, com batch-norm dobrada nas convoluções. É a carga densa mais
# pesada da série, exatamente onde os arrays de MAC da NPU mais se destacam.
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/vgg/model/vgg19-bn-7.onnx"
MODEL_PATH = "vgg19-bn-7.onnx"

DEVICES = ["CPU", "NPU"]
ITERATIONS = 200
INPUT_SHAPE = [1, 3, 224, 224]  # batch=1, RGB, 224x224

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[+] A descarregar modelo VGG-19 (~574MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Download concluído.")

def bench_sync(core, model, device, dummy_input):
    # Latência: um pedido de cada vez, síncrono (batch-1).
    compiled = core.compile_model(model, device)
    input_layer = compiled.input(0)
    request = compiled.create_infer_request()
    request.infer({input_layer.any_name: dummy_input})  # aquecimento
    start = time.time()
    for _ in range(ITERATIONS):
        request.infer({input_layer.any_name: dummy_input})
    avg_ms = (time.time() - start) / ITERATIONS * 1000
    return avg_ms

def bench_async(core, model, device, dummy_input):
    # Débito: vários pedidos em voo para manter o pipeline cheio.
    compiled = core.compile_model(model, device)
    input_layer = compiled.input(0)
    queue = ov.AsyncInferQueue(compiled)
    for _ in range(8):  # primar o pipeline
        queue.start_async({input_layer.any_name: dummy_input})
    queue.wait_all()
    start = time.time()
    for _ in range(ITERATIONS):
        queue.start_async({input_layer.any_name: dummy_input})
    queue.wait_all()
    throughput = ITERATIONS / (time.time() - start)  # inferências/segundo
    return throughput

def run_comparison():
    core = ov.Core()
    devices = core.available_devices
    print(f"\n[+] Dispositivos de IA detetados no Linux: {devices}")

    targets = [d for d in DEVICES if d in devices]
    missing = [d for d in DEVICES if d not in devices]
    for d in missing:
        print(f"[-] AVISO: '{d}' não foi encontrado, será ignorado.")
    if not targets:
        print("[-] ERRO: nenhum dos dispositivos-alvo está disponível.")
        return

    download_model()
    print(f"[+] A ler o modelo {MODEL_PATH}...")
    model = core.read_model(MODEL_PATH)
    model.reshape({model.input(0).any_name: INPUT_SHAPE})  # fixar batch estático
    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    sync, asyncr = {}, {}
    for device in targets:
        print(f"[+] A avaliar {device} (compilar + benchmark)...")
        sync[device] = bench_sync(core, model, device, dummy_input)
        asyncr[device] = bench_async(core, model, device, dummy_input)

    print("\n========== RESULTADOS (VGG-19, 224x224) ==========")
    print(f"{'Dispositivo':12}{'Latência (ms)':>16}{'Débito (inf/s)':>18}")
    for device in targets:
        print(f"{device:12}{sync[device]:>16.2f}{asyncr[device]:>18.0f}")
    print("==================================================")

    # Comparação direta quando ambos os dispositivos estão presentes.
    if "CPU" in targets and "NPU" in targets:
        lat = sync["CPU"] / sync["NPU"]
        thr = asyncr["NPU"] / asyncr["CPU"]
        print(f"NPU vs CPU -> latência: {lat:.2f}x menor | débito: {thr:.2f}x maior")

if __name__ == "__main__":
    run_comparison()
