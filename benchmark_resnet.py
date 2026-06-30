import os
import time
import urllib.request
import openvino as ov
import numpy as np

# Modelo realista (ResNet-50) para demonstrar a vantagem da NPU sobre a CPU.
# Ao contrário do MNIST (demasiado pequeno, dominado por overhead de despacho),
# o ResNet-50 tem computação suficiente para a NPU se destacar.
MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx"
MODEL_PATH = "resnet50-v1-12.onnx"

DEVICES = ["CPU", "GPU", "NPU"]
ITERATIONS = 200
INPUT_SHAPE = [1, 3, 224, 224]  # batch=1, RGB, 224x224

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[+] A descarregar modelo ResNet-50 (~97MB)...")
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

    print("\n========== RESULTADOS (ResNet-50, 224x224) ==========")
    print(f"{'Dispositivo':12}{'Latência (ms)':>16}{'Débito (inf/s)':>18}")
    for device in targets:
        print(f"{device:12}{sync[device]:>16.2f}{asyncr[device]:>18.0f}")
    print("=====================================================")

    # Comparação direta de cada acelerador face à CPU (referência).
    if "CPU" in targets:
        for dev in ("GPU", "NPU"):
            if dev in targets:
                lat = sync["CPU"] / sync[dev]
                thr = asyncr[dev] / asyncr["CPU"]
                print(f"{dev} vs CPU -> latência: {lat:.2f}x menor | débito: {thr:.2f}x maior")

if __name__ == "__main__":
    run_comparison()
