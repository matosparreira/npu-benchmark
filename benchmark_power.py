import os
import time
import glob
import urllib.request
import openvino as ov
import numpy as np

# Mede o consumo de energia por dispositivo (CPU / GPU / NPU) via Intel RAPL.
#
# LIMITAÇÃO IMPORTANTE: o RAPL mede o domínio "package-0" — o pacote inteiro do
# SoC, onde a CPU, a iGPU e a NPU partilham o mesmo contador de energia. Não há
# domínio RAPL dedicado por acelerador. Portanto não medimos a potência isolada
# de cada bloco; medimos a potência TOTAL do pacote enquanto a carga corre em
# cada dispositivo e subtraímos o idle. O resultado útil é a energia por
# inferência (mJ/inf) e a eficiência (inf/s por watt) de cada dispositivo.
#
# Requer leitura de /sys/class/powercap/intel-rapl:0/energy_uj, que por omissão
# é root-only (mitigação do CVE-2020-8694). Dá acesso uma vez com:
#   sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj
# (reverte no reboot).

MODEL_URL = "https://github.com/onnx/models/raw/main/validated/vision/classification/resnet/model/resnet50-v1-12.onnx"
MODEL_PATH = "resnet50-v1-12.onnx"

DEVICES = ["CPU", "GPU", "NPU"]
INPUT_SHAPE = [1, 3, 224, 224]
DURATION_S = 10.0   # carga sustentada por dispositivo (boa média de potência)
IDLE_S = 5.0        # baseline em repouso

RAPL = "/sys/class/powercap/intel-rapl:0"

def rapl_available():
    try:
        with open(f"{RAPL}/energy_uj") as f:
            f.read()
        return True
    except Exception:
        return False

def read_energy_j():
    # Energia acumulada em joules (o contador é em microjoules).
    with open(f"{RAPL}/energy_uj") as f:
        return int(f.read()) / 1e6

def read_max_range_j():
    with open(f"{RAPL}/max_energy_range_uj") as f:
        return int(f.read()) / 1e6

MAX_RANGE_J = None

def energy_delta(start, end):
    # Lida com o wraparound do contador RAPL.
    if end >= start:
        return end - start
    return end - start + MAX_RANGE_J

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("[+] A descarregar modelo ResNet-50 (~97MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[+] Download concluído.")

def measure_idle():
    print(f"[+] A medir baseline em repouso ({IDLE_S:.0f}s)...")
    e0 = read_energy_j()
    t0 = time.time()
    time.sleep(IDLE_S)
    dt = time.time() - t0
    de = energy_delta(e0, read_energy_j())
    return de / dt  # watts

def measure_device(core, model, device, dummy_input):
    # Carga assíncrona sustentada (pipeline cheio) durante DURATION_S.
    compiled = core.compile_model(model, device)
    input_layer = compiled.input(0)
    queue = ov.AsyncInferQueue(compiled)
    feed = {input_layer.any_name: dummy_input}
    for _ in range(8):  # aquecimento
        queue.start_async(feed)
    queue.wait_all()

    count = 0
    e0 = read_energy_j()
    t0 = time.time()
    deadline = t0 + DURATION_S
    while time.time() < deadline:
        queue.start_async(feed)  # bloqueia quando o pipeline está cheio
        count += 1
    queue.wait_all()
    dt = time.time() - t0
    de = energy_delta(e0, read_energy_j())

    return {
        "throughput": count / dt,
        "power": de / dt,                 # watts (pacote, durante a carga)
        "energy_per_inf_mj": de / count * 1000.0,
    }

def run():
    if not rapl_available():
        print("[-] ERRO: não consigo ler", f"{RAPL}/energy_uj", "(root-only).")
        print("    Dá acesso uma vez com:")
        print("    sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj")
        return

    global MAX_RANGE_J
    MAX_RANGE_J = read_max_range_j()

    core = ov.Core()
    devices = core.available_devices
    print(f"\n[+] Dispositivos de IA detetados no Linux: {devices}")
    targets = [d for d in DEVICES if d in devices]
    if not targets:
        print("[-] ERRO: nenhum dos dispositivos-alvo está disponível.")
        return

    download_model()
    print(f"[+] A ler o modelo {MODEL_PATH}...")
    model = core.read_model(MODEL_PATH)
    model.reshape({model.input(0).any_name: INPUT_SHAPE})
    dummy_input = np.random.randn(*INPUT_SHAPE).astype(np.float32)

    idle_w = measure_idle()
    print(f"[+] Potência em repouso (pacote): {idle_w:.2f} W")

    res = {}
    for device in targets:
        print(f"[+] A medir {device} (carga sustentada {DURATION_S:.0f}s)...")
        res[device] = measure_device(core, model, device, dummy_input)

    print("\n========== ENERGIA (ResNet-50, pacote RAPL) ==========")
    print(f"{'Disp.':6}{'Débito':>10}{'Potência':>11}{'Ativa':>9}{'mJ/inf':>10}{'inf/s/W':>10}")
    print(f"{'':6}{'(inf/s)':>10}{'(W)':>11}{'(W)':>9}{'':10}{'(efic.)':>10}")
    for d in targets:
        r = res[d]
        active = r["power"] - idle_w
        eff = r["throughput"] / r["power"]
        print(f"{d:6}{r['throughput']:>10.0f}{r['power']:>11.2f}{active:>9.2f}"
              f"{r['energy_per_inf_mj']:>10.1f}{eff:>10.2f}")
    print("======================================================")
    print(f"Repouso: {idle_w:.2f} W | 'Ativa' = potência do pacote menos repouso.")
    print("Nota: RAPL mede o pacote inteiro (CPU+GPU+NPU partilham o domínio).")

if __name__ == "__main__":
    run()
