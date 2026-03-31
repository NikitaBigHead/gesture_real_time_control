# ZMQ RealSense Streamer

Система передачи данных **Orange Pi → ПК** по протоколу ZeroMQ (PUB/SUB).

## Что передаётся

| Поток | Порт | Формат |
|-------|------|--------|
| RGB   | 5550 | JPEG (сжатый, качество 80%) |
| Depth | 5551 | raw uint16, 640×480, единицы мм |
| Audio | 5552 | raw int16 PCM, 16 кГц, моно |

Каждое сообщение: `[topic_bytes][timestamp_float64][payload]`

---

## Быстрый старт

### 1. ПК (сервер) — запустить первым

```bash
pip install -r requirements_server.txt
python server.py
```

### 2. Orange Pi (клиент)

```bash
pip install -r requirements_client.txt
# Install pyrealsense2 on ARM (see below)
# Edit SERVER_IP in client.py
python list_microphones.py
python client.py
```

---

## Установка pyrealsense2 на Orange Pi (ARM)

```bash
# Option 1 — package from the Intel repository
sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key F6E65AC044F831AC
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main"
sudo apt update
sudo apt install librealsense2-dev python3-pyrealsense2

# Option 2 — build from source
# https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md
```

---

## Настройка (`client.py`)

```python
SERVER_IP    = "192.168.1.100"  # <- IP of your PC
RS_WIDTH     = 640
RS_HEIGHT    = 480
RS_FPS       = 30
JPEG_QUALITY = 80               # 0–100, lower means less traffic
AUDIO_RATE   = 16000            # Hz
MICROPHONE_DEVICE_ID = None     # None = first available input
```

You can also choose the microphone at launch:

```bash
python client.py --mic-id 2
```

---

## Добавление обработки на сервере

Откройте `server.py` и замените тело заглушек:

```python
def handle_rgb(frame: np.ndarray, send_ts: float) -> None:
    # frame — BGR uint8 (640×480×3)
    # Your code: YOLO, MediaPipe, cv2.imshow, video recording...
    results = model(frame)

def handle_depth(depth: np.ndarray, send_ts: float) -> None:
    # depth — uint16 (640×480), values in mm
    # Your code: point cloud, SLAM, obstacle detection...
    pass

def handle_audio(samples: np.ndarray, send_ts: float) -> None:
    # samples — int16 PCM (1024 samples, 16 kHz)
    # Your code: Whisper, Vosk, WAV recording...
    pass
```

---

## Архитектура

```
Orange Pi                          ПК
─────────────────────              ─────────────────────
RealSense ──► RGBStreamer  ──PUB──► RGBReceiver  ──► handle_rgb()
             DepthStreamer ──PUB──► DepthReceiver ──► handle_depth()
Микрофон ──► AudioStreamer ──PUB──► AudioReceiver ──► handle_audio()
```

- Паттерн **PUB/SUB**: клиент `connect`, сервер `bind`
- Если клиент опережает сервер — старые кадры отбрасываются (`SNDHWM=10`)
- Каждый поток в отдельном потоке Python (`threading.Thread`)
- При отсутствии RealSense или микрофона клиент автоматически переключается на заглушки-генераторы

---

## Проверка сети

```bash
# On the PC, make sure the ports are open
sudo ufw allow 5550:5552/tcp

# Check the connection from Orange Pi
python -c "import zmq; ctx=zmq.Context(); s=ctx.socket(zmq.REQ); s.connect('tcp://192.168.1.100:5550'); print('OK')"
```
