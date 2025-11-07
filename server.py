# server.py — faster-whisper, MIC+SYS capture, per-stream translation with UI control + RMS fix
import asyncio, json, threading
from typing import Set, Dict

import numpy as np
import pyaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from faster_whisper import WhisperModel

# -------------------- CONFIG --------------------
MODEL_NAME       = "tiny"
SAMPLE_RATE      = 16000
FRAME_MS         = 20

CHUNK_SEC_MIC    = 2.0     # better for mic
CHUNK_SEC_SYS    = 2.5

FALLBACK_RATE    = 48000

MIC_IDX          = 21
SYS_IDX          = 1
MIC_CHANNELS     = 2
SYS_CHANNELS     = 2

VALID_TARGETS = {"hi","en","mr","gu","bn","ta","te","ml","kn"}

TARGET_MIC = "hi"
TARGET_SYS = "hi"

# -------------------- APP / WS --------------------
app = FastAPI()

class Broadcast:
    def __init__(self):
        self.clients: Set[WebSocket] = set()
        self._lock = asyncio.Lock()
    async def register(self, ws: WebSocket):
        await ws.accept()
        async with self._lock:
            self.clients.add(ws)
    async def unregister(self, ws: WebSocket):
        async with self._lock:
            self.clients.discard(ws)
    async def send(self, text: str):
        async with self._lock:
            dead=[]
            for ws in list(self.clients):
                try: await ws.send_text(text)
                except: dead.append(ws)
            for ws in dead: self.clients.discard(ws)

broadcast = Broadcast()

@app.get("/")
def home():
    return HTMLResponse(open("index_home.html","r",encoding="utf-8").read())

@app.get("/mic")
def mic():
    return HTMLResponse(open("index_mic.html","r",encoding="utf-8").read())

@app.get("/sys")
def sys():
    return HTMLResponse(open("index_sys.html","r",encoding="utf-8").read())

@app.get("/video")
def video():
    return HTMLResponse(open("index_video.html","r",encoding="utf-8").read())


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await broadcast.register(ws)
    try:
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except:
                continue
            if msg.get("type") == "set_target":
                scope = str(msg.get("scope","")).upper()
                lang  = str(msg.get("lang","")).lower()
                if lang in VALID_TARGETS and scope in {"MIC","SYS"}:
                    global TARGET_MIC, TARGET_SYS
                    if scope=="MIC":
                        TARGET_MIC = lang
                        await ws.send_text(json.dumps({"type":"ack_target","scope":"MIC","lang":TARGET_MIC}))
                    else:
                        TARGET_SYS = lang
                        await ws.send_text(json.dumps({"type":"ack_target","scope":"SYS","lang":TARGET_SYS}))
    except WebSocketDisconnect:
        pass
    finally:
        await broadcast.unregister(ws)

# -------------------- TRANSLATION --------------------
_translators: Dict[str, object] = {}

def translate_text(text: str, lang: str) -> str:
    if not text or lang not in VALID_TARGETS:
        return text
    try:
        tr = _translators.get(lang)
        if tr is None:
            from deep_translator import GoogleTranslator
            tr = GoogleTranslator(source="auto", target=lang)
            _translators[lang] = tr
        return tr.translate(text)
    except: return text

# -------------------- AUDIO --------------------
pa = pyaudio.PyAudio()

def _open_stream(idx: int, ch: int):
    # try 16000
    try:
        return pa.open(
            format=pyaudio.paInt16,
            channels=ch,
            rate=16000,
            input=True,
            input_device_index=idx,
            frames_per_buffer=int(16000*(FRAME_MS/1000.0)),
        ), 16000
    except:
        # try 48000
        return pa.open(
            format=pyaudio.paInt16,
            channels=ch,
            rate=48000,
            input=True,
            input_device_index=idx,
            frames_per_buffer=int(48000*(FRAME_MS/1000.0)),
        ), 48000


def _downmix_to_mono(i16: np.ndarray, ch: int) -> np.ndarray:
    if ch == 1: return i16
    return i16.reshape(-1, ch).mean(axis=1).astype(np.int16)

def _resample_linear(f32: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or f32.size == 0: return f32
    old = len(f32)
    new = int(old*(dst_rate/src_rate))
    return np.interp(np.linspace(0,old,new,endpoint=False), np.arange(old), f32).astype(np.float32)

def _capture_loop(idx, ch, label, model, loop):
    stream, rate = _open_stream(idx, ch)

    frame_size   = int(rate*(FRAME_MS/1000.0))
    chunk_target = int(SAMPLE_RATE * (CHUNK_SEC_MIC if label=="MIC" else CHUNK_SEC_SYS))
    print(f"[{label}] Capturing @{rate}Hz device={idx}")

    buf = np.empty(0, dtype=np.float32)
    while True:
        data = stream.read(frame_size, exception_on_overflow=False)
        i16  = np.frombuffer(data, dtype=np.int16)
        i16  = _downmix_to_mono(i16, ch)
        f32  = i16.astype(np.float32)/32768.0
        if rate != SAMPLE_RATE:
            f32 = _resample_linear(f32, rate, SAMPLE_RATE)

        buf = np.concatenate((buf, f32)) if buf.size else f32
        if buf.size < chunk_target:
            continue

        audio = buf[:chunk_target]
        buf   = buf[chunk_target:]
        audio = np.ascontiguousarray(audio.astype(np.float32))

        # RMS SILENCE FILTER (fixes MIC random hallucinations)
        rms = float(np.sqrt(np.mean(audio**2)))
        if label=="MIC" and rms < 0.005:
            continue

        segs, info = model.transcribe(audio, beam_size=1, language=None, vad_filter=True)
        text = " ".join(s.text.strip() for s in segs).strip()
        if not text:
            continue

        if label=="MIC":
            out = translate_text(text, TARGET_MIC)
        else:
            out = translate_text(text, TARGET_SYS)

        payload = f"[{label}] {out}"
        asyncio.run_coroutine_threadsafe(broadcast.send(payload), loop)

# -------------------- START --------------------
@app.on_event("startup")
async def _startup():
    print("[INIT] Loading faster-whisper:", MODEL_NAME)
    model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
    loop  = asyncio.get_event_loop()

    threading.Thread(target=_capture_loop,args=(MIC_IDX,MIC_CHANNELS,"MIC",model,loop),daemon=True).start()
    threading.Thread(target=_capture_loop,args=(SYS_IDX,SYS_CHANNELS,"SYS",model,loop),daemon=True).start()

# uvicorn server:app --host 0.0.0.0 --port 8000

