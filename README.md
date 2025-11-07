# 🌬️ Shruti — Real-Time Audio Translator (MIC + System Audio)

Shruti is a low-latency real-time speech translation system that can translate both:

- **Microphone live speech** (your voice)
- **System audio** (YouTube / Movies / PC audio)

It uses:
- Download https://vb-audio.com/Cable/  for **minimum latency** + **high accuracy** for system audio
- `faster-whisper` (GPU accelerated)
- WebSockets (real-time push)
- Deep Translator (Google Translate)
- FastAPI backend
- Pyaudio input streaming

This project was built in a hackathon — our focus was **minimum latency** + **high practicality**.

---

## 🚀 Features

| Feature | Status |
|--------|-------|
| Microphone real-time speech → translated subtitles | ✅ |
| System-audio (Stereo Mix / VB Cable) translation | ✅ |
| Automatic language detection | ✅ |
| GPU support (CUDA) → < 200ms inference | ✅ |
| UI to choose output target language | ✅ |
| Silence filtering (VAD + RMS) | ✅ |
| Multi-language target (hi/en/mr/gu/bn/ta/te/ml/kn) | ✅ |


---

## 🧠 How it works

1. Audio is continuously captured from Mic + System simultaneously
2. Faster-Whisper detects the language (auto detect)
3. VAD/RMS removes silent parts (reduces hallucination)
4. Text is translated to target language via GoogleTranslator
5. UI displays translated text live through WebSocket stream

```bash
pip install -r requirements.txt
