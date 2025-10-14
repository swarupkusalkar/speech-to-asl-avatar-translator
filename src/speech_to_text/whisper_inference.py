# src/speech_to_text/whisper_inference.py
import argparse
import os
from faster_whisper import WhisperModel

def transcribe_file(path, model_size="base", device="cuda", compute_type="float16"):
    """
    Transcribe using faster-whisper.
    model_size: tiny, base, small, medium, large
    device: "cuda" or "cpu"
    compute_type (recommended):
      - "float16" for GPU
      - "int8_float16" or "int8" for CPU quantized (faster, less memory)
      - "float32" safe fallback
    """
    print(f"[INFO] device={device}, model={model_size}, compute_type={compute_type}")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    segments, info = model.transcribe(path, beam_size=5)
    text = " ".join([seg.text.strip() for seg in segments]).strip()
    return text, info

def download_audio_from_youtube(url, outtmpl="downloads/%(id)s.%(ext)s"):
    from yt_dlp import YoutubeDL
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": outtmpl,
        "quiet": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
    }
    os.makedirs(os.path.dirname(outtmpl), exist_ok=True)
    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filename = ydl.prepare_filename(info)
        base, _ = os.path.splitext(filename)
        return base + ".mp3"

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", type=str, help="Path to audio/video file")
    p.add_argument("--youtube", type=str, help="YouTube URL to download audio from")
    p.add_argument("--model", type=str, default="base", help="model size (tiny, base, small, medium, large)")
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    p.add_argument("--compute", type=str, default=None, help="compute_type (float16/int8/int8_float16/float32)")
    args = p.parse_args()

    if args.youtube:
        print("[INFO] Downloading audio from YouTube...")
        audio_file = download_audio_from_youtube(args.youtube, outtmpl="downloads/%(id)s.%(ext)s")
    elif args.file:
        audio_file = args.file
    else:
        raise SystemExit("Provide --file <path> or --youtube <url>")

    # Choose compute_type if not provided
    compute = args.compute
    if compute is None:
        compute = "float16" if args.device == "cuda" else "int8"

    txt, info = transcribe_file(audio_file, model_size=args.model, device=args.device, compute_type=compute)
    print("\n=== TRANSCRIPT ===\n")
    print(txt)
    print("\n=== INFO ===\n")
    print(info)
