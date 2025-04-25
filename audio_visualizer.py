import librosa
import numpy as np
import cv2
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import VideoClip
import argparse
import os

# === Parse Args ===
parser = argparse.ArgumentParser(description="Animated audio visualizer")
parser.add_argument("input_path", help="Path to your MP3 file (e.g., ./mysong.mp3)")
args = parser.parse_args()

# === Folder Setup ===
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === File Paths ===
INPUT_PATH = args.input_path
BASE_NAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{BASE_NAME}_visualizer.mp4")

# === Settings ===
FPS = 30
HEIGHT, WIDTH = 720, 1280
CENTER = (WIDTH // 2, HEIGHT // 2)

# === Load Audio ===
y, sr = librosa.load(INPUT_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
samples_per_frame = int(sr / FPS)

# === STFT ===
S = np.abs(librosa.stft(y, n_fft=2048, hop_length=samples_per_frame))
frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
bass_idx = np.where(frequencies < 200)[0]
mid_idx = np.where((frequencies >= 200) & (frequencies < 2000))[0]
treble_idx = np.where(frequencies >= 2000)[0]

def get_energies(frame_idx):
    if frame_idx >= S.shape[1]:
        return 0, 0, 0
    bass = np.mean(S[bass_idx, frame_idx])
    mid = np.mean(S[mid_idx, frame_idx])
    treb = np.mean(S[treble_idx, frame_idx])
    return bass, mid, treb

# === Waveform Amplitude ===
frame_amplitudes = librosa.util.frame(y, frame_length=samples_per_frame, hop_length=samples_per_frame)
waveform_amplitude = np.mean(np.abs(frame_amplitudes), axis=0)

# === Frame Generator ===
# Persistent state to hold previous energies
prev_bass, prev_mid, prev_treb = 0, 0, 0
alpha = 0.15  # smoothing factor

def make_frame(t):
    global prev_bass, prev_mid, prev_treb

    frame_idx = int(t * FPS)
    raw_bass, raw_mid, raw_treb = get_energies(frame_idx)

    # === Smooth the energy for stable motion
    bass = alpha * raw_bass + (1 - alpha) * prev_bass
    mid = alpha * raw_mid + (1 - alpha) * prev_mid
    treb = alpha * raw_treb + (1 - alpha) * prev_treb
    prev_bass, prev_mid, prev_treb = bass, mid, treb

    # === Create a dark background that pulses slightly with bass
    base_brightness = int(20 + min(bass * 0.01, 1) * 50)
    img = np.full((HEIGHT, WIDTH, 3), base_brightness, dtype=np.uint8)

    # === HSV Color shifting: hue based on time or treble
    hue = int((t * 30 + treb * 0.01) % 180)
    color_hsv = np.uint8([[[hue, 255, 255]]])  # full saturation, value
    core_color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

    # === Radii based on smoothed energy
    scale = lambda v: int(np.clip(v * 0.015, 0, 1) * 250)
    bass_radius = scale(bass) + 50
    mid_radius = bass_radius + scale(mid)
    treb_radius = mid_radius + scale(treb)

    # === Helper to draw smooth glowing circles
    def draw_filled_circle(img, center, radius, color, alpha):
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, thickness=-1)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # === Draw blended concentric glow layers
    img = draw_filled_circle(img, CENTER, treb_radius, core_color, 0.2)
    img = draw_filled_circle(img, CENTER, mid_radius, core_color, 0.35)
    img = draw_filled_circle(img, CENTER, bass_radius, core_color, 0.5)

    # Optional: soft outer glow
    glow_radius = treb_radius + 30
    img = draw_filled_circle(img, CENTER, glow_radius, core_color, 0.05)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# === Generate Video ===
audioclip = AudioFileClip(INPUT_PATH)
videoclip = VideoClip(make_frame, duration=duration).with_audio(audioclip).with_fps(FPS)
videoclip.write_videofile(OUTPUT_PATH, codec='libx264', audio_codec='aac')
