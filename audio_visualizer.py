import librosa
import numpy as np
import cv2
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.VideoClip import VideoClip
import argparse
import os

# === Parse Arguments ===
parser = argparse.ArgumentParser(description="Living blob audio visualizer")
parser.add_argument("input_path", help="Path to your MP3 file (e.g., ./song.mp3)")
args = parser.parse_args()

# === Folder Setup ===
OUTPUT_FOLDER = "output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === File Paths ===
INPUT_PATH = args.input_path
BASE_NAME = os.path.splitext(os.path.basename(INPUT_PATH))[0]
OUTPUT_PATH = os.path.join(OUTPUT_FOLDER, f"{BASE_NAME}_visualizer.mp4")

# === Visual Settings ===
FPS = 30
HEIGHT, WIDTH = 720, 1280
CENTER = (WIDTH // 2, HEIGHT // 2)
alpha = 0.15  # smoothing factor

# === Load Audio ===
y, sr = librosa.load(INPUT_PATH, sr=None)
duration = librosa.get_duration(y=y, sr=sr)
samples_per_frame = int(sr / FPS)

# === STFT Analysis ===
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

# === Globals to store previous energy
prev_bass, prev_mid, prev_treb = 0, 0, 0

# === Helper: Draw Organic Blob ===
def draw_blob(img, center, base_radius, energy, color, time, thickness=0):
    num_points = 180
    points = []

    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        wiggle = np.sin(angle * 6 + time * 4)  # wavy distortion
        dynamic_radius = base_radius + wiggle * 20 + energy * 0.05
        x = int(center[0] + dynamic_radius * np.cos(angle))
        y = int(center[1] + dynamic_radius * np.sin(angle))
        points.append((x, y))

    pts = np.array(points, np.int32).reshape((-1, 1, 2))

    if thickness == 0:
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], color)
        return cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
    else:
        return cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

# === Frame Generator ===
def make_frame(t):
    global prev_bass, prev_mid, prev_treb

    frame_idx = int(t * FPS)
    raw_bass, raw_mid, raw_treb = get_energies(frame_idx)

    # === Smooth energies
    bass = alpha * raw_bass + (1 - alpha) * prev_bass
    mid = alpha * raw_mid + (1 - alpha) * prev_mid
    treb = alpha * raw_treb + (1 - alpha) * prev_treb
    prev_bass, prev_mid, prev_treb = bass, mid, treb

    # === Background pulse
    base_brightness = int(20 + min(bass * 0.01, 1) * 50)
    img = np.full((HEIGHT, WIDTH, 3), base_brightness, dtype=np.uint8)

    # === HSV color shifting
    hue = int((t * 30 + treb * 0.01) % 180)
    color_hsv = np.uint8([[[hue, 255, 255]]])
    core_color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()

    # === Radius based on bass energy
    scale = lambda v: int(np.clip(v * 0.015, 0, 1) * 250)
    base_radius = scale(bass) + 80

    # === Draw the breathing blob
    img = draw_blob(img, CENTER, base_radius, bass, core_color, t)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === Generate Video ===
audioclip = AudioFileClip(INPUT_PATH)
videoclip = VideoClip(make_frame, duration=duration).with_audio(audioclip).with_fps(FPS)
videoclip.write_videofile(OUTPUT_PATH, codec='libx264', audio_codec='aac')
