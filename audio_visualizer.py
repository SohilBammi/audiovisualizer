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
def make_frame(t):
    frame_idx = int(t * FPS)
    bass, mid, treble = get_energies(frame_idx)

    img = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    scale = lambda v: int(np.clip(v * 0.015, 0, 1) * 250)

    # Radii based on frequency energy
    bass_radius = scale(bass) + 50
    mid_radius = bass_radius + scale(mid)
    treb_radius = mid_radius + scale(treble)

    # Color layers (BGR format)
    bass_color = (60, 60, 255)   # Reddish
    mid_color = (60, 255, 60)    # Green
    treb_color = (255, 60, 255)  # Magenta

    def draw_filled_circle(img, center, radius, color, alpha):
        overlay = img.copy()
        cv2.circle(overlay, center, radius, color, thickness=-1)
        return cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw concentric circles from outermost to innermost for layering
    img = draw_filled_circle(img, CENTER, treb_radius, treb_color, 0.25)
    img = draw_filled_circle(img, CENTER, mid_radius, mid_color, 0.4)
    img = draw_filled_circle(img, CENTER, bass_radius, bass_color, 0.6)

    # Optional glow around everything
    glow_radius = treb_radius + 20
    img = draw_filled_circle(img, CENTER, glow_radius, (100, 100, 255), 0.08)

    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# === Generate Video ===
audioclip = AudioFileClip(INPUT_PATH)
videoclip = VideoClip(make_frame, duration=duration).with_audio(audioclip).with_fps(FPS)
videoclip.write_videofile(OUTPUT_PATH, codec='libx264', audio_codec='aac')
