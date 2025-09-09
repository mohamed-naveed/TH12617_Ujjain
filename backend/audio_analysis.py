# audio_analysis.py
import librosa

def analyze_audio(audio_path, panic_threshold=0.06):
    try:
        # Load audio file
        y, sr = librosa.load(audio_path, sr=None)
        # Calculate energy
        energy = sum(abs(y)) / len(y)
        # Panic decision
        panic_status = "PANIC" if energy > panic_threshold else "NORMAL"
        return {"energy": round(energy, 4), "status": panic_status}
    except Exception as e:
        return {"error": str(e)}

# Test the module
audio_file = "dataset/audio/crowd_panic.wav"
print(analyze_audio(audio_file))
