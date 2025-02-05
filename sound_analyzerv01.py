import sys
import os
import time
import librosa
import numpy as np
import matplotlib.pyplot as plt
import torch
from pyannote.audio import Pipeline
import whisper
from spleeter.separator import Separator
import scipy.signal
import soundfile as sf
import sounddevice as sd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QFileDialog, QLabel,
    QVBoxLayout, QWidget, QTextEdit, QProgressBar, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QObject, QRunnable, QThreadPool
from speechbrain.inference import EncoderClassifier
from transformers import pipeline
import librosa.display
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from audio_utils import load_audio_in_chunks, process_audio_in_chunks, get_device, separate_frequency_band
from matplotlib.widgets import Cursor
from pyannote.core import Segment, Timeline

sys.path.append('C:/Users/matej/Music/v2/speechbrain')

class ProgressEmitter(QObject):
    progress = pyqtSignal(int)
    finished = pyqtSignal(object)

class ModelLoader(QRunnable):
    def __init__(self, model_name, model_path, device):
        super().__init__()
        self.model_name = model_name
        self.model_path = model_path
        self.device = device
        self.signals = ProgressEmitter()

    def run(self):
        try:
            model = EncoderClassifier.from_hparams(source=self.model_path, savedir=f"tmp_{self.model_name}_model", run_opts={"device": self.device})
            self.signals.finished.emit((self.model_name, model, None))
        except Exception as e:
            self.signals.finished.emit((self.model_name, None, str(e)))

class AudioAnalysisTask(QRunnable):
    def __init__(self, file_path, whisper_model):
        super().__init__()
        self.file_path = file_path
        self.whisper_model = whisper_model
        self.signals = ProgressEmitter()

    def run(self):
        try:
            audio, sr = librosa.load(self.file_path, sr=None)
            duration = librosa.get_duration(y=audio, sr=sr)
            
            # Rozdělení audio signálu na menší části pro sledování pokroku
            chunk_duration = 10  # Délka každé části v sekundách
            num_chunks = int(np.ceil(duration / chunk_duration))
            transcript = ""

            for i in range(num_chunks):
                start = i * chunk_duration
                end = min((i + 1) * chunk_duration, duration)
                audio_chunk = audio[int(start * sr):int(end * sr)]
                temp_file = f"temp_chunk_{i}.wav"
                sf.write(temp_file, audio_chunk, sr)

                chunk_transcript = self.whisper_model.transcribe(temp_file, language="cs")["text"]
                transcript += chunk_transcript + " "

                # Emitování pokroku
                progress = int(((i + 1) / num_chunks) * 100)
                self.signals.progress.emit(progress)

                # Odstranění dočasného souboru
                os.remove(temp_file)

            self.signals.finished.emit((transcript.strip(), duration))
        except Exception as e:
            self.signals.finished.emit((f"Chyba při analýze zvuku: {str(e)}", 0))

class SeparateVoiceTask(QRunnable):
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.signals = ProgressEmitter()

    def run(self):
        try:
            start_time = time.time()
            separator = Separator('spleeter:2stems')
            separator.separate_to_file(self.file_path, 'separated_audio')
            for i in range(100):  # Simulace pokroku
                time.sleep(0.1)
                self.signals.progress.emit(i + 1)
            remaining_time = time.time() - start_time
            self.signals.finished.emit(remaining_time)
        except Exception as e:
            self.signals.finished.emit(0)

class AudioAnalyzer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.device = get_device()
        self.initUI()
        self.models = {}
        self.model_configs = [
            ("music", "speechbrain/music-detection"),
            ("tv", "speechbrain/tv-detection"),
            ("street", "speechbrain/street-detection"),
            ("intimate", "speechbrain/intimate-detection"),
            ("emotion", "speechbrain/emotion-recognition-wav2vec2"),
        ]
        self.load_models()
        self.playing = False
        self.stream = None
        self.audio_signal = None  # Uloží audio signál pro přehrávání
        self.sample_rate = None  # Uloží vzorkovací frekvenci
        self.audio_position = 0  # Track the current position in the audio signal

    def load_models(self):
        self.text_output.append(f"Začínám načítat modely... (Zařízení: {self.device})")
        for model_name, model_path in self.model_configs:
            loader = ModelLoader(model_name, model_path, self.device)
            loader.signals.finished.connect(self.on_model_loaded)
            QThreadPool.globalInstance().start(loader)

    def on_model_loaded(self, result):
        model_name, model, error = result
        if model:
            self.models[model_name] = model
            self.text_output.append(f"Model {model_name} byl úspěšně načten.")
        else:
            self.text_output.append(f"Nepodařilo se načíst model {model_name}. Chyba: {error}")
        
        if len(self.models) + len([m for m in self.model_configs if m[0] not in self.models]) == len(self.model_configs):
            self.text_output.append("Načítání modelů dokončeno.")
            self.update_ui_based_on_loaded_models()

    def update_ui_based_on_loaded_models(self):
        self.classify_sounds_button.setEnabled("music" in self.models)
        self.detect_emotion_button.setEnabled("emotion" in self.models)
        self.analyze_environment_button.setEnabled(any(m in self.models for m in ["music", "tv", "street", "intimate"]))

    def initUI(self):
        self.setWindowTitle("Audio Analyzátor")
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.label = QLabel("Vyberte zvukový soubor:")
        layout.addWidget(self.label)

        self.select_button = QPushButton("Otevřít soubor")
        self.select_button.clicked.connect(self.open_file)
        layout.addWidget(self.select_button)

        self.analyze_button = QPushButton("Analyzovat zvuk")
        self.analyze_button.clicked.connect(self.analyze_audio)
        layout.addWidget(self.analyze_button)

        self.text_output = QTextEdit()
        self.text_output.setReadOnly(True)
        layout.addWidget(self.text_output)

        self.spectrogram_label = QLabel()
        layout.addWidget(self.spectrogram_label)

        self.separate_button = QPushButton("Separovat hlasy a hudbu")
        self.separate_button.clicked.connect(self.separate_voice)
        layout.addWidget(self.separate_button)

        self.detect_emotion_button = QPushButton("Detekovat emoce")
        self.detect_emotion_button.clicked.connect(self.detect_emotions)
        self.detect_emotion_button.setEnabled(False)  # Začíná jako zakázané
        layout.addWidget(self.detect_emotion_button)

        self.classify_sounds_button = QPushButton("Klasifikovat zvuky")
        self.classify_sounds_button.clicked.connect(self.classify_sounds)
        self.classify_sounds_button.setEnabled(False)  # Začíná jako zakázané
        layout.addWidget(self.classify_sounds_button)
        
        self.detect_speakers_button = QPushButton("Rozpoznat mluvčí")
        self.detect_speakers_button.clicked.connect(self.detect_speakers)
        layout.addWidget(self.detect_speakers_button)
        
        self.detect_keywords_button = QPushButton("Hledat klíčová slova")
        self.detect_keywords_button.clicked.connect(self.detect_keywords)
        layout.addWidget(self.detect_keywords_button)
        
        self.analyze_tone_button = QPushButton("Analyzovat tón řeči")
        self.analyze_tone_button.clicked.connect(self.analyze_tone)
        layout.addWidget(self.analyze_tone_button)
        
        self.analyze_environment_button = QPushButton("Analyzovat zvukové prostředí")
        self.analyze_environment_button.clicked.connect(self.analyze_environment)
        self.analyze_environment_button.setEnabled(False)  # Začíná jako zakázané
        layout.addWidget(self.analyze_environment_button)
        
        self.generate_visual_report_button = QPushButton("Generovat vizuální přehled")
        self.generate_visual_report_button.clicked.connect(self.generate_visual_report)
        layout.addWidget(self.generate_visual_report_button)

        self.detect_background_sounds_button = QPushButton("Detekovat pozadí")
        self.detect_background_sounds_button.clicked.connect(self.detect_background_sounds)
        layout.addWidget(self.detect_background_sounds_button)

        self.detect_volume_changes_button = QPushButton("Detekovat změny hlasitosti")
        self.detect_volume_changes_button.clicked.connect(self.detect_volume_changes)
        layout.addWidget(self.detect_volume_changes_button)
        
        self.select_frequency_button = QPushButton("Vybrat frekvenci ze spektrogramu")
        self.select_frequency_button.clicked.connect(self.select_frequency_from_spectrogram)
        layout.addWidget(self.select_frequency_button)
        
        self.separate_frequency_button = QPushButton("Separovat zvuk na vybrané frekvenci")
        self.separate_frequency_button.clicked.connect(self.separate_frequency)
        layout.addWidget(self.separate_frequency_button)

        self.play_pause_button = QPushButton("Přehrát zvuk")
        self.play_pause_button.clicked.connect(self.play_pause_audio)
        layout.addWidget(self.play_pause_button)

        self.track_button = QPushButton("Track Voice on Selected Frequency")
        self.track_button.clicked.connect(self.track_voice_on_frequency)
        layout.addWidget(self.track_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.file_path = ""
        self.whisper_model = whisper.load_model("large", device=self.device)
        self.diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="YOUR_TOKEN_HERE")
        self.diarization_pipeline = self.diarization_pipeline.to(self.device)
        self.asr_pipeline = pipeline("automatic-speech-recognition", model="MehdiHosseiniMoghadam/wav2vec2-large-xlsr-53-Czech", device=0 if self.device.type == "cuda" else -1)

    def show_error(self, title, message):
        QMessageBox.critical(self, title, message)

    def open_file(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Otevřít zvukový soubor", "", "Audio Files (*.wav *.mp3 *.m4a)"
        )
        if file_name:
            self.file_path = file_name
            self.label.setText(f"Vybraný soubor: {os.path.basename(file_name)}")

    def analyze_audio(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before analyzing.")
            return
        
        self.text_output.setText("🟢 Přepisuji zvuk do textu...")
        self.progress_bar.setValue(0)
        
        task = AudioAnalysisTask(self.file_path, self.whisper_model)
        task.signals.progress.connect(self.update_progress)
        task.signals.finished.connect(self.on_analysis_finished)
        
        QThreadPool.globalInstance().start(task)

    def on_analysis_finished(self, result):
        transcript, duration = result
        output_transcription = "transcriptions.txt"
        with open(output_transcription, "a", encoding="utf-8") as f:
            f.write(f"Soubor: {self.file_path}\nPřepis: {transcript}\n\n")

        self.text_output.append(
            f"📃 Přepis: {transcript}\nVýsledky uloženy do {output_transcription}.\n⏳ Doba zpracování: {duration:.2f} sekund."
        )
        self.progress_bar.setValue(0)  # Resetuj na 0 po dokončení

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def classify_sounds(self):
        if not self.file_path:
            self.show_error("Není vybrán soubor", "Prosím, vyberte zvukový soubor před klasifikací zvuků.")
            return

        if "music" not in self.models:
            self.show_error("Model není k dispozici", "Model pro klasifikaci zvuků není načten.")
            return

        try:
            self.text_output.append("🟢 Klasifikuji zvuky...")
            
            def process_chunk(audio_chunk, sr):
                embeddings = self.models["music"].encode_batch(audio_chunk.unsqueeze(0))
                return torch.mean(embeddings)

            avg_embedding = process_audio_in_chunks(self.file_path, process_chunk, device=self.device)
            sound_category = "Hudba" if avg_embedding > 0.5 else "Hlas"

            output_path = "classified_sounds.txt"
            with open(output_path, "a") as f:
                f.write(f"Soubor: {self.file_path} -> Klasifikace: {sound_category}\n")

            self.text_output.append(
                f"🔍 Zvuky byly klasifikovány jako: {sound_category}\nVýsledky uloženy do {output_path}."
            )
        except Exception as e:
            self.show_error("Chyba klasifikace", f"Během klasifikace zvuků došlo k chybě: {str(e)}")

    def detect_emotions(self):
        if not self.file_path:
            self.show_error("Není vybrán soubor", "Prosím, vyberte zvukový soubor před detekcí emocí.")
            return

        if "emotion" not in self.models:
            self.show_error("Model není k dispozici", "Model pro detekci emocí není načten.")
            return

        try:
            start_time = time.time()
            self.text_output.setText("🟢 Detekuji emoce...")
            
            def process_chunk(audio_chunk, sr):
                if sr != 16000:
                    audio_chunk = librosa.resample(audio_chunk.cpu().numpy(), sr, 16000)
                    audio_chunk = torch.tensor(audio_chunk, device=self.device)
                return predictions.squeeze()

            avg_predictions = process_audio_in_chunks(self.file_path, process_chunk, sr=16000, device=self.device)
            predicted_emotion = torch.argmax(avg_predictions).item()
            emotions = ["Neutral", "Calm", "Happy", "Sad", "Angry", "Fearful", "Disgust", "Surprised"]
            detected_emotion = emotions[predicted_emotion]

            output_path = "detected_emotions.txt"
            with open(output_path, "a") as f:
                f.write(f"Soubor: {self.file_path}\nDetekovaná emoce: {detected_emotion}\n\n")

            remaining_time = time.time() - start_time
            self.text_output.append(
                f"🔍 Detekovaná emoce: {detected_emotion}\nVýsledky uloženy do {output_path}.\n⏳ Doba zpracování: {remaining_time:.2f} sekund."
            )
        except Exception as e:
            self.show_error("Chyba detekce emocí", f"Během detekce emocí došlo k chybě: {str(e)}")

    def separate_voice(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before separating voice and music.")
            return

        try:
            self.text_output.setText("🟢 Separuji hlas a hudbu...")
            self.progress_bar.setValue(0)

            task = SeparateVoiceTask(self.file_path)
            task.signals.progress.connect(self.update_progress)
            task.signals.finished.connect(self.on_separation_finished)
            QThreadPool.globalInstance().start(task)
        except Exception as e:
            self.show_error("Separation Error", f"An error occurred during voice and music separation: {str(e)}")

    def on_separation_finished(self, remaining_time):
        self.text_output.append(
            f"🎶 Separace dokončena. Výsledky uloženy ve složce 'separated_audio'.\n⏳ Doba zpracování: {remaining_time:.2f} sekund."
        )
        self.progress_bar.setValue(0)  # Resetuj na 0 po dokončení

    def detect_speakers(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before detecting speakers.")
            return
    
        try:
            self.text_output.append("🟢 Analyzuji mluvčí...")
            diarization = self.diarization_pipeline(self.file_path)
            
            speakers = set()
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                speakers.add(speaker)
        
            self.text_output.append(f"🔍 Identifikováno {len(speakers)} mluvčích.")
        
            # Print detailed information about each speaker
            for i, speaker in enumerate(speakers, 1):
                speaker_time = sum([segment.duration for segment, _, spk in diarization.itertracks(yield_label=True) if spk == speaker])
                self.text_output.append(f"  Mluvčí {i}: celková doba mluvení {speaker_time:.2f} sekund")
        
        except Exception as e:
            self.show_error("Speaker Detection Error", f"An error occurred during speaker detection: {str(e)}")
            self.text_output.append(f"Detailní chybová zpráva: {str(e)}")

    def detect_keywords(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before detecting keywords.")
            return
        
        try:
            self.text_output.append("🟢 Hledám klíčová slova...")
            keywords = ["miláčku", "lásko", "pojď sem", "přijď večer"]
            transcript = self.whisper_model.transcribe(self.file_path, language="cs")["text"]
            found_keywords = [word for word in keywords if word in transcript]
            
            if found_keywords:
                self.text_output.append(f"🔍 Nalezená klíčová slova: {', '.join(found_keywords)}")
            else:
                self.text_output.append("✅ Žádná klíčová slova nebyla nalezena.")
        except Exception as e:
            self.show_error("Keyword Detection Error", f"An error occurred during keyword detection: {str(e)}")

    def analyze_tone(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before analyzing tone.")
            return

        try:
            self.text_output.append("🟢 Analyzuji tón řeči...")
            audio_signal, sample_rate = librosa.load(self.file_path, sr=None)
            rms_energy = librosa.feature.rms(y=audio_signal)
            avg_energy = np.mean(rms_energy)
            tone = "Stres" if avg_energy > 0.05 else "Klid"
            self.text_output.append(f"🔍 Detekovaný tón řeči: {tone}")
        except Exception as e:
            self.show_error("Tone Analysis Error", f"An error occurred during tone analysis: {str(e)}")

    def analyze_environment(self):
        if not self.file_path:
            self.show_error("Není vybrán soubor", "Prosím, vyberte zvukový soubor před analýzou prostředí.")
            return

        if not any(m in self.models for m in ["music", "tv", "street", "intimate"]):
            self.show_error("Modely nejsou k dispozici", "Žádné modely pro analýzu prostředí nejsou načteny.")
            return

        try:
            self.text_output.append("🟢 Analyzuji zvukové prostředí...")
            
            def process_chunk(audio_chunk, sr):
                results = {}
                for model_name in ["music", "tv", "street", "intimate"]:
                    if model_name in self.models:
                        embeddings = self.models[model_name].encode_batch(audio_chunk.unsqueeze(0))
                        results[model_name] = torch.mean(embeddings).item()
                return results

            avg_results = process_audio_in_chunks(self.file_path, process_chunk, device=self.device)
            
            detected_sounds = [model_name.capitalize() for model_name, value in avg_results.items() if value > 0.5]

            if detected_sounds:
                self.text_output.append(f"🔍 Detekované zvuky: {', '.join(detected_sounds)}")
            else:
                self.text_output.append("✅ Žádné specifické zvuky nebyly detekovány.")
        except Exception as e:
            self.show_error("Chyba analýzy prostředí", f"Během analýzy zvukového prostředí došlo k chybě: {str(e)}")

    def generate_visual_report(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before generating a visual report.")
            return

        try:
            self.text_output.append("🟢 Generuji vizuální přehled...")
            audio_signal, sample_rate = librosa.load(self.file_path, sr=None)

            # Vytvoření seznamu barevných map
            color_maps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'coolwarm', 'Spectral', 'jet']

            # Vytvoření nového okna pro spektrogramy
            spectrogram_window = QWidget()
            spectrogram_layout = QVBoxLayout()
            spectrogram_window.setLayout(spectrogram_layout)
            spectrogram_window.setWindowTitle("Spektrogramy")

            for cmap in color_maps:
                fig, ax = plt.subplots(figsize=(10, 4))
                img = librosa.display.specshow(librosa.amplitude_to_db(np.abs(librosa.stft(audio_signal)), ref=np.max),
                                               y_axis='hz', x_axis='time', ax=ax, cmap=cmap)
                fig.colorbar(img, ax=ax, format="%+2.0f dB")
                ax.set_title(f'Spektrogram ({cmap})')
                
                canvas = FigureCanvas(fig)
                spectrogram_layout.addWidget(canvas)

                # Uložení obrázku
                plt.savefig(f"spectrogram_{cmap}.png")
                plt.close(fig)

            spectrogram_window.show()
            
            self.text_output.append("📊 Vizuální přehledy byly vygenerovány a uloženy jako 'spectrogram_<colormap>.png'.")
        except Exception as e:
            self.show_error("Visual Report Error", f"An error occurred during visual report generation: {str(e)}")

    def detect_background_sounds(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before detecting background sounds.")
            return
        
        try:
            self.text_output.append("🟢 Detekuji zvuky na pozadí...")
            audio_signal, sample_rate = librosa.load(self.file_path, sr=None)
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_signal, sr=sample_rate)
            mean_centroid = np.mean(spectral_centroid)
            
            if mean_centroid < 2000:
                detected_sound = "Nízkofrekvenční zvuk (např. televize, mluvení)"
            elif mean_centroid < 4000:
                detected_sound = "Středofrekvenční zvuk (např. hudba, ulice)"
            else:
                detected_sound = "Vysokofrekvenční zvuk (např. ostré zvuky, intimní zvuky)"
            
            self.text_output.append(f"🔍 Detekovaný typ zvuku: {detected_sound}")
        except Exception as e:
            self.show_error("Background Sound Detection Error", f"An error occurred during background sound detection: {str(e)}")

    def detect_volume_changes(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before detecting volume changes.")
            return
        
        try:
            self.text_output.append("🟢 Detekuji změny hlasitosti...")
            audio_signal, sample_rate = librosa.load(self.file_path, sr=None)
            rms = librosa.feature.rms(y=audio_signal)[0]
            changes = np.diff(rms)
            significant_changes = np.where(np.abs(changes) > np.std(changes))[0]
            
            self.text_output.append(f"🔍 Detekováno {len(significant_changes)} významných změn hlasitosti.")
        except Exception as e:
            self.show_error("Volume Change Detection Error", f"An error occurred during volume change detection: {str(e)}")

    def select_frequency_from_spectrogram(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before selecting frequency.")
            return

        try:
            self.text_output.append("🟢 Zobrazuji spektrogram pro výběr frekvence...")
            audio_signal, sample_rate = librosa.load(self.file_path, sr=None)
            fig, ax = plt.subplots(figsize=(12, 6))
            S = librosa.stft(audio_signal)
            S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
            img = librosa.display.specshow(S_db, sr=sample_rate, x_axis='time', y_axis='hz', ax=ax)
            ax.set_xlabel('Time (seconds)')
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            ax.set_title('Spektrogram s frekvencemi hlasů a časovými úseky mluvčích')

            # Zvýraznění frekvencí hlasů
            pitches, magnitudes = librosa.piptrack(y=audio_signal, sr=sample_rate)
            times = librosa.frames_to_time(np.arange(pitches.shape[1]), sr=sample_rate)
            for t, pitch in zip(times, pitches):
                if pitch.any():
                    ax.plot(t * np.ones_like(pitch), pitch, 'r.', markersize=2)

            # Detekce mluvčích a jejich časových úseků
            diarization = self.diarization_pipeline(self.file_path)
            colors = plt.cm.rainbow(np.linspace(0, 1, len(set(diarization.labels()))))
            for i, (segment, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
                ax.axvspan(segment.start, segment.end, alpha=0.2, color=colors[int(speaker.split('_')[1])-1])
                ax.text(segment.start, ax.get_ylim()[1], f'Speaker {speaker}', 
                        verticalalignment='top', fontsize=8, color=colors[int(speaker.split('_')[1])-1])

            cursor = Cursor(ax, useblit=True, color='green', linewidth=1)
            self.selected_frequency = None

            def onclick(event):
                if event.ydata is not None and event.xdata is not None:
                    self.selected_frequency = event.ydata
                    selected_time = event.xdata
                    plt.close(fig)
                    self.text_output.append(f"🔍 Vybraná frekvence: {self.selected_frequency:.2f} Hz v čase {selected_time:.2f} s")

            fig.canvas.mpl_connect('button_press_event', onclick)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            self.show_error("Spectrogram Error", f"An error occurred while displaying the spectrogram: {str(e)}")

    def separate_frequency(self):
        if not self.file_path:
            self.show_error("No File Selected", "Please select an audio file before separating frequency.")
            return

        if self.selected_frequency is None:
            self.show_error("No Frequency Selected", "Please select a frequency from the spectrogram first.")
            return

        try:
            self.text_output.append(f"🟢 Separuji zvuk na frekvenci {self.selected_frequency:.2f} Hz...")
            audio_signal, sample_rate = librosa.load(self.file_path, sr=None)
            separated_signal = separate_frequency_band(audio_signal, sample_rate, self.selected_frequency)
            output_file = "separated_frequency.wav"
            sf.write(output_file, separated_signal, sample_rate)
            self.text_output.append(f"🎶 Zvuk na frekvenci {self.selected_frequency:.2f} Hz byl separován a uložen do {output_file}.")
        except Exception as e:
            self.show_error("Separation Error", f"An error occurred during frequency separation: {str(e)}")

    def play_pause_audio(self):
        if self.audio_signal is None or self.sample_rate is None:
            self.load_audio()  # Načti audio, pokud ještě nebylo načteno

        if self.playing:
            sd.stop()  # Pozastaví přehrávání
            self.playing = False
            self.play_pause_button.setText("Přehrát zvuk")
            self.text_output.append("⏸️ Přehrávání pozastaveno.")
        else:
            self.audio_position = 0  # Reset position when starting playback
            self.stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, callback=self.audio_callback)
            self.stream.start()
            self.playing = True
            self.play_pause_button.setText("Pozastavit zvuk")
            self.text_output.append("🔊 Zvuk byl přehrán.")

    def audio_callback(self, outdata, frames, time, status):
        if status:
            print(status)
        if self.audio_position + frames > len(self.audio_signal):
            # We've reached the end of the audio
            remaining = len(self.audio_signal) - self.audio_position
            outdata[:remaining] = self.audio_signal[self.audio_position:].reshape(-1, 1)
            outdata[remaining:] = 0
            self.stream.stop()
            self.playing = False
            self.play_pause_button.setText("Přehrát zvuk")
            raise sd.CallbackStop()
        else:
            chunk = self.audio_signal[self.audio_position:self.audio_position + frames]
            outdata[:] = chunk.reshape(-1, 1)
            self.audio_position += frames

    def load_audio(self):
        if not self.file_path:
            self.show_error("Není vybrán soubor", "Prosím, vyberte zvukový soubor před přehráváním.")
            return  # Přidáno pro ukončení funkce, pokud není vybrán soubor

        try:
            self.audio_signal, self.sample_rate = librosa.load(self.file_path, sr=None)
        except Exception as e:
            self.show_error("Chyba načítání zvuku", f"Během načítání zvuku došlo k chybě: {str(e)}")

    def track_voice_on_frequency(self):
        """Track and visualize voice on the selected frequency."""
        if self.selected_frequency is None:
            self.show_error("No Frequency Selected", "Please select a frequency from the spectrogram first.")
            return

        try:
            self.text_output.append(f"Tracking voice on frequency {self.selected_frequency:.2f} Hz...")
            
            # Load the audio file
            self.audio_signal, self.sample_rate = librosa.load(self.file_path, sr=None)
            print(f"Loaded audio file: {self.file_path}, Sample Rate: {self.sample_rate}, Signal Length: {len(self.audio_signal)}")

            # Create a bandpass filter around the selected frequency
            nyquist = 0.5 * self.sample_rate
            low = (self.selected_frequency - 50) / nyquist  # 50 Hz bandwidth
            high = (self.selected_frequency + 50) / nyquist
            b, a = scipy.signal.butter(4, [low, high], btype='band')

            # Filter the audio signal
            filtered_signal = scipy.signal.lfilter(b, a, self.audio_signal)

            # Create a new figure for visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.set_title(f'Tracking voice on frequency {self.selected_frequency:.2f} Hz')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')

            # Plot the filtered signal
            times = np.arange(len(filtered_signal)) / self.sample_rate
            line, = ax.plot(times, filtered_signal, color='blue')

            # Add grid for better visualization
            ax.grid(True)

            # Play the filtered audio
            sd.play(filtered_signal, samplerate=self.sample_rate)
            print("Playing filtered audio...")
            sd.wait()  # Wait until the sound is finished playing
            print("Finished playing audio.")

            plt.show()
        except Exception as e:
            self.show_error("Tracking Error", f"An error occurred during tracking: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    analyzer = AudioAnalyzer()
    analyzer.show()
    sys.exit(app.exec())
