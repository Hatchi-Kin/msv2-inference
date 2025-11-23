import librosa
import numpy as np
import onnxruntime as ort
import soundfile as sf


class OpenL3Inference:
    def __init__(self, model_path, use_gpu=False):
        self.sr = 48000
        self.n_mels = 128
        self.frame_size = 2048
        self.hop_size = 242
        self.a_min = 1e-10
        self.d_range = 80
        self.db_ref = 1.0

        # OpenL3 specific parameters
        self.patch_samples = int(1 * self.sr)
        self.hop_time = 1.0
        self.hop_samples = int(self.hop_time * self.sr)
        self.x_size = 199
        self.y_size = 128

        # CPU-first execution providers (GPU optional if available)
        providers = ["CPUExecutionProvider"]
        if use_gpu:
            providers.insert(0, "CUDAExecutionProvider")

        try:
            self.session = ort.InferenceSession(model_path, providers=providers)
            active_provider = self.session.get_providers()[0]
            print(f"ONNX Runtime initialized with: {active_provider}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX Runtime: {e}")

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _compute_mel_spectrogram(self, audio):
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.frame_size,
            hop_length=self.hop_size,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sr / 2,
            power=1.0,  # Magnitude spectrum (Essentia type="magnitude")
            norm="slaney",  # Match Slaney
        )

        melbands = S.T  # (time, n_mels)

        # Log scaling and normalization
        melbands = 10.0 * np.log10(np.maximum(self.a_min, melbands))
        melbands -= 10.0 * np.log10(np.maximum(self.a_min, self.db_ref))
        melbands = np.maximum(melbands, melbands.max() - self.d_range)

        return melbands

    def _preprocess_audio(self, audio_file):
        # Load audio
        audio, _ = librosa.load(audio_file, sr=self.sr, mono=True)

        patches = []

        # Slide over audio with 1s window and hop
        n_samples = len(audio)

        for i in range(0, n_samples, self.hop_samples):
            chunk = audio[i : i + self.patch_samples]

            # If chunk is smaller than patch_samples, pad it
            if len(chunk) < self.patch_samples:
                pad_length = self.patch_samples - len(chunk)
                chunk = np.pad(chunk, (0, pad_length), mode="constant")

            # Compute mel for this chunk

            S = librosa.feature.melspectrogram(
                y=chunk,
                sr=self.sr,
                n_fft=self.frame_size,
                hop_length=self.hop_size,
                n_mels=self.n_mels,
                fmin=0,
                fmax=self.sr / 2,
                power=1.0,
                norm="slaney",
                center=False,
            )

            melbands = S.T

            # Post-processing
            melbands = 10.0 * np.log10(np.maximum(self.a_min, melbands))
            melbands -= 10.0 * np.log10(np.maximum(self.a_min, self.db_ref))
            melbands = np.maximum(melbands, melbands.max() - self.d_range)
            melbands -= np.max(melbands)

            # Check shape
            # We need (199, 128)

            if melbands.shape[0] < self.x_size:
                # Pad if slightly short
                pad_width = ((0, self.x_size - melbands.shape[0]), (0, 0))
                melbands = np.pad(melbands, pad_width, mode="constant")
            elif melbands.shape[0] > self.x_size:
                melbands = melbands[: self.x_size, :]

            patches.append(melbands)

        if not patches:
            return np.empty((0, self.x_size, self.y_size, 1))

        batch = np.array(patches)
        # Add channel dim: (batch, time, freq, 1)
        batch = batch[..., np.newaxis]
        batch = np.transpose(batch, (0, 2, 1, 3))  # (batch, 128, 199, 1)

        return batch.astype(np.float32)

    def run(self, audio_file):
        batch = self._preprocess_audio(audio_file)
        if batch.shape[0] == 0:
            return np.zeros(512)

        # Run inference
        outputs = self.session.run([self.output_name], {self.input_name: batch})
        embeddings = outputs[0]

        # Return mean pooled embedding
        return np.mean(embeddings, axis=0)
