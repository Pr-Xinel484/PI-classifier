#Persistence image classification of bird songs
# 5 bird species: Bewick's Wren, Northern Cardinal, American Robin, Song Sparrow, Northern Mockingbird
import matplotlib
matplotlib.use('Agg')
import Persistence_image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
import os
import hashlib
import joblib
import time
import threading
from scipy.signal import butter, sosfiltfilt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARCHIVE_DIR = os.path.join(BASE_DIR, 'archive')


def resolve_wav_dir():
    candidates = [
        os.path.join(ARCHIVE_DIR, 'wavfiles'),
        os.path.join(BASE_DIR, 'wavfiles'),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find wavfiles directory. Tried: {candidates}"
    )


WAV_DIR = resolve_wav_dir()

#from the archive folder, load the csv file with the bird song data
metadata_path = os.path.join(ARCHIVE_DIR, 'bird_songs_metadata.csv')
if not os.path.exists(metadata_path):
    raise FileNotFoundError(f"Missing metadata file: {metadata_path}")
data = pd.read_csv(metadata_path)
required_columns = {'name', 'filename'}
if not required_columns.issubset(data.columns):
    missing = required_columns.difference(data.columns)
    raise ValueError(f"Missing required metadata columns: {sorted(missing)}")

#extract the columns 'name' and 'filename' and select 50 samples from each species
species = ['Bewick\'s Wren', 'Northern Cardinal', 'American Robin', 'Song Sparrow', 'Northern Mockingbird']
samples_per_species = 5
new_samples_per_species = 2
samples = []
species_selected = {}
species_new_candidates = {}
for sp in species:
    species_rows = data[data['name'] == sp]
    if species_rows.empty:
        raise ValueError(f"No metadata rows found for species: {sp}")

    # Keep only files that actually exist on disk to avoid runtime file-not-found crashes.
    candidate_filenames = species_rows['filename'].tolist()
    existing_filenames = [
        fn for fn in candidate_filenames
        if os.path.exists(os.path.join(WAV_DIR, fn))
    ]

    if len(existing_filenames) < samples_per_species + new_samples_per_species:
        raise ValueError(
            f"Not enough existing audio files for species {sp}. "
            f"Required {samples_per_species + new_samples_per_species}, found {len(existing_filenames)}"
        )

    sp_samples = existing_filenames[:samples_per_species]
    species_selected[sp] = sp_samples
    species_new_candidates[sp] = existing_filenames[samples_per_species:]
    samples.extend(sp_samples)

if len(samples) != samples_per_species * len(species):
    raise RuntimeError("Unexpected total sample count after species filtering.")

#load the audio files from the wavfiles folder
from scipy.io import wavfile
audio_data = []
sample_rates = []
loaded_filenames = []
file_signatures = []
total_samples = len(samples)
load_start = time.perf_counter()
print(f"[progress] Loading audio files: 0/{total_samples}")
for filename in samples:
    filepath = os.path.join(WAV_DIR, filename)
    stat = os.stat(filepath)
    file_signatures.append((filename, int(stat.st_size), int(stat.st_mtime_ns)))
    sr, audio = wavfile.read(filepath)
    if audio is None or np.asarray(audio).size == 0:
        raise ValueError(f"Empty audio signal for file: {filename}")
    audio_data.append(audio)
    sample_rates.append(int(sr))
    loaded_filenames.append(filename)

    loaded_count = len(loaded_filenames)
    if loaded_count % 10 == 0 or loaded_count == total_samples:
        elapsed = time.perf_counter() - load_start
        print(f"[progress] Loading audio files: {loaded_count}/{total_samples} | elapsed {elapsed:.1f}s")

#Perstence image pipeline by Taken embeddings of the audio data
def takens_embedding(audio, dimension=3, delay_samples=1):
    if dimension < 2:
        raise ValueError("Takens embedding dimension must be at least 2.")
    if delay_samples < 1:
        raise ValueError("Takens embedding delay in samples must be at least 1.")

    n = len(audio)
    if n <= (dimension - 1) * delay_samples:
        raise ValueError("Audio length must be at least (dimension - 1) * delay_samples.")

    max_start = n - (dimension - 1) * delay_samples
    start_idx = np.arange(0, max_start, delay_samples)
    embedded = np.array([audio[i:i + dimension * delay_samples:delay_samples] for i in start_idx])
    if embedded.ndim != 2 or embedded.shape[1] != dimension:
        raise RuntimeError("Unexpected Takens embedding shape.")

    return embedded


def delay_ms_to_samples(sr, delay_ms):
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if delay_ms <= 0:
        raise ValueError("delay_ms must be positive.")
    return max(1, int(round(sr * delay_ms / 1000.0)))


def dominant_frequency_delay_samples(audio, sr, min_delay=1, max_delay=128, period_fraction=0.25):
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")

    x = normalize_audio(audio)
    if x.size < 8:
        return int(min_delay)

    window = np.hanning(x.size)
    spectrum = np.fft.rfft(x * window)
    magnitude = np.abs(spectrum)
    if magnitude.size <= 1:
        return int(min_delay)

    magnitude[0] = 0.0
    freqs = np.fft.rfftfreq(x.size, d=1.0 / sr)
    dom_idx = int(np.argmax(magnitude))
    dom_freq_hz = float(freqs[dom_idx])
    if not np.isfinite(dom_freq_hz) or dom_freq_hz <= 0.0:
        return int(min_delay)

    period_samples = float(sr / dom_freq_hz)
    delay = int(round(period_fraction * period_samples))
    return int(np.clip(delay, min_delay, max_delay))


def compute_persistence_image(audio, sr, dimension=3, delay_ms=None, w=0.1, res=(20, 20), sigma=0.01, box=None):
    if delay_ms is None:
        max_delay = max(1, min(128, int(len(audio) // 4)))
        delay_samples = dominant_frequency_delay_samples(audio, sr, min_delay=1, max_delay=max_delay, period_fraction=0.25)
    else:
        delay_samples = delay_ms_to_samples(sr, delay_ms)

    embedded = takens_embedding(audio, dimension=dimension, delay_samples=delay_samples)
    pd = Persistence_image.PD(embedded, sparse=1)
    pi_result = pd.PI(w=w, res=res, sigma=sigma, box=box)
    vec = pi_result[0]
    expected_dim = 2 * res[0] * res[1]
    if vec.shape[0] != expected_dim:
        raise RuntimeError(f"Unexpected PI feature length {vec.shape[0]}, expected {expected_dim}")
    if not np.isfinite(vec).all():
        raise FloatingPointError("PI vector contains non-finite values")
    return vec


def normalize_audio(audio):
    arr = np.asarray(audio, dtype=float)
    if arr.ndim == 2:
        arr = np.mean(arr, axis=1)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("Audio must be a non-empty 1D signal after mono conversion.")

    arr = arr - np.mean(arr)
    scale = np.std(arr)
    if scale > 0:
        arr = arr / scale
    return arr


def bandpass_filter(signal, sr, low_hz=500.0, high_hz=10000.0, order=4):
    nyq = 0.5 * sr
    low = max(1.0, low_hz) / nyq
    high = min(high_hz, nyq * 0.99) / nyq
    if not (0 < low < high < 1):
        return signal
    sos = butter(order, [low, high], btype='bandpass', output='sos')
    return sosfiltfilt(sos, signal)


def spectral_entropy(window):
    spec = np.fft.rfft(window)
    power = np.abs(spec) ** 2
    total = float(np.sum(power))
    if total <= 0:
        return 1.0
    p = power / total
    p = np.clip(p, 1e-12, 1.0)
    ent = -np.sum(p * np.log(p)) / np.log(p.size)
    return float(ent)


def spectral_centroid(window, sr):
    """Compute spectral centroid in Hz."""
    spec = np.fft.rfft(window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0/sr)
    total_power = np.sum(power)
    if total_power <= 0:
        return 0.0
    centroid_hz = np.sum(freqs * power) / total_power
    return float(centroid_hz)


def spectral_bandwidth(window, sr):
    """Compute spectral bandwidth in Hz (std dev of frequencies weighted by power)."""
    spec = np.fft.rfft(window)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(window), d=1.0/sr)
    total_power = np.sum(power)
    if total_power <= 0:
        return 0.0
    centroid = np.sum(freqs * power) / total_power
    variance = np.sum(power * (freqs - centroid) ** 2) / total_power
    bandwidth_hz = np.sqrt(max(0.0, variance))
    return float(bandwidth_hz)


def spectral_rolloff(window, sr, roll_percent=0.85):
    spec = np.fft.rfft(window)
    power = np.abs(spec) ** 2
    if power.size == 0:
        return 0.0
    cumulative = np.cumsum(power)
    total = float(cumulative[-1])
    if total <= 0:
        return 0.0
    threshold = roll_percent * total
    idx = int(np.searchsorted(cumulative, threshold, side='left'))
    idx = min(max(idx, 0), power.size - 1)
    freqs = np.fft.rfftfreq(len(window), d=1.0 / sr)
    return float(freqs[idx])


def spectral_flatness(window):
    spec = np.fft.rfft(window)
    power = np.abs(spec) ** 2
    power = np.clip(power, 1e-12, None)
    gmean = float(np.exp(np.mean(np.log(power))))
    amean = float(np.mean(power))
    if amean <= 0:
        return 0.0
    return float(gmean / amean)


def zero_crossing_rate(window):
    if len(window) < 2:
        return 0.0
    signs = np.signbit(window)
    crossings = np.count_nonzero(signs[1:] != signs[:-1])
    return float(crossings / (len(window) - 1))


def hz_to_mel(freq_hz):
    return 2595.0 * np.log10(1.0 + freq_hz / 700.0)


def mel_to_hz(freq_mel):
    return 700.0 * (10.0 ** (freq_mel / 2595.0) - 1.0)


def mel_filterbank(sr, n_fft, n_mels=26, fmin=0.0, fmax=None):
    if fmax is None:
        fmax = 0.5 * sr

    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    bins = np.floor((n_fft + 1) * hz / sr).astype(int)

    n_freqs = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_freqs), dtype=float)

    for i in range(n_mels):
        left = int(np.clip(bins[i], 0, n_freqs - 1))
        center = int(np.clip(bins[i + 1], 0, n_freqs - 1))
        right = int(np.clip(bins[i + 2], 0, n_freqs - 1))

        if center <= left:
            center = min(left + 1, n_freqs - 1)
        if right <= center:
            right = min(center + 1, n_freqs - 1)

        if center > left:
            fb[i, left:center] = (np.arange(left, center) - left) / (center - left)
        if right > center:
            fb[i, center:right] = (right - np.arange(center, right)) / (right - center)

    return fb


def mfcc_vector(window, sr, n_mfcc=13, n_mels=26):
    n_fft = len(window)
    if n_fft < 16:
        return np.zeros(n_mfcc, dtype=float)

    win = np.hamming(n_fft)
    spec = np.fft.rfft(window * win, n=n_fft)
    power = np.abs(spec) ** 2
    fb = mel_filterbank(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=0.0, fmax=0.5 * sr)
    mel_energies = fb.dot(power)
    log_mel = np.log(np.clip(mel_energies, 1e-12, None))

    # DCT-II implemented directly to avoid extra dependencies.
    n = log_mel.size
    k = np.arange(n_mfcc)[:, None]
    n_idx = np.arange(n)[None, :]
    dct_basis = np.cos(np.pi / n * (n_idx + 0.5) * k)
    coeffs = dct_basis.dot(log_mel)
    return coeffs.astype(float)


def extract_active_windows(
    audio,
    sr,
    window_seconds=0.25,
    hop_seconds=0.125,
    max_windows=12,
    min_windows=4,
    energy_quantile=0.6,
    bandpass_low_hz=500.0,
    bandpass_high_hz=10000.0,
    snr_db_threshold=6.0,
    spectral_entropy_max=0.9
):
    if sr <= 0:
        raise ValueError("Sample rate must be positive.")
    if max_windows < 1 or min_windows < 1 or min_windows > max_windows:
        raise ValueError("Need 1 <= min_windows <= max_windows.")

    signal = normalize_audio(audio)
    signal = bandpass_filter(signal, sr, low_hz=bandpass_low_hz, high_hz=bandpass_high_hz)

    # Adaptive window sizing: if not provided, choose a per-recording window
    # that yields a reasonable number of candidate windows before top-k selection.
    if window_seconds is None:
        duration_s = max(signal.size / float(sr), 1e-6)
        target_candidates = max(2 * max_windows, 8)
        adaptive_window_s = (2.0 * duration_s) / (target_candidates + 1.0)
        window_seconds = float(np.clip(adaptive_window_s, 0.12, 0.35))
    if hop_seconds is None:
        hop_seconds = 0.5 * float(window_seconds)

    if not (window_seconds > 0 and hop_seconds > 0):
        raise ValueError("window_seconds and hop_seconds must be positive.")

    win = max(16, int(round(window_seconds * sr)))
    hop = max(1, int(round(hop_seconds * sr)))

    if signal.size < win:
        return [signal]

    starts = np.arange(0, signal.size - win + 1, hop, dtype=int)
    windows = [signal[s:s + win] for s in starts]
    energies = np.array([float(np.mean(w * w)) for w in windows], dtype=float)

    # Keep only the most energetic windows in descending-energy order.
    selected_idx = np.argsort(energies)[::-1][:max_windows]

    return [windows[i] for i in selected_idx]


def compute_recording_feature(audio, sr, pi_config):
    windows = extract_active_windows(
        audio,
        sr,
        window_seconds=pi_config['window_seconds'],
        hop_seconds=pi_config['hop_seconds'],
        max_windows=pi_config['max_windows'],
        min_windows=pi_config['min_windows'],
        energy_quantile=pi_config['energy_quantile'],
        bandpass_low_hz=pi_config['bandpass_low_hz'],
        bandpass_high_hz=pi_config['bandpass_high_hz'],
        snr_db_threshold=pi_config['snr_db_threshold'],
        spectral_entropy_max=pi_config['spectral_entropy_max']
    )

    segment_features = []
    include_mfcc = bool(pi_config.get('include_mfcc', True))
    n_mfcc = int(pi_config.get('n_mfcc', 13))
    n_mels = int(pi_config.get('n_mels', 26))
    segment_mfcc = []

    for segment in windows:
        vec = compute_persistence_image(
            segment,
            sr,
            dimension=pi_config['dimension'],
            delay_ms=pi_config.get('delay_ms'),
            w=pi_config['w'],
            res=pi_config['res'],
            sigma=pi_config['sigma'],
            box=pi_config.get('box')
        )
        segment_features.append(vec)
        if include_mfcc:
            segment_mfcc.append(mfcc_vector(segment, sr, n_mfcc=n_mfcc, n_mels=n_mels))

    seg_arr = np.array(segment_features, dtype=float)
    if seg_arr.ndim != 2 or seg_arr.shape[0] < 1:
        raise RuntimeError("No valid segment-level PI features were produced.")

    # PI features: mean + std + the first 2 window PI vectors.
    feature_mean = np.mean(seg_arr, axis=0)
    feature_std = np.std(seg_arr, axis=0)
    pi_dim = int(seg_arr.shape[1])
    keep_windows = int(pi_config.get('max_windows', 2))
    first_window_pi = np.zeros((keep_windows, pi_dim), dtype=float)
    n_copy = min(keep_windows, int(seg_arr.shape[0]))
    first_window_pi[:n_copy] = seg_arr[:n_copy]

    rec_feature_parts = [feature_mean, feature_std, first_window_pi.reshape(-1)]
    if include_mfcc:
        mfcc_arr = np.asarray(segment_mfcc, dtype=float)
        mfcc_mean = np.mean(mfcc_arr, axis=0)
        mfcc_std = np.std(mfcc_arr, axis=0)
        rec_feature_parts.extend([mfcc_mean, mfcc_std])

    rec_feature = np.concatenate(rec_feature_parts)

    if not np.isfinite(rec_feature).all():
        raise FloatingPointError("Recording feature contains non-finite values")

    return rec_feature, int(seg_arr.shape[0])


def compute_pi_tuple_for_segment(segment, sr, pi_config):
    delay_ms = pi_config.get('delay_ms')
    if delay_ms is None:
        max_delay = max(1, min(128, int(len(segment) // 4)))
        delay_samples = dominant_frequency_delay_samples(segment, sr, min_delay=1, max_delay=max_delay, period_fraction=0.25)
    else:
        delay_samples = delay_ms_to_samples(sr, delay_ms)

    embedded = takens_embedding(segment, dimension=pi_config['dimension'], delay_samples=delay_samples)
    pd_obj = Persistence_image.PD(embedded, sparse=1)
    return pd_obj.PI(
        w=pi_config['w'],
        res=pi_config['res'],
        sigma=pi_config['sigma'],
        box=pi_config.get('box')
    )

#5 classes classifier based on the persistence images
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.metrics import recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA


def dataset_fingerprint(X_arr, y_arr):
    hasher = hashlib.sha256()
    hasher.update(np.ascontiguousarray(X_arr).tobytes())
    hasher.update(np.asarray(y_arr, dtype=str).tobytes())
    return hasher.hexdigest()


def build_feature_cache_key(file_sigs, pi_config, species_list, filenames):
    hasher = hashlib.sha256()
    for name, size, mtime_ns in file_sigs:
        hasher.update(str((name, size, mtime_ns)).encode('utf-8'))
    hasher.update(str(tuple(species_list)).encode('utf-8'))
    hasher.update(str(tuple(filenames)).encode('utf-8'))
    hasher.update(str(sorted(pi_config.items())).encode('utf-8'))
    return hasher.hexdigest()


def score_random_forest_classifier(X_train, y_train, X_eval, y_eval, n_estimators=300, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    y_hat = clf.predict(X_eval)
    return float(np.mean(y_hat == y_eval))


def repeated_stratified_rf_cv(
    X,
    y,
    n_splits,
    n_repeats=5,
    n_estimators=300,
    base_random_state=42
):
    scores = []
    for rep in range(n_repeats):
        cv = StratifiedKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=base_random_state + rep
        )
        for fold_i, (tr_idx, va_idx) in enumerate(cv.split(X, y), start=1):
            fold_score = score_random_forest_classifier(
                X[tr_idx],
                y[tr_idx],
                X[va_idx],
                y[va_idx],
                n_estimators=n_estimators,
                random_state=base_random_state + rep
            )
            scores.append(fold_score)
            print(
                f"[progress] RF repeated CV rep {rep + 1}/{n_repeats}, "
                f"fold {fold_i}/{n_splits}: accuracy {fold_score:.3f}"
            )
    return np.array(scores, dtype=float)


def stratified_subset_indices(y_labels, per_class=10):
    idx = []
    y_arr = np.asarray(y_labels)
    for cls in np.unique(y_arr):
        cls_idx = np.where(y_arr == cls)[0]
        take = min(per_class, len(cls_idx))
        idx.extend(cls_idx[:take].tolist())
    return np.array(idx, dtype=int)


def tune_windowing(audio_data, sample_rates, y_labels, base_config):
    candidate_configs = []
    for window_seconds in [0.20, 0.25, 0.35]:
        for hop_ratio in [0.5, 0.75]:
            for max_windows in [4]:
                cfg = dict(base_config)
                cfg['window_seconds'] = window_seconds
                cfg['hop_seconds'] = window_seconds * hop_ratio
                cfg['max_windows'] = max_windows
                cfg['min_windows'] = min(base_config['min_windows'], max_windows)
                candidate_configs.append(cfg)

    tune_idx = stratified_subset_indices(y_labels, per_class=10)
    X_tune_audio = [audio_data[i] for i in tune_idx]
    X_tune_sr = [sample_rates[i] for i in tune_idx]
    y_tune = np.asarray(y_labels)[tune_idx]

    min_class_count = int(pd.Series(y_tune).value_counts().min())
    cv_splits = max(2, min(3, min_class_count))
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    best_cfg = dict(base_config)
    best_score = -1.0
    print(f"[progress] Tuning windowing on {len(y_tune)} samples with {len(candidate_configs)} candidates")

    for i, cfg in enumerate(candidate_configs, start=1):
        rec_features = []
        for audio, sr in zip(X_tune_audio, X_tune_sr):
            rec_vec, _ = compute_recording_feature(audio, int(sr), cfg)
            rec_features.append(rec_vec)
        X_tune = np.asarray(rec_features, dtype=float)

        fold_scores = []
        for tr_idx, va_idx in cv.split(X_tune, y_tune):
            s = score_random_forest_classifier(
                X_tune[tr_idx],
                y_tune[tr_idx],
                X_tune[va_idx],
                y_tune[va_idx],
                n_estimators=200,
                random_state=42
            )
            fold_scores.append(s)

        score_mean = float(np.mean(fold_scores))
        if score_mean > best_score:
            best_score = score_mean
            best_cfg = cfg

        if i % 3 == 0 or i == len(candidate_configs):
            print(f"[progress] Window tuning: {i}/{len(candidate_configs)} | best CV acc {best_score:.3f}")

    print(f"[progress] Best windowing config CV accuracy: {best_score:.3f}")
    print(f"[progress] Best windowing config: window_seconds={best_cfg['window_seconds']}, hop_seconds={best_cfg['hop_seconds']}, max_windows={best_cfg['max_windows']}")
    return best_cfg


def compute_feature_matrix_for_config(audio_data, sample_rates, pi_config, file_sigs, species_list, filenames, y_labels, cache_dir='cache', reuse_cache=False):
    os.makedirs(cache_dir, exist_ok=True)
    feature_cache_path = os.path.join(cache_dir, 'bird_song_pi_features.joblib')
    feature_cache_key = build_feature_cache_key(file_sigs, pi_config, species_list, filenames)
    X = None
    y_out = np.asarray(y_labels)

    if reuse_cache and os.path.exists(feature_cache_path):
        try:
            feature_payload = joblib.load(feature_cache_path)
            if feature_payload.get('cache_key') == feature_cache_key:
                X_cached = feature_payload.get('X')
                y_cached = feature_payload.get('y')
                if X_cached is not None and y_cached is not None and X_cached.shape[0] == y_cached.shape[0]:
                    X = X_cached
                    y_out = y_cached
                    print(f"Loaded PI feature cache from {feature_cache_path}")
                    print(f"[progress] PI feature generation skipped (cache hit) for {X.shape[0]} samples")
        except Exception as exc:
            print(f"Failed to load feature cache ({exc}). Recomputing PI features.")

    if X is None:
        n_jobs = int(pi_config.get('n_jobs', -1))

        def _extract_one(audio_sr):
            audio, sr = audio_sr
            return compute_recording_feature(audio, sr, pi_config)

        total_pi = len(audio_data)
        pi_start = time.perf_counter()
        print(f"[progress] Computing PI features in parallel (n_jobs={n_jobs}): 0/{total_pi}")

        extraction_results = joblib.Parallel(n_jobs=n_jobs, prefer='processes')(
            joblib.delayed(_extract_one)(item)
            for item in zip(audio_data, sample_rates)
        )

        recording_features = [res[0] for res in extraction_results]
        segment_counts = [res[1] for res in extraction_results]

        expected_feature_dim = recording_features[0].shape[0]
        for rec_vec in recording_features[1:]:
            if rec_vec.shape[0] != expected_feature_dim:
                raise RuntimeError("Feature dimension mismatch across samples")

        elapsed = time.perf_counter() - pi_start
        mean_segments = float(np.mean(segment_counts)) if segment_counts else 0.0
        print(
            f"[progress] Computing PI features: {total_pi}/{total_pi} "
            f"| avg segments {mean_segments:.2f} | elapsed {elapsed:.1f}s"
        )

        X = np.array(recording_features)
        feature_payload = {
            'cache_key': feature_cache_key,
            'X': X,
            'y': y_out,
            'pi_config': pi_config,
            'filenames': list(filenames),
            'species': list(species_list),
        }
        joblib.dump(feature_payload, feature_cache_path)
        print(f"Saved PI feature cache to {feature_cache_path}")

    return np.asarray(X, dtype=float), np.asarray(y_out)


def plot_cached_pi_vectors_for_class(
    X,
    y_labels,
    target_class,
    pi_config,
    n_examples=4,
    output_path=None
):
    y_arr = np.asarray(y_labels)
    class_idx = np.where(y_arr == target_class)[0]
    if class_idx.size == 0:
        print(f"[plot] No samples found for class: {target_class}")
        return

    n_show = min(n_examples, int(class_idx.size))
    sel = class_idx[:n_show]

    pi_dim = 2 * pi_config['res'][0] * pi_config['res'][1]
    if X.shape[1] < 2 * pi_dim:
        raise RuntimeError(
            f"Feature vector too short for PI mean/std slices: got {X.shape[1]}, need at least {2 * pi_dim}."
        )

    res_x, res_y = pi_config['res']
    for col, idx in enumerate(sel, start=1):
        feat = X[idx]
        pi_mean = feat[:pi_dim]
        pi0 = pi_mean[:res_x * res_y].reshape((res_x, res_y))
        pi1 = pi_mean[res_x * res_y:].reshape((res_x, res_y))

        # Reuse the shared heatmap utility from Persistence_image.py.
        pi_tuple = (
            pi_mean,
            pi0,
            pi1,
            np.empty((0, 2), dtype=float),
            np.empty((0, 2), dtype=float),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
        )
        fig, _ = Persistence_image.plot_PI_heatmap(pi_tuple, res=pi_config['res'])
        fig.suptitle(f"Cached PI heatmap ({target_class}) sample {col}")

        if output_path is not None:
            root, ext = os.path.splitext(output_path)
            ext = ext if ext else '.png'
            sample_output = f"{root}_sample_{col}{ext}"
            fig.savefig(sample_output, dpi=150)
            print(f"[plot] Saved cached PI heatmap to {sample_output}")
            plt.close(fig)
        else:
            # Avoid GUI backend calls when running outside the main thread.
            if threading.current_thread() is threading.main_thread():
                plt.show()
            else:
                print("[plot] Skipping plt.show() outside main thread; closing figure.")
                plt.close(fig)


def evaluate_model_suite(X_train, y_train, random_state=42):
    class_counts_train = pd.Series(y_train).value_counts()
    cv_splits = min(5, int(class_counts_train.min()))
    if cv_splits < 2:
        raise ValueError("Need at least 2 training samples per class for cross-validation")
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    models = {
        'LogisticRegression': make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight='balanced', random_state=random_state)
        ),
        'LinearSVM': make_pipeline(
            StandardScaler(),
            LinearSVC(class_weight='balanced', max_iter=8000, random_state=random_state)
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=400,
            max_depth=16,
            min_samples_leaf=2,
            class_weight='balanced_subsample',
            random_state=random_state,
            n_jobs=-1
        ),
    }

    results = []
    for model_name, model in models.items():
        f1_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_macro', n_jobs=-1)
        ba_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        results.append({
            'name': model_name,
            'model': model,
            'f1_macro_mean': float(np.mean(f1_scores)),
            'f1_macro_std': float(np.std(f1_scores)),
            'bal_acc_mean': float(np.mean(ba_scores)),
            'bal_acc_std': float(np.std(ba_scores)),
        })

    results = sorted(results, key=lambda r: (r['f1_macro_mean'], r['bal_acc_mean']), reverse=True)
    return results


base_pi_config = {
    'dimension': 3,
    'delay_ms': None,
    'w': 0.1,
    'res': (10, 10),
    'sigma': 0.001,
    'box': None,
    'window_seconds': None,
    'hop_seconds': None,
    'max_windows': 8,
    'min_windows': 4,
    'energy_quantile': 0.5,
    'bandpass_low_hz': 400.0,
    'bandpass_high_hz': 20000.0,
    'snr_db_threshold': 6.0,
    'spectral_entropy_max': 0.9,
    'n_jobs': 4,
    'include_mfcc': True,
    'n_mfcc': 13,
    'n_mels': 26,
}

def main():
    y = data.set_index('filename').loc[loaded_filenames]['name'].values
    window_tuning_enabled = False
    reuse_feature_cache = True

    # Step 1: PI + MFCC + PCA default pipeline.
    pi_config = dict(base_pi_config)
    print("\n=== Experiment: PI+MFCC + PCA (default) ===")

    if window_tuning_enabled:
        pi_config = tune_windowing(audio_data, sample_rates, y, pi_config)

    X, y_exp = compute_feature_matrix_for_config(
        audio_data,
        sample_rates,
        pi_config,
        file_signatures,
        species,
        loaded_filenames,
        y,
        cache_dir='cache',
        reuse_cache=reuse_feature_cache,
    )

    if X.shape[0] != y_exp.shape[0]:
        raise RuntimeError(f"Feature-label size mismatch: X has {X.shape[0]}, y has {y_exp.shape[0]}")
    if not np.isfinite(X).all():
        raise FloatingPointError("Feature matrix X contains non-finite values")

    plot_cached_pi_vectors_for_class(
        X,
        y_exp,
        target_class="Bewick's Wren",
        pi_config=pi_config,
        n_examples=4,
        output_path=os.path.join('cache', 'bewicks_wren_cached_pi_heatmap.png')
    )

    class_counts = pd.Series(y_exp).value_counts()
    if (class_counts < 2).any():
        too_small = class_counts[class_counts < 2]
        raise ValueError(f"Each class needs at least 2 samples before split. Failing classes: {too_small.to_dict()}")

    new_filenames = []
    new_labels = []
    for sp in species:
        sp_new = species_new_candidates[sp][:new_samples_per_species]
        if len(sp_new) < new_samples_per_species:
            raise RuntimeError(
                f"Not enough held-out files for new-data test for species {sp}. "
                f"Required {new_samples_per_species}, found {len(sp_new)}"
            )
        new_filenames.extend(sp_new)
        new_labels.extend([sp] * len(sp_new))

    print(f"\n[progress] Building new-data feature set: 0/{len(new_filenames)}")
    new_start = time.perf_counter()

    def _extract_new_feature(filename):
        filepath = os.path.join(WAV_DIR, filename)
        sr_new, audio_new = wavfile.read(filepath)
        if audio_new is None or np.asarray(audio_new).size == 0:
            raise ValueError(f"Empty audio signal in new-data test file: {filename}")
        rec_vec, n_segments = compute_recording_feature(audio_new, int(sr_new), pi_config)
        return rec_vec, n_segments

    n_jobs_new = int(pi_config.get('n_jobs', -1))
    new_results = joblib.Parallel(n_jobs=n_jobs_new, prefer='processes')(
        joblib.delayed(_extract_new_feature)(filename)
        for filename in new_filenames
    )

    X_new = np.array([res[0] for res in new_results], dtype=float)
    new_segment_counts = [res[1] for res in new_results]
    elapsed_new = time.perf_counter() - new_start
    mean_new_segments = float(np.mean(new_segment_counts)) if new_segment_counts else 0.0
    print(
        f"[progress] Building new-data feature set: {len(new_filenames)}/{len(new_filenames)} "
        f"| avg segments {mean_new_segments:.2f} | elapsed {elapsed_new:.1f}s"
    )

    y_new = np.array(new_labels)


def evaluate_rf_on_seed(X_all, y_all, X_new_eval, y_new_eval, rf_cfg, seed):
    X_train, X_test, y_train, y_test = train_test_split(
        X_all,
        y_all,
        test_size=0.2,
        random_state=seed,
        stratify=y_all,
    )
    if X_new_eval.shape[1] != X_train.shape[1]:
        raise RuntimeError(
            f"New-data feature dimension mismatch: expected {X_train.shape[1]}, got {X_new_eval.shape[1]}"
        )

    pca_variance = float(rf_cfg.get('pca_variance', 0.95))
    rf_params = {k: v for k, v in rf_cfg.items() if k != 'pca_variance'}

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    X_new_s = scaler.transform(X_new_eval)

    pca = PCA(n_components=pca_variance, svd_solver='full')
    X_train_p = pca.fit_transform(X_train_s)
    X_test_p = pca.transform(X_test_s)
    X_new_p = pca.transform(X_new_s)

    model = RandomForestClassifier(random_state=seed, n_jobs=-1, **rf_params)
    model.fit(X_train_p, y_train)

    y_pred_holdout = model.predict(X_test_p)
    y_pred_new = model.predict(X_new_p)
    new_recall_per_class = recall_score(
        y_new_eval,
        y_pred_new,
        labels=species,
        average=None,
        zero_division=0,
    )

    result = {
        'seed': seed,
        'holdout_acc': float(accuracy_score(y_test, y_pred_holdout)),
        'holdout_f1_macro': float(f1_score(y_test, y_pred_holdout, average='macro')),
        'new_acc': float(accuracy_score(y_new_eval, y_pred_new)),
        'new_f1_macro': float(f1_score(y_new_eval, y_pred_new, average='macro')),
        'new_recall_per_class': np.asarray(new_recall_per_class, dtype=float),
    }
    return result


    rf_grid = [
        {'n_estimators': 120, 'max_depth': 8, 'min_samples_leaf': 4, 'class_weight': 'balanced_subsample', 'pca_variance': 0.90},
        {'n_estimators': 120, 'max_depth': 10, 'min_samples_leaf': 3, 'class_weight': 'balanced_subsample', 'pca_variance': 0.95},
        {'n_estimators': 160, 'max_depth': 10, 'min_samples_leaf': 3, 'class_weight': 'balanced_subsample', 'pca_variance': 0.90},
        {'n_estimators': 160, 'max_depth': 12, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample', 'pca_variance': 0.95},
        {'n_estimators': 200, 'max_depth': 12, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample', 'pca_variance': 0.90},
        {'n_estimators': 200, 'max_depth': 14, 'min_samples_leaf': 2, 'class_weight': 'balanced_subsample', 'pca_variance': 0.95},
    ]

    # Steps 2 and 3: repeated split stability + RF hyperparameter sweep.
    seed_list = [42, 52, 62, 72, 82]
    grid_summary = []
    best_cfg = None
    best_score = -1.0

    print(f"\n=== RF Grid Search with Repeated Splits ({len(seed_list)} seeds) ===")
    for i, rf_cfg in enumerate(rf_grid, start=1):
        per_seed = []
        for seed in seed_list:
            per_seed.append(evaluate_rf_on_seed(X, y_exp, X_new, y_new, rf_cfg=rf_cfg, seed=seed))

        holdout_acc_vals = np.array([r['holdout_acc'] for r in per_seed], dtype=float)
        holdout_f1_vals = np.array([r['holdout_f1_macro'] for r in per_seed], dtype=float)
        new_acc_vals = np.array([r['new_acc'] for r in per_seed], dtype=float)
        new_f1_vals = np.array([r['new_f1_macro'] for r in per_seed], dtype=float)

        row = {
            'cfg': rf_cfg,
            'holdout_acc_mean': float(np.mean(holdout_acc_vals)),
            'holdout_acc_std': float(np.std(holdout_acc_vals)),
            'holdout_f1_mean': float(np.mean(holdout_f1_vals)),
            'holdout_f1_std': float(np.std(holdout_f1_vals)),
            'new_acc_mean': float(np.mean(new_acc_vals)),
            'new_acc_std': float(np.std(new_acc_vals)),
            'new_f1_mean': float(np.mean(new_f1_vals)),
            'new_f1_std': float(np.std(new_f1_vals)),
            'per_seed': per_seed,
        }
        grid_summary.append(row)

        # Rank by unseen-data macro-F1 first, then unseen accuracy.
        rank_score = row['new_f1_mean'] + 0.1 * row['new_acc_mean']
        if rank_score > best_score:
            best_score = rank_score
            best_cfg = row

        print(
            f"[progress] RF cfg {i}/{len(rf_grid)} | "
            f"new f1_macro {row['new_f1_mean']:.3f} +/- {row['new_f1_std']:.3f} | "
            f"new acc {row['new_acc_mean']:.3f} +/- {row['new_acc_std']:.3f}"
        )

    print("\n=== RF Grid Summary (sorted by new-data macro-F1) ===")
    grid_summary_sorted = sorted(
        grid_summary,
        key=lambda r: (r['new_f1_mean'], r['new_acc_mean']),
        reverse=True,
    )
    for r in grid_summary_sorted:
        cfg = r['cfg']
        print(
            f"n_est={cfg['n_estimators']}, depth={cfg['max_depth']}, leaf={cfg['min_samples_leaf']} | "
            f"holdout acc {r['holdout_acc_mean']:.3f}+/-{r['holdout_acc_std']:.3f}, "
            f"holdout f1 {r['holdout_f1_mean']:.3f}+/-{r['holdout_f1_std']:.3f} | "
            f"new acc {r['new_acc_mean']:.3f}+/-{r['new_acc_std']:.3f}, "
            f"new f1 {r['new_f1_mean']:.3f}+/-{r['new_f1_std']:.3f}"
        )

    print("\n=== Best RF Config (Repeated-Split Stability) ===")
    print(best_cfg['cfg'])
    print(
        f"holdout acc {best_cfg['holdout_acc_mean']:.3f} +/- {best_cfg['holdout_acc_std']:.3f} | "
        f"holdout f1_macro {best_cfg['holdout_f1_mean']:.3f} +/- {best_cfg['holdout_f1_std']:.3f}"
    )
    print(
        f"new acc {best_cfg['new_acc_mean']:.3f} +/- {best_cfg['new_acc_std']:.3f} | "
        f"new f1_macro {best_cfg['new_f1_mean']:.3f} +/- {best_cfg['new_f1_std']:.3f}"
    )

    recall_matrix = np.vstack([r['new_recall_per_class'] for r in best_cfg['per_seed']])
    recall_mean = np.mean(recall_matrix, axis=0)
    recall_std = np.std(recall_matrix, axis=0)

    print("\n=== Best RF Per-Class New-Data Recall (mean +/- std) ===")
    for i, cls in enumerate(species):
        print(f"{cls}: {recall_mean[i]:.3f} +/- {recall_std[i]:.3f}")


if __name__ == '__main__':
    main()
