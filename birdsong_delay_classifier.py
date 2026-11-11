"""Birdsong classification with sliding-window persistence image features.

This script builds a 5-species classifier from files in archive/wavfiles using
the persistence-image pipeline defined in persistence_image.py.

Pipeline summary:
1) Read metadata and keep the 5 most frequent species.
2) For each recording, split waveform into overlapping windows.
3) Convert each window to a delay-embedding point cloud.
4) Compute persistence-image vectors (H0+H1) per window.
5) Aggregate per-recording by [mean, std] of window vectors.
6) Train and evaluate a multi-class classifier.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.io import wavfile
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from persistence_image import PD


@dataclass
class FeatureConfig:
	window_seconds: float = 1.0
	hop_seconds: float = 0.5
	points_per_window: int = 220
	embedding_dim: int = 3
	embedding_delay: int = 2
	pi_resolution: Tuple[int, int] = (24, 24)
	pi_sigma: float = 0.03
	pi_weight: float = 0.10
	max_files_per_species: Optional[int] = 120
	random_state: int = 42


def load_wav_mono_float(path: Path) -> Tuple[int, np.ndarray]:
	sr, x = wavfile.read(path)
	x = np.asarray(x)

	if x.ndim == 2:
		x = x.mean(axis=1)

	if np.issubdtype(x.dtype, np.integer):
		max_abs = np.iinfo(x.dtype).max
		x = x.astype(np.float32) / max_abs
	else:
		x = x.astype(np.float32)

	return sr, x


def sliding_windows(x: np.ndarray, sr: int, window_seconds: float, hop_seconds: float) -> List[np.ndarray]:
	window_n = int(window_seconds * sr)
	hop_n = int(hop_seconds * sr)

	if window_n <= 0 or hop_n <= 0:
		raise ValueError("window_seconds and hop_seconds must be > 0")

	if x.size < window_n:
		return []

	windows = []
	for start in range(0, x.size - window_n + 1, hop_n):
		seg = x[start : start + window_n]
		# Center and normalize each segment to reduce amplitude bias.
		seg = seg - np.mean(seg)
		std = np.std(seg)
		if std > 1e-8:
			seg = seg / std
		windows.append(seg)
	return windows


def delay_embedding(window: np.ndarray, n_points: int, emb_dim: int, delay: int) -> np.ndarray:
	needed = (emb_dim - 1) * delay + 1
	if window.size < needed:
		return np.empty((0, emb_dim), dtype=float)

	# Uniformly downsample to keep Rips computation tractable.
	sample_idx = np.linspace(0, window.size - 1, num=min(n_points, window.size), dtype=int)
	s = window[sample_idx]

	if s.size < needed:
		return np.empty((0, emb_dim), dtype=float)

	rows = s.size - (emb_dim - 1) * delay
	cloud = np.zeros((rows, emb_dim), dtype=float)
	for j in range(emb_dim):
		cloud[:, j] = s[j * delay : j * delay + rows]
	return cloud


def persistence_image_vector(cloud: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
	pd_obj = PD(cloud, max_dimension=2, type="point_cloud")
	vec, *_ = pd_obj.PI(
		w=cfg.pi_weight,
		res=cfg.pi_resolution,
		sigma=cfg.pi_sigma,
	)
	return vec.astype(np.float32)


def clip_feature_vector(path: Path, cfg: FeatureConfig) -> Optional[np.ndarray]:
	try:
		sr, x = load_wav_mono_float(path)
	except Exception:
		return None

	windows = sliding_windows(x, sr, cfg.window_seconds, cfg.hop_seconds)
	if not windows:
		return None

	window_features: List[np.ndarray] = []
	for w in windows:
		cloud = delay_embedding(w, cfg.points_per_window, cfg.embedding_dim, cfg.embedding_delay)
		if cloud.shape[0] < max(12, cfg.embedding_dim + 2):
			continue
		try:
			fv = persistence_image_vector(cloud, cfg)
		except Exception:
			continue
		window_features.append(fv)

	if not window_features:
		return None

	arr = np.vstack(window_features)
	return np.concatenate([arr.mean(axis=0), arr.std(axis=0)], axis=0)


def read_birdsong_metadata(project_root: Path) -> pd.DataFrame:
	metadata_path = project_root / "archive" / "bird_songs_metadata.csv"
	if not metadata_path.is_file():
		raise FileNotFoundError(f"Missing metadata: {metadata_path}")
	df = pd.read_csv(metadata_path)
	if "species" not in df.columns and "name" in df.columns:
		df = df.rename(columns={"name": "species"})
	missing = {"species", "filename"} - set(df.columns)
	if missing:
		raise ValueError(f"Missing metadata columns: {sorted(missing)}")
	return df


def build_dataset(project_root: Path, cfg: FeatureConfig) -> Tuple[np.ndarray, np.ndarray, List[str], int, set[str]]:
	wav_root = project_root / "archive" / "wavfiles"
	df = read_birdsong_metadata(project_root)
	top5 = list(df["species"].value_counts().head(5).index)
	df = df[df["species"].isin(top5)].copy()

	label_map = {sp: i for i, sp in enumerate(top5)}
	species_names = top5

	rng = np.random.default_rng(cfg.random_state)
	X_list: List[np.ndarray] = []
	y_list: List[int] = []
	used_files = 0
	processed_filenames: set[str] = set()

	file_list: List[Tuple[str, str]] = []
	for sp in top5:
		sp_df = df[df["species"] == sp].copy()

		if cfg.max_files_per_species is not None and len(sp_df) > cfg.max_files_per_species:
			pick_idx = rng.choice(len(sp_df), size=cfg.max_files_per_species, replace=False)
			sp_df = sp_df.iloc[pick_idx]

		for fname in sp_df["filename"].tolist():
			wav_path = wav_root / fname
			if wav_path.exists():
				file_list.append((fname, sp))

	print(f"[Dataset] Found {len(file_list)} candidate files across 5 species.")
	for fname, sp in tqdm(file_list, desc="Extracting features", unit="file"):
		wav_path = wav_root / fname
		feat = clip_feature_vector(wav_path, cfg)
		if feat is None:
			continue

		X_list.append(feat)
		y_list.append(label_map[sp])
		used_files += 1
		processed_filenames.add(fname)

	if not X_list:
		raise RuntimeError("No features were extracted. Check archive paths and parameters.")

	X = np.vstack(X_list)
	y = np.asarray(y_list, dtype=int)
	print(f"[Dataset] Extracted {used_files} features; feature dimension: {X.shape[1]}")
	return X, y, species_names, used_files, processed_filenames


def train_and_evaluate(X: np.ndarray, y: np.ndarray, class_names: Sequence[str], random_state: int) -> Dict[str, object]:
	counts = np.bincount(y)
	use_stratify = bool(counts.size > 0 and np.min(counts) >= 2)
	stratify_arg = y if use_stratify else None
	n_samples = int(y.size)
	n_classes = int(np.unique(y).size)

	if n_samples < 4:
		raise ValueError("Need at least 4 samples to perform train/test split.")

	if use_stratify:
		min_test_count = n_classes
		default_test_count = int(np.ceil(0.25 * n_samples))
		test_count = max(default_test_count, min_test_count)
		# Keep enough training samples so each class can remain represented.
		max_allowed_test = n_samples - n_classes
		test_count = min(test_count, max_allowed_test)
		if test_count <= 0:
			test_count = max(1, int(np.floor(0.25 * n_samples)))
		test_size = test_count / n_samples
	else:
		test_size = 0.25

	print(f"[Split] Training on {int(n_samples * (1 - test_size))} samples, testing on {int(n_samples * test_size)} samples.")
	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=test_size,
		random_state=random_state,
		stratify=stratify_arg,
	)

	print("[Training] Initializing RandomForest classifier...")
	model = make_pipeline(
		StandardScaler(),
		RandomForestClassifier(
			n_estimators=350,
			random_state=random_state,
			n_jobs=-1,
			class_weight="balanced_subsample",
		),
	)
	print("[Training] Fitting model...")
	model.fit(X_train, y_train)
	pred = model.predict(X_test)

	acc = accuracy_score(y_test, pred)
	all_labels = list(range(len(class_names)))
	report = classification_report(
		y_test,
		pred,
		labels=all_labels,
		target_names=list(class_names),
		digits=3,
		zero_division=0,
	)
	cm = confusion_matrix(y_test, pred, labels=all_labels)
	print(f"[Training] Validation accuracy: {acc:.4f}")
	return {
		"model": model,
		"accuracy": acc,
		"report": report,
		"confusion_matrix": cm,
		"train_size": int(y_train.size),
		"test_size": int(y_test.size),
		"scaler": model.named_steps["standardscaler"],
	}


def test_on_novel_data(
    model: object,
    project_root: Path,
    cfg: FeatureConfig,
    class_names: Sequence[str],
    trained_filenames: set[str],
    max_novel_samples: Optional[int] = 30,
) -> Optional[Dict[str, object]]:
	"""Evaluate on completely held-out novel files from the dataset."""
	wav_root = project_root / "archive" / "wavfiles"
	df = read_birdsong_metadata(project_root)
	top5 = list(df["species"].value_counts().head(5).index)
	df = df[df["species"].isin(top5)].copy()

	label_map = {sp: i for i, sp in enumerate(class_names)}

	X_novel: List[np.ndarray] = []
	y_novel: List[int] = []
	novel_count = 0

	for class_idx, sp in enumerate(class_names):
		sp_df = df[df["species"] == sp].copy()
		class_quota = None if max_novel_samples is None else divmod(max_novel_samples, len(class_names))[0] + int(class_idx < max_novel_samples % len(class_names))
		class_count = 0
		for fname in sp_df["filename"].tolist():
			if class_quota is not None and class_count >= class_quota:
				break
			wav_path = wav_root / fname
			if not wav_path.exists() or fname in trained_filenames:
				continue

			feat = clip_feature_vector(wav_path, cfg)
			if feat is None:
				continue

			X_novel.append(feat)
			y_novel.append(label_map[sp])
			novel_count += 1
			class_count += 1

	if not X_novel:
		print("[Novel Test] No novel recordings found for held-out evaluation.")
		return None

	X_novel = np.vstack(X_novel)
	y_novel = np.asarray(y_novel, dtype=int)

	print(f"[Novel Test] Evaluating on {novel_count} completely held-out recordings...")
	pred_novel = model.predict(X_novel)
	acc_novel = accuracy_score(y_novel, pred_novel)

	all_labels = list(range(len(class_names)))
	report_novel = classification_report(
		y_novel,
		pred_novel,
		labels=all_labels,
		target_names=list(class_names),
		digits=3,
		zero_division=0,
	)
	cm_novel = confusion_matrix(y_novel, pred_novel, labels=all_labels)
	print(f"[Novel Test] Held-out accuracy: {acc_novel:.4f}")
	return {
		"accuracy": acc_novel,
		"report": report_novel,
		"confusion_matrix": cm_novel,
		"n_samples": novel_count,
	}


def parse_args() -> argparse.Namespace:
	p = argparse.ArgumentParser(description="5-species birdsong classification with persistence images")
	p.add_argument("--project-root", type=str, default=".", help="Project root containing archive/")
	p.add_argument("--window-seconds", type=float, default=1.0)
	p.add_argument("--hop-seconds", type=float, default=0.5)
	p.add_argument("--points-per-window", type=int, default=220)
	p.add_argument("--embedding-dim", type=int, default=3)
	p.add_argument("--embedding-delay", type=int, default=2)
	p.add_argument("--pi-res", type=int, nargs=2, default=[10,10], metavar=("NX", "NY"))
	p.add_argument("--pi-sigma", type=float, default=0.03)
	p.add_argument("--pi-weight", type=float, default=0.10)
	p.add_argument("--max-files-per-species", type=int, default=10)
	p.add_argument("--max-novel-samples", type=int, default=10, help="Max samples for held-out test")
	p.add_argument("--random-state", type=int, default=42)
	return p.parse_args()


def main() -> None:
	args = parse_args()
	cfg = FeatureConfig(
		window_seconds=args.window_seconds,
		hop_seconds=args.hop_seconds,
		points_per_window=args.points_per_window,
		embedding_dim=args.embedding_dim,
		embedding_delay=args.embedding_delay,
		pi_resolution=(args.pi_res[0], args.pi_res[1]),
		pi_sigma=args.pi_sigma,
		pi_weight=args.pi_weight,
		max_files_per_species=args.max_files_per_species,
		random_state=args.random_state,
	)

	root = Path(args.project_root)
	print("\n" + "="*60)
	print("Birdsong PI Classifier - 5-Species Classification")
	print("="*60)
	print(f"Config: window={cfg.window_seconds}s, hop={cfg.hop_seconds}s, pi_res={cfg.pi_resolution}")
	print()

	X, y, species, used_files, trained_files = build_dataset(root, cfg)
	print()

	out = train_and_evaluate(X, y, species, cfg.random_state)
	model = out.pop("model")
	out.pop("scaler", None)

	print()
	print("="*60)
	print("VALIDATION SET RESULTS")
	print("="*60)
	print(f"Classes: {species}")
	print(f"Total recordings used: {used_files}")
	print(f"Feature dimension: {X.shape[1]}")
	print(f"Train/Test: {out['train_size']}/{out['test_size']}")
	print(f"Validation Accuracy: {out['accuracy']:.4f}")
	print("\nClassification report:\n")
	print(out["report"])
	print("Confusion matrix (rows=true, cols=pred):")
	print(out["confusion_matrix"])


	print()
	print("="*60)
	novel_results = test_on_novel_data(model, root, cfg, species, trained_files, args.max_novel_samples)
	print("="*60)
	if novel_results:
		print(f"Novel Test Accuracy: {novel_results['accuracy']:.4f}")
		print(f"Held-out samples: {novel_results['n_samples']}")
		print("\nClassification report:\n")
		print(novel_results["report"])
		print("Confusion matrix (rows=true, cols=pred):")
		print(novel_results["confusion_matrix"])
	print("="*60)


if __name__ == "__main__":
	main()
