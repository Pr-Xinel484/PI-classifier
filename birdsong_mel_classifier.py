"""Bird-song 5-species classification with persistence images.

This script builds persistence-image (PI) features from mel spectrograms using
sublevel-set filtrations, then trains and evaluates a classifier with
group-aware data splitting to reduce leakage across clips from the same
recording session.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import joblib
import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
	balanced_accuracy_score,
	classification_report,
	confusion_matrix,
	f1_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

import persistence_image


@dataclass
class Config:
	workspace: str = "."
	metadata_csv: str = "archive/bird_songs_metadata.csv"
	wav_dir: str = "archive/wavfiles"
	cache_dir: str = "cache"
	model_path: str = "cache/bird_song_pi_5species_classifier.joblib"
	results_path: str = "cache/bird_song_pi_5species_results.json"

	species: Tuple[str, ...] = (
		"Song Sparrow",
		"Northern Mockingbird",
		"Northern Cardinal",
		"American Robin",
		"Bewick's Wren",
	)

	random_state: int = 42
	max_files_per_class: int = 10
	min_duration_s: float = 2.0
	sample_rate: int = 22050

	segment_seconds: float = 3.0
	segment_hop_seconds: float = 1.5
	max_windows_per_file: int = 6

	n_mels: int = 96
	n_fft: int = 1024
	hop_length: int = 256
	fmin: int = 1000
	fmax: int = 10000

	pi_res: Tuple[int, int] = (20, 20)
	pi_sigma: float = 0.02
	pi_weight_w: float = 0.05

	test_size: float = 0.2
	split_attempts: int = 120


def _feature_cache_key(cfg: Config) -> str:
	key_fields = {
		"species": cfg.species,
		"max_files_per_class": cfg.max_files_per_class,
		"sample_rate": cfg.sample_rate,
		"segment_seconds": cfg.segment_seconds,
		"segment_hop_seconds": cfg.segment_hop_seconds,
		"max_windows_per_file": cfg.max_windows_per_file,
		"n_mels": cfg.n_mels,
		"n_fft": cfg.n_fft,
		"hop_length": cfg.hop_length,
		"fmin": cfg.fmin,
		"fmax": cfg.fmax,
		"pi_res": cfg.pi_res,
		"pi_sigma": cfg.pi_sigma,
		"pi_weight_w": cfg.pi_weight_w,
		"min_duration_s": cfg.min_duration_s,
		"random_state": cfg.random_state,
	}
	raw = json.dumps(key_fields, sort_keys=True).encode("utf-8")
	return hashlib.sha256(raw).hexdigest()[:16]


def load_metadata(cfg: Config) -> pd.DataFrame:
	metadata_path = Path(cfg.workspace) / cfg.metadata_csv
	wav_dir = Path(cfg.workspace) / cfg.wav_dir
	if not metadata_path.exists():
		raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
	if not wav_dir.exists():
		raise FileNotFoundError(f"WAV directory not found: {wav_dir}")

	df = pd.read_csv(metadata_path)
	required_cols = {"id", "name", "sound_type", "filename"}
	missing = required_cols.difference(df.columns)
	if missing:
		raise ValueError(f"Missing metadata columns: {sorted(missing)}")

	df = df[df["name"].isin(cfg.species)].copy()
	df = df[df["sound_type"].fillna("").str.contains("song", case=False, na=False)].copy()
	df["filepath"] = df["filename"].apply(lambda fn: str((wav_dir / fn).resolve()))
	df = df[df["filepath"].apply(lambda p: Path(p).exists())].copy()

	if df.empty:
		raise ValueError("No usable rows found after metadata filtering.")

	# Cap files per class for tractable experiments while preserving balance.
	capped_parts: List[pd.DataFrame] = []
	rng = np.random.default_rng(cfg.random_state)
	for species_name, part in df.groupby("name", sort=False):
		if len(part) > cfg.max_files_per_class:
			idx = rng.choice(len(part), size=cfg.max_files_per_class, replace=False)
			part = part.iloc[np.sort(idx)]
		capped_parts.append(part)

	out = pd.concat(capped_parts, ignore_index=True)
	out = out[out["filepath"].apply(lambda p: librosa.get_duration(path=p) >= cfg.min_duration_s)].copy()
	out.reset_index(drop=True, inplace=True)
	return out


def window_starts(signal_length: int, window_len: int, hop_len: int) -> List[int]:
	if signal_length <= window_len:
		return [0]
	starts = list(range(0, signal_length - window_len + 1, hop_len))
	if starts[-1] != signal_length - window_len:
		starts.append(signal_length - window_len)
	return starts


def compute_pi_for_audio(
	y: np.ndarray,
	sr: int,
	cfg: Config,
	rng: np.random.Generator,
) -> np.ndarray:
	win_len = int(cfg.segment_seconds * sr)
	hop_len = int(cfg.segment_hop_seconds * sr)
	starts = window_starts(len(y), win_len, hop_len)

	if len(starts) > cfg.max_windows_per_file:
		starts = sorted(rng.choice(starts, size=cfg.max_windows_per_file, replace=False).tolist())

	vecs: List[np.ndarray] = []
	for st in starts:
		seg = y[st : st + win_len]
		if seg.size == 0:
			continue

		S = librosa.feature.melspectrogram(
			y=seg,
			sr=sr,
			n_mels=cfg.n_mels,
			n_fft=cfg.n_fft,
			hop_length=cfg.hop_length,
			fmin=cfg.fmin,
			fmax=cfg.fmax,
			power=2.0,
		)
		S_db = librosa.power_to_db(S, ref=np.max)

		# Build a smooth scalar field for sublevel filtration.
		surface = -S_db
		surface = (surface - np.mean(surface)) / (np.std(surface) + 1e-8)

		X = np.arange(surface.shape[0], dtype=float)
		Y = np.arange(surface.shape[1], dtype=float)
		pd_obj = persistence_image.PD((X, Y, surface), type="sublevel_set")
		vec, *_ = pd_obj.PI(
			w=cfg.pi_weight_w,
			res=cfg.pi_res,
			sigma=cfg.pi_sigma,
		)
		vecs.append(vec.astype(np.float32))

	if not vecs:
		raise ValueError("No persistence vectors computed for audio clip.")

	arr = np.vstack(vecs)
	mean_vec = arr.mean(axis=0)
	std_vec = arr.std(axis=0)
	return np.concatenate([mean_vec, std_vec], axis=0)


def build_or_load_features(cfg: Config, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
	cache_dir = Path(cfg.workspace) / cfg.cache_dir
	cache_dir.mkdir(parents=True, exist_ok=True)
	file_snapshot = [(str(row.filename), str(row.id), str(row.name), int(Path(row.filepath).stat().st_size), int(Path(row.filepath).stat().st_mtime_ns)) for row in df.itertuples(index=False)]
	snapshot_hash = hashlib.sha256(json.dumps(file_snapshot, sort_keys=True).encode("utf-8")).hexdigest()[:12]
	cache_file = cache_dir / f"pi_features_{_feature_cache_key(cfg)}_{snapshot_hash}.joblib"

	if cache_file.exists():
		data = joblib.load(cache_file)
		return data["X"], data["y"], data["groups"], data["filenames"]

	rng = np.random.default_rng(cfg.random_state)
	X_rows: List[np.ndarray] = []
	y_rows: List[str] = []
	group_rows: List[str] = []
	fn_rows: List[str] = []

	total = len(df)
	for i, row in enumerate(df.itertuples(index=False), start=1):
		path = row.filepath
		try:
			y, sr = librosa.load(path, sr=cfg.sample_rate, mono=True)
			feat = compute_pi_for_audio(y, sr, cfg, rng)
		except Exception:
			continue

		X_rows.append(feat)
		y_rows.append(row.name)
		group_rows.append(str(row.id))
		fn_rows.append(row.filename)

		if i % 100 == 0 or i == total:
			print(f"Processed {i}/{total} files")

	if not X_rows:
		raise RuntimeError("Feature extraction produced zero usable samples.")

	X = np.vstack(X_rows)
	y = np.array(y_rows)
	groups = np.array(group_rows)

	joblib.dump(
		{
			"X": X,
			"y": y,
			"groups": groups,
			"filenames": fn_rows,
			"config": asdict(cfg),
		},
		cache_file,
	)
	return X, y, groups, fn_rows


def _class_distribution(y: np.ndarray) -> Dict[str, float]:
	vals, cnts = np.unique(y, return_counts=True)
	total = float(len(y))
	return {str(v): float(c / total) for v, c in zip(vals, cnts)}


def _dist_error(target: Dict[str, float], candidate: Dict[str, float]) -> float:
	keys = sorted(set(target.keys()).union(candidate.keys()))
	return float(sum(abs(target.get(k, 0.0) - candidate.get(k, 0.0)) for k in keys))


def best_group_holdout_split(
	y: np.ndarray,
	groups: np.ndarray,
	test_size: float,
	random_state: int,
	attempts: int,
) -> Tuple[np.ndarray, np.ndarray]:
	target_dist = _class_distribution(y)
	best_train: np.ndarray | None = None
	best_test: np.ndarray | None = None
	best_score = float("inf")

	for i in range(attempts):
		splitter = GroupShuffleSplit(
			n_splits=1,
			test_size=test_size,
			random_state=random_state + i,
		)
		tr_idx, te_idx = next(splitter.split(np.zeros(len(y)), y, groups=groups))
		if set(y[tr_idx]) != set(y) or set(y[te_idx]) != set(y):
			continue
		if any(len(set(groups[tr_idx][y[tr_idx] == label])) < 5 for label in np.unique(y)):
			continue
		te_dist = _class_distribution(y[te_idx])
		score = _dist_error(target_dist, te_dist)
		if score < best_score:
			best_score = score
			best_train = tr_idx
			best_test = te_idx

	if best_train is None or best_test is None:
		raise ValueError("Unable to create a group-aware holdout with all species and five independent training groups per species. Add data or increase --max-files-per-class.")

	return best_train, best_test


def select_model(
	X_train: np.ndarray,
	y_train_int: np.ndarray,
	groups_train: np.ndarray,
	random_state: int,
) -> Tuple[Pipeline, Dict[str, float]]:
	cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=random_state)

	candidates: Dict[str, Pipeline] = {
		"rf": Pipeline(
			[
				("scaler", StandardScaler()),
				(
					"clf",
					RandomForestClassifier(
						n_estimators=400,
						max_depth=None,
						min_samples_leaf=1,
						n_jobs=-1,
						class_weight="balanced_subsample",
						random_state=random_state,
					),
				),
			]
		),
		"svm": Pipeline(
			[
				("scaler", StandardScaler()),
				(
					"clf",
					SVC(
						kernel="rbf",
						C=4.0,
						gamma="scale",
						class_weight="balanced",
						probability=False,
						random_state=random_state,
					),
				),
			]
		),
	}

	cv_scores: Dict[str, float] = {}
	for name, pipe in candidates.items():
		scores = cross_val_score(
			pipe,
			X_train,
			y_train_int,
			cv=cv,
			groups=groups_train,
			scoring="f1_macro",
			n_jobs=-1,
		)
		cv_scores[name] = float(np.mean(scores))
		print(f"CV macro-F1 ({name}) = {cv_scores[name]:.4f}")

	best_name = max(cv_scores, key=cv_scores.get)
	best_pipe = candidates[best_name]
	best_pipe.fit(X_train, y_train_int)
	return best_pipe, cv_scores


def run_pipeline(cfg: Config) -> Dict[str, object]:
	print("Loading and filtering metadata...")
	df = load_metadata(cfg)
	print("Class counts after filtering:")
	print(df["name"].value_counts().sort_index())

	print("Building/loading PI features...")
	X, y, groups, filenames = build_or_load_features(cfg, df)
	print(f"Usable samples: {len(y)}")

	le = LabelEncoder()
	y_int = le.fit_transform(y)

	tr_idx, te_idx = best_group_holdout_split(
		y=y,
		groups=groups,
		test_size=cfg.test_size,
		random_state=cfg.random_state,
		attempts=cfg.split_attempts,
	)

	X_train, X_test = X[tr_idx], X[te_idx]
	y_train_int, y_test_int = y_int[tr_idx], y_int[te_idx]
	groups_train = groups[tr_idx]

	print("Selecting model with group-aware CV...")
	model, cv_scores = select_model(X_train, y_train_int, groups_train, cfg.random_state)

	y_pred_int = model.predict(X_test)
	y_test = le.inverse_transform(y_test_int)
	y_pred = le.inverse_transform(y_pred_int)

	macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
	bal_acc = float(balanced_accuracy_score(y_test, y_pred))
	cm = confusion_matrix(y_test, y_pred, labels=list(le.classes_))
	report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

	print(f"Holdout macro-F1: {macro_f1:.4f}")
	print(f"Holdout balanced accuracy: {bal_acc:.4f}")

	artifact = {
		"model": model,
		"label_encoder": le,
		"classes": list(le.classes_),
		"config": asdict(cfg),
		"cv_scores": cv_scores,
		"holdout_macro_f1": macro_f1,
		"holdout_balanced_accuracy": bal_acc,
	}

	model_path = Path(cfg.workspace) / cfg.model_path
	model_path.parent.mkdir(parents=True, exist_ok=True)
	joblib.dump(artifact, model_path)

	results_payload = {
		"classes": list(le.classes_),
		"n_samples": int(len(y)),
		"n_train": int(len(tr_idx)),
		"n_test": int(len(te_idx)),
		"cv_scores": cv_scores,
		"holdout_macro_f1": macro_f1,
		"holdout_balanced_accuracy": bal_acc,
		"confusion_matrix": cm.tolist(),
		"classification_report": report_dict,
		"class_counts": {k: int(v) for k, v in pd.Series(y).value_counts().sort_index().to_dict().items()},
		"test_class_counts": {k: int(v) for k, v in pd.Series(y_test).value_counts().sort_index().to_dict().items()},
		"test_filenames": [filenames[i] for i in te_idx],
	}

	results_path = Path(cfg.workspace) / cfg.results_path
	results_path.parent.mkdir(parents=True, exist_ok=True)
	with open(results_path, "w", encoding="utf-8") as f:
		json.dump(results_payload, f, indent=2)

	return results_payload


def parse_args(argv: Sequence[str] | None = None) -> Config:
	parser = argparse.ArgumentParser(description="Train 5-species bird-song PI classifier")
	parser.add_argument("--workspace", type=str, default=".")
	parser.add_argument("--max-files-per-class", type=int, default=350)
	parser.add_argument("--sample-rate", type=int, default=22050)
	parser.add_argument("--segment-seconds", type=float, default=3.0)
	parser.add_argument("--segment-hop-seconds", type=float, default=1.5)
	parser.add_argument("--max-windows-per-file", type=int, default=6)
	parser.add_argument("--pi-res", type=int, default=20)
	parser.add_argument("--pi-sigma", type=float, default=0.02)
	parser.add_argument("--pi-weight-w", type=float, default=0.05)
	parser.add_argument("--test-size", type=float, default=0.2)
	parser.add_argument("--split-attempts", type=int, default=120)
	parser.add_argument("--random-state", type=int, default=42)
	args = parser.parse_args(argv)

	return Config(
		workspace=args.workspace,
		max_files_per_class=args.max_files_per_class,
		sample_rate=args.sample_rate,
		segment_seconds=args.segment_seconds,
		segment_hop_seconds=args.segment_hop_seconds,
		max_windows_per_file=args.max_windows_per_file,
		pi_res=(args.pi_res, args.pi_res),
		pi_sigma=args.pi_sigma,
		pi_weight_w=args.pi_weight_w,
		test_size=args.test_size,
		split_attempts=args.split_attempts,
		random_state=args.random_state,
	)


def main() -> None:
	cfg = parse_args()
	results = run_pipeline(cfg)
	print("Done.")
	print(json.dumps({
		"holdout_macro_f1": results["holdout_macro_f1"],
		"holdout_balanced_accuracy": results["holdout_balanced_accuracy"],
		"cv_scores": results["cv_scores"],
	}, indent=2))


if __name__ == "__main__":
	main()
