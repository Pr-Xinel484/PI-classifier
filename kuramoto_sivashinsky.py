#Classification of the anisotropicKuramoto-Sivashinsky using persistence image

import persistence_image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time
import argparse
from pathlib import Path

#Numerical simulation of 2D anisotropic Kuramoto-Sivashinsky equation by Fourier spectral method with ETDRK4 time stepping and parameter r
def kuramoto_sivashinsky_2d(
    Lx,
    Ly,
    Nx,
    Ny,
    dt,
    T,
    r,
    snapshot_targets=None,
    runtime_checks=True,
    check_every_steps=200
):
    if Nx <= 0 or Ny <= 0:
        raise ValueError("Nx and Ny must be positive integers.")
    if dt <= 0 or T <= 0:
        raise ValueError("dt and T must be positive.")
    if runtime_checks and check_every_steps <= 0:
        raise ValueError("check_every_steps must be positive when runtime_checks=True.")

    # Spatial grid
    x = np.linspace(0, Lx, Nx, endpoint=False)
    y = np.linspace(0, Ly, Ny, endpoint=False)
    X, Y = np.meshgrid(x, y)

    #Low apmlitude white noise initial condition
    u = 0.1 * np.random.randn(Ny, Nx)

    # Fourier wave numbers
    kx = 2 * np.pi * np.fft.fftfreq(Nx, d=Lx/Nx)
    ky = 2 * np.pi * np.fft.fftfreq(Ny, d=Ly/Ny)
    KX, KY = np.meshgrid(kx, ky)

    # Linear operator in Fourier space for: -Δu - Δ^2u -> (k^2 - k^4) u_hat
    k2 = KX**2 + KY**2
    linear_operator = k2 - k2**2

    # ETDRK4 coefficients (Kassam-Trefethen contour integral formulation)
    E = np.exp(dt * linear_operator)
    E2 = np.exp(dt * linear_operator / 2)
    M = 16
    roots = np.exp(1j * np.pi * (np.arange(1, M + 1) - 0.5) / M)
    LR = dt * linear_operator[:, :, None] + roots[None, None, :]
    Q = dt * np.real(np.mean((np.exp(LR / 2) - 1) / LR, axis=2))
    f1 = dt * np.real(np.mean((-4 - LR + np.exp(LR) * (4 - 3 * LR + LR**2)) / LR**3, axis=2))
    f2 = dt * np.real(np.mean((2 + LR + np.exp(LR) * (-2 + LR)) / LR**3, axis=2))
    f3 = dt * np.real(np.mean((-4 - 3 * LR - LR**2 + np.exp(LR) * (4 - LR)) / LR**3, axis=2))

    def nonlinear_hat(u_hat_local):
        # Nonlinear term: r*(u_x)^2 + (u_y)^2, computed in physical space
        ux = np.real(np.fft.ifft2(1j * KX * u_hat_local))
        uy = np.real(np.fft.ifft2(1j * KY * u_hat_local))
        return np.fft.fft2(r * ux**2 + uy**2)

    # Time stepping with ETDRK4
    num_steps = int(T / dt)
    if num_steps < 1:
        raise ValueError("T/dt must be at least 1 time step.")

    u_hat = np.fft.fft2(u)
    snapshots = []
    snapshot_times = []

    target_steps = []
    if snapshot_targets is not None:
        for t_target in snapshot_targets:
            if 0 <= t_target <= T:
                step_idx = int(round(t_target / dt))
                step_idx = min(max(step_idx, 0), num_steps)
                target_steps.append(step_idx)

        target_steps = sorted(set(target_steps))

        if 0 in target_steps:
            u0 = np.real(np.fft.ifft2(u_hat))
            snapshots.append(u0.copy())
            snapshot_times.append(0.0)

    for step in range(num_steps):
        Nv = nonlinear_hat(u_hat)
        a = E2 * u_hat + Q * Nv
        Na = nonlinear_hat(a)
        b = E2 * u_hat + Q * Na
        Nb = nonlinear_hat(b)
        c = E2 * a + Q * (2 * Nb - Nv)
        Nc = nonlinear_hat(c)
        u_hat = E * u_hat + f1 * Nv + 2 * f2 * (Na + Nb) + f3 * Nc

        if runtime_checks and (((step + 1) % check_every_steps == 0) or (step + 1 == num_steps)):
            if not np.isfinite(u_hat).all():
                raise FloatingPointError(f"Non-finite values detected in u_hat at step {step + 1}/{num_steps}, r={r}.")

        if snapshot_targets is not None and (step + 1) in target_steps:
            u_snapshot = np.real(np.fft.ifft2(u_hat))
            snapshots.append(u_snapshot.copy())
            snapshot_times.append((step + 1) * dt)

    # Inverse Fourier transform to get solution in physical space
    u = np.real(np.fft.ifft2(u_hat))

    if snapshot_targets is not None:
        return X, Y, u, snapshots, snapshot_times
    return X, Y, u


def normalize_height_variance(u, eps=1e-12):
    # Normalize each frame to zero mean and unit variance for visual consistency.
    centered = u - np.mean(u)
    std = np.std(centered)
    if std < eps:
        return np.zeros_like(u)
    return centered / std


def save_contour_snapshots(X, Y, frames, frame_times, r, output_path):
    normalized_frames = [np.clip(normalize_height_variance(frame), -3.0, 3.0) for frame in frames]

    fig, axes = plt.subplots(1, len(normalized_frames), figsize=(6 * len(normalized_frames), 5), constrained_layout=True)
    if len(normalized_frames) == 1:
        axes = [axes]

    contour_set = None
    for i, ax in enumerate(axes):
        contour_set = ax.contourf(X, Y, normalized_frames[i], levels=50, cmap='viridis', vmin=-3.0, vmax=3.0)
        ax.set_title(f'r={r}, t={frame_times[i]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    fig.colorbar(contour_set, ax=axes, label='normalized u')
    fig.suptitle('Anisotropic KSE Contour Snapshots')
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def save_contour_gif(X, Y, frames, frame_times, r, output_path, fps=10):
    normalized_frames = [np.clip(normalize_height_variance(frame), -3.0, 3.0) for frame in frames]

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    contour_holder = {'artist': None}

    def update(i):
        if contour_holder['artist'] is not None:
            for coll in contour_holder['artist'].collections:
                coll.remove()
        contour_holder['artist'] = ax.contourf(
            X,
            Y,
            normalized_frames[i],
            levels=50,
            cmap='viridis',
            vmin=-3.0,
            vmax=3.0
        )
        ax.set_title(f'Anisotropic KSE contour | r={r} | t={frame_times[i]:.2f}')
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    anim = FuncAnimation(fig, update, frames=len(normalized_frames), interval=1000 // fps)
    anim.save(output_path, writer=PillowWriter(fps=fps))
    plt.close(fig)

#Persistence image subspace discriminant analysis and classification
def build_pi_feature(surface, w=0.1, res=(20, 20), sigma=0.01):
    pd = persistence_image.PD(surface, type='sublevel_set')
    vec, _, _, _, _, _, _ = pd.PI(w=w, res=res, sigma=sigma)
    return vec


def stratified_train_test_split(X, y, train_ratio=0.7, seed=42):
    rng = np.random.default_rng(seed)
    train_idx = []
    test_idx = []

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        cls_idx = rng.permutation(cls_idx)
        n_train = max(1, int(train_ratio * len(cls_idx)))
        n_train = min(n_train, len(cls_idx) - 1)
        train_idx.extend(cls_idx[:n_train].tolist())
        test_idx.extend(cls_idx[n_train:].tolist())

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def train_subspace_classifier(X_train, y_train, subspace_dim=8):
    model = {}
    for cls in np.unique(y_train):
        Xc = X_train[y_train == cls]
        mu = Xc.mean(axis=0)
        Xc_centered = Xc - mu

        # Compute class subspace basis using SVD.
        _, _, Vt = np.linalg.svd(Xc_centered, full_matrices=False)
        d = min(subspace_dim, Vt.shape[0])
        U = Vt[:d].T
        model[cls] = {'mu': mu, 'U': U}

    return model


def predict_subspace_classifier(model, X):
    classes = sorted(model.keys())
    y_pred = []

    for x in X:
        residuals = []
        for cls in classes:
            mu = model[cls]['mu']
            U = model[cls]['U']
            xc = x - mu
            proj = U @ (U.T @ xc)
            residual = np.linalg.norm(xc - proj)
            residuals.append(residual)

        y_pred.append(classes[int(np.argmin(residuals))])

    return np.array(y_pred)


def confusion_counts(y_true, y_pred, class_order):
    mat = np.zeros((len(class_order), len(class_order)), dtype=int)
    class_to_i = {c: i for i, c in enumerate(class_order)}
    for yt, yp in zip(y_true, y_pred):
        mat[class_to_i[yt], class_to_i[yp]] += 1
    return mat


def stratified_k_fold_indices(y, n_splits=5, seed=42):
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(n_splits)]

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) < n_splits:
            raise ValueError(
                f"Class {cls} has only {len(cls_idx)} samples; needs at least n_splits={n_splits}."
            )

        cls_idx = rng.permutation(cls_idx)
        cls_split = np.array_split(cls_idx, n_splits)
        for fold_id in range(n_splits):
            folds[fold_id].extend(cls_split[fold_id].tolist())

    fold_pairs = []
    all_idx = np.arange(len(y))
    for fold_id in range(n_splits):
        test_idx = np.array(sorted(folds[fold_id]), dtype=int)
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_idx] = False
        train_idx = all_idx[train_mask]
        fold_pairs.append((train_idx, test_idx))

    return fold_pairs


def cross_validate_subspace_classifier(
    X,
    y,
    class_order,
    subspace_dim=8,
    n_splits=5,
    seed=42
):
    fold_splits = stratified_k_fold_indices(y, n_splits=n_splits, seed=seed)
    fold_accuracies = []
    fold_confusions = []

    for train_idx, test_idx in fold_splits:
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_test_fold = X[test_idx]
        y_test_fold = y[test_idx]

        model_fold = train_subspace_classifier(X_train_fold, y_train_fold, subspace_dim=subspace_dim)
        y_pred_fold = predict_subspace_classifier(model_fold, X_test_fold)
        acc_fold = float(np.mean(y_pred_fold == y_test_fold))
        cm_fold = confusion_counts(y_test_fold, y_pred_fold, class_order)

        fold_accuracies.append(acc_fold)
        fold_confusions.append(cm_fold)

    fold_accuracies = np.array(fold_accuracies, dtype=float)
    cm_sum = np.sum(np.array(fold_confusions, dtype=int), axis=0)
    return fold_accuracies, cm_sum


def build_dataset_cache_path(prefix, r_values, n_samples_per_class, pi_w, pi_res, pi_sigma, dt, T):
    r_tag = "-".join(f"{r:.2f}" for r in r_values)
    res_tag = f"{pi_res[0]}x{pi_res[1]}"
    fname = (
        f"{prefix}_r_{r_tag}_n_{n_samples_per_class}_"
        f"w_{pi_w}_res_{res_tag}_sigma_{pi_sigma}_dt_{dt}_T_{T}.npz"
    )
    return Path(fname)


def generate_or_load_pi_dataset(
    r_values,
    n_samples_per_class,
    pi_w,
    pi_res,
    pi_sigma,
    Lx,
    Ly,
    Nx,
    Ny,
    dt,
    T,
    seed_offset,
    cache_path,
    enable_runtime_logging=True
):
    if cache_path.exists():
        cached = np.load(cache_path)
        X_cached = cached["X_features"]
        y_cached = cached["y_labels"]
        if enable_runtime_logging:
            print(f"[cache] Loaded dataset from {cache_path}")
        return X_cached, y_cached

    X_features = []
    y_labels = []
    expected_feature_dim = None

    for r_cls in r_values:
        if enable_runtime_logging:
            print(f"[runtime] Generating class r={r_cls} with {n_samples_per_class} samples")

        for sample_id in range(n_samples_per_class):
            np.random.seed(seed_offset + int(100 * r_cls) + sample_id)
            _, _, surface = kuramoto_sivashinsky_2d(
                Lx,
                Ly,
                Nx,
                Ny,
                dt,
                T,
                r_cls,
                runtime_checks=True,
                check_every_steps=250
            )
            feature_vec = build_pi_feature(surface, w=pi_w, res=pi_res, sigma=pi_sigma)

            if expected_feature_dim is None:
                expected_feature_dim = feature_vec.shape[0]
            elif feature_vec.shape[0] != expected_feature_dim:
                raise RuntimeError(
                    f"Feature dimension mismatch: expected {expected_feature_dim}, got {feature_vec.shape[0]} "
                    f"for r={r_cls}, sample={sample_id}."
                )

            if not np.isfinite(feature_vec).all():
                raise FloatingPointError(
                    f"Non-finite values in PI feature vector for r={r_cls}, sample={sample_id}."
                )

            X_features.append(feature_vec)
            y_labels.append(r_cls)

    X_features = np.array(X_features, dtype=float)
    y_labels = np.array(y_labels, dtype=float)

    np.savez_compressed(
        cache_path,
        X_features=X_features,
        y_labels=y_labels,
        r_values=np.array(r_values, dtype=float),
        n_samples_per_class=np.array([n_samples_per_class], dtype=int),
        pi_w=np.array([pi_w], dtype=float),
        pi_res=np.array(pi_res, dtype=int),
        pi_sigma=np.array([pi_sigma], dtype=float),
        dt=np.array([dt], dtype=float),
        T=np.array([T], dtype=float),
        seed_offset=np.array([seed_offset], dtype=int),
    )
    if enable_runtime_logging:
        print(f"[cache] Saved dataset to {cache_path}")

    return X_features, y_labels


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kuramoto-Sivashinsky persistence-image classification")
    parser.add_argument("--samples-per-class", type=int, default=50)
    parser.add_argument("--novel-samples-per-class", type=int, default=10)
    parser.add_argument("--nx", type=int, default=256)
    parser.add_argument("--ny", type=int, default=256)
    parser.add_argument("--lx", type=float, default=512)
    parser.add_argument("--ly", type=float, default=512)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    args = parser.parse_args(argv)
    if args.samples_per_class < max(2, args.cv_folds) or args.novel_samples_per_class < 1:
        parser.error("Not enough samples for train/test and cross-validation")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    r_values = [1.25, 1.75]
    w, res, sigma = 0.1, (10, 10), 0.01
    def dataset(prefix, n, seed):
        cache_path = args.cache_dir / build_dataset_cache_path(prefix, r_values, n, w, res, sigma, args.dt, args.duration)
        tag = f"{args.nx}x{args.ny}_{args.lx:g}x{args.ly:g}_"
        cache_path = cache_path.with_name(tag + cache_path.name)
        return generate_or_load_pi_dataset(r_values, n, w, res, sigma, args.lx, args.ly, args.nx, args.ny, args.dt, args.duration, seed, cache_path)
    start = time.perf_counter()
    X, y = dataset("ks_pi_train", args.samples_per_class, 1000)
    X_train, X_test, y_train, y_test = stratified_train_test_split(X, y, train_ratio=0.7, seed=7)
    model = train_subspace_classifier(X_train, y_train, subspace_dim=5)
    y_pred = predict_subspace_classifier(model, X_test)
    X_novel, y_novel = dataset("ks_pi_novel", args.novel_samples_per_class, 100000)
    y_novel_pred = predict_subspace_classifier(model, X_novel)
    cv_scores, cv_cm = cross_validate_subspace_classifier(X, y, r_values, subspace_dim=5, n_splits=args.cv_folds, seed=11)
    print(f"Classes: {r_values}")
    print(f"Holdout accuracy: {np.mean(y_pred == y_test):.3f}")
    print("Holdout confusion matrix:\n", confusion_counts(y_test, y_pred, r_values))
    print(f"Novel accuracy: {np.mean(y_novel_pred == y_novel):.3f}")
    print("Novel confusion matrix:\n", confusion_counts(y_novel, y_novel_pred, r_values))
    print(f"{args.cv_folds}-fold CV: {np.round(cv_scores, 3)}; mean {cv_scores.mean():.3f}")
    print("CV confusion matrix:\n", cv_cm)
    print(f"Elapsed: {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
