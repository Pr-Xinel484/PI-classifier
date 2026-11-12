# Persistence Images: Birdsong Classification & Kuramoto–Sivashinsky Experiments

Constructing **persistence images (PIs)** from persistent homology and using those features in machine-learning experiments. The repository covers two applications:

1. **Birdsong classification:** extract topological features from audio signals or their mel spectrograms, then classify bird species.
2. **Anisotropic Kuramoto–Sivashinsky (KS) equation:** simulate 2D surfaces, compute sublevel-set persistence images, and classify the parameter setting.

The PI implementation uses [GUDHI](https://gudhi.inria.fr/) to compute persistence diagrams and integrates weighted Gaussian kernels over a birth–lifetime grid to form fixed-length feature vectors for dimensions H0 and H1. 

## Project structure

```text
.
├── persistence_image.py              # Shared point-cloud / sublevel-set PI implementation
├── birdsong_mel_classifier.py        # Main birdsong pipeline (group-aware evaluation)
├── birdsong_delay_classifier.py      # Alternative waveform delay-embedding experiment
├── kuramoto_sivashinsky.py           # KS simulator, PI classifier, CLI
├── examples/
│   ├── persistence_image_trials.py   # Noisy circle / torus visualizations
│   └── compare_kuramoto_images.py    # H1 images for two KS parameter values
├── tests/                             # Small synthetic smoke tests (no external dataset)
├── legacy/                            # Earlier original experiments, for reference
├── requirements.txt
├── requirements-dev.txt
└── .github/workflows/tests.yml
```



## Quick start (does not require bird recordings)

Generate a noisy two-circle point-cloud example and save its H1 persistence image:

```bash
python -m examples.persistence_image_trials circle
```

Or generate a small torus example (more expensive due to the Rips complex):

```bash
python -m examples.persistence_image_trials torus --points 12
```

Compare the H1 persistence images of two KS simulations:

```bash
python -m examples.compare_kuramoto_images --grid-size 32 --duration 1
```

## Kuramoto–Sivashinsky classification

The solver uses a Fourier spectral discretization and ETDRK4 time-stepping. It generates surfaces for anisotropy parameters `r = 1.25` and `r = 1.75`, converts them into sublevel-set PIs, and fits a subspace-based classifier with held-out and cross-validation results.


## Paper and attribution

The persistence-image method was introduced in:

> Adams, H., Emerson, T., Kirby, M., Neville, R., Peterson, C., Shipman, P., Chepushtanova, S., Hanson, E., Motta, F., & Ziegelmeier, L. (2017). **Persistence Images: A Stable Vector Representation of Persistent Homology.** *Journal of Machine Learning Research*, 18, 1–35. [Paper](https://jmlr.org/papers/v18/16-337.html).


