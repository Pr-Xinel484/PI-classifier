"""Visualize persistence images on a noisy circle or torus."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persistence_image import PD

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("shape", choices=["circle", "torus"])
    p.add_argument("--points", type=int, default=12)
    p.add_argument("--resolution", type=int, default=40)
    p.add_argument("--output-dir", type=Path, default=Path("outputs"))
    a = p.parse_args()
    if a.points < 5 or a.resolution < 1:
        p.error("--points must be >= 5 and --resolution >= 1")
    rng = np.random.default_rng(42)
    angles = np.linspace(0, 2*np.pi, a.points, endpoint=False)
    if a.shape == "circle":
        pts = np.vstack([np.column_stack((.25*np.cos(angles)-2, .25*np.sin(angles)-1)), np.column_stack((np.cos(angles)+2,np.sin(angles)))])
    else:
        t, u = np.meshgrid(angles, angles)
        pts = np.column_stack(((2+.5*np.cos(t)).ravel()*np.cos(u).ravel(),(2+.5*np.cos(t)).ravel()*np.sin(u).ravel(),np.sin(t).ravel()))
    pts += rng.normal(0,.05,pts.shape)
    _, _, h1, _, _, _, extent = PD(pts, sparse=1 if a.shape == "torus" else None).PI(w=.1,res=(a.resolution,a.resolution),sigma=.03)
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121, projection="3d" if a.shape == "torus" else None)
    if a.shape == "circle": ax.scatter(pts[:,0],pts[:,1],s=9)
    else: ax.scatter(pts[:,0],pts[:,1],pts[:,2],s=5)
    ax.set_title(a.shape.title())
    ax = fig.add_subplot(122)
    im = ax.imshow(h1.T, origin="lower", aspect="auto", extent=extent)
    ax.set(title="H1 persistence image",xlabel="Birth",ylabel="Lifetime")
    fig.colorbar(im,ax=ax)
    fig.tight_layout()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    dest = a.output_dir / f"{a.shape}_persistence_image.png"
    fig.savefig(dest,dpi=150)
    plt.close(fig)
    print(f"Saved {dest}")
if __name__ == "__main__": main()
