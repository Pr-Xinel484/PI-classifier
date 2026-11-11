"""Compare H1 persistence images of two simulated Kuramoto-Sivashinsky surfaces."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from persistence_image import PD
from kuramoto_sivashinsky import kuramoto_sivashinsky_2d

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grid-size",type=int,default=64)
    p.add_argument("--duration",type=float,default=2)
    p.add_argument("--dt",type=float,default=.01)
    p.add_argument("--output",type=Path,default=Path("outputs/kuramoto_pi_comparison.png"))
    a=p.parse_args()
    if a.grid_size < 8 or a.dt <= 0 or a.duration < a.dt: p.error("Invalid grid/dt/duration")
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,r in zip(axes,[1.,2.]):
        np.random.seed(42)
        _,_,surface=kuramoto_sivashinsky_2d(Lx=32,Ly=32,Nx=a.grid_size,Ny=a.grid_size,dt=a.dt,T=a.duration,r=r)
        _,_,pi,_,_,_,extent=PD(surface,type="sublevel_set").PI(w=.1,res=(40,40),sigma=.01)
        im=ax.imshow(pi.T,origin="lower",aspect="auto",extent=extent)
        ax.set(title=f"H1 persistence image r={r:g}",xlabel="Birth",ylabel="Lifetime")
        fig.colorbar(im,ax=ax)
    fig.tight_layout()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(a.output,dpi=150)
    plt.close(fig)
    print(f"Saved {a.output}")
if __name__ == "__main__": main()
