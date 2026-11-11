#Generate some PI's for illustraion
import numpy as np
import matplotlib.pyplot as plt
import Persistence_image
import Kuramoto_Sivashinsky_classifcation as KS


surface_1 = KS.kuramoto_sivashinsky_2d(Lx=32, Ly=32, Nx=64, Ny=64,dt=0.01, T = 10, r= 1)
surface_2 = KS.kuramoto_sivashinsky_2d(Lx=32, Ly=32, Nx=64, Ny=64,dt=0.01, T = 10, r= 2)
pd_1 = Persistence_image.PD(surface_1, type='sublevel_set')
pd_2 = Persistence_image.PD(surface_2, type='sublevel_set')
pi_1 = pd_1.PI(w=0.1, res=(50, 50), sigma=0.01)
pi_2 = pd_2.PI(w=0.1, res=(50, 50), sigma=0.01)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
im1 = axs[0].imshow(pi_1[2].T, origin='lower', aspect='auto', extent=pi_1[6], cmap='viridis')
axs[0].set_title('Persistence Image (H1) - r=1, t =10')
axs[0].set_xlabel('Birth')
axs[0].set_ylabel('Lifetime')

im2 = axs[1].imshow(pi_2[2].T, origin='lower', aspect='auto', extent=pi_2[6], cmap='viridis')
axs[1].set_title('Persistence Image (H1) - r=2, t =10')
axs[1].set_xlabel('Birth')
axs[1].set_ylabel('Lifetime')

plt.tight_layout()
plt.show()
