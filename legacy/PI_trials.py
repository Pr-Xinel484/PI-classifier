import Persistence_image
import numpy as np
import matplotlib.pyplot as plt
import gudhi as gd
# Example usage for noisy circles, noisy torus
#user input circle or torus to choos the example
input_str = input("Choose example: 'circle' or 'torus': ")
if input_str == 'circle':
    # Example usage for noisy circles
    if __name__ == "__main__":
        # Noisy sample points of two circles
        num_points = 100
        noise_level = 0.05
        angles_1 = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    angles_2 = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = np.array([(0.25*np.cos(a)-2 + noise_level * np.random.randn(),
                        0.25*np.sin(a)-1 + noise_level * np.random.randn()) for a in angles_1])
    points = np.vstack([points, np.array([(np.cos(a)+2 + noise_level * np.random.randn(),
                                           np.sin(a) + noise_level * np.random.randn()) for a in angles_2])])

    # Create persistence diagram
    pd = Persistence_image.PD(points)

    # Compute persistence image
    w = 0.1  # weight parameter
    res = (200, 200)  # resolution of the persistence image
    pi = pd.PI(w=w, res=res, sigma=0.01)
    # Plot point cloud next to H1 persistence image (without H0 panel).
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(points[:, 0], points[:, 1], color='blue', s=10)
    axs[0].set_aspect('equal', 'box')

    axs[0].legend()

    h1_extent = pi[6]
    im_h1 = axs[1].imshow(pi[2].T, origin='lower', aspect='auto', extent=h1_extent, cmap='viridis')
    axs[1].set_title('Persistence Image (H1)')
    axs[1].set_xlabel('Birth')
    axs[1].set_ylabel('Lifetime')

    plt.tight_layout()
    plt.show()

    # H1 points are returned by PI as birth-lifetime coordinates.
    bl_dim1 = pi[4]
    h1_birth = bl_dim1[:, 0]
    h1_lifetime = bl_dim1[:, 1]
    h1_death = h1_birth + h1_lifetime

    # Make 3 plots: native GUDHI PD (birth-death), H1 birth-lifetime PD, and H1 PI.
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    # Native GUDHI plot in birth-death coordinates.
    gd.plot_persistence_diagram(pd.persistence_diagram, axes=axs[0])
    axs[0].set_title('Persistence Diagram (Birth-Death)')

    # Plot persistence diagram in birth-persistence coordinates
    axs[1].scatter(h1_birth, h1_lifetime, color='green', s=10, label='dim 1')
    axs[1].set_xlim(h1_extent[0], h1_extent[1])
    axs[1].set_ylim(h1_extent[2], h1_extent[3])
    axs[1].set_xlabel('Birth')
    axs[1].set_ylabel('Lifetime')
    axs[1].set_title('Persistence Diagram (Birth-Lifetime Coordinates)')
    axs[1].legend()

    # Plot persistence image
    im = axs[2].imshow(pi[2].T, origin='lower', aspect='auto', extent=h1_extent, cmap='viridis')
    axs[2].set_xlim(h1_extent[0], h1_extent[1])
    axs[2].set_ylim(h1_extent[2], h1_extent[3])
    axs[2].set_title('Persistence Image (H1)')
    axs[2].set_xlabel('Birth')
    axs[2].set_ylabel('Lifetime')
   # fig.colorbar(im, ax=axs[2])
    plt.tight_layout()
    plt.show()

    # Demonstration of PI stability under noise by computing ratio of PI distance to Wasserstein distance across varying noise levels.
    noise_levels = [0.01, 0.03, 0.05]
    original_points = np.array([(0.25*np.cos(a)-2, 0.25*np.sin(a)-1) for a in angles_1])
    original_points = np.vstack([original_points, np.array([(np.cos(a)+2, np.sin(a)) for a in angles_2])])
    original_pd = Persistence_image.PD(original_points)
    original_pi = original_pd.PI(w=w, res=res, sigma=0.01)

    ratios = []
    for noise in noise_levels:
        noisy_points = np.array([(0.25*np.cos(a)-2 + noise * np.random.randn(),
                                  0.25*np.sin(a)-1 + noise * np.random.randn()) for a in angles_1])
        noisy_points = np.vstack([noisy_points, np.array([(np.cos(a)+2 + noise * np.random.randn(),
                                                           np.sin(a) + noise * np.random.randn()) for a in angles_2])])
        noisy_pd = Persistence_image.PD(noisy_points)
        noisy_pi = noisy_pd.PI(w=w, res=res, sigma=0.01)
        # Convert persistence diagrams to numpy arrays (dimension-filtered, H1 only)
        original_h1_bd = np.array([[birth, death] for dim, (birth, death) in original_pd.persistence_diagram if dim == 1 and death != float('inf')], dtype=np.float64)
        noisy_h1_bd = np.array([[birth, death] for dim, (birth, death) in noisy_pd.persistence_diagram if dim == 1 and death != float('inf')], dtype=np.float64)
        # Compute bottleneck distance between original PD and noisy PD (H1 only)
        wasserstein_distance = gd.bottleneck_distance(original_h1_bd, noisy_h1_bd) if original_h1_bd.size > 0 and noisy_h1_bd.size > 0 else 0.0
        # Compute L1 distance between original PI and noisy PI
        pi_distance = np.linalg.norm(original_pi[0] - noisy_pi[0], ord=1)
        # Compute ratio of PI distance to Wasserstein distance
        ratio = pi_distance / wasserstein_distance if wasserstein_distance > 0 else float('inf')
        ratios.append(ratio)

    # Plot ratio of distances against noise level
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(noise_levels, ratios, marker='o', linewidth=2, markersize=8, color='blue', label='PI Distance / Wasserstein Distance')
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('Ratio (PI Distance / Wasserstein Distance)')
    ax.set_title('Stability of Persistence Images under Noise')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()







elif input_str == 'torus':
    if __name__ == "__main__":
        num_points = 50
        noise_level = 0.05
        angles_1 = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        angles_2 = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
        points = np.array([((2 + 0.5*np.cos(a))*np.cos(b) + noise_level * np.random.randn(),
                            (2 + 0.5*np.cos(a))*np.sin(b) + noise_level * np.random.randn(),
                            1*np.sin(a) + noise_level * np.random.randn()) for a in angles_1 for b in angles_2])
        pd = Persistence_image.PD(points, sparse=1)
        # Compute persistence image
        w = 0.05  # weight parameter
        res = (200, 200)  # resolution of the persistence image
        pi = pd.PI(w=w, res=res, sigma=0.03)
        # Plot 3D point cloud next to H1 persistence image.
        fig = plt.figure(figsize=(12, 5))
        axs0 = fig.add_subplot(121, projection='3d')
        axs1 = fig.add_subplot(122)
        axs0.scatter(points[:, 0], points[:, 1], points[:, 2], color='blue', s=10, label='Point Cloud')
        axs0.set_title('3D Point Cloud')
        axs0.legend()

        h1_extent = pi[6]
        im_h1 = axs1.imshow(pi[2].T, origin='lower', aspect='auto', extent=h1_extent, cmap='viridis')
        axs1.set_title('Persistence Image (H1)')
        axs1.set_xlabel('Birth')
        axs1.set_ylabel('Lifetime')
        fig.colorbar(im_h1, ax=axs1)
        plt.tight_layout()
        plt.show()
else:
    print("Invalid input. Please choose 'circle' or 'torus'.")
