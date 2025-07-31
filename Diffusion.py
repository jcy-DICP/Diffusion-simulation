import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from matplotlib import rcParams
import scienceplots
from scipy.special import erfinv

# Set global font parameters
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']
rcParams['mathtext.fontset'] = 'stixsans'
rcParams['text.usetex'] = False

# Physical parameters
D =1.597e-9  # Diffusion coefficient (m²/s)
Lx = 1000e-6  # Space range1000 μm
time_points = [0, 5, 10, 15,20,25]  # Target time

# Generate mesh
x = np.linspace(-Lx / 2, Lx / 2, 1000)
y = np.linspace(-1e-6, 1e-6, 10)
X, Y = np.meshgrid(x, y)

# Result storage list
results = []

with plt.style.context(['science', 'no-latex']):
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    # Calculate one-dimensional diffusion
    for i, t in enumerate(time_points):
        # Calculate one-dimensional concentration
        t_safe = max(t, 1e-10)
        c_1d = 0.5*special.erfc(X[0, :] / (2 * np.sqrt(D * t_safe)))

        # Calculate the X value corresponding to the target concentration
        c_target = 0.326
        z = erfinv(1 - 2*c_target)  #Use inverse function of error function
        x_value = 2 * np.sqrt(D * t_safe) * z
        x_um = x_value * 1e6  # Convert to μm unit
        results.append((t, x_um))

        # Expand to two-dimensional
        c = np.tile(c_1d, (len(y), 1))

        # Drawing
        im = axes.flat[i].imshow(c,
                                 extent=[-Lx / 2 * 1e6, Lx / 2 * 1e6, -1, 1],
                                 aspect='auto',
                                 cmap='viridis',
                                 vmin=0, vmax=1)

        # Add line of the target concentration in phase separation frontier
        CS = axes.flat[i].contour(
            X * 1e6, Y * 1e6, c,
            levels=[c_target],
            colors='grey',
            linestyles='--',
            linewidths=1.5
        )
        axes.flat[i].clabel(CS, inline=True, fmt=f'{c_target}', fontsize=8)

    # Unified color bars
    plt.colorbar(im, ax=axes.ravel().tolist(), location='right', label='Concentration')
plt.savefig('DMK-D1.597-C0.326.tiff', dpi=900, bbox_inches='tight')
plt.show()

# Print the time and position corresponding to the target concentration.
print("\nTime(s) | X (μm)")
print("---------------------")
for t, x in results:
    print(f"{t:^7} | {x:.2f}")  # Keep two decimal places


