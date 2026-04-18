import numpy as np
import matplotlib.pyplot as plt

# Load your raw XFOIL text data (skips the messy text header)
data = np.loadtxt("final_polar.txt", skiprows=12)
cl = data[:, 1]
cd = data[:, 2]
cm = data[:, 4]

# Create a professional 1x2 graph layout
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Graph 1: The Drag Polar (Cd vs Cl)
ax1.plot(cd, cl, 'b-', linewidth=2)
ax1.set_xlabel('Drag Coefficient (Cd)')
ax1.set_ylabel('Lift Coefficient (Cl)')
ax1.set_title('Optimized Drag Polar (Type 2, Re=50k)')
ax1.grid(True, linestyle='--', alpha=0.7)

# Graph 2: The Stability Curve (Cm vs Cl)
ax2.plot(cm, cl, 'r-', linewidth=2)
ax2.axvline(0, color='black', linewidth=1.5, linestyle='--') # The critical Zero line
ax2.set_xlabel('Pitching Moment (Cm)')
ax2.set_ylabel('Lift Coefficient (Cl)')
ax2.set_title('Pitching Moment vs Lift')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()