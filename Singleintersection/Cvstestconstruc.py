import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


""" curve1x = [[(0.0),(0.0)],[(0.27),(0.06)],[(0.58),(0.13)],[(1.0),(0.22)],[(1.41),(0.31)],[(1.75),(0.37)],[(2.16),(0.44)]
           ,[(2.58),(0.51)],[(3.16),(0.6)],[(3.27),(0.67)],[(4.0),(0.7)],[(4.4),(0.74)],[(4.8),(0.78)],[(5.18),(0.81)],
           [(5.63),(0.84)],[(5.97),(0.86)],[(6.33),(0.87)],[(6.88),(0.89)],[(7.26),(0.89)],[(7.83),(0.89)],
           [(8.28),(0.89)],[(8.62),(0.88)],[(9.14),(0.85)],[(9.45),(0.84)],[(9.76),(0.82)], [(10.0),(0.80)]] """

""" spline = make_interp_spline(x, y, k=3)  # k=3 for cubic spline
x_smooth = np.linspace(x.min(), x.max(), 500)  # Increase number of points for smoothness
y_smooth = spline(x_smooth) """

import numpy as np
import matplotlib.pyplot as plt

def curve_function1(x):
    return -0.001*x**3 + 0.0042*x**2 + 0.11*x

def curve_function2(x):
    return 0.001*x**3 - 0.0042*x**2 - 0.11*x + 1

""" def curve_function1(x):
    return 0.1*x

def curve_function2(x):
    return -0.3*x + 1
 """
# Generate 500 points for x between 0 and 10
x_values1 = np.linspace(0, 10, 800)
y_values1 = curve_function1(x_values1)

x_values2 = np.linspace(0, 10, 800)
y_values2 = curve_function2(x_values2)
#plt.figure(figsize=(15,7))
# Plot the curve
fig, ax = plt.subplots(figsize=(15,7))
ax.set_xlim(0, 10.0)
ax.set_ylim(0.0, 1.0)
ax.plot(x_values1, y_values1, color='blue')
ax.plot(x_values2, y_values2, color='blue')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.spines['bottom'].set_visible(False)

ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
ax.grid(True)
plt.savefig('NoscreenshotCurve2.tif', dpi=300, format='tiff', bbox_inches='tight', pad_inches=0)
plt.show()