import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
line1, line2, = ax.plot(x, y1, 'b-', x, y2, 'g-')

for phase in np.linspace(0, 10*np.pi, 100):
    line1.set_ydata(np.sin(0.5 * x + phase))
    line2.set_ydata(np.cos(0.5 * x + phase))
    fig.canvas.draw()

plt.ioff()
plt.show()
