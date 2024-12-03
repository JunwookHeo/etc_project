import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact
 
 
def interactive_plot(amplitude, frequency):
    x = np.linspace(0, 2 * np.pi, 1000)
    y = amplitude * np.sin(frequency * x)
    plt.figure(figsize=(8, 4))
    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Interactive Sine Wave')
    plt.grid(True)
    plt.show()
 
 
interact(interactive_plot, amplitude=(1, 5, 0.1), frequency=(1, 10, 0.1))