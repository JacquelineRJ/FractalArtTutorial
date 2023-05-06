# Original Tutorial: https://towardsdatascience.com/create-stunning-fractal-art-with-python-a-tutorial-for-beginners-c83817fcb64b

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass

def sequence(c, n=7) -> list:
    z_list = list()
    
    z = 0
    for _ in range(n):
        z = z ** 2 + c
        z_list.append(z)
    
    return z_list

def is_stable(c, n_iterations=20):
    z = 0
    
    for _ in range(n_iterations):
        z = z ** 2 + c
        
        if abs(z) > 2:
            return False
    return True


def candidate_values(xmin, xmax, ymin, ymax, pixel_density):
    # Generate a 2D grid of real and imaginary values
    real = np.linspace(xmin, xmax, num=int((xmax-xmin) * pixel_density))
    imag = np.linspace(ymin, ymax, num=int((ymax-ymin) * pixel_density))
    
    # Cross each row of `xx` with each column of `yy` to create a grid of values
    xx, yy = np.meshgrid(real, imag)
    
    # Combine the real and imaginary parts into complex numbers
    matrix = xx + 1j * yy
    
    return matrix


@dataclass
class Mandelbrot:
    max_iterations: int
    
    def escape_count(self, c: complex) -> int:
        z = 0
        for i in range(self.max_iterations):
            z = z ** 2 + c
            if abs(z) > 2:
                return i
        return self.max_iterations
    
    def stability(self, c: complex) -> float:
        return self.escape_count(c) / self.max_iterations
    
    def is_stable(self, c: complex) -> bool:
        # Return True only when stability is 1
        return self.stability(c) == 1

    @staticmethod
    def candidate_values(xmin, xmax, ymin, ymax, pixel_density):
        real = np.linspace(xmin, xmax, num=int((xmax-xmin) * pixel_density))
        imag = np.linspace(ymin, ymax, num=int((ymax-ymin) * pixel_density))

        xx, yy = np.meshgrid(real, imag)
        matrix = xx + 1j * yy

        return matrix
    
    
    def plot(self, xmin, xmax, ymin, ymax, pixel_density=64, cmap="gist_heat"):
        c = Mandelbrot.candidate_values(xmin, xmax, ymin, ymax, pixel_density)
        
        # Apply `stability` over all elements of `c`
        c = np.vectorize(self.stability)(c)
        
        plt.imshow(c, cmap=cmap, extent=[0, 1, 0, 1])
        plt.gca().set_aspect("equal")
        plt.axis('off')
        plt.tight_layout()
        plt.show()


df = pd.DataFrame()
df['element'] = [f"z_{i}" for i in range(7)]

# Random numbers
cs = [0, 1, -1, 2, 0.25, -.1]

for c in cs:
    df[f"c={c}"] = sequence(c)

df2 = pd.DataFrame()
df2['c'] = [c for c in cs]
df2['Stable'] = [is_stable(c) for c in cs]

print(df)
print(df2)
print("\nOnly stable variables can create Mandelbrot's set!")
print("--------------------------------------------------")

# create matrix of candidate values we can iterate madelbrot over
c = candidate_values(-2, 0.7, -1.2, 1.2, pixel_density=1024)
mandelbrot = Mandelbrot(max_iterations=30)
mandelbrot.plot(
    xmin=-2, xmax=0.5, 
    ymin=-1.5, ymax=1.5, 
    pixel_density=1024,
)


