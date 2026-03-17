import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

from functions import force_hermitian_symmetry
p = True
N = 100 
x, dx = np.linspace(0, 2*np.pi, N, endpoint=False, retstep=True)
f_x = 1 + 2*np.sin(x) + 10*np.sin(5*x) + 3*np.cos(5*x)
F_k = fft(f_x)
print("Před úpravou pro Hermitian symetrii:")
for i in range(len(F_k)):
    if abs(F_k[i]) > 1e-10:  # Tiskneme pouze nenulové koeficienty
        print(f"F_k[{i}] = {F_k[i]}")

# 1. Vygeneruješ si náhodnou první polovinu (indexy 1 až 49)


# 2. Tu druhou polovinu (indexy 51 až 99) vytvoříš jako zrcadlo
# np.flip pole otočí (aby 1 odpovídalo 99)
# np.conj otočí znaménka u 'j'
print("\nPo úpravě pro Hermitian symetrii:")
F_k = force_hermitian_symmetry(F_k)
for i in range(len(F_k)):
    if abs(F_k[i]) > 1e-10:  # Tiskneme pouze nenulové koeficienty
        print(f"F_k[{i}] = {F_k[i]}")