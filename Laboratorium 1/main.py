import math
import numpy as np
from numpy.linalg import LinAlgError


def cylinder_area(r:float,h:float):
    """Obliczenie pola powierzchni walca. 
    Szczegółowy opis w zadaniu 1.
    
    Parameters:
    r (float): promień podstawy walca 
    h (float): wysokosć walca
    
    Returns:
    float: pole powierzchni walca 
    """
    if r <= 0 or h <= 0:
        return float('NaN')
    else:
        return 2*math.pi*(r**2) + 2*math.pi*r*h


def fib(n:int):
    """Obliczenie pierwszych n wyrazów ciągu Fibonnaciego.
    Szczegółowy opis w zadaniu 3.
    
    Parameters:
    n (int): liczba określająca ilość wyrazów ciągu do obliczenia 
    
    Returns:
    np.ndarray: wektor n pierwszych wyrazów ciągu Fibonnaciego.
    """
    memory = [1, 1]
    if not isinstance(n, int) or n <= 0:
        return None
    elif n == 1:
        return [1]
    else:
        for i in range(2, n):
            memory.append(float(memory[i-1]+memory[i-2]))
    return np.array([memory])



def matrix_calculations(a:float):
    """Funkcja zwraca wartości obliczeń na macierzy stworzonej 
    na podstawie parametru a.  
    Szczegółowy opis w zadaniu 4.
    
    Parameters:
    a (float): wartość liczbowa 
    
    Returns:
    touple: krotka zawierająca wyniki obliczeń 
    (Minv, Mt, Mdet) - opis parametrów w zadaniu 4.
    """
    m = np.array([[a, 1, -a], [0, 1, 1], [-a, a, 1]])
    try:
        minv = np.linalg.inv(m)
    except LinAlgError:
        minv = float('NaN')
    mt = m.transpose()
    mdet = np.linalg.det(m)
    return minv, mt, mdet

def custom_matrix(m:int, n:int):
    """Funkcja zwraca macierz o wymiarze mxn zgodnie 
    z opisem zadania 7.  
    
    Parameters:
    m (int): ilość wierszy macierzy
    n (int): ilość kolumn macierzy  
    
    Returns:
    np.ndarray: macierz zgodna z opisem z zadania 7.
    """
    if isinstance(m, int) and isinstance(n, int) and n > 0 and m > 0: #n and m are int and > 0
        matrix = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                if i > j:
                    matrix[i,j] = i
                else:
                    matrix[i,j] = j
        return matrix
    else:
        return None