# GRADIENTE DESCENDENTE
from cap04 import Vector, dot

def sum_of_squares(v: Vector) -> float:
   """Computa a soma de elementos quadrados em v"""
   return dot(v, v)
# Muitas vezes, teremos que maximizar ou minimizar essas funções, ou seja, determinar a
# entrada v que produz o maior (ou menor) valor possível.

#Estimando o gradiente:
# gradiente: o vetor das derivadas parciais
from typing import Callable
def difference_quotient(f: Callable[[float], float],
                        x: float,
                        h: float) -> float:
   return(f(x + h) - f(x)) / h

def square(x: float) -> float:
   return x * x

def derivative(x: float) -> float:
   return 2 * x

xs = range(-10, 11)
actuals = [derivative(x) for x in xs]
estimates = [difference_quotient(square, x, h=0.001) for x in xs]

# plote para indicar que eles são essencialmente os mesmos
import matplotlib.pyplot as plt
plt.title("Derivadas reais X Estimaivas")
plt.plot(xs, actuals, 'rx', label='Real')                # vermelho x
plt.plot(xs, estimates, 'b+', label='Estimativa')        # azul +
plt.legend(loc=9)
plt.show()

def partial_difference_quotient(f: Callable[[Vector], float],
                                v: Vector,
                                i: int,
                                h: float) -> float:
   """Retorna o quociente parcial das diferenças i de f em v"""
   w = [v_j + (h if j == i else 0)                       # adicione h somente ao elemento i de v
        for j, v_j in enumerate(v)]
   return (f(w) - f(v)) / h

def estimate_gradient(f: Callable[[Vector], float],
                      v: Vector,
                      h: float = 0.0001):
   return [partial_difference_quotient(f, v, i, h)
           for i in range(len(v))]

# Usando o Gradiente
import random
from cap04 import distance, add, scalar_multiply

def gradient_step(v: Vector, gradient: Vector, step_size: float) -> Vector:
   """Move 'step-size' na direção de 'gradient' a partir de 'v'"""
   assert len(v) == len(gradient)
   step = scalar_multiply(step_size, gradient)
   return add(v, step)

def sum_of_squares_gradient(v: Vector) -> Vector:
   return [2 * v_i for v_i in v]

#selecione um ponto de partida aleatório:
v = [random.uniform(-10, 10) for i in range(3)]

for epoch in range(1000):
   grad = sum_of_squares_gradient(v)                        # Compute o gradiente em v
   v = gradient_step(v, grad, -0.01)                        # dê um passo negativo para o gradiente
   print(epoch, v)

assert distance(v, [0, 0, 0]) < 0.001                       # v deve ser próximo de 0


# USANDO O GRADIENTE DESCENDENTE PARA AJUSTAR MODELOS
# x vai de -50 a 49, y é sempre 20 * x + 5
inputs = [(x, 20 * + 5) for x in range(-50, 50)]

def linear_gradient(x: float, y: float, theta: Vector) -> Vector:
   slope, intercept = theta
   predicted = slope * x + intercept                        # A previsão do modelo.
   error = (predicted - y)                                  # o erro é (previsto - real)
   squared_error = error ** 2                               # Vamos minimizar o erro quadrático
   grad = [2 * error * x, 2 * error]                        # usando seu gradiente
   return grad