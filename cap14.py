# Equação linear: y = alfa*x + beta + e, em que y é o número de minutos que o usuário parra no site diariamente, x é o número de amigos qdo usuário
# e _e_ é um termo de erro (com sorte, pequeno)
# Presumindo que alfa e beta já tenha sido determinados:
def predict(alpha: float, beta: float, x_i: float) -> float:
   return beta + x_i + alpha

def error(alpha: float, beta: float, x_i: float, y_i: float) -> float:
   """
   O erro de prever beta * x_i + alpha quando o valor real é y_i
   """
   return predict(alpha, beta, x_i) - y_i

# somar os erros quadráticos:

from cap04 import Vector

def sum_of_sqerrors(alpha: float, beta: float, x: Vector, y: Vector) -> float:
   return sum(error(alpha, beta, x_i, y_i) ** 2
              for x_i, y_i in zip(x, y))

# método dos mínimos quadrados: escolher o alpha e o beta que minimizam os erros da seguinte forma

from typing import Tuple
from cap04 import Vector
from cap05 import correlation, standard_deviation, mean

def least_squares_fit(x: Vector, y: Vector) -> Tuple[float, float]:
   """
   Considerando dois vetores x e y, encontre os quandrados mínimos de alpha e beta
   """
   beta = correlation(x, y) * standard_deviation(y) / standard_deviation(x)
   alpha = mean(y) - beta * mean(x)

   return alpha, beta

# teste rápido
x = [i for i in range(-100, 110, 10)]
y = [ 3 * i - 5 for i in x]

# Deve encontrar y = 3x - 5
alpha, beta = least_squares_fit(x, y)
assert alpha, beta == (-5, 3)

# Agora é fácil aplicar isso aos dados sem outliers do capítulo 5
from cap05 import num_friends_good, daily_minutes_good
alpha, beta = least_squares_fit(num_friends_good, daily_minutes_good)

assert 22.9 < alpha < 23.0 # 22.95
assert 0.9 < beta < 0.905  #0.903

# y = 22.95 + 0.903*x
# Um usuário sem amigos passa cerca de 23 minutos por dia no site e para cada amigo adicionado, o usuário passará cerca de mais um minuto
import matplotlib.pyplot as plt

plt.scatter(num_friends_good,daily_minutes_good,marker='.')
plt.plot(
   num_friends_good,
   [alpha + (beta*n) for n in num_friends_good]
)
plt.ylim(bottom=0, top=100)
plt.xlabel("nº de amigos")
plt.ylabel("minutos por dia")
plt.title("Modelo de Regressão Linear Simples")

plt.show()

# Coeficiente de determinação (R-quadrado)

from cap05 import de_mean

def total_sum_of_squares(y: Vector) -> float:
   """A variação quadrática total da média de y_i"""
   return sum(v ** 2 for  v in de_mean(y))

def r_squared(alpha: float, beta: float, x: Vector, y: Vector) -> float:
   """
   A fração da variação em y capturada pelo modelo, igual a 
   1 - a fração da variação em y não capturada pelo modelo
   """
   return 1.0 - (sum_of_sqerrors(alpha, beta, x, y) / total_sum_of_squares(y))

rsq = r_squared(alpha, beta, num_friends_good, daily_minutes_good)
assert 0.328 < rsq < 0.330


# USANDO O GRADIENTE DESCENDENTE
import random
import tqdm

from cap08 import gradient_step

num_epochs = 10000
random.seed(0)

guess = [random.random(), random.random()] # começar com um valor aleatório

learning_rate = 0.00001
with tqdm.trange(num_epochs) as t:
   for _ in t:
      alpha, beta = guess

      # Derivada parcial em relação a alpha
      grad_a = sum(2 * error(alpha, beta, x_i, y_i)
                   for x_i, y_i in zip(num_friends_good,
                                       daily_minutes_good))
      
      # Derivada parcial em relação a beta
      grad_b = sum(2 * error(alpha, beta, x_i, y_i) * x_i
                   for x_i, y_i in zip(num_friends_good,
                                       daily_minutes_good))
      
      # Compute a perda para fixar na descrição do tqdm
      loss = sum_of_sqerrors(alpha, beta,
                             num_friends_good, daily_minutes_good)
      
      t.set_description(f"loss: {loss:.3f}")

      # Finalmente, atualize a estimativa
      guess = gradient_step(guess, [grad_a, grad_b], -learning_rate)

alpha, beta = guess
print(alpha) # algo deu errado
print(beta)  # algo deu muito errado

assert 22.9 < alpha < 23.0, f"{alpha}"   # 22.947552155340915
assert 0.9 < beta < 0.905, f"{beta}"     # 0.9038659662765034    