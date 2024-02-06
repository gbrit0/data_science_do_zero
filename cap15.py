from cap04 import dot, Vector

def predict(x: Vector, beta: Vector) -> float:
   """Pressupõe que o primeiro elemento de x é 1"""
   return dot(x, beta)

from typing import List

def error(x: Vector, y: float, beta: Vector) -> float:
   return predict(x, beta) - y

def squared_error(x: Vector, y: float, beta: Vector) -> float:
   return error(x, y, beta) ** 2

x = [1, 2, 3]
y = 30
beta = [4, 4, 4] # logo, a previsão = 4 + 8 + 12 = 24

assert error(x, y, beta) == -6
assert squared_error(x, y, beta) == 36

def sqerror_gradient(x: Vector, y: float, beta: Vector) -> Vector:
   err = error(x, y, beta)
   return [2 * err * x_i for x_i in x]

assert sqerror_gradient(x, y, beta) == [-12, -24, -36]

import random
import tqdm
from cap04 import vector_mean
from cap08 import gradient_step

inputs: List[List[float]] = [[1.,49,4,0],[1,41,9,0],[1,40,8,0],[1,25,6,0],[1,21,1,0],[1,21,0,0],[1,19,3,0],[1,19,0,0],[1,18,9,0],[1,18,8,0],[1,16,4,0],[1,15,3,0],[1,15,0,0],[1,15,2,0],[1,15,7,0],[1,14,0,0],[1,14,1,0],[1,13,1,0],[1,13,7,0],[1,13,4,0],[1,13,2,0],[1,12,5,0],[1,12,0,0],[1,11,9,0],[1,10,9,0],[1,10,1,0],[1,10,1,0],[1,10,7,0],[1,10,9,0],[1,10,1,0],[1,10,6,0],[1,10,6,0],[1,10,8,0],[1,10,10,0],[1,10,6,0],[1,10,0,0],[1,10,5,0],[1,10,3,0],[1,10,4,0],[1,9,9,0],[1,9,9,0],[1,9,0,0],[1,9,0,0],[1,9,6,0],[1,9,10,0],[1,9,8,0],[1,9,5,0],[1,9,2,0],[1,9,9,0],[1,9,10,0],[1,9,7,0],[1,9,2,0],[1,9,0,0],[1,9,4,0],[1,9,6,0],[1,9,4,0],[1,9,7,0],[1,8,3,0],[1,8,2,0],[1,8,4,0],[1,8,9,0],[1,8,2,0],[1,8,3,0],[1,8,5,0],[1,8,8,0],[1,8,0,0],[1,8,9,0],[1,8,10,0],[1,8,5,0],[1,8,5,0],[1,7,5,0],[1,7,5,0],[1,7,0,0],[1,7,2,0],[1,7,8,0],[1,7,10,0],[1,7,5,0],[1,7,3,0],[1,7,3,0],[1,7,6,0],[1,7,7,0],[1,7,7,0],[1,7,9,0],[1,7,3,0],[1,7,8,0],[1,6,4,0],[1,6,6,0],[1,6,4,0],[1,6,9,0],[1,6,0,0],[1,6,1,0],[1,6,4,0],[1,6,1,0],[1,6,0,0],[1,6,7,0],[1,6,0,0],[1,6,8,0],[1,6,4,0],[1,6,2,1],[1,6,1,1],[1,6,3,1],[1,6,6,1],[1,6,4,1],[1,6,4,1],[1,6,1,1],[1,6,3,1],[1,6,4,1],[1,5,1,1],[1,5,9,1],[1,5,4,1],[1,5,6,1],[1,5,4,1],[1,5,4,1],[1,5,10,1],[1,5,5,1],[1,5,2,1],[1,5,4,1],[1,5,4,1],[1,5,9,1],[1,5,3,1],[1,5,10,1],[1,5,2,1],[1,5,2,1],[1,5,9,1],[1,4,8,1],[1,4,6,1],[1,4,0,1],[1,4,10,1],[1,4,5,1],[1,4,10,1],[1,4,9,1],[1,4,1,1],[1,4,4,1],[1,4,4,1],[1,4,0,1],[1,4,3,1],[1,4,1,1],[1,4,3,1],[1,4,2,1],[1,4,4,1],[1,4,4,1],[1,4,8,1],[1,4,2,1],[1,4,4,1],[1,3,2,1],[1,3,6,1],[1,3,4,1],[1,3,7,1],[1,3,4,1],[1,3,1,1],[1,3,10,1],[1,3,3,1],[1,3,4,1],[1,3,7,1],[1,3,5,1],[1,3,6,1],[1,3,1,1],[1,3,6,1],[1,3,10,1],[1,3,2,1],[1,3,4,1],[1,3,2,1],[1,3,1,1],[1,3,5,1],[1,2,4,1],[1,2,2,1],[1,2,8,1],[1,2,3,1],[1,2,1,1],[1,2,9,1],[1,2,10,1],[1,2,9,1],[1,2,4,1],[1,2,5,1],[1,2,0,1],[1,2,9,1],[1,2,9,1],[1,2,0,1],[1,2,1,1],[1,2,1,1],[1,2,4,1],[1,1,0,1],[1,1,2,1],[1,1,2,1],[1,1,5,1],[1,1,3,1],[1,1,10,1],[1,1,6,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,4,1],[1,1,9,1],[1,1,9,1],[1,1,4,1],[1,1,2,1],[1,1,9,1],[1,1,0,1],[1,1,8,1],[1,1,6,1],[1,1,1,1],[1,1,1,1],[1,1,5,1]]

def least_squares_fit(xs: List[Vector],
                      ys: List[float],
                      learning_rate: float = 0.001,
                      num_steps: int = 1000,
                      batch_size: int = 1) -> Vector:
   """
   Encontre o beta que  minimiza a soma dos erros quadráticos
   pressupondo que o modelo y = dot(x, beta).
   """
   # Comece com uma estimativa aleatória
   guess = [random.random() for _ in xs[0]]

   for _ in tqdm.trange(num_steps, desc="least squares fit"):
      for start in range(0, len(xs), batch_size):
         batch_xs = xs[start:start+batch_size]
         batch_ys = ys[start:start+batch_size]

         gradient = vector_mean([sqerror_gradient(x, y, guess)
                                 for x, y in zip(batch_xs, batch_ys)])
         guess = gradient_step(guess, gradient, -learning_rate)

   return guess

from cap05 import daily_minutes_good
from cap08 import gradient_step

random.seed(0)
learning_rate = 0.001

beta = least_squares_fit(inputs, daily_minutes_good, learning_rate, 5000, 25)
assert 30.50 < beta[0] < 30.70         # constante
assert 0.96 < beta[1] < 1.00           # número de amigos
assert -1.89 < beta[2] < -1.85         # hpras trabalhadas por dia
assert 0.91 < beta[3] < 0.94           # tem PhD

from cap14 import total_sum_of_squares

def multiple_r_squared(xs: List[Vector], ys: Vector, beta: Vector) -> float:
   sum_of_squared_erros = sum(error(x, y, beta) ** 2
                                    for x, y in zip(xs, ys))
   return 1.0 - sum_of_squared_erros / total_sum_of_squares(ys)

assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta) < 0.68

from typing import TypeVar, Callable

X = TypeVar('X')        # tipo genéricom para os dados
Stat = TypeVar('Stat')  # tipo genérico para "estatísticas"

def bootstrap_sample(data: List[X]) -> List[X]:
   """Tira amostras aleatoriamente de elementos len(data) com substituição"""
   return[random.choice(data) for _ in data]

def bootstrap_statistics(data: List[X],
                         stats_fn: Callable[[List[X]], Stat],
                         num_samples: int) -> List[Stat]:
   """avlia stats_fn em amostras de bootstrap num_samples dos dados"""
   return [stats_fn(bootstrap_sample(data)) for _ in range(num_samples)]

# 101 pontos muito próximos de 100
close_to_100 = [99.5 + random.random() for _ in range(101)]

#101 pontos, 50 próximos de 0, 50 próximos de 200
far_from_100 = ([99.5 + random.random()] + 
                [random.random() for _ in range(50)] + 
                [200 + random.random() for _ in range(50)])

from cap05 import median, standard_deviation

medians_close = bootstrap_statistics(close_to_100, median, 100)

medians_far = bootstrap_statistics(far_from_100, median, 100)


assert standard_deviation(medians_close) < 1
assert standard_deviation(medians_far) > 90

from typing import Tuple

import datetime

def estimate_sample_beta(pairs: List[Tuple[Vector, float]]):
   x_sample = [x for x, _ in pairs]
   y_sample = [y for _, y in pairs]
   beta = least_squares_fit(x_sample, y_sample, learning_rate, 5000, 25)
   print("bootsrap sample", beta)
   return beta

# Isso demora alguns minutos
bootstrap_betas = bootstrap_statistics(list(zip(inputs, daily_minutes_good)),
                                        estimate_sample_beta,
                                        100)

bootstrap_standard_errors = [
   standard_deviation([beta[i] for beta in bootstrap_betas])
   for i in range(4)]

print(bootstrap_standard_errors)

from cap06 import normal_cdf

def p_value(beta_hat_j: float, sigma_hat_j: float) -> float:
   if beta_hat_j > 0:
      # se o coeficiente for positivo, temos que computar o dobro
      # da probabilidade de ver um valor *maior* do que esse
      return 2 * (1 - normal_cdf(beta_hat_j / sigma_hat_j))
   else:
      # caso contrário, o dobro da probabilidade de ver um valor *menor*
      return 2* normal_cdf(beta_hat_j / sigma_hat_j)
   
assert p_value(30.58, 1.27)   < 0.001  # constant term
assert p_value(0.972, 0.103)  < 0.001  # num_friends
assert p_value(-1.865, 0.155) < 0.001  # work_hours
assert p_value(0.923, 1.249)  > 0.4    # phd

# REGULARIZAÇÃO

# alpha é um hiperparâmetro que controla o rigor da penalidade.
# Às vezes, ele é chamado de lambda, mas esse símbolo já tem significado no Python
def ridge_penalty(beta: Vector, alpha: float) -> float:
   return alpha * dot(beta[1:], beta[1:])

def squared_error_ridge(x: Vector,
                        y: float,
                        beta: Vector,
                        alpha: float) -> float:
   """Estime o erro mais a penalidade de cume em beta"""
   return error(x, y, beta) ** 2 + ridge_penalty(beta, alpha)

# conectando isso ao gradiente descendente
from cap04 import add

def ridge_penalty_gradient(beta: Vector, alpha: float) -> Vector:
   """Gradiente apenas da penalidade de cume"""
   return [0.] + [2 * alpha * beta_j for beta_j in beta[1:]]

def sqerror_ridge_gradient(x: Vector,
                           y: float,
                           beta: Vector,
                           alpha: float) -> Vector:
   """
   O gradiente correspondente ao termo de erro quadrático i,
   incluindo a penalidade de cume y
   """
   return add(sqerror_gradient(x, y, beta),
              ridge_penalty_gradient(beta, alpha))

def least_squares_fit_ridge(xs: List[Vector],
                            ys: List[float],
                            alpha: float,
                            learning_rate: float,
                            num_steps: int,
                            batch_size: int = 1) -> Vector:
    # Start guess with mean
    guess = [random.random() for _ in xs[0]]

    for i in range(num_steps):
        for start in range(0, len(xs), batch_size):
            batch_xs = xs[start:start+batch_size]
            batch_ys = ys[start:start+batch_size]

            gradient = vector_mean([sqerror_ridge_gradient(x, y, guess, alpha)
                                    for x, y in zip(batch_xs, batch_ys)])
            guess = gradient_step(guess, gradient, -learning_rate)

    return guess

beta_0 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.0, #alpha
                                 learning_rate, 5000, 25)
# [30.514484888280567, 0.9748440502839583, -1.8506798946882428, 0.9143200926682622]

assert 5 < dot(beta_0[1:], beta_0[1:]) < 6
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0) < 0.69

beta_0_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 0.1,  # alpha
                                 learning_rate, 5000, 25)
# [30.8, 0.95, -1.83, 0.54]
assert 4 < dot(beta_0_1[1:], beta_0_1[1:]) < 5
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_0_1) < 0.69


beta_1 = least_squares_fit_ridge(inputs, daily_minutes_good, 1,  # alpha
                                 learning_rate, 5000, 25)
# [30.6, 0.90, -1.68, 0.10]
assert 3 < dot(beta_1[1:], beta_1[1:]) < 4
assert 0.67 < multiple_r_squared(inputs, daily_minutes_good, beta_1) < 0.69

beta_10 = least_squares_fit_ridge(inputs, daily_minutes_good,10,  # alpha
                                 learning_rate, 5000, 25)
# [28.3, 0.67, -0.90, -0.01]
assert 1 < dot(beta_10[1:], beta_10[1:]) < 2
assert 0.5 < multiple_r_squared(inputs, daily_minutes_good, beta_10) < 0.6