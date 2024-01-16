# HIPÓTESE E INFERÊNCIA

#Teste de Hipóteses: Lançando uma moeda
from typing import Tuple
import math

def normal_approximation_to_binomial(n: int, p: float) -> Tuple[float, float]:
   """Retorna mu e sigma correspondentes Binomial(n, p)"""
   mu = p * n
   sigma = math.sqrt(p * (1 - p) * n)
   return mu, sigma

from cap06 import normal_cdf

# o normal_cdf é a probabilidade de a variávels estar abaixo de um limite
normal_probability_below = normal_cdf

# Está acima do limite se não está abaixo do limite
def normal_probability_above(lo: float,
                             mu: float = 0,
                             sigma: float = 1) -> float:
   """A probabilidade de que um N(mu, sigma) seja maior do que lo"""
   return 1 - normal_cdf(lo, mu, sigma)

# Está entre se é menor do que hi, mas não menor do que lo
def normal_probability_between(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
   """A probabilidade de que um N(mu, sigma) esteja entre lo e hi"""
   return normal_cdf(hi, mu, sigma) - normal_cdf(lo, mu, sigma)

# Está fora se não está entre
def normal_probability_outside(lo: float,
                               hi: float,
                               mu: float = 0,
                               sigma: float = 1) -> float:
   """A probabilidade de quem N(mu, sigma) não esteja entre lo e hi"""
   return 1 - normal_probability_between(lo, hi, mu, sigma)

from cap06 import inverse_normal_cdf

def normal_upper_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
   """Retorna o z para o qual P(Z <= z) = probabilidade"""
   return inverse_normal_cdf(probability, mu, sigma)

def normal_lower_bound(probability: float,
                       mu: float = 0,
                       sigma: float = 1) -> float:
   """Retorna o z para o qual P(Z >= z) = probabilidade"""
   return inverse_normal_cdf(1 - probability, mu, sigma)

def normal_two_sided_bounds(probability: float,
                            mu: float = 0,
                            sigma: float = 1) -> Tuple[float, float]:
   """
   Retorna os limites simétricos (relativos à média) que contêm a probabilidade especificada
   """
   tail_probability = (1 - probability) / 2

   # O limite superior deve estar abaixo de tail_probability
   upper_bound = normal_lower_bound(tail_probability, mu, sigma)

   # O li,ite inferior deve estar acima de tail_probability
   lower_bound = normal_upper_bound(tail_probability, mu, sigma)

   return lower_bound, upper_bound

mu_0, sigma_0 = normal_approximation_to_binomial(1000, 0.5)

# Significância - disposição para obter um erro tipo 1 ("Falso Positivo") e recusar a H_0 mesmo se ela for verdadeira
# Teste para recusar H_0 se X estiver fora dos limites:
# (469, 531)
lower_bound, upper_bound = normal_two_sided_bounds(0.95, mu_0, sigma_0)
print(lower_bound)
print(upper_bound)

# Potência do teste - probabilidade de não ocorrer um erro tipo 2 (quando aceitamos uma hipótese falsa)

# Limites de 95% baseados na premissa de que p é o 0.5
lo, hi = normal_two_sided_bounds(0.95, mu_0, sigma_0)

# mu e sigma reais baseados em p = 0.55
mu_1, sigma_1 = normal_approximation_to_binomial(1000, 0.55)

# um erro tipo 2 ocorre quando falhamos em rejeitar a hipótese nula,
# o que ocorre quando X ainda está no intervalo original
type_2_probability = normal_probability_between(lo, hi, mu_1, sigma_1)
power = 1 - type_2_probability               # 0.887

hi = normal_upper_bound(0.95, mu_0, sigma_0)
type_2_probability = normal_probability_below(hi, mu_1, sigma_1)
power = 1 - type_2_probability
print(power)

# p-Valor

def two_sided_p_value(x: float, mu: float = 0, sigma: float = 1) -> float:
   """
   Qual é a probabilidade de observar um valor pelo menos tão extremo
   quanto x (em qualquer direção) se os valores vêm de um N(mu, sigma)?
   """
   if x >= mu:
      # x é maior do que a média, então a coroa é qualquer valor maior do que x
      return 2 * normal_probability_above(x, mu, sigma)
   else:
      # x é menor do que a média, então a coroa é qualquer valor menor do que x
      return 2 * normal_probability_below(x, mu, sigma)
   
print(two_sided_p_value(529.5, mu_0, sigma_0))              #0.062

import random
extreme_value_count = 0
for _ in range(1000):
   num_heads = sum(1 if random.random() < 0.5 else 0              # Conte o nº de caras
                   for _ in range(1000))                          # em mil lançamentos,
   if num_heads >= 530 or num_heads <= 470:                       # e conte as vezes em que
      extreme_value_count += 1                                    # o nº é 'extremo'

# o p-value era 0.062 => ~62 valores extremos em 1000
assert 59 < extreme_value_count < 65, f"{extreme_value_count}"

print(two_sided_p_value(531.5, mu_0, sigma_0))                    # 0.046345287837786575