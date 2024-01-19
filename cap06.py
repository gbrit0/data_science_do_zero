#PROBABILIDADE
#Probabilidade Condicional
import enum, random

# Um Enum é um conjunto tipado de valores enumerados que deixa o código mais descritivo e legível
class Kid(enum.Enum):
   BOY = 0
   GIRL = 1

def random_kid() -> Kid:
   return random.choice([Kid.BOY, Kid.GIRL])             # Escolha aleatória entre garoto e garota

both_girls = 0
older_girl = 0
either_girl = 0

random.seed(0)

for _ in range(10000):
   younger = random_kid()

   older = random_kid()

   if older == Kid.GIRL:
      older_girl += 1
   
   if older == Kid.GIRL and younger == Kid.GIRL:
      both_girls += 1

   if older == Kid.GIRL or younger == Kid.GIRL:
      either_girl += 1

print("P(both | older):", both_girls / older_girl)             #0.5007 ~ 1/2
print("P(both | either):", both_girls / either_girl)           #0.3311 ~ 1/3

# O TEOREMA DE BAYES

# VARIÁVEIS ALEATÓRIAS

# DISTRIBUIÇÕES CONTÍNUAS

def uniform_pdf(x: float) -> float:             #PDF - função densidade de probabilidade
   return 1 if 0 <= x < 1 else 0

def uniform_cdf(x: float) -> float:             #CDF - função de distribuição cumulativa
   """Retorna a probabilidade de uma variável aleatória uniforme ser <= x"""
   if x < 0: return 0               # a aleatória uniforme nunca é menor que 0
   elif x < 1: return x             # ex.: P(X <= 0.4) = 0.4
   else: return 1                   # a aleatória uniforme sempre é menor do que 1

y = [uniform_cdf(x) for x in range(-1,3)]
x = [x for x in range(-1,3)]

import matplotlib.pyplot as plt

plt.plot(x,y,linestyle='solid')
plt.title("CDF uniforme")
plt.axis([-1,2,-0.5,1.5])
# plt.show()

# DISTRIBUIÇÃO NORMAL
import math

SQRT_TWO_PI =  math.sqrt(2 * math.pi)

def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
   return (math.exp(-(x-mu) ** 2 / (2 * (sigma ** 2))) / (SQRT_TWO_PI * sigma))

xs = [x / 10.0 for x in range(-50, 50)]
plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs], '-', label='mu=0,sigma=1')
plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs], '--', label='mu=0,sigma=2')
plt.plot(xs,[normal_pdf(x,sigma=0.5) for x in xs], ':', label='mu=0,sigma=0.5')
plt.plot(xs,[normal_pdf(x,mu=-1) for x in xs], '-.', label='mu=-1,sigma=1')
plt.legend()
plt.title("Various Normal pdfs")
# plt.show()

def normal_cdf(x: float, mu: float = 0, sigma: float = 1) -> float:
   return (1 + math.erf((x - mu) / math.sqrt(2) / sigma)) / 2

xs = [ x / 10.0 for x in range(-50,50)]
plt.plot(xs,[normal_cdf(x,sigma=1) for x in xs],'-',label='mu=o,sigma=1')
plt.plot(xs,[normal_cdf(x,sigma=2) for x in xs], '--', label='mu=0,sigma=2')
plt.plot(xs,[normal_cdf(x,sigma=0.5) for x in xs], ':', label='mu=0,sigma=0.5')
plt.plot(xs,[normal_cdf(x,mu=-1) for x in xs], '-.', label='mu=-1,sigma=1')
plt.legend(loc=4)          # no canto inferior direito
plt.title("Varias CDF's Normais")
# plt.show()

# Ocasionalmente é preciso inverter a normal_cdf para obter o valor correspondente à probabilidade especificada
# Não existe uma forma simples de calcular mas como a normal_cdf é contínua e cresce estritamente, podemos usar uma busca binária
def inverse_normal_cdf(p: float,
                       mu: float = 0,
                       sigma: float = 1,
                       tolerance: float = 0.00001) -> float:
   """Encontre o inverso aproximado usando a pesquisa binária"""

   # se não for padrão, compute o padrão e redimensione
   if mu != 0 or sigma != 1:
      return mu + sigma * inverse_normal_cdf(p, tolerance=tolerance)
   low_z = -10.0                             # normal_cdf(-10) é (muito próxima de) 0
   hi_z = 10.0                               # normal_cdf(10) é (muito próxima de) 1
   while hi_z - low_z > tolerance:
      mid_z = (low_z + hi_z) / 2             #Considere o ponto médio
      mid_p = normal_cdf(mid_z)              # e o valor da CDF
      if mid_p < p:
         low_z = mid_z                       # O ponto médio é muito baixo, procure um maior
      else:
         hi_z = mid_z                        # O ponto médio é muito alto, procure um menor

   return mid_z

#TEOREMA DO LIMITE CENTRAL

def bernoulli_trial(p: float) -> int:
   """Retorna 1 com probabilidade p e 0 com o probabilidade 1-p"""
   return 1 if random.random() < p else 0

def binomial(n: int, p: float) -> int:
   """Retorna a soma de n trials bernoulli(p)"""
   return sum(bernoulli_trial(p) for _ in range(n))

from collections import Counter

def binomial_histogram(p: float, n: int, num_points: int) -> None:
   """Seleciona pontos de um Binomial(n, p) e plota seu histograma"""
   data = [binomial(n, p) for _ in range(num_points)]

   # use um gráfico de barras para indicar as amostras de binomiais
   histogram = Counter(data)
   plt.bar([x - 0.4 for x in histogram.keys()],
           [v / num_points for v in histogram.values()],
           0.8,
           color='0.75')
   
   mu = p * n
   sigma = math.sqrt(n * p * (1 - p))

   # use um gráfico de linhas para indicar a aproximação normal
   xs = range(min(data), max(data) + 1)
   ys = [normal_cdf(i + 0.5, mu, sigma) - normal_cdf(i - 0.5, mu, sigma)
         for i in xs]
   plt.plot(xs,ys)
   plt.title("Binomial Distribution vs. Normal Approximation")
   plt.show()

binomial_histogram(0.75, 100, 10000)