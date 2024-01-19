#  TRABALHANDO COM DADOS

# Explorando dados Unidimensionais

from typing import List, Dict
from collections import Counter
import math

import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
   """Coloque o ponto perto do próximo mínimo múltiplo de bucket_size"""
   return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
   """Coloca os pontos em buckets e conta o número de pontos em cada bucket"""
   return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
   histogram = make_histogram(points, bucket_size)
   plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
   plt.title(title)
   plt.show()

import random
from cap06 import inverse_normal_cdf

random.seed(0)

# uniforme entre -100 e 100
uniform = [200 * random.random() - 100 for _ in range(10000)]

# distribuição normal com média 0, desvio-padrão 57
normal = [57 * inverse_normal_cdf(random.random())
          for _ in range(10000)]

plot_histogram(uniform, 10, "Histograma Uniforme")  

plot_histogram(normal, 10, "Histograma Normal")

# as duas distibuições têm pontos max e min muito diferentes mas
# determinar isso não é suficiente pra explicar a diferença entre os histogramas

# Duas Dimensões
def random_normal() -> float:
   """Retorna um ponto aleatório de uma distribuição normal padrão"""
   return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [ -x + random_normal() / 2 for x in xs]

plot_histogram(ys1, 10, "ys1")
plot_histogram(ys2, 10, "ys2")

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Distribuições conjuntas muito diferentes")
plt.show()

# a diferença entre as distribuições conjuntas é bem diferente e essa diferença também 
# aparece quando analisamos as correlações

from cap05 import correlation

print(correlation(xs, ys1))      # 0.9010493686379609
print(correlation(xs, ys2))      # -0.8920981526880033
