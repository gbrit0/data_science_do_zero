from typing import List
from collections import Counter

def raw_majority_vote(labels: List[str]) -> str:
   votes = Counter(labels)
   winner, _ = votes.most_common(1)[0]
   return winner

assert raw_majority_vote(['a','b','c','b']) == 'b'

def majority_vote(labels: List[str]) -> str:
   """Supõe que os rótulos estão classificados do mais próximo para o mais distante"""
   vote_counts = Counter(labels)
   winner, winner_count = vote_counts.most_common(1)[0]

   num_winners = len([count
                      for count in vote_counts.values()
                      if count == winner_count])
   
   if num_winners == 1:
      return winner                          # vencedor único, então retorne isso
   else:
      return majority_vote(labels[:-1])      # tente novamente sem o mais distante
   

# Empate, então primeiro analise 4, depois 'b'
assert majority_vote(['a','b','c','b','a']) == 'b'

# Com essa função é fácil criar um classificador:

from typing import NamedTuple
from cap04 import Vector, distance

# class LabeledPoint(NamedTuple):
#    point: Vector
#    label: str

class LabeledPoint(NamedTuple):
    point: List[float]
    label: str


def knn_classify(k: int,
                 labeled_points: List[LabeledPoint],
                 new_point: Vector) -> str:
   
   # Classifique os pontos rotulados do mais próximo para o mais distante
   by_distance = sorted(labeled_points,
                        key=lambda lp: distance(lp.point, new_point))
   
   # Encontre os rótulos dos k mais próximos
   k_nearest_labels = [lp.label for lp in by_distance[:k]]

   # e receba seus votos
   return majority_vote(k_nearest_labels)


# Exemplo: O Conjunto de Dados Iris

import requests

data = requests.get(
   "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
)

with open('iris.dat', 'w') as f:
   f.write(data.text)

import csv
from collections import defaultdict
from typing import List, Dict, Union, NamedTuple, Tuple

# def parse_iris_row(row: List[str]) -> LabeledPoint:
#    """
#    sepal_length, sepal_width, petal_length, petal_width, class
#    """
#    # Cópia do vetor original tirando a classe, que é o q queremos conseguir prever
#    measurements = [float(value) for value in row[:-1]]

#    # a classe é p. ex. "Iris-virginica"; queremos só "virginica"
#    label = row[-1].split("-")[-1]

#    return LabeledPoint(measurements, label)

# with open('iris.dat') as f:
#    reader = csv.reader(f)
#    iris_data = [parse_iris_row(row) for row in reader]


def parse_iris_row(row: List[str]) -> Union[LabeledPoint, None]:
    """
    sepal_length, sepal_width, petal_length, petal_width, class
    """
    # Verifica se a linha tem pelo menos 5 elementos
    if len(row) < 5:
        return None

    # Cópia do vetor original tirando a classe, que é o que queremos conseguir prever
    measurements = [float(value) for value in row[:-1]]

    # A classe é, por exemplo, "Iris-virginica"; queremos só "virginica"
    label = row[-1].split("-")[-1]

    return LabeledPoint(measurements, label)

# Restante do seu código...

with open('iris.dat') as f:
    reader = csv.reader(f)
    iris_data = [parse_iris_row(row) for row in reader if parse_iris_row(row) is not None]

# Também agruparemos apenas os pontos por espécie/rótulo para plotá-los
points_by_species: Dict[str, List[Vector]] = defaultdict(list)
for iris in iris_data:
   points_by_species[iris.label].append(iris.point)

from matplotlib import pyplot as plt
metrics = ['sepal length', 'sepal width', 'petal length', 'petal width']
pairs = [(i, j) for i in range(4) for j in range(4) if i < j]
marks = ['+', '.', 'x']

fig, ax = plt.subplots(2,3)

for row in range(2):
   for col in range(3):
      i, j = pairs[3 * row + col]
      ax[row][col].set_title(f"{metrics[i]} vs {metrics[j]}", fontsize=8)
      ax[row][col].set_xticks([])
      ax[row][col].set_yticks([])

      for mark, (species, points) in zip(marks, points_by_species.items()):
         xs = [point[i] for point in points]
         ys = [point[j] for point in points]
         ax[row][col].scatter(xs, ys, marker=mark, label=species)

ax[-1][-1].legend(loc='lower right', prop={'size': 6})
plt.show()

import random
from cap11 import split_data

random.seed(12)

iris_train, iris_test = split_data(iris_data, 0.7)
assert len(iris_train) == 0.7 * 150
assert len(iris_test) == 0.3 * 150

# conte quantas vezes identificamos (previsto, real)
confusion_matrix: Dict[Tuple[str, str], int] = defaultdict(int)
num_correct = 0

for iris in iris_test:
   predicted = knn_classify(5, iris_train, iris.point)
   actual = iris.label

   if predicted == actual:
      num_correct += 1

   confusion_matrix[(predicted, actual)] += 1

pct_corret = num_correct / len(iris_test)
print(pct_corret, confusion_matrix)


# A maldição da dimensionalidade

def random_point(dim: int) -> Vector:
   return [random.random() for _ in range(dim)]

def random_distances(dim: int, num_pairs: int) -> List[float]:
   return [distance(random_point(dim), random_point(dim))
           for _ in range(num_pairs)]

# Para cada dimensão de 1 a 100 vamos computar 10 mil distâncias
# e usá-las para computar a distância média entre os pontos e a distância mínima
# entre os pontos em cada dimensão

import tqdm
dimensions = range(1,101)

avg_distances = []
min_distances = []

random.seed(0)
for dim in tqdm.tqdm(dimensions, desc="Maldição da Dimensionalidade"):
   distances = random_distances(dim, 10_000)          # 10 mil pares aleatórios
   avg_distances.append(sum(distances) / 10_000)      # obtenha a média
   min_distances.append(min(distances))               # obtenha a mínima

xs = [dimension for dimension in dimensions]
plt.plot(xs,avg_distances,label='distância média')
plt.plot(xs,min_distances,label='distância mínima')
plt.legend(loc=0)
plt.xlabel("nº de dimensões")
# plt.xticks([])
plt.title("10000 Distâncias Aleatórias")

plt.show()

# o fator mais problemático é a relação entre a distância mais próxima e a distância média

min_avg_ratio = [min_dist / avg_dist
                 for min_dist, avg_dist in zip(min_distances, avg_distances)]

plt.plot(xs, min_avg_ratio)
plt.xlabel("nº de dimensões")
plt.title("Distância Mínima/Distância Média")

plt.show()

# 50 números aleatórios entre 0 e 1

numbers = [random.random() for _ in range(50)]
numbers.sort()

# Criar um gráfico de dispersão
plt.scatter(range(len(numbers)), numbers, marker='o', color='blue', label='Pontos Aleatórios')

# Adicionar rótulos e título
plt.xticks([])
plt.ylabel('Valores')
plt.title('Gráfico de Pontos Aleatórios em Uma Dimensão')

# Adicionar legenda
plt.legend()

# Exibir o gráfico
plt.show()