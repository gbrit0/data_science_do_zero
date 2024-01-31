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
    measurements: List[float]
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
