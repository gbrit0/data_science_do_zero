import random
from typing import TypeVar, List, Tuple
X = TypeVar('X')  # tipo genérico para representar um ponto de dados

# Separar parte dos dados para treino e parte para teste
def split_data(data: List[X], prob: float) -> Tuple[List[X], List[X]]:
   """Divida os dados em frações [prob, 1 - prob]"""
   data = data[:]                                     # Faça uma cópia superficial
   random.shuffle(data)                               # porque o shuffle modifica a lista.
   cut = int(len(data) * prob)                        # Use prob para encontrar um limiar
   return data[:cut], data[cut:]                      # e dividir a lista aleatória nesse ponto

data = [n for n in range(1000)]
train, test = split_data(data, 0.75)

# As proporções devem estar corretas
assert len(train) == 750
assert len(test) == 250

# E os dados originais devem ser preservados (em alguma ordem)
assert sorted(train + test) == data

# Muitas vazes há pares de variáveis de entrada e variáveis de saída
Y = TypeVar('Y')  # tipo genérico para representar variáveis de saída

def train_test_split(xs: List[X],
                     ys: List[Y],
                     test_pct: float) -> Tuple[List[X], List[X], List[Y], List[Y]]:
   # Gere e divida os índices
   idxs = [i for i in range(len(xs))]
   train_idxs, test_idxs = split_data(idxs, 1 - test_pct)

   return([xs[i] for i in train_idxs],       # x_train
          [xs[i] for i in test_idxs],        # x_test
          [ys[i] for i in train_idxs],       # y_train
          [ys[i] for i in test_idxs])        # y_test

xs = [x for x in range(1000)]          # xs's são 1 ... 1000
ys = [2 * x for x in xs]               # cada y_i é o dobro de x_i
x_train, x_test, y_train, y_test = train_test_split(xs, ys, 0.25)

# Verifique se as proporções estão corretas
assert len(x_train) == len(y_train) == 750
assert len(x_test) == len(y_test) == 250

# Verifique se os pontos de dados correspondentes estão emparelhados corretamente
assert all(y == 2 * x for x, y in zip(x_train, y_train))
assert all(y == 2 * x for x, y in zip(x_test, y_test))


# Correção

def accuracy(tp: int, fp: int, fn: int, tn: int) -> float:
   correct = tp + tn
   total = tp + fp + fn + tn
   return correct / total

assert accuracy(70, 4930, 13930, 981070) == 0.98114

def precision(tp: int, fp: int, fn: int, tn: int) -> float:
   return tp / (tp+fp)

assert precision(70, 4930, 13930, 981070) == 0.014

def recall(tp: int, fp: int, fn: int, tn: int) -> float:
   return tp / (tp + fn)

assert recall(70, 4930, 13930, 981070) == 0.005, f"{recall(70, 4930, 13930, 981070)}"

def f1_score(tp: int, fp: int, fn: int, tn: int) -> float:
   p = precision(tp, fp, fn, tn)
   r = recall(tp, fp, fn, tn)

   return 2 * p * r / (p + r)