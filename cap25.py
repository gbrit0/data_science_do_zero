# MapReduce
from typing import List
from collections import Counter

def tokenize(document: str) -> List[str]:
   """Basta dividir no espaço em branco"""
   return document.split()

def word_count_old(documents: List[str]):
   """Contagem de palavras sem o MapReduce"""
   return Counter(word
                  for document in documents
                  for word in tokenize(document))

from typing import Iterator, Tuple

def wc_mapper(document: str) -> Iterator[Tuple[str, int]]:
   """Para cada palavra no documento, emita (word, 1)"""
   for word in tokenize(document):
      yield(word, 1)

from typing import Iterable

def wc_reducer(word: str,
               counts: Iterable[int]) -> Iterator[Tuple[str, int]]:
   """Some as contagens da palavra"""
   yield (word, sum(counts))

from collections import defaultdict

def word_count(documents: List[str]) -> List[Tuple[str, int]]:
   """Conte as palavras nos documentos de entrada usando o MapReduce"""

   collector = defaultdict(list)          # para armazenar os valores agrupados

   for document in documents:
      for word, count in wc_mapper(document):
         collector[word].append(count)

   return [output
           for word, counts in collector.items()
           for output in wc_reducer(word, counts)]

from typing import Callable, Iterable, Any, Tuple

# Um par de chave/valor é apenas uma tupla de 2 elementos
KV = Tuple[Any, Any]

# Um Mapper é uma função que retorna um iteravel de pares de chave/valor
Mapper = Callable[..., Iterable[KV]]

# Um Reducer é uma função que recebe uma chave e um iterável de valores e retorna um par de chave/valor
Reducer = Callable[[Any, Iterable], KV]

def map_reduce(inputs: Iterable,
               mapper: Mapper,
               reducer: Reducer) -> List[KV]:
   """Execute o MapReduce nas entradas usando o mapper e o reducer"""
   collector = defaultdict(list)

   for input  in inputs:
      for key, value in mapper(input):
         collector[key].append(value)

   return [output
           for key, values in collector.items()
           for output in reducer(key, values)]

# word_counts = map_reduce(documents, wc_mapper, wc_reducer)

def values_reducer(values_fn: Callable) -> Reducer:
   """Retorne um reducer que aplique o values_fn sem seus valores"""
   def reduce(key, values: Iterable) -> KV:
      return (key, values_fn(values))
   
   return reduce

sum_reducer = values_reducer(sum)
max_reducer = values_reducer(max)
min_reducer = values_reducer(min)
count_distinct_reducer = values_reducer(lambda values: len(set(values)))

assert sum_reducer("key", [1, 2, 3, 3]) == ("key", 9)
assert min_reducer("key", [1, 2, 3, 3]) == ("key", 1)
assert max_reducer("key", [1, 2, 3, 3]) == ("key", 3)
assert count_distinct_reducer("key", [1, 2, 3, 3]) == ("key", 3)

def data_science_day_mapper(status_update: dict) -> Iterable:
   """Gera (day_of_week, 1) se o status_update mencionar "data science" """

   if 'data science' in status_update["text"].lower():
      day_of_week = status_update["created_at"].weekday()
      yield(day_of_week, 1)

# data_science_days = map_reduce(status_updates,
#                                data_science_day_mapper,
#                                sum_reducer)
      
def words_per_user_mapper(status_update: dict):
   user = status_update["username"]
   for word in tokenize(status_update["text"]):
      yield(user, (word, 1))

def most_popular_word_reducer(user: str,
                               words_and_counts: Iterable[KV]):
   """
   Para uma sequência de pares (word, count)
   retorne a plavra com a maior contagem total
   """
   word_counts = Counter()
   for word, count in words_and_counts:
      word_count[word] += count
   word, count = word_counts.most_common(1)[0]

   yield (user, (word, count))

# user_words = map_reduce(status_updates,
#                         words_per_user_mapper,
#                         most_popular_word_reducer)
   

def liker_mapper(status_update: dict):
   user = status_update["username"]
   for liker in status_update["liked_by"]:
      yield (user, liker)

# distinct_likers_per_user = map_reduce(status_updates,
#                                       liker_mapper,
#                                       count_distinct_reducer)
      
from typing import NamedTuple

class Entry(NamedTuple):
   name: str
   i: int
   j: int
   value: float

def matrix_multiply_mapper(num_rows_a: int, num_cols_b: int) -> Mapper:
   # C[x][y] = A[x][0] * B[0][y] + ... + A[x][m] * B[m][y]

   # então, um elemento A[i][j] entra em todos os C[i][y] com o coef B[j][y]
   # e um elemento B[i][j] entra em todos os C[x][j] com o coef A[x][i]
   def mapper(entry: Entry):
      if entry.name == "A":
         for y in range(num_cols_b):
            key = (entry.i, y)                  # qual elemento de C
            value = (entry.j, entry.value)      # qual entrada na soma
            yield (key, value)
      else:
         for x in range(num_rows_a):
            key = (x, entry.j)                  # qual elemento de C
            value = (entry.i, entry.value)      # qual entrada na soma
            yield (key, value)
   
   return mapper

def matrix_multiply_reducer(key: Tuple[int, int],
                            indexed_values: Iterable[Tuple[int, int]]):
   results_by_index = defaultdict(list)

   for index, value in indexed_values:
      results_by_index[index].append(value)

   # Multiplique os valores das posições por dois valores
   # (um de A e outro de B) e some todos
   sumproduct = sum(values[0] * values[1]
                    for values in results_by_index.values()
                    if len(values) == 2)
   
   if sumproduct != 0.0:
      yield(key, sumproduct)


