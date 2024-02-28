from typing import NamedTuple

class User(NamedTuple):
   id: int
   name: str

users = [
   User(0, "Hero"), User(1, "Dunn"), User(2, "Sue"),
   User(3, "Chi"), User(4, "Thor"), User(5, "Clive"),
   User(6, "Hicks"), User(7, "Devin"), User(8, "Kate"),
   User(9, "Klein")
]

friendship_pairs = [
   (0, 1), (0, 2), (1, 2), (1, 3), (2, 3), (3, 4),
   (4, 5), (5, 6), (5, 7), (6, 8), (7, 8), (8, 9)
]

# Será mais fácil trabalhar com as amizades usando um dict:
from typing import Dict, List

Friendships = Dict[int, List[int]]

friendships: Friendships = {user.id: [] for user in users}

from collections import deque
Path = List[int]

def shortest_paths_from(from_user_id: int,
                        frinedships: Friendships) -> Dict[int, List[Path]]:
   # Um dicionário de user_id para *todos* os caminhos mais curtos até esse usuário
   shortest_paths_to: Dict[int, List[Path]] = {from_user_id: [[]]}

   # Uma fila de (usuários anterior, próximo usuário) que precisamos verificar.
   # Começa com todos os pares (from _user, friend_of_from_user)
   frontier = deque((from_user_id, friend_id)
                    for friend_id in friendships[from_user_id])
   
   # Continue até esvaziarmos a fila
   while frontier:
      # Remova o próximo par da fila
      prev_user_id, user_id = frontier.popleft()

      paths_to_prev_user = shortest_paths_to[prev_user_id]
      new_paths_to_user = [path + [user_id] for path in paths_to_prev_user]

      # Talvez já saibamos o caminho mais curto para o user_id
      old_paths_to_user = shortest_paths_to.get(user_id, [])

      # Qual é o caminho mais curto para cá identificado até agora?
      if old_paths_to_user:
         min_path_length = len(old_paths_to_user[0])
      else:
         min_path_length = float('inf')

      # mantenha apenas os caminhos não muito longos e relativamente novos
      new_paths_to_user = [path
                           for path in new_paths_to_user
                           if len(path) <= min_path_length
                           and path not in old_paths_to_user]
      
      shortest_paths_to[user_id] = old_paths_to_user + new_paths_to_user

      # Adicione vizinhos inéditos à fronteira
      frontier.extend((user_id, friend_id)
                      for friend_id in friendships[user_id]
                      if friend_id not in shortest_paths_to)
      
   return shortest_paths_to

def farness(user_id: int) -> float:
   """A soma dos comprimentos dos caminhos mais curtos até os outros usuários"""
   return sum(len(paths[0])
            for paths in shortest_paths[user_id].values())


for i, j in friendship_pairs:
   friendships[i].append(j)
   friendships[j].append(i)

assert friendships[4] == [3, 5]
assert friendships[8] == [6, 7, 9]

shortest_paths = {user.id: shortest_paths_from(user.id, friendships)
                  for user in users}

betweenness_centrality = {user.id: 0.0 for user in users}

for source in users:
   for target_id, paths in shortest_paths[source.id].items():
      if source.id < target_id:              # não conte duas vezes
         num_paths = len(paths)              # quantos caminhos mais curtos?
         contrib = 1 / num_paths             # contribuição para a centralidade
         for path in paths:
            for between_id in path:
               if between_id not in [source.id, target_id]:
                  betweenness_centrality[between_id] += contrib



closeness_centrality = {user.id: 1 / farness(user.id) for user in users}

from cap04 import Matrix, make_matrix, shape

def matrix_times_matrix(m1: Matrix, m2: Matrix) -> Matrix:
   nr1, nc1 = shape(m1)
   nr2, nc2 = shape(m2)
   assert nc1 == nr2, "deve ter (nº de colunas em mq) == (nº de linhas em m2)"

   def entry_fn(i: int, j: int) -> float:
      """Produto escalar da linha i de m1 pela coluna j de m2"""
      return sum(m1[i][k] * m2[k][j] for k in range(nc1))
   
   return make_matrix(nr1, nc2, entry_fn)

from cap04 import Vector, dot

def matrix_times_vector(m: Matrix, v: Vector) -> Vector:
   nr, nc = shape(m)
   n = len(v)
   assert nc == n, "deve ter o (n° de colunas em m) == (n° de elementos em v)"

   return [dot(row, v) for row in m]   # a saída tem o comprimento nr

from typing import Tuple
import random
from cap04 import magnitude, distance

def find_eigenvector(m: Matrix,
                     tolerance: float = 0.00001) -> Tuple[Vector, float]:
   guess = [random.random() for _ in m]

   while True:
      result = matrix_times_vector(m, guess)       # transforme o palpite
      norm = magnitude(result)                     # compute a norma
      next_guess = [x/ norm for x in result]       # redimensione

      if distance(guess, next_guess) < tolerance:
         # convergência, então retorne (autovetor, autovalor)
         return next_guess, norm
      guess = next_guess

def entry_fn(i: int, j: int):
   return 1 if (i, j) in friendship_pairs or (j, i) in friendship_pairs else 0

n = len(users)
adjacency_matrix = make_matrix(n, n, entry_fn)

for row in adjacency_matrix:
   print(row)

eigenvector_centralities, _ = find_eigenvector(adjacency_matrix)

matrix_times_vector(adjacency_matrix, eigenvector_centralities)

endorsements = [(0, 1), (1, 0), (0, 2), (2, 0), (1, 2),
                (2, 1), (1, 3), (2, 3), (3, 4), (5, 4),
                (5, 6), (7, 5), (6, 8), (8, 7), (8, 9)]

from collections import Counter

endorsement_counts = Counter(target for source, target in endorsements)

import tqdm

def page_rank(users: List[User],
              endorsements: List[Tuple[int, int]],
              damping: float = 0.85,
              num_iters: int = 100) -> Dict[int, float]:
   # Compute quantas pessoas cada pessoa recomenda
   outgoing_counts = Counter(target for source, target in endorsements)

   # Inicialmente, distribua o PageRank uniformemente
   num_users = len(users)
   pr = {user.id : 1 / num_users for user in users}

   # A pequena fração do PageRank que cada nó obtém a cada iteração
   base_pr = (1 - damping) / num_users

   for iter in tqdm.trange(num_iters):
      next_pr = {user.id : base_pr for user in users} # Comece com base_pr

      for source, target in endorsements:
         # Adicione uma fração amortecida do pr da origem ao destino
         next_pr[target] += damping * pr[source] / outgoing_counts[source]

      pr = next_pr
   
   return pr

pr = page_rank(users, endorsements)

# Thor (user_id 4) tem uma classificação de página mais alta do que os demais
assert pr[4] > max(page_rank
                   for user_id, page_rank in pr.items()
                   if user_id != 4)