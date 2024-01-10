#usar um alias para definir vector como uma list de floats
from typing import List
Vector = List[float]
# As lists do Python não são vetores de modo que se torna necessária a construção das ferramentas aritméticas
# para realizar cálculos
def add(v: Vector, w: Vector) -> Vector:
   """Soma os elementos correspondentes"""
   assert len(v) == len(w), "vetores devem ter o mesmo tamanho"
   return [v_i + w_i for v_i, w_i in zip(v, w)]

assert add([1,2,3], [4,5,6]) == [5,7,9], "a adição não funcionou"

def subtract(v: Vector, w: Vector) -> Vector:
   """Subtrai os elementos correspondentes"""
   assert len(v) == len(w), "vetores devem ter o mesmo tamanho"
   return [v_i - w_i for v_i, w_i in zip(v, w)]

assert subtract([5,7,9], [4,5,6]) == [1,2,3], "a subtração não funcionou"

#somar uma lista de vetores por componente
def vector_sum(vectors: List[Vector]) -> Vector:
   """Soma todos os elementos correspondentes"""
   #Verifique se os vetores não estão vazios
   assert vectors, "no vectors provided!" # ia colocar em portugês mas em inglês fica mais bonito kkkkk

   #Verifique se os vetores são do mesmo tamanho
   num_elements = len(vectors[0])
   assert all(len(v) == num_elements for v in vectors), "different sizes!"       # a função all() retorna True
                                                                                 # se uma função boolena é True
                                                                                 # para todos os valores num
                                                                                 # iterável, se o iterável
                                                                                 # estiver vazio, retorna True

   # o i-ésimo elemento do resultado é a soma de todo vector[i]
   return [sum(vector[i] for vector in vectors)
           for i in range(num_elements)]

assert vector_sum([[1,2], [3,4], [5,6],[7,8]]) == [16,20]

#multiplicação de vetor por escalar
def scalar_multiply(c: float, v: Vector) -> Vector:
   """Multiplica cada elemento por c"""
   return [c*v_i for v_i in v]

assert scalar_multiply(2, [1,2,3]) == [2,4,6]

#assim podemos computar a média dos componentes de uma lista de vetores
def vector_mean(vectors: List[Vector]) -> Vector:
   """Computa a média dos elementos"""
   n = len(vectors)
   return scalar_multiply(1/n, vector_sum(vectors))

assert vector_mean([[1,2],[3,4],[5,6]]) == [3,4]

#Produto escalar - retorna o comprimento da projeção de um vetor sobre outro
def dot(v: Vector, w: Vector) -> float:
   """Computa v_1 * w_1 + ... + v_n * w_n"""
   assert len(v) == len(w), "vectors must be same length"

   return sum(v_i * w_i for v_i, w_i in zip(v, w))

assert dot([1,2,3],[4,5,6]) == 32

# De modo que a soma dos quadrados de um vetor, dot(v, v) é a projeção de um vetor sobre si mesmo
def sum_of_squares(v: Vector) -> float:
   """Retorna v_1 * v_1 + ... + v_n * v_n"""       # ou v_1 ** 2 + ... + v_n ** 2
   return dot(v, v)

assert sum_of_squares([1,2,3]) == 14

#Podemos usar esse resultado para calcular a magnitude, comprimento, do vetor:
import math

def magnitude(v:Vector) -> float:
   """Retorna o comprimento de v"""
   return math.sqrt(sum_of_squares(v))

assert magnitude([3,4]) == 5

# Agora conseguimos calcular a distância entre dois vetores:
def squared_distance(v: Vector, w: Vector) -> float:
   """Computa (v_1 - w_1) ** 2 + ... + (v_n - w_n) ** 2"""
   return sum_of_squares(subtract(v,w))         # subtract foi a função criada no início do arquivo

def distance(v: Vector, w: Vector) -> float:
   """Computa a distância entre v e w"""
   return math.sqrt(squared_distance(v, w))

# MATRIZES - coleções multidimensionais de números
Matrix = List[List[float]]       # Outro alias de tipo

from typing import Tuple

def shape(A: Matrix) -> Tuple[int, int]:
   """Retorna nº de linhas de A e nº de colunas de A"""
   num_rows = len(A)
   num_cols = len(A[0]) if A else 0             # nº de elementos na primeira linha
   return num_rows, num_cols

assert shape([[1,2,3], [4,5,6]]) == (2,3)       # 2 linhas, 3 colunas

def get_row(A: Matrix, i: int) -> Vector:
   """Retorna a linha i de A (como um Vector)"""
   return [A[i]]           # A[i] já está na linha i

def get_column(A: Matrix, j: int) -> Vector:
   """Retorna a coluna j de A (como um Vector)"""
   return [A_i[j]                # elemento j da linha A_i
           for A_i in A]         # para cada linha A_i

from typing import Callable

def make_matrix(num_rows: int,
                num_columns: int,
                entry_fn: Callable[[int, int], float]) -> Matrix:
   """
   Retorna uma matriz num_rows x num_cols
   cuja entrada (i,j) é entry_fn(i,j)
   """
   return [[entry_fn(i,j)                       #com i, crie uma lista
            for j in range(num_columns)]        #[entry_fn(i,0), ...]
            for i in range(num_rows)]           #crie uma lista para cada i

def identity_matrix(n: int) -> Matrix:
   """Retorna a matrix de identidade n x n"""
   return make_matrix(n, n, lambda i, j: 1 if i == j else 0)

print(f"{identity_matrix(5)}\n")

# Matrizes para representar relações binárias

#No exemplo do capitulo 1 a situação era o seguinte:

friendships_pairs = [ # pares de id's representando as conexões, amizades.
    (0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)
]

result_matrix = make_matrix(
   10,   #10 linhas
   10,   #10 colunas
   lambda i, j: 1 if (i,j) in friendships_pairs       # função geradora da matriz:
      or (j,i) in friendships_pairs else 0            # se existe o par (i,j) ou (j,i)
                                                      # então 1 se não 0
   )

for row in result_matrix:
   print(row)