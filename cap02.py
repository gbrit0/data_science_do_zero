# IPython é um shell de Python completo a bastante útil para análise de dados
# pip install ipython
# para usar no terminal do vscode bata digitar ipython na linha de comando dentro do venv

# FUNÇÕES
# Em Python definimos as funções usando o def:
def double(x): 
   """
      Nesse ponto, você coloca um docstring opcional para descrever a função.
      Por exemplo, esta função multiplica a entrada por 2.
   """
   return x * 2

def apply_to_one(f):
   """Chama a função f usando 1 como argumento"""
   return f(1)

my_double = double            #refere-se à função x já definida
x = apply_to_one(my_double)   #igual a 2

#funções lambdas são pequenas funções anônimas
y = apply_to_one(lambda x: x + 4)

another_double = lambda x: 2 * x #não faça isso

def another_double(x):
   """Faça isso"""
   return 2 * x

#Os parâmetros da função também podem receber argumentos padrão:
def my_print(message = "my default message"):
   print(message)

my_print("hello") #imprime 'hello'
my_print()        #imprime 'my default message'

#STRINGS

#A barra invertida '\' serve para codificar caracteres esoeciais. Como o caractere tab: \t
tab_string = "\t"    # representa o caractere tab
len(tab_string)      # é 1

#para usar o carctere de barra invertida como em nosmes de diretórios
# podemos criar strings brutas com r""
not_tab_string = r"\t"     # representa os caracteres '\' e 't'
len(not_tab_string)        # é 2

#f-string
first_name = "Gabriel"
last_name = "Ribeiro"

full_name = f"{first_name} {last_name}"
print(full_name)

#EXCEÇÕES
try:
   print(0 / 0)
except ZeroDivisionError:
   print("cannot divide by zero")

#LISTAS
x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

last = x[-1]            #igual a 9, 'Pythonic' para o último elemento
penultimate = x[-2]     #igual a 8, 'Pythonic' para o penúltimo elemento
x[0] = -1               #agora x é [-1, 1, 2, ..., 8, 9]

#É possível utilizar colchetes para fatiar as listas. A fatia i:j contém 
#os elementos de i (incluído) a j (não incluído). Se o início da fatia não for
#explicitado, ele começará no início da lista; se o final não for indicado,
#ela terminará no final da lista

first_three = x[:3]
three_to_end = x[3:]
one_to_four = x[1:5]
last_three = x[-3:]

#uma fatia pode receber um terceiro argumento, stride (passo)
every_third = x[::3]             #[-1, 3, 6, 9]
five_to_three = x[5:2:-1]        #[5, 4, 3]

#Para concatenar as listas é possível usar o .extend(['lista']) ou a adição x = [1, 2, 3] y = x + [4, 5, 6]; x é inalterado
#outro modo é o método .append(0)
x = [1, 2, 3]
x.append(0)          #x agora é [1, 2, 3, 0]

#Muitas vezes é conveniente descompactar as listas quando sabemos quantos elementos elas contém
x, y = [1, 2]        #agora x é 1, y é 2
#usa-se sublinhado _ para indicar o valor que será descartado:
_, y = [1, 2]        #agora y == 2, não considerou o primeiro elemento

#TUPLAS
#são como listas mas não podem ser modificadas
#são uma forma eficaz de usar funções para retornar múltiplos valores:
def sum_and_product(x, y):
   return (x + y), (x * y)

sp = sum_and_product(2,3)        #sp é (5, 6)
s, p = sum_and_product(5, 10)    #s é 15, p é 50

#As tuplas (e as listas) podem ser usadas em atribuições múltiplas:
x, y = 1, 2       #agora x é 1 e y é 2
x, y = y, x       #forma 'Pythonic' de trocar variáveis; agora x é 2 e y é 1

#DICIONÁRIOS
# Associa valores a chaves
grades = {"Joel": 80, "Tim":95}
#para pesquisar o valor de uma chave, pode-se usar colchetes:
joels_grade = grades["Joel"]        #igual a 80

#aparecerá KeyError se procurar por uma chave que não está no dicionário
try:
   kates_grade = grades["Kate"]
except KeyError:
   print("no grade for Kate!")

#Para verificar a existência de uma chave é possível usar o in:
joel_has_grade = "Joel" in grades         #Verdadeiro
kate_has_grade = "Kate" in grades         #Falso
#o mpetodo .get() retorna um valor padrão ao invés de uma exceção:
joels_grade = grades.get("Joel", 0)    #igual a 80
kates_grade = grades.get("Kate", 0)    #igual a 0
no_ones_grade = grades.get("No One")   #o padrão é None
#é possível atribuir valores usando colchetes:
grades["Tim"] = 99         #substitui o valor anterior
grades["Kate"] = 100       #adiciona uma terceira entrada

#DEFAULTDICTC
#Para usar os defaultdicts devemos importá-los das collections:
from collections import defaultdict
document = ["Lista", "supostamente", "aleatória"]
word_counts = defaultdict(int)      #int() produz 0
for word in document: #document é uma lista de palavras
   word_counts[word] += 1

#CONTADORES
# O Counter (contador) converte uma sequencia de valores em algo precido com um defaultdict(int) mapeando as chaves correspondentes às contagens
from collections import Counter
c = Counter([0, 1, 2, 0])  #c é basicamente {0: 2, 1: 1, 2: 1}

#Uma instância Counter contém um método most_common bastante útil:

#imprima as 10 palavras mais comuns e suas contagens
for word, count in word_counts.most_common(10):
   print(word, count)

#CONJUNTO (set) - coleção de elementos *distintos*
primes_bellow_10 = {2, 3, 5, 7} #Não funciona com conjuntos bvazios pos {} é um dict vazio e não um set. Nesse cso deve-se usar o set()
s = set()
s.add(1)       #s agora é {1}
s.add(2)       #s é {1, 2}
s.add(2)       #s ainda é {1, 2}

#FLUXO DE CONTROLE 
parity = "even" if x % 2 == 0 else "odd"     #operador ternário

for x in range(10):
   if x == 3:
      continue #vá imediatamente para a próxima iteração
   if x == 5:
      break
   print(x)

#VERACIDADE
# Todos os exemplos a seguir são falsos:
#    False
#    None
#    [] uma list vazia
#    {} um dict vazio
#    ""
#    set()
#    0
#    0.0
   
#CLASSIFICAÇÃO
# Em Python, toda lista tem um método sort()
x = [4, 1, 3, 2]
y = sorted(x)     # y agora é [1, 2, 3, 4], x não mudou
x.sort()          # agora x é [1, 2, 3, 4]

#COMPREENSÃO DE LISTAS
# para transformar uma lista em outra, devemos selecionar alguns elementos, transformá-los ou fazer as duas coisas.
#As compreensões de listas são a forma Pythonic de fazer isso

#TESTES AUTOMATIZADOS A ASSERÇÃO
def smallest_item(xs):
   assert xs, "empty list has no smallest item"
   return min(xs)

assert smallest_item([10, 20, 5, 40]) == 5
assert smallest_item([1, 0, -1, 2]) == -1

#PROGRAMAÇÃO ORIENTADA A OBJETOS
# definição de classes para encapsular dados e funções
class CountingClicker:
   """A classe pode/deve ter um docstring, como as funções"""
   def __init__(self, count = 0):
      self.count = count

   def __repr__(self):
      return f"CountingCLicker(count={self.count})"
   
   def click(self, num_times = 1):
      """Clique no contador algumas vezes."""
      self.count += num_times

   def read(self):
      return self.count
   
   def reset(self):
      self.count = 0

#usando o assert para escerever casos de test para o contador:
clicker = CountingClicker()
assert clicker.read() == 0, "clicker should start with count 0"
clicker.click()
clicker.click()
assert clicker.read() == 2, "after two clicks, clicker should have count 2"
clicker.reset()
assert clicker.read() == 0, "after reset, clicker should be back to 0"

#De vez em quando cria-se subclasses que herdam algumas funcionalidades de uma classe pai

#A subclasse herda todo o comportamento da classe pai.
class NoResetClicker(CountingClicker):
   # Esta classe te os mesmos métodos da CountingClicker

   # Mas seu método reset não faz nada.
   def reset(self):
      pass

clicker2 = NoResetClicker()
assert clicker2.read() == 0
clicker2.click()
assert clicker2.read() == 1
clicker2.reset()
assert clicker2.read() == 1, "reset shouldn't do anything"

#ITERÁVEIS E GERADORES
def generate_range(n):
   i = 0;
   while i < n:
      yield i
      i += 1

for i in generate_range(10):
   print(f"i: {i}")


#também é possível criar geradores colocando compreensões de for entre parênteses:
evens_bellow_20 = (i for i in generate_range(20) if i%2 == 0) #não faz nada até que a iteração seja promovida (usando for ou next)

# a função enumerate() transforma valores de uma lista em paras (index, value):

names = ["Alice", "Bob", "Charlie", "Debbie"]
for i, name in enumerate(names):
   print(f"name {i} is {name}")

#ALEATORIEDADE
#geração de números aleatórios pode ser feita com o módulo random
import random
random.seed(10) # assim, obteremos semper os mesmos resultados
four_unifor_randoms = [random.random() for _ in range(4)]
#random.random() produz números uniformemente entre 0 e 1
#random.randrange() recebe um ou dois argumentos e retorna um elemento escolhido aleatoriamente no range indicado
#random.shuffle() reordena aleatoriamente os elementos de uma lista
#random.choice() escolhe aleatoriamente
#random.sample() escolhe aleatoriamente sem substituição(sem repetição) 
lottery_numbers = range(60)
winning_numbers = random.sample(lottery_numbers, 6)
print(winning_numbers) #[50, 17, 15, 9, 52, 43] e.g.

#EXPRESSÕES REGULARES
import re
#re.match verifica se o início de uma string corresponde à expressão regular
#re.search verifica se alguma parte da string corresponde à expressão regular

#ZIP E DESCOMPACTAÇÃO DE ARGUMENTO
#zip = compactação1 
list1 = ['a', 'b', 'c']
list2 = [1, 2, 3]

pairs = [pair for pair in zip(list1, list2)] # é [('a', 1), ('b', 2), ('c', 3)]

#também dá pra extrair uma lista
letters, numbers = zip(*pairs)   # (*) executa a descompactação de argumento é o mesmo que chamar: letters, numbers = zip(('a', 1), ('b', 2), ('c', 3)) 

#ARGS E KWARGS
def doubler(f):
   """Aqui definimos uma nova função que mantém uma referência a f"""
   def g(x):
      return 2 * f(x)
   
   #E retorna a nova função
   return g

#Funciona em alguns casos:
def f1(x):
   return x + 1

g = doubler(f1)
assert g(3) == 8, "(3 + 1) * 2 should be equal 8"
assert g(-1) == 0, "(-1 + 1) * 2 should be equal 0"

#No entanto não funciona com funções que recebem mais de um argumento:
def f2(x, y):
   return x + y

g = doubler(f2)
try:
   g(1, 2)
except TypeError:
   print("as defined, g only takes one argument")

#É preciso especificar função que receba argumentos arbitrários usando a descompactação de argumentos:

def magic(*args, **kwargs):
   print("unammed args:", args)
   print("keyword args:", kwargs)

magic(1, 2, key="word", key2="word2")
# Imprime:
# unammed args: (1, 2)
# keyword args: {'key': 'word', 'key2': 'word2'}

#args é uma tupla dos argumentos, sem nome
#kwargs é um dict com os argumentos, nomeados

def doubler_correct(f):
   """Funciona para qualquer entrada recebiade por f"""
   def g(*args, **kwargs):
      """Todo argumento fornecdo para g deve ser transmitido para f"""
      return 2 * f(*args, **kwargs)
   return g

g = doubler_correct(f2)
assert g(1, 2) == 6, "doubler should work now"

#ANOTAÇÕES DE TIPO
from typing import Union, List, Optional, Dict, Iterable, Tuple, Callable