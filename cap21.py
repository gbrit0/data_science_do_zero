data = [ ("big data", 100, 15), ("Hadoop", 95, 25), ("Python", 75, 50),
         ("R", 50, 40), ("machine learning", 80, 20), ("statistics", 20, 60),
         ("data science", 60, 70), ("analytics", 90, 3),
         ("team player", 85, 85), ("dynamic", 2, 90), ("synergies", 70, 0),
         ("actionable insights", 40, 30), ("think out of the box", 45, 10),
         ("self-starter", 30, 50), ("customer focus", 65, 15),
         ("thought leadership", 35, 35)]

def fix_unicode(text: str) -> str:
   return text.replace(u"\u2019", "'")

import random

def generate_using_bigrams() -> str:
   current = "."     # indica que a próxima palavra iniciará uma frase
   result = []
   while True:
      next_words_candidates = transitions[current]
      current = random.choice(next_words_candidates)
      result.append(current)
      if current == ".": return " ".join(result)

from matplotlib import pyplot as plt

def text_size(total: int) -> float:
   """É igual a 8 se o total for 0, 28 se o total for 200"""
   return 8 + total / 200 * 20

for word, job_popularity, resume_popularity in data:
   plt.text(job_popularity, resume_popularity, word,
            ha='center', va='center',
            size=text_size(job_popularity + resume_popularity))

plt.xlabel("Popularidade nos Anúncios de Empregos")
plt.ylabel("Populadirade nos Currículos")
plt.axis([0, 100, 0, 100])
plt.xticks([])
plt.yticks([])
# plt.show()

import re
from bs4 import BeautifulSoup
import requests

url = "http://radar.oreilly.com/2010/06/what-is-data-science.html"
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

content = soup.find("div", "entry-content")                        # Encontre o article-body div
regex = r"[\w']+|[\.]"                                           # associa a uma palavra ou a um ponto

document = []

for paragraph in content("p"):
   words = re.findall(regex, fix_unicode(paragraph.text))
   document.extend(words)

from collections import defaultdict

transitions = defaultdict(list)
for prev, current in zip(document, document[1:]):
   transitions[prev].append(current)

trigram_transitions = defaultdict(list)
starts = []

for prev, current, next in zip(document, document[1:], document[2:]):
   if prev == ".":            # se a "palavra" foi um ponto
      starts.append(current)  # então esta é uma palavra inicial

   trigram_transitions[(prev, current)].append(next)
     
def generate_using_trigrams() -> str:
   current = random.choice(starts)              # escolha uma palavra inicial aleatória
   prev = "."                                   # e coloque um '.' antes dela
   result = [current]
   while True:
      next_word_candidates = trigram_transitions[(prev, current)]
      next_word = random.choice(next_word_candidates)

      prev, current = current, next_word
      result.append(current)

      if current == ".":
         return " ".join(result)
      
from typing import List, Dict

# Digite o alias para se referir às gramáticcas depois
Grammar = Dict[str, List[str]]

grammar = {
   "_S" : ["_NP _VP"],
   "_NP" : ["_N",
            "_A _NP _P _A _N"],
   "_VP" : ["_V",
            "_V _NP"],
   "_N" : ["data science", "Python", "regression"],
   "_A" : ["big", "linear", "logistic"],
   "_P" : ["about", "near"],
   "_V" : ["learns", "trains", "test", "is"]
}

def is_terminal(token: str) -> bool:
   return token[0] != "_"

def expand(grammar: Grammar, tokens: List[str]) -> List[str]:
   for i, token in enumerate(tokens):
      # Se este for um token terminal, pule-o
      if is_terminal(token): continue

      # Se não, é um token não terminal
      # então escolhermos uma substituição aleatoriamente
      replacement = random.choice(grammar[token])

      if is_terminal(replacement):
         tokens[i] = replacement
      else:
         # A substituição será, por exemplo, "_NP _VP", então temos que
         # dividi-la em espaços e integrá-la
         tokens = tokens[:i] + replacement.split() + tokens[(i + 1):]

      # Agora, chame expand na nova lista de tokens.
      return expand(grammar, tokens)
   
   return tokens
   
def generate_sentence(grammar: Grammar) -> List[str]:
   return expand(grammar, ["_S"])

from typing import Tuple
import random

def roll_a_die() -> int:
   return random.choice([1, 2, 3, 4, 5, 6])

def direct_sample() -> Tuple[int, int]:
   d1 = roll_a_die()
   d2 = roll_a_die()
   return d1, d1 + d2

def random_y_given_x(x: int) -> int:
   """É bem provável que seja x + 1, x + 2, ..., x + 6"""
   return x + roll_a_die()

def random_x_given_y(y: int) -> int:
   if y <= 7:
      # se o total for menor ou igual a 7 ou menos, é provável que o primeiro dado seja
      # 1, 2, ..., (total -  1)
      return random.randrange(1, y)
   else:
      # se o total for maior ou igual a 7, é bem provável que o primeiro dados seja
      # (total - 6), (total - 5), ..., 6
      return random.randrange(y - 6, 7)
   
def gibbs_sample(num_iters: int = 100) -> Tuple[int, int]:
   x, y = 1, 2    # não importa
   for _ in range(num_iters):
      x = random_x_given_y(y)
      y = random_y_given_x(x)
   return x, y

def compare_distibutions(num_samples: int = 1000) -> Dict[int, List[int]]:
   counts = defaultdict(lambda: [0, 0])
   for _ in range(num_samples):
      counts[gibbs_sample()][0] += 1
      counts[direct_sample()][1] += 1
   return counts

def sample_from(weights: List[float]) -> int:
   """Retorna i com probabilidade weights[i] / sum(weigthts)"""
   total = sum(weights)
   rnd = total * random.random()      # uniforme entre 0 e total
   for i, w in enumerate(weights):
      rnd -= w
      if rnd <= 0: return i

from collections import Counter
# Execute mil vezes e conte
draws = Counter(sample_from([0.1, 0.1, 0.8]) for _ in range(1000))

assert 10 < draws[0] < 190 # deve ser ~10%, o teste é muito flexível
assert 10 < draws[1] < 190 # deve ser ~10%, o teste é muito flexível
assert 650 < draws[2] < 950 # deve ser ~80%, o teste é muito flexível
assert draws[0] + draws[1] + draws[2] == 1000

documents = [
   ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
   ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
   ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
   ["R", "Python", "statistics", "regression", "probability"],
   ["machine learning", "regression", "decision trees", "libsvm"],
   ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
   ["statistics", "probability", "mathematics", "theory"],
   ["machine learning", "scikit-learn", "Mahout", "neural networks"],
   ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
   ["Hadoop", "Java", "MapReduce", "Big Data"],
   ["statistics", "R", "statsmodels"],
   ["C++", "deep learning", "artificial intelligence", "probability"],
   ["pandas", "R", "Python"],
   ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
   ["libsvm", "regression", "support vector machines"]
]

K = 4

# Quantas vezes cada tópico é atribuído a cada documento
# uma lista de Counters, um para cada documento
document_topic_counts = [Counter() for _ in documents]

# Quantas vezes cada palavra é atribuída a cada tópico
# uma lisra de Counters, um para cada tópico
topic_word_counts = [Counter() for _ in range(K)]

# O número total de palavras atribuídas a cada tópico
# uma lista de números, um para cada tópico
topic_counts = [0 for _ in range(K)]

# O número total de palavras contidas em cada documento
# uma lista de números, um para cada documento
document_lenghts =[len(document) for document in documents]

# O número de palavras distintas
distinct_words = set(word for document in documents for word in document)
W = len(distinct_words)

# o número de documentos
D = len(documents)


def p_topic_given_document(topic: int, d: int, alpha: float = 0.1) -> float:
   """
   A fração de palavras no documento 'd' atribuídas ao 'tópico' (mais a suavização)
   """
   return ((document_topic_counts[d][topic] + alpha) /
           (document_lenghts[d] + K * alpha))


def p_word_given_topic(word: str, topic: int, beta: float = 0.1) -> float:
   """
   A fração de palavras atribuídas ao 'tópico' iguais à 'palavra' (mais a suavização)
   """
   return ((topic_word_counts[topic][word] + beta) /
           (topic_counts[topic] + W * beta))


def topic_weight(d: int, word: str, k: int) -> float:
   """
   Para um certo documento e uma certa palavra nesse documento, retorne o peso do tópico k
   """
   return p_word_given_topic(word, k) * p_topic_given_document(k, d)


def choose_new_topic(d: int, word: str) -> str:
   return sample_from([topic_weight(d, word, k)
                       for k in range(K)])


random.seed(0)
document_topics = [[random.randrange(K) for word in document]
                   for document in documents]

for d in range(D):
   for word, topic in zip(documents[d], document_topics[d]):
      document_topic_counts[d][topic] += 1
      topic_word_counts[topic][word] += 1
      topic_counts[topic] += 1

import tqdm

for iter in tqdm.trange(1000):
   for d in range(D):
      for i, (word, topic) in enumerate(zip(documents[d], document_topics[d])):

         # remova esta palavra/tópico das contagens para que naõ influencie os pesos
         document_topic_counts[d][topic] -= 1
         topic_word_counts[topic][word] -= 1
         topic_counts[topic] -= 1
         document_lenghts[d] -= 1

         # escolha um novo tópico com base nos pesos
         new_topic = choose_new_topic(d, word)
         document_topics[d][i] = new_topic

         # e agora o adicione novamente às contagens
         document_topic_counts[d][new_topic] += 1
         topic_word_counts[new_topic][word] += 1
         topic_counts[new_topic] += 1
         document_lenghts[d] += 1


for k, word_counts in enumerate(topic_word_counts):
   for word, count in word_counts.most_common():
      if count > 0:
         print(k, word, count)


from cap04 import dot, Vector
import math

def cosine_similarity(v1: Vector, v2: Vector) -> float:
   return dot(v1, v2) / math.sqrt(dot(v1, v1) * dot(v2, v2))

assert cosine_similarity([1., 1, 1], [2., 2, 2]) == 1, "mesma direção"
assert cosine_similarity([-1, -1], [2., 2]) == -1, "direção oposta"
assert cosine_similarity([1., 0], [0., 1]) == 0, "ortogonal"

colors = ["red", "green", "blue", "yellow", "black", ""]
nouns = ["bed", "car", "boat", "cat"]
verbs = ["is", "was", "seems"]
adverbs = ["very", "quite", "extremely", ""]
adjectives = ["slow", "fast", "soft", "hard"]

def make_sentence() -> str:
   return " ".join([
      "The",
      random.choice(colors),
      random.choice(nouns),
      random.choice(verbs),
      random.choice(adverbs),
      random.choice(adjectives),
      "."
   ])

NUM_SENTENCES = 50

random.seed(0)
sentences = [make_sentence() for _ in range(NUM_SENTENCES)]

from cap19 import Tensor

class Vocabulary:
   def __init__(self, words: List[str] = None) -> None:
      self.w2i: Dict[str, int] = {}          # mapeamento palavra -> word_id
      self.i2w: Dict[int, str] = {}          # mapeamento word_id -> palavra

      for word in (words or []):             # Se houver palavras, adicione-as
         self.add(word)

   @property
   def size(self) -> int:
      """Há quantas palavras no vocabulário"""
      return len(self.w2i)
   
   def add(self, word: str) -> None:
      if word not in self.w2i:                  # Se a palavra for nova:
         word_id = len(self.w2i)                # Encontre o próximo ID.
         self.w2i[word] = word_id               # Adicione ao mapa palavra -> word_id.
         self.i2w[word_id] = word               # Adicione ao mapa word_id -> palavra.

   def get_id(self, word: str):
      """Retorne o id da palavra (ou None)"""
      return self.w2i.get(word)
   
   def get_word(self, word_id: int) -> str:
      """Retorne a palavra com o id fornecido (ou None)"""
      return self.i2w.get(word_id)
   
   def one_hot_encode(self, word:str) -> Tensor:
      word_id = self.get_id(word)
      assert word_id is not None, f"unknown word {word}"

      return [1.0 if i == word_id else 0.0 for i in range(self.size)]
   
vocab = Vocabulary(["a", "b", "c"])
assert vocab.size == 3, "há 3 palavras no vocabulário"
assert vocab.get_id("b") == 1, "b deve ter word_id 1"
assert vocab.one_hot_encode("b") == [0, 1, 0]
assert vocab.get_id("z") is None, "z não está no vocabulário"
assert vocab.get_word(2) == "c", "word_id 2 deve ser c"
vocab.add("z")
assert vocab.size == 4, "agora  há 4 palavras no vocabulário"
assert vocab.get_id("z") == 3, "agora z deve trer o id 3"
assert vocab.one_hot_encode("z") == [0, 0, 0, 1]

import json

def save_vocab(vocab: Vocabulary, filename: str) -> None:
   with open(filename, 'w') as f:
      json.dump(vocab.w2i, f)       # é só salvar o w2i

def load_vocabulary(filename: str) -> Vocabulary:
   vocab = Vocabulary()
   with open(filename) as f:
      # Carregue o w2i e gere o i2w a partir dele
      vocab.w2i = json.load(f)
      vocab.i2w = {id: word for word, id in vocab.w2i.items()}

   return vocab

from typing import Iterable
from cap19 import Layer, Tensor, random_tensor, zeros_like

class Embedding(Layer):
   def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
      self.num_embeddings = num_embeddings
      self.embedding_dim = embedding_dim

      # Um vetor de tamanho embedding_dim para cada incorporação desejada
      self.embeddings = random_tensor(num_embeddings, embedding_dim)
      self.grad = zeros_like(self.embeddings)

      # Salve o ID da última entrada
      self.last_input_id = None

   def forward(self, input_id: int) -> Tensor:
      """Basta selecionar o vetor de incorporação correspondente ao id da entrada"""
      self.input_id = input_id               # lembre-se para usar na retropropagação

      return self.embeddings[input_id]
   
   def backward(self, gradient: Tensor) -> None:
      # Zere o gradiente correspondente à última entrada
      # Isso é muito mais barato do que criar um tensor sempre que necessário
      if self.last_input_id is not None:
         zero_row = [0 for _ in range(self.embedding_dim)]
         self.grad[self.last_input_id] = zero_row

      self.last_input_id = self.input_id
      self.grad[self.input_id] = gradient

   def params(self) -> Iterable[Tensor]:
      return [self.embeddings]
   
   def grads(self) -> Iterable[Tensor]:
      return [self.grad]
   
class TextEmbedding(Embedding):
   def __init__(self, vocab: Vocabulary, embedding_dim: int) -> None:
      # Chame o construtor da superclasse
      super().__init__(vocab.size, embedding_dim)

      # E continue com ovaocabulário
      self.vocab = vocab

   def __getitem__(self, word: str) -> Tensor:
      word_id = self.vocab.get_id(word)
      if word_id is not None:
         return self.embeddings[word_id]
      else:
         return None
      
   