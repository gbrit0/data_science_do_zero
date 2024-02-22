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

from typing import Tuple, random

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

