from typing import Set
import re

def tokenize(text: str) -> Set[str]:
   text = text.lower()                                # Converta para minúsculas
   all_words = re.findall("[a-z0-9']+", text)         # extraia as palavras e
   return set(all_words)                              # remova as duplicatas

assert tokenize("Data Science is science") == {"data", "science", "is"}

from typing import NamedTuple
class Message(NamedTuple):
   text: str
   is_spam: bool

from typing import List, Tuple, Dict, Iterable
import math
from collections import defaultdict

class NaiveBayesClassifier:
   def __init__(self, k: float = 0.5) -> None:
      self.k = k # fator de suavização

      self.tokens: Set[str] = set()
      self.token_spam_counts: Dict[str, int] = defaultdict(int)
      self.token_ham_counts: Dict[str, int] = defaultdict(int)
      self.spam_messages = self.ham_messages = 0

   def train(self, messages: Iterable[Message]) -> None:
      for message in messages:
         # Incremente as contagens de mensagens
         if message.is_spam:
            self.spam_messages += 1
         else:
            self.ham_messages += 1

         # Incremente as contagens de palavras
         for token in tokenize(message.text):
            self.tokens.add(token)
            if message.is_spam:
               self.token_spam_counts[token] += 1
            else:
               self.token_ham_counts[token] += 1
            
   def _probabilities(self, token: str) -> Tuple[float, float]:
      """Retorna P(token | spam) e P(token | ham)"""
      spam = self.token_spam_counts[token]
      ham = self.token_ham_counts[token]

      p_token_spam = (spam + self.k) / (self.spam_messages + 2 * self.k)
      p_token_ham = (ham + self.k) / (self.ham_messages + 2 * self.k)

      return p_token_spam, p_token_ham
   
   def predict(self, text: str) -> float:
      text_tokens = tokenize(text)
      log_prob_if_spam = log_prob_if_ham = 0.0

      # Itere em cada palavra do vocabulário
      for token in self.tokens:
         prob_if_spam, prob_if_ham = self._probabilities(token)

         # Se o *token* aparecer na mensagem
         # adicione o log da probabilidade de vê-lo
         if token in text_tokens:
            log_prob_if_spam += math.log(prob_if_spam)
            log_prob_if_ham += math.log(prob_if_ham)
         # Se não, adicione o log da probabilibade de _não_ vê-lo,
         # que corresponde a log(1 - probabilidade de vê-lo)
         else:
            log_prob_if_spam += math.log(1.0 - prob_if_spam)
            log_prob_if_ham += math.log(1.0 - prob_if_ham)
      
      prob_if_spam = math.exp(log_prob_if_spam)
      prob_if_ham = math.exp(log_prob_if_ham)

      return prob_if_spam / (prob_if_spam + prob_if_ham)
   

# Testando o Modelo


messages = [Message("spam rules", is_spam=True),
            Message("ham rules", is_spam=False),
            Message("hello ham", is_spam=False)]

model = NaiveBayesClassifier(k=0.5)
model.train(messages)

# Verificando as contagens

assert model.tokens == {'spam', 'ham', 'rules', 'hello'}
assert model.spam_messages == 1
assert model.ham_messages == 2
assert model.token_spam_counts == {'spam': 1, 'rules': 1}
assert model.token_ham_counts == {'ham': 2, 'rules': 1, 'hello': 1}

#  Fazer a previsão manualmente =( 
text = "hello spam"

probs_if_spam = [
   (1 + 0.5) / (1 + 2 * 0.5),          # "spam" (presente)
   1 - (0 + 0.5) / (1 + 2 * 0.5),      # "ham" (ausente)
   1 - (1 + 0.5) / (1 + 2 * 0.5),      # "rules" (ausente)
   (0 + 0.5) / (1 + 2 * 0.5)           # "hello" (presente)
]

probs_if_ham = [
   (0 + 0.5) / ( 2 + 2 * 0.5),         # "spam" (presente)
   1 - (2 + 0.5) / (2 + 2 * 0.5),      # "ham" (ausente)
   1 - (1 + 0.5) / (2 + 2 * 0.5),      # "rules" (ausente)
   (1 + 0.5) / (2 + 2 * 0.5)           # "hello" (presente)
]

p_if_spam = math.exp(sum(math.log(p) for p in probs_if_spam))
p_if_ham = math.exp(sum(math.log(p) for p in probs_if_ham))


# Deu erro de 10^-16 "AssertionError: modelo: 0.8350515463917526 - cálculo manual:0.8350515463917525"
assert model.predict(text) == p_if_spam / (p_if_spam + p_if_ham), f"modelo: {model.predict(text)} - cálculo manual:{p_if_spam / (p_if_spam + p_if_ham)}"     # Aproximadamente 0.83

# Agora utilizaremos dados reais

# Um conjunto de dados popular porém antigo é o que usaremos: public corpus do Spam Assassin

# Script para fazer o download e descompactação:

from io import BytesIO        # tratar byter como um arquivo.
import requests               # para baixar arquivos, que
import tarfile                # estão no formato .tar.bz

BASE_URL = "https://spamassassin.apache.org/old/publiccorpus/"
FILES = ["20021010_easy_ham.tar.bz2",
         "20021010_hard_ham.tar.bz2",
         "20021010_spam.tar.bz2"]

# Os dados ficarão nas subpastas /spam, /easy_ham e /hard_ham

OUTPUT_DIR = 'spam_data'

for filename in FILES:
   # Use solicitações para obter o conteúde dos arquivos em cada URL
   content = requests.get(f"{BASE_URL}/{filename}").content

   # Encapsule os bytes na memória para usá-los como um 'arquivo'
   fin = BytesIO(content)

   # Extraia todos os arquivos para o diretório de saída especificado
   with tarfile.open(fileobj=fin, mode='r:bz2') as tf:
      tf.extractall(OUTPUT_DIR)


import glob, re
import tqdm
# Modifique o caminho para indicar o local dos arquivos
path = 'spam_data/*/*'

data: List[Message] = []

# glob.glob retorna todos os nomes que arquivos que correspondem ao path curinga definido anteriormente
for filename in tqdm.tqdm(glob.glob(path), desc="Extrair os assuntos dos e-mails:"):
   is_spam = "ham" not in filename

   # Existem alguns caracteres de lixo nos e-mails; o errors='ignore' os ignora em vez de gerar uma exceção
   with open(filename, errors='ignore') as email_file:
      for line in email_file:
         if line.startswith("Subject: "):
            subject = line.lstrip("Subject: ")
            data.append(Message(subject, is_spam))
            break # o arquivo foi finalizado


import random
from cap11 import split_data

random.seed(0)
train_messages, test_messages = split_data(data, 0.75)

model = NaiveBayesClassifier()
model.train(train_messages)

from collections import Counter

predictions = [(message, model.predict(message.text))
               for message in test_messages]

# Presuma que spam_probability > 0.5 corresponde à previsão de spam
# e conte as combinações de (real is_spam, previsto is_spam)
confusion_matrix =  Counter((message.is_spam, spam_probability > 0.5)
                            for message, spam_probability in predictions)

print(confusion_matrix)

def p_spam_given_token(token: str, model: NaiveBayesClassifier) -> float:
   # Não se recomenda chamr métodos privados, mas é por uma boa causa
   probs_if_spam, probs_if_ham = model._probabilities(token)

   return probs_if_spam / (probs_if_spam + probs_if_ham)

words =  sorted(model.tokens, key=lambda t: p_spam_given_token(t, model))

print("spammiest_words", words[-10:])
print("hammiest_words", words[:10])