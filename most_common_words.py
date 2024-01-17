# most_common_words.py
import sys
from collections import Counter

#passe o número de palavras como primeiro argumento
try:
   num_words = int(sys.argv[1])
except:
   print("usage: most_common_words.py num_words")
   sys.exit(1)  # código de saída diferente de zero indica erro

counter = Counter(word.lower()                     # palavras em minúsculo
                  for line in sys.stdin
                  for word in line.strip().split() # divida nos espaços
                  if word)                         # ignore palavras vazias

for word, count in counter.most_common(num_words):
   sys.stdout.write(str(count))
   sys.stdout.write("\t")
   sys.stdout.write(word)
   sys.stdout.write("\n")

# para executar digite: type arquivo.txt | python most_common_words.py <nº de palavras>