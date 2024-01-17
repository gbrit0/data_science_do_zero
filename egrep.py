#egrep.py
import sys, re

# sys.argv é a lista de argumentos da linha de comando
# sys.argv[0] é o nome do programa
# sys.argv[1] será o regex especificado na linha de comando
regex = sys.argv[1]

# para cada linha inserida no script
for line in sys.stdin:
   # se corresponder ao regex, escreva em stdout
   if re.search(regex, line):
      sys.stdout.write(line)
