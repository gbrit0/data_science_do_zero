# OBTENDO DADOS

# stdin e stdout:
   # arquivos egrep.py line_count.py lipsum.txt most_common.words.py teste1.txt

# Lendo Arquivos

# para trabalhar com arquivos de texto, o primeiro passo é obter um objeto de arquivo usando open:

# 'r' significa somente leitura, é o padrão se não for definido
file_for_reading = open('reading_file.txt', 'r')
file_for_reading2 = open('reading_file.txt')

# 'w' é gravar -> destrói tudo o que está no arquivo!
file_for_writing = open('writing_file.txt', 'w')

# 'a' é acrescentar -> adiciona aldo ao final do arquivo
file_for_appending = open('appendign_file.txt', 'a')

# não esqueça de fechar os arquivos quando acabar
file_for_reading.close()
file_for_reading2.close()
file_for_writing.close()
file_for_appending.close()

# como é fácil esquecer de fechar os arquivos, sempre utilizar um bloco with, pois serão fechados automaticamente ao final
with open(filename) as f:
   data = function_that_gets_data_from(f)

# nesse ponto, f já foi fechado, então nem tente usá-lo
process(data)

# Para ler um arquivo inteiro basta iterar nas linhas usando um for
import regex as re

starts_with_hash = 0

with open('input.txt') as f:
   for line in f:                                  # analise cada linha do arquivo
      if re.match("^#",line):                      # use um regex para determinar se começa com '#'
         starts_with_hash += 1                     # se sim, adicione 1 à contagem

def get_domain(email_address: str) ->str:
   """Divida em '@' e retorne o último trecho"""
   return email_address.lower().split("@")[-1]

# dois testes
assert get_domain('joelgrus@gmail.com') == 'gmail.com'
assert get_domain('joel@m.datasciencester.com') == 'm.datasciencester.com'

from collections import Counter
with open('email_adresses.txt', 'r') as f:
   domain_counts = Counter(get_domain(line.strip())
                           for line in f
                           if '@' in line)
   
# Arquivos Delimitados
   
with open('tab_delimited_stock_prices.txt', 'w') as f:      # escreve a data, a ação e o preço
    f.write("""6/20/2014\tAAPL\t90.91
6/20/2014\tMSFT\t41.68
6/20/2014\tFB\t64.5
6/19/2014\tAAPL\t91.86
6/19/2014\tMSFT\t41.51
6/19/2014\tFB\t64.34
""")

def process(date: str, symbol: str, closing_price: float) -> None:
    # Imaginge that this function actually does something.
    assert closing_price > 0.0


import csv

with open('tab_delimited_stock_prices.csv') as f:
   tab_reader = csv.reader(f, delimiter='\t')
   for row in tab_reader:
      date = row[0]
      symbol = row[1]
      closing_price = float(row[2])
      process(date, symbol, closing_price)

# se o arquivo tiver cabeçalhos:
with open('colon_delimited_stock_prices.txt', 'w') as f:
    f.write("""date:symbol:closing_price
6/20/2014:AAPL:90.91
6/20/2014:MSFT:41.68
6/20/2014:FB:64.5
""")

with open('colon_delimited_stock_prices.txt') as f:
   colon_reader = csv.DictReader(f, delimiter=':')
   for dict_row in colon_reader:
      date = dict_row["date"]
      symbol = dict_row["symbol"]
      closing_price = float(dict_row["closing_price"])
      process(date, symbol, closing_price)

# da mesma forma é possível gravar os dados delimitados usando o csv.writer:
today_prices = {'AAPL': 90.91, 'MSFT': 41.68, 'FB': 64.5}

with open('comma_delimited_stock_prices.txt', 'w') as f:
   csv_writer = csv.writer(f, delimiter=',')
   for stock, price in today_prices.items():
      csv_writer.writerow([stock, price])

# o csv.writer sempre funciona quando os campos têm vírgulas, Um editor de texto convencional, não. Por exemplo:
results = [["test1", "sucess", "Monday"],
           ["test2", "sucess, kind of", "Tuesday"],
           ["test3", "failure, kind of", "Wednesday"],
           ["test4", "failure, utter", "Thursday"]]

# não faça isso!
with open('bad_csv.txt', 'w') as f:
   for row in results:
      f.write(",".join(map(str, row)))       # talvez tenha muitas vírgulas!
      f.write("\n")                          # a linha também pode ter muitas newlines!

# Extraindo Dados da Internet

# HTML e Análise de Dados
from bs4 import BeautifulSoup
import requests

url = ("https://raw.githubusercontent.com/joelgrus/data/master/getting-data.html")
html = requests.get(url).text
soup = BeautifulSoup(html, 'html5lib')

# Por exemplo: ara encontrar a primeira tag <p> (e seu conteúdo)
first_paragraph = soup.find('p')    # ou apenas soup.p
first_paragraph_text = soup.p.text
first_paragraph_words = soup.p.text.split()

first_paragraph_id = soup.p['id']            # gera KeyError se não houver 'id'
first_paragraph_id2 = soup.p.get('id')       # retorna None se não houver 'id'

# obter múltiplas tags ao mesmo tempo:
all_paragraphs = soup.find_all('p')          # ou apenas soup('p')
paragraphs_with_ids = [p for p in soup('p') if p.get('id')]

importants_paragraphs = soup('p', {'class' : 'importatn'})
importants_paragraphs2 = soup('p', 'important')
importants_paragraphs3 = [p for p in soup('p')
                          if 'important' in p.get('class', [])]