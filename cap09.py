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
filename = 'nome_do_arquivo.extensao'
def function_that_gets_data_from():
   # suponha uma função de extração de dados
   return True

with open(filename) as f:
   data = function_that_gets_data_from(f)

# nesse ponto, f já foi fechado, então nem tente usá-lo
   
def process() -> None:
    # Imaginge that this function actually does something.
   assert 1 + 1 == 2

process(data)

# Para ler um arquivo inteiro basta iterar nas linhas usando um for
import re

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

# combinação dos métodos para um resultado mais elaborado:
# encontrar todos os elementos <span> contidos em um elemento <div>:

# Aviso: retornará o mesmo <span> várias vezes
# se ele estiver em vários <div>
# Ficar atento a isso:
spans_inside_divs = [span
                     for div in soup('div')                 # para cada <div> na página
                     for span in div('span')]               # encontre cada <span> dentro dele

# Exemplo: Monitorando o Congresso

url = "https://www.house.gov/representatives"
text = requests.get(url).text
soup = BeautifulSoup(text, "html5lib")

all_urls = [a['href']
            for a in soup('a')
            if a.has_attr('href')]

print(len(all_urls))                                        # 967, o que é demais

# Deve começar com http:// ou https://
# Deve terminar com .house.gov ou .house.gov/
regex = r"^https?://.*\.house\.gov/?$"

# Escrevendo alguns testes:
assert re.match(regex, "http://joel.house.gov")
assert re.match(regex, "https://joel.house.gov")
assert re.match(regex, "http://joel.house.gov/")
assert re.match(regex, "https://joel.house.gov/")
assert not re.match(regex, "joel.house.gov")
assert not re.match(regex, "http://joel.house.com")
assert not re.match(regex, "https://joel.house.gov/biography")

good_urls = [url for url in all_urls if re.match(regex, url)]

print(len(good_urls))                                       # 876

good_urls = list(set(good_urls))                            # set -> coleção de elementos únicos. Eliminar as duplicatas

print(len(good_urls))                                       # 438

html = requests.get('https://jayapal.house.gov').text
soup = BeautifulSoup(html, 'html5lib')

# Use um conjunto porque os links talvez surjam várias vezes
links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}

print(links)

from typing import Dict, Set

press_releases: Dict[str, Set[str]] = {}

for house_url in good_urls:
   html = requests.get(house_url).text
   soup = BeautifulSoup(html, 'html5lib')
   pr_links = {a['href'] for a in soup('a') if 'press releases' in a.text.lower()}
   print(f"{house_url}: {pr_links}")
   press_releases[house_url] = pr_links

# identificar comunicados de imprensa com a palavra "dados"
def paragraph_mentions(text: str, keyword: str) -> bool:
   """Retorna True se um <p> no texto menciona [keyword]"""
   soup = BeautifulSoup(text, 'html5lib')
   paragraphs = [p.get_text() for p in soup('p')]

   return any(keyword.lower() in paragraph.lower()
              for paragraph in paragraphs)

# um teste:
text = """<body><h1>Facebook</h1><p>Twitter</p>"""
assert paragraph_mentions(text, "twitter")
assert not paragraph_mentions(text, "facebook")

for house_url, pr_links in press_releases.items():
   for pr_link in pr_links:
      url = f"{house_url}/{pr_link}"
      text = requests.get(url).text

      if paragraph_mentions(text, 'data'):
         print(f"{house_url}")
         break    # fim da atividade em house_url

# Usando API's
# JSON e XML
      
# { "title" : "Data Science Book",
#   "author" : "Joel Grus",
#   "publicationYear" : 2019,
#   "topics" : [ "data", "science", "data science"] }
      
import json
serialized = """{ "title" : "Data Science Book",
  "author" : "Joel Grus",
  "publicationYear" : 2019,
  "topics" : [ "data", "science", "data science"] }"""

# analise o JSON para criar um dict do Python
deserialized = json.loads(serialized)
assert deserialized["publicationYear"] == 2019
assert "data science" in deserialized["topics"]

# <Book>
#    <Title>Data Science Book</Title>
#    <Author>Joel Grus</Author>
#    <PublicationYear>2014</PublicationYear>
#    <Topics>
#       <Topic>data</Topic>
#       <Topic>science</Topic>
#       <Topic>data science</Topic>
#    </Topics>
# </Book>
import requests, json
github_user = "gbrit0"
endpoint = f"https://api.github.com/users/{github_user}/repos"

repos = json.loads(requests.get(endpoint).text)    # list de dicts

# com isso é possível definir os meses e dias da semana com mais probabilidade de se criar um repositório, por exemplo
# porém as datas na resposta são strings -> é preciso tratar
from collections import Counter
from dateutil.parser import parse

dates = [parse(repo["created_at"]) for repo in repos]
month_counts = Counter(date.month for date in dates)
weekday_counts = Counter(date.weekday() for date in dates)

last_5_repositories = sorted(repos,
                             key=lambda r: r["pushed_at"],
                             reverse=True)[:5]

last_5_languages = [repo["language"]
                    for repo in last_5_repositories]

# ----------------------------------------------------------------------------------------------------------
# ---------------------------------------------- USANDO O TWYTHON ------------------------------------------
# ----------------------------------------------------------------------------------------------------------
import json

with open('credentials.json') as file:
   credentials = json.load(file)

CONSUMER_KEY = credentials.get('api_key')
CONSUMER_SECRET = credentials.get('api_key_secret')

# agora podemos criar uma instância do cliente:
import webbrowser
from twython import Twython

# configure um cliente temporário para recuperar uma URL de autenticação
temp_client = Twython(CONSUMER_KEY, CONSUMER_SECRET)
temp_creds = temp_client.get_authentication_tokens()
url = temp_creds['auth_url']

# agora, acesse a URL para autorizar o aplicativo e obter um PIN
print(f"go visit {url} and get the PIN code and paste it bellow")
webbrowser.open(url)
PIN_CODE = input("please enter the PIN code: ")

# agora, usamos o PIN_CODE para obter os tokens reais
auth_client = Twython(CONSUMER_KEY,
                      CONSUMER_SECRET,
                      temp_creds['oauth_token'],
                      temp_creds['oauth_token_secret'])

final_step = auth_client.get_authorized_tokens(PIN_CODE)
ACCESS_TOKEN = final_step['oauth_token']
ACCESS_TOKEN_SECRET = final_step['oauth_token_secret']

# e obter uma nova instância do Twython com eles
twitter = Twython(CONSUMER_KEY,
                  CONSUMER_SECRET,
                  ACCESS_TOKEN,
                  ACCESS_TOKEN_SECRET)

# salvando os tokens obtidos para não precisar criar uma instância de cliente novamente:

access_tokens = {
   "ACCESS_TOKEN": ACCESS_TOKEN,
   "ACCESS_TOKEN_SECRET": ACCESS_TOKEN_SECRET
   }

with open('access_tokens.json', 'w') as file:
   json.dump(access_tokens, file)

# com ums instância do Twython autenticada, podemos começar a fazer pesquisas:

# pesquise tweets que contenham a expressão 'data science'
for status in twitter.search(q='"data science"')["statuses"]:
   user = status["user"]["screen_name"]
   text = status["text"]
   with open('test.txt', 'w') as file:
      file.writelines({user: text})