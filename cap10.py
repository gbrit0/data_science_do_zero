#  TRABALHANDO COM DADOS

# Explorando dados Unidimensionais

from typing import List, Dict
from collections import Counter
import math

import matplotlib.pyplot as plt

def bucketize(point: float, bucket_size: float) -> float:
   """Coloque o ponto perto do próximo mínimo múltiplo de bucket_size"""
   return bucket_size * math.floor(point / bucket_size)

def make_histogram(points: List[float], bucket_size: float) -> Dict[float, int]:
   """Coloca os pontos em buckets e conta o número de pontos em cada bucket"""
   return Counter(bucketize(point, bucket_size) for point in points)

def plot_histogram(points: List[float], bucket_size: float, title: str = ""):
   histogram = make_histogram(points, bucket_size)
   plt.bar(histogram.keys(), histogram.values(), width=bucket_size)
   plt.title(title)
   plt.show()

import random
from cap06 import inverse_normal_cdf

def random_normal() -> float:
    """Returns a random draw from a standard normal distribution"""
    return inverse_normal_cdf(random.random())

random.seed(0)

# uniforme entre -100 e 100
uniform = [200 * random.random() - 100 for _ in range(10000)]

# distribuição normal com média 0, desvio-padrão 57
normal = [57 * inverse_normal_cdf(random.random())
          for _ in range(10000)]

plot_histogram(uniform, 10, "Histograma Uniforme")  

plot_histogram(normal, 10, "Histograma Normal")

# as duas distibuições têm pontos max e min muito diferentes mas
# determinar isso não é suficiente pra explicar a diferença entre os histogramas

# Duas Dimensões
def random_normal() -> float:
   """Retorna um ponto aleatório de uma distribuição normal padrão"""
   return inverse_normal_cdf(random.random())

xs = [random_normal() for _ in range(1000)]
ys1 = [ x + random_normal() / 2 for x in xs]
ys2 = [ -x + random_normal() / 2 for x in xs]

plot_histogram(ys1, 10, "ys1")
plot_histogram(ys2, 10, "ys2")

plt.scatter(xs, ys1, marker='.', color='black', label='ys1')
plt.scatter(xs, ys2, marker='.', color='gray', label='ys2')
plt.xlabel('xs')
plt.ylabel('ys')
plt.legend(loc=9)
plt.title("Distribuições conjuntas muito diferentes")
plt.show()

# a diferença entre as distribuições conjuntas é bem diferente e essa diferença também 
# aparece quando analisamos as correlações

from cap05 import correlation

print(correlation(xs, ys1))      # 0.9010493686379609
print(correlation(xs, ys2))      # -0.8920981526880033

# Muitas Dimensões
# ao lidar com muitas dimensões devemos determinar as relações entre elas
# uma abordagem é analisar a matriz de correlação
from cap04 import Matrix, Vector, make_matrix

def correlation_matrix(data: List[Vector]) -> Matrix:
   """
   Retorna a matriz len(data) x len(data), na qual a entrada (i, j) é a correlação entre data[i] e data[j]
   """
   def correlation_ij(i: int, j: int) -> float:
      return correlation(data[i], data[j])
   
   return make_matrix(len(data), len(data), correlation_ij)

# Just some random data to show off correlation scatterplots
num_points = 100

def random_row() -> List[float]:
   row = [0.0, 0, 0, 0]
   row[0] = random_normal()
   row[1] = -5 * row[0] + random_normal()
   row[2] = row[0] + row[1] + 5 * random_normal()
   row[3] = 6 if row[2] > -2 else 0
   return row

random.seed(0)
   # each row has 4 points, but really we want the columns
corr_rows = [random_row() for _ in range(num_points)]

corr_data = [list(col) for col in zip(*corr_rows)]

# corr_data é uma lista com quatro vetores 100-d
num_vectors = len(corr_data)
fig, ax = plt.subplots(num_vectors, num_vectors)

for i in range(num_vectors):
   for j in range(num_vectors):

      # Disperse a column_j no eixo x e a column_i no eixo y
      if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])

      # a menos que i == j, nesse caso, mostre o nome da séria
      else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                              xycoords = 'axes fraction',
                              ha='center', va='center')
         
      # Em seguida, oculte os rótulos dos eixos, exceto pelos gráficos à esquerda e na parte inferior
      if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
      if j > 0: ax[i][j].yaxis.set_visible(False)

# Corrija os rótulos dos eixos no canto superior esquerdo e no canto inferiro direito,
# pois só haverá texto nesses gráficos
ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
ax[0][0].set_ylim(ax[0][1].get_ylim())
plt.show()

# Usando NamedTuples
import datetime

stock_price = {'closing_price': 102.06,
               'date': datetime.date(204, 8, 29),
               'symbol': 'AAPL'}

from collections import namedtuple

StockPrice = namedtuple('StockPrice', ['symbol', 'date', 'closing_price'])
price =  StockPrice('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03

from typing import NamedTuple

class StockPrice(NamedTuple):
   symbol: str
   date: datetime.date
   closing_price: float

   def is_high_tech(self) -> bool:
      """Como é uma classe, também podemos adicionar métodos"""
      return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
   
price = StockPrice('MSFT', datetime.date(2016, 12, 14), 106.03)

assert price.symbol == 'MSFT'
assert price.closing_price == 106.03
assert price.is_high_tech()

# dataclasses
from dataclasses import dataclass

@dataclass
class StockPrice2:
   symbol: str
   date: datetime.date
   closing_price: float

   def is_high_tech(self) -> bool:
      """Como é uma classe, também podemos adicionar métodos"""
      return self.symbol in ['MSFT', 'GOOG', 'FB', 'AMZN', 'AAPL']
   
price2 = StockPrice2('MSFT', datetime.date(2018, 12, 14), 106.03)

assert price2.symbol == 'MSFT'
assert price2.closing_price == 106.03
assert price2.is_high_tech()

# divida as ações
price2.closing_price /= 2
print(f"{price2.closing_price}")
assert price2.closing_price == 53.015, f"{price2.closing_price}"

# como essa é uma classe regular, adicione novos campos da forma que quiser, o que a deixa sucetível a erros
price2.cosing_price = 75
print(price2)        # StockPrice2(symbol='MSFT', date=datetime.date(2018, 12, 14), closing_price=53.015)

# Limpando e Estruturando
# antes usávamos assim:
# closing_price = float(row[2])
# entretanto é possível reduzir a propensão a erros se a análise for feita em uma função testável

from dateutil.parser import parse

def parse_row(row: List[str]) -> StockPrice:
   symbol, date, closing_price = row
   return StockPrice(symbol=symbol,
                     date=parse(date).date(),
                     closing_price=float(closing_price))

# Agora teste a função
stock = parse_row(["MSFT", "2018-12-14", "106.03"])

assert stock.symbol == "MSFT"
assert stock.date == datetime.date(2018, 12, 14)
assert stock.closing_price == 106.03

# e se houver dados inválidos?

from typing import Optional
import re

def try_parse_row(row: List[str]) -> Optional[StockPrice]:
   symbol, date_, closing_price_ = row
   # Os símbolos das ações devem estar em letras maiúsculas
   if not re.match(r"[A-Z]+$", symbol):
      return None
   
   try:
      date = parse(date_).date()
   except ValueError:
      return None
   
   try:
      closing_price = float(closing_price_)
   except ValueError:
      return None
   
   return StockPrice(symbol, date, closing_price)

# Deve retornar None em caso de erros
assert try_parse_row(["MSFT0", "2018-12-14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12--14", "106.03"]) is None
assert try_parse_row(["MSFT", "2018-12-14", "x"]) is None

# mas deve retornar o mesmo que antes se os dados forem válidos
assert try_parse_row(["MSFT", "2018-12-14", "106.03"]) == stock

# podemos ler e retornar apenas as linhas válidas
import csv

data: List[StockPrice] = []

with open("comma_delimited_stock_prices.csv") as f:
   reader = csv.reader(f)
   for row in reader:
      maybe_stock = try_parse_row(row)
      if maybe_stock is None:
         print(f"skipping invalid row: {row}")
      else:
         data.append(maybe_stock)

# um bom próximo passo é procurar outliers usando as técnicas indicadas na seção "Explorando os Dados"
         
with open("comma_delimited_stock_prices.csv") as f:      # abra o arquivo
   reader = csv.reader(f)
   for row in reader:                                    # para cada linha
      data.append(parse(row[1]).year)                    # capture o valor do ano

plot_histogram(data, 10, "Histograma de anos")           # imprima um histograma dos anos em buckets de 10 anos

# Manipulando Dados
with open('stocks.csv') as f:
   reader = csv.DictReader(f)
   rows = [[row['Symbol'], row['Date'], row['Close']]
           for row in reader]
   
# skip header
maybe_data = [try_parse_row(row) for row in rows]

# Make sure they all loaded successfully:
assert maybe_data
assert all(sp is not None for sp in maybe_data)

# This is just to make mypy happy
data = [sp for sp in maybe_data if sp is not None]

max_aapl_price = max(stock_price.closing_price
                     for stock_price in data
                     if stock_price.symbol == "AAPL")

from collections import defaultdict

max_prices: Dict[str, float] = defaultdict(lambda: float('-inf'))

for sp in data:
   symbol, closing_price = sp.symbol, sp.closing_price
   if closing_price > max_prices[symbol]:
      max_prices[symbol] = closing_price

# agrupando os preços por símbolo
from typing import List
from collections import defaultdict

# colete os preços por símbolo:
import datetime

prices: Dict[str, List[StockPrice]] = defaultdict(list)

for sp in data:
   prices[sp.symbol].append(sp)

# Classifique os preços por data
prices = {symbol: sorted(symbol_prices)
          for symbol, symbol_prices in prices.items()}

def pct_change(yesterday: StockPrice, today: StockPrice) -> float:
    return today.closing_price / yesterday.closing_price - 1

class DailyChange(NamedTuple):
    symbol: str
    date: datetime.date
    pct_change: float

def day_over_day_changes(prices: List[StockPrice]) -> List[DailyChange]:
    """
    Assumes prices are for one stock and are in order
    """
    return [DailyChange(symbol=today.symbol,
                        date=today.date,
                        pct_change=pct_change(yesterday, today))
            for yesterday, today in zip(prices, prices[1:])]

all_changes = [change
               for symbol_prices in prices.values()
               for change in day_over_day_changes(symbol_prices)]

max_change = max(all_changes, key=lambda change: change.pct_change)
assert max_change.symbol == 'AAPL'
assert max_change.date == datetime.date(1997, 8, 6)
assert 0.33 < max_change.pct_change < 0.34

min_change = min(all_changes, key=lambda change: change.pct_change)
assert min_change.symbol == 'AAPL'
assert min_change.date == datetime.date(2000, 9, 29)
assert -0.52 < min_change.pct_change < -0.51

changes_by_month: List[DailyChange] = {month: [] for month in range(1, 13)}

for change in all_changes:
   changes_by_month[change.date.month].append(change)

avg_daily_change = {
   month: sum(change.pct_change for change in changes) / len(changes)
   for month, changes in changes_by_month.items()
}
# Outubro é o melhor mês
assert avg_daily_change[10] == max(avg_daily_change.values())

# Redimensionamento
from cap04 import distance

# alturas em polegadas
a_to_b = distance([63, 150], [67, 160])      # 10.77
a_to_c = distance([63, 150], [70, 171])      # 22.13
b_to_c = distance([67, 160], [70, 171])      # 11.40

# alturas em centímetros