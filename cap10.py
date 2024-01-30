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
a_to_b = distance([160, 150], [170.2, 160])      # 14.28
a_to_c = distance([160, 150], [177.8, 171])      # 27.53
b_to_c = distance([170.2, 160], [177.8, 171])      # 13.37

from typing import Tuple

from cap04 import vector_mean
from cap05 import standard_deviation

def scale(data: List[Vector]) -> Tuple[Vector, Vector]:
   """retorna a média e o desvio-padrão de cada posição"""
   dim = len(data[0])

   means = vector_mean(data)
   stdevs = [standard_deviation([vector[i] for vector in data])
             for i in range(dim)]
   
   return means, stdevs

vectors = [[-3, -1, 1], [-1, 0, 1], [1, 1, 1]]
means, stdevs = scale(vectors)

assert means == [-1, 0, 1]
assert stdevs == [2, 1, 0]

# Agora, criamos um novo conjunto de dados:
def rescale(data: List[Vector]) -> List[Vector]:
   """
   Redimensiona os dados de entrada para que cada posição tenha média 0 e desvio-padrão
   1. (Deixa a posição como está se o desvio-padrão for 0)
   """
   dim = len(data[0])
   means, stdevs = scale(data)

   # Faça uma cópia de cada vetor
   rescaled = [v[:] for v in data]

   for v in rescaled:
      for i in range(dim):
         if stdevs[i] > 0:
            v[i] = (v[i] - means[i]) / stdevs[i]

   return rescaled

means, stdevs = scale(rescale(vectors))
assert means == [0, 0, 1]
assert stdevs == [1, 1, 0]

# alturas em polegadas
a_to_b = distance([63, 150], [67, 160])      # 10.77
a_to_c = distance([63, 150], [70, 171])      # 22.13
b_to_c = distance([67, 160], [70, 171])      # 11.40

vet_em_pol = [[63, 150], [67, 160], [70, 171]]

# alturas em centímetros
a_to_b = distance([160, 150], [170.2, 160])      # 14.28
a_to_c = distance([160, 150], [177.8, 171])      # 27.53
b_to_c = distance([170.2, 160], [177.8, 171])      # 13.37

vet_em_cm = [[160, 150], [170.2, 160], [177.8, 171]]

means_pol, stdevs_pol = scale(rescale(vet_em_pol))
means_cm, stdevs_cm = scale(rescale(vet_em_cm))

import tqdm
import random 

for i in tqdm.tqdm(range(100)):
   # faça algo devagar
   _ = [random.random() for _ in range(1000000)]

from typing import List
def primes_up_to(n: int) -> List[int]:
   primes = [2]

   with tqdm.trange(3, n) as t:
      for i in t:
         # i é primo se não for divisível por nenhum primo menor
         i_is_prime = not any(i % p == 0 for p in primes)
         if i_is_prime:
            primes.append(i)

         t.set_description(f"{len(primes)} primes")
   
   return primes

my_primes = primes_up_to(100_000) # o sublinhado serve somento como um separador para facilitar a visualização do número

# Redução de Dimensionalidade
# análise de componente principal (PCA)
pca_data = [
[20.9666776351559,-13.1138080189357],
[22.7719907680008,-19.8890894944696],
[25.6687103160153,-11.9956004517219],
[18.0019794950564,-18.1989191165133],
[21.3967402102156,-10.8893126308196],
[0.443696899177716,-19.7221132386308],
[29.9198322142127,-14.0958668502427],
[19.0805843080126,-13.7888747608312],
[16.4685063521314,-11.2612927034291],
[21.4597664701884,-12.4740034586705],
[3.87655283720532,-17.575162461771],
[34.5713920556787,-10.705185165378],
[13.3732115747722,-16.7270274494424],
[20.7281704141919,-8.81165591556553],
[24.839851437942,-12.1240962157419],
[20.3019544741252,-12.8725060780898],
[21.9021426929599,-17.3225432396452],
[23.2285885715486,-12.2676568419045],
[28.5749111681851,-13.2616470619453],
[29.2957424128701,-14.6299928678996],
[15.2495527798625,-18.4649714274207],
[26.5567257400476,-9.19794350561966],
[30.1934232346361,-12.6272709845971],
[36.8267446011057,-7.25409849336718],
[32.157416823084,-10.4729534347553],
[5.85964365291694,-22.6573731626132],
[25.7426190674693,-14.8055803854566],
[16.237602636139,-16.5920595763719],
[14.7408608850568,-20.0537715298403],
[6.85907008242544,-18.3965586884781],
[26.5918329233128,-8.92664811750842],
[-11.2216019958228,-27.0519081982856],
[8.93593745011035,-20.8261235122575],
[24.4481258671796,-18.0324012215159],
[2.82048515404903,-22.4208457598703],
[30.8803004755948,-11.455358009593],
[15.4586738236098,-11.1242825084309],
[28.5332537090494,-14.7898744423126],
[40.4830293441052,-2.41946428697183],
[15.7563759125684,-13.5771266003795],
[19.3635588851727,-20.6224770470434],
[13.4212840786467,-19.0238227375766],
[7.77570680426702,-16.6385739839089],
[21.4865983854408,-15.290799330002],
[12.6392705930724,-23.6433305964301],
[12.4746151388128,-17.9720169566614],
[23.4572410437998,-14.602080545086],
[13.6878189833565,-18.9687408182414],
[15.4077465943441,-14.5352487124086],
[20.3356581548895,-10.0883159703702],
[20.7093833689359,-12.6939091236766],
[11.1032293684441,-14.1383848928755],
[17.5048321498308,-9.2338593361801],
[16.3303688220188,-15.1054735529158],
[26.6929062710726,-13.306030567991],
[34.4985678099711,-9.86199941278607],
[39.1374291499406,-10.5621430853401],
[21.9088956482146,-9.95198845621849],
[22.2367457578087,-17.2200123442707],
[10.0032784145577,-19.3557700653426],
[14.045833906665,-15.871937521131],
[15.5640911917607,-18.3396956121887],
[24.4771926581586,-14.8715313479137],
[26.533415556629,-14.693883922494],
[12.8722580202544,-21.2750596021509],
[24.4768291376862,-15.9592080959207],
[18.2230748567433,-14.6541444069985],
[4.1902148367447,-20.6144032528762],
[12.4332594022086,-16.6079789231489],
[20.5483758651873,-18.8512560786321],
[17.8180560451358,-12.5451990696752],
[11.0071081078049,-20.3938092335862],
[8.30560561422449,-22.9503944138682],
[33.9857852657284,-4.8371294974382],
[17.4376502239652,-14.5095976075022],
[29.0379635148943,-14.8461553663227],
[29.1344666599319,-7.70862921632672],
[32.9730697624544,-15.5839178785654],
[13.4211493998212,-20.150199857584],
[11.380538260355,-12.8619410359766],
[28.672631499186,-8.51866271785711],
[16.4296061111902,-23.3326051279759],
[25.7168371582585,-13.8899296143829],
[13.3185154732595,-17.8959160024249],
[3.60832478605376,-25.4023343597712],
[39.5445949652652,-11.466377647931],
[25.1693484426101,-12.2752652925707],
[25.2884257196471,-7.06710309184533],
[6.77665715793125,-22.3947299635571],
[20.1844223778907,-16.0427471125407],
[25.5506805272535,-9.33856532270204],
[25.1495682602477,-7.17350567090738],
[15.6978431006492,-17.5979197162642],
[37.42780451491,-10.843637288504],
[22.974620174842,-10.6171162611686],
[34.6327117468934,-9.26182440487384],
[34.7042513789061,-6.9630753351114],
[15.6563953929008,-17.2196961218915],
[25.2049825789225,-14.1592086208169]
]

from cap04 import subtract, Vector, vector_mean

def de_mean(data: List[Vector]) -> List[Vector]:
   """Centraliza novamente os dados para que todas as dimensões tenham média 0"""
   mean = vector_mean(data)
   return [subtract(vector, mean) for vector in data]

from cap04 import magnitude

def direction(w: Vector) -> Vector:
   mag = magnitude(w)
   return [w_i / mag for w_i in w]

from cap04 import dot

def directional_variance(data: List[Vector], w: Vector) -> float:
   """Retorna a variação de x na direção de w"""
   w_dir = direction(w)
   return sum(dot(v, w_dir) ** 2 for v in data)

def directional_variance_gradient(data: List[Vector], w: Vector) -> Vector:
   """O gradiente da variação direcional em relação a w"""
   w_dir = direction(w)
   return [sum(2 * dot(v, w_dir) * v[i] for v in data)
           for i in range(len(w))]

from cap08 import gradient_step

def first_principal_component(dat: List[Vector],
                              n: int = 100,
                              step_size: float = 0.1 ) -> Vector:
   # comece com um valor aleatório:
   guess = [1.0 for _ in data[0]]

   with tqdm.trange(n) as t:
      for _ in t:
         dv = directional_variance(data, guess)
         gradient = directional_variance_gradient(data, guess)
         guess = gradient_step(guess, gradient, step_size)
         t.set_description(f"dv: {dv:.3f}")

   return direction(guess)

from cap04 import scalar_multiply

def project(v: Vector, w: Vector) -> Vector:
   """Retorne a projeção de v na direção w"""
   projection_lenght = dot(v,w)
   return scalar_multiply(projection_lenght, w)

from cap04 import subtract

def remove_projection_from_vector(v: Vector, w: Vector) -> Vector:
   """projeta v em w e subtrai o resultado de v"""
   return subtract(v, project(v, w))

def remove_projection(data: List[Vector], w: Vector) -> List[Vector]:
   return [remove_projection_from_vector(v,w) for v in data]

def pca(data: List[Vector], num_components: int) -> List[Vector]:
    components: List[Vector] = []
    for _ in range(num_components):
        component = first_principal_component(data)
        components.append(component)
        data = remove_projection(data, component)

    return components

def transform_vector(v: Vector, components: List[Vector]) -> Vector:
    return [dot(v, w) for w in components]

def transform(data: List[Vector], components: List[Vector]) -> List[Vector]:
    return [transform_vector(v, components) for v in data]

def main():

    # I don't know why this is necessary
    plt.gca().clear()
    plt.close()

    import random
    from cap06 import inverse_normal_cdf

    random.seed(0)

    # uniform between -100 and 100
    uniform = [200 * random.random() - 100 for _ in range(10000)]

    # normal distribution with mean 0, standard deviation 57
    normal = [57 * inverse_normal_cdf(random.random())
              for _ in range(10000)]

    plot_histogram(uniform, 10, "Uniform Histogram")



    plt.savefig('im/working_histogram_uniform.png')
    plt.gca().clear()
    plt.close()

    plot_histogram(normal, 10, "Normal Histogram")


    plt.savefig('im/working_histogram_normal.png')
    plt.gca().clear()

    from cap05 import correlation

    print(correlation(xs, ys1))      # about 0.9
    print(correlation(xs, ys2))      # about -0.9



    from typing import List

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

    # corr_data is a list of four 100-d vectors
    num_vectors = len(corr_data)
    fig, ax = plt.subplots(num_vectors, num_vectors)

    for i in range(num_vectors):
        for j in range(num_vectors):

            # Scatter column_j on the x-axis vs column_i on the y-axis,
            if i != j: ax[i][j].scatter(corr_data[j], corr_data[i])

            # unless i == j, in which case show the series name.
            else: ax[i][j].annotate("series " + str(i), (0.5, 0.5),
                                    xycoords='axes fraction',
                                    ha="center", va="center")

            # Then hide axis labels except left and bottom charts
            if i < num_vectors - 1: ax[i][j].xaxis.set_visible(False)
            if j > 0: ax[i][j].yaxis.set_visible(False)

    # Fix the bottom right and top left axis labels, which are wrong because
    # their charts only have text in them
    ax[-1][-1].set_xlim(ax[0][-1].get_xlim())
    ax[0][0].set_ylim(ax[0][1].get_ylim())

    # plt.show()



    plt.savefig('im/working_scatterplot_matrix.png')
    plt.gca().clear()
    plt.close()
    plt.clf()

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

    from typing import List

    def primes_up_to(n: int) -> List[int]:
        primes = [2]

        with tqdm.trange(3, n) as t:
            for i in t:
                # i is prime if no smaller prime divides it.
                i_is_prime = not any(i % p == 0 for p in primes)
                if i_is_prime:
                    primes.append(i)

                t.set_description(f"{len(primes)} primes")

        return primes

    my_primes = primes_up_to(100_000)



    de_meaned = de_mean(pca_data)
    fpc = first_principal_component(de_meaned)
    assert 0.923 < fpc[0] < 0.925
    assert 0.382 < fpc[1] < 0.384

if __name__ == "__main__": main()