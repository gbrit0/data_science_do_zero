Tensor = list

from typing import List

def shape(tensor: Tensor) -> List[int]:
   """Função auxiliar para encontrar a forma do tensor"""
   sizes: List[int] = []
   while isinstance(tensor, list):
      sizes.append(len(tensor))
      tensor = tensor[0]

   return sizes

assert shape([1, 2, 3]) == [3]
assert shape([[1,2], [3,4], [5,6]]) == [3, 2]

def is_1d(tensor: Tensor) -> bool:
   """
   Se o tensor [0] é uma lista, é um tensor de ordem superior.
   Se não, o tensor é unidimensional (ou, seja, um vetor).
   """
   return not isinstance(tensor[0], list)

assert is_1d([1, 2, 3])
assert not is_1d([[1, 2], [3, 4]])

def tensor_sum(tensor: Tensor) -> float:
   """Soma todos os valores do tensor"""
   if is_1d(tensor):
      return sum(tensor)      # apenas uma lista de floats, use a soma do Python
   else:
      return sum(tensor_sum(tensor_i)     # Chame tensor_sum em cada linha
                 for tensor_i in tensor)  # e some esses resultados.
   
assert tensor_sum([1, 2, 3]) == 6
assert tensor_sum([[1, 2], [3, 4]]) == 10

from typing import Callable

def tensor_apply(f: Callable[[float], float], tensor: Tensor) -> Tensor:
   """Aplica f elementwise"""
   if is_1d(tensor):
      return [f(x) for x in tensor]
   else:
      return [tensor_apply(f, tensor_i) for tensor_i in tensor]
   
assert tensor_apply(lambda x: x + 1, [1, 2, 3]) == [2, 3, 4]
assert tensor_apply(lambda x: 2 * x, [[1, 2], [3, 4]]) == [[2, 4], [6, 8]]

def zeros_like(tensor: Tensor) -> Tensor:
   return tensor_apply(lambda _: 0.0, tensor)

assert zeros_like([1, 2, 3]) == [0, 0, 0]
assert zeros_like([[1, 2], [3, 4]]) == [[0, 0], [0, 0]]

def tensor_combine(f: Callable[[float, float], float],
                   t1: Tensor,
                   t2: Tensor) -> Tensor:
   """Aplica f aos elementos correspondentes de t1 e t2"""
   if is_1d(t1):
      return [f(x, y) for x, y in zip(t1, t2)]
   else:
      return [tensor_combine(f, t1_i, t2_i)
              for t1_i, t2_i in zip(t1, t2)]
   
import operator
assert tensor_combine(operator.add, [1, 2, 3], [4, 5, 6]) == [5, 7, 9]
assert tensor_combine(operator.mul, [1, 2, 3], [4, 5, 6]) == [4, 10, 18]

from typing import Iterable, Tuple

class Layer:
   """
   Nossas redes neurais serão compostas por Layers que sabem
   computar as entradas "para frente" e propragas gradientes "para trás"
   """
   def forward(self, input):
      """
      Observe que não há tipos. Não indicaremos expressamente os tipos
      de entradas que serão recebidos pelas camadas nem os tipos de saídas
      que elas retornarão
      """
      raise NotImplementedError
   
   def backward(self, gradient):
      """
      Da mesma forma, não indicaremos expressamente o formato do gradiente.
      Cabe ao usuário (eu) avaliar se está fazendo as coisas de forma razoável
      """
      raise NotImplementedError
   
   def params(self) -> Iterable[Tensor]:
      """
      Retorna o sparâmetros dessa camada. Como a implementação padrão não retorna naada,
      se houver uma camada sem parâmetros, você não precisará implementar isso
      """
      return()
   
   def grads(self) -> Iterable[Tensor]:
      """
      Retorna os gradientes na mesma ordem dos params()
      """
      return()
   

from cap18 import sigmoid

class Sigmoid(Layer):
   def forward(self, input: Tensor) -> Tensor:
      """
      Aplique sigmoid em todos os elementos do tensor de
      entrada e salve os resultados para usar na retropropragação.
      """
      self.sigmoids = tensor_apply(sigmoid, input)
      return self.sigmoids
   
   def backward(self, gradient: Tensor) -> Tensor:
      return tensor_combine(lambda sig, grad: sig * (1 - sig) * grad,
                            self.sigmoids, gradient)
   
import random
from cap06 import inverse_normal_cdf

def random_uniform(*dims: int) -> Tensor:
   if len(dims) == 1:
      return [random.random() for _ in range(dims[0])]
   else:
      return [random_uniform(*dims[1:]) for _ in range(dims[0])]
   
def random_normal(*dims: int,
                  mean: float = 0.0,
                  variance: float = 1.0) -> Tensor:
   if len(dims) == 1:
      return [mean + variance * inverse_normal_cdf(random.random())
              for _ in range(dims[0])]
   else:
      return [random_normal(*dims[1:], mean=mean, variance=variance)
              for _ in range(dims[0])]
   
assert shape(random_uniform(2, 3, 4)) == [2, 3, 4]
assert shape(random_normal(5, 6, mean=10)) == [5, 6]

def random_tensor(*dims: int, init: str = 'normal') -> Tensor:
   if init == 'normal':
      return random_normal(*dims)
   elif init == 'uniform':
      return random_uniform(*dims)
   elif init == 'xavier':
      variance = len(dims) / sum(dims)
      return random_normal(*dims, variance=variance)
   else:
      raise ValueError(f"unknown init: {init}")
   
from cap04 import dot

class Linear(Layer):
   def __init__(self,
                input_dim: int,
                output_dim: int,
                init: str = 'xavier') -> None:
      """
      Uma camada de neurônios output_dim com pesos input_dim (e um viés).
      """
      self.input_dim = input_dim
      self.output_dim = output_dim

      # self.w[o] representa os pesos do neurônio 'o'
      self.w = random_tensor(output_dim, input_dim, init=init)

      # self.b[o] representa o termo de viés do neurônio 'o'
      self.b = random_tensor(output_dim, init=init)

   def forward(self, input: Tensor) -> Tensor:
      # Salve a entrada pra usar na transmissão para trás (retropropagação)
      self.input = input

      # Retorne o vetor das saídas dos neurônios
      return [dot(input, self.w[o]) + self.b[o]
               for o in range(self.output_dim)]
   
   def backward(self, gradient: Tensor) -> Tensor:
      # Cada b[o] é adicionao ao output[o], indicando que o gradiente de b é igual ao gradiente de saída
      self.b_grad = gradient
      
      # Cada w[o][i] multiplica o input[i] e é adicionado ao output[o]
      # Portanto, seu gradiente é input[i] * gradient[o]
      self.w_grad = [[self.input[i] * gradient[o]
                        for i in range(self.input_dim)]
                        for o in range(self.output_dim)]
      
      # Cada input[i] multiplica o w[o][i] e é adicionado ao output[o]
      # Portanto, seu gradiente é a soma de w[o][i] * gradient[o] em todas as saídas
      return [sum(self.w[o][i] * gradient[o] for o in range(self.output_dim))
               for i in range(self.input_dim)]
   
   def params(self) -> Iterable[Tensor]:
      return [self.w, self.b]
   
   def grads(self) -> Iterable[Tensor]:
      return [self.w_grad, self.b_grad]
   
from typing import List

class Sequential(Layer):
   """
   Uma camada é uma sequência de outras camadas.
   Cabe a você avaliar se já coerência entre a saída de uma camada e a entrada da próxima
   """
   def __init__(self, layers: List[Layer]) -> None:
      self.layers = layers

   def forward(self, input):
      """Só avance a entrada pelas camadas em sequência"""
      for layer in self.layers:
         input = layer.forward(input)
      return input
   
   def backward(self, gradient):
      """Só retropropague o gradiente pelas camadas na sequência inversa"""
      for layer in reversed(self.layers):
         gradient = layer.backward(gradient)
      return gradient
   
   def params(self) -> Iterable[List]:
      """Só retorne os params de cada camada"""
      return (param for layer in self.layers for param in layer.params())
   
   def grads(self) -> Iterable[List]:
      """Só retorne os grads de cada camada"""
      return (grad for layer in self.layers for grad in layer.grads())
   
xor_net = Sequential([
   Linear(input_dim=2, output_dim=2),
   Sigmoid(),
   Linear(input_dim=2, output_dim=1),
   Sigmoid()
])

class Loss:
   def loss(self, predicted: Tensor, actual: Tensor) -> float:
      """Qual é a qualidade das previsões? (Os números maiores são os piores.)"""
      raise NotImplementedError
   
   def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
      """Como a perda muda à medida que mudam as previsões?"""
      raise NotImplementedError
   
class SSE(Loss):
   """A função de perda que computa a soma dos erros quadráticos"""
   def loss(self, predicted: Tensor, actual: Tensor) -> float:
      # Compute o tensor das diferenças quadráticas
      squared_error = tensor_combine(
         lambda predicted, actual: (predicted - actual) ** 2,
         predicted,
         actual
      )
      # E some tudo
      return tensor_sum(squared_error)
   
   def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
      return tensor_combine(
         lambda predicted, actual: 2 * (predicted - actual),
         predicted,
         actual
      )
   
class Optimizer:
   """
   O otimizador atualiza os pesos de uma camada (no local) usando informações conhecidas
   pela camada ou pelo otimizador (ou por ambos)
   """
   def step(self, layer: Layer) -> None:
      raise NotImplementedError
   
class GradientDescendent(Optimizer):
   def __init__(self, learning_rate: float = 0.1) -> None:
      self.lr = learning_rate

   def step(self, layer: Layer) -> None:
      for param, grad in zip(layer.params(), layer.grads()):
         """Atualize o param usando um passo de gradiente"""
         param[:] = tensor_combine(
            lambda param, grad: param - grad * self.lr,
            param,
            grad
         )

class Momentum(Optimizer):
   def __init__(self,
                learning_rate: float,
                momentum: float = 0.9) -> None:
      self.lr = learning_rate
      self.mo = momentum
      self.updates: List[Tensor] = []     # média móvel

   def step(self, layer: Layer) -> None:
      # Se não houver atualizaçãoes anteriores, comece com zeros
      if not self.updates:
         self.updates = [zeros_like(grad) for grad in layer.grads()]

      for update, param, grad in zip(self.updates,
                                     layer.params(),
                                     layer.grads()):
         # Aplique o momentum
         update[:] = tensor_combine(
            lambda u, g: self.mo * u + (1 - self.mo) * g,
            update,
            grad
         )

         # Em seguida, dê um passo de gradiente
         param[:] = tensor_combine(
            lambda p, u: p - self.lr * u,
            param,
            update
         )

import math 

def tanh(x: float) -> float:
   # Se x for muito grande ou muito pequeno, tanh será (essencialmente) 1 ou -1
   # Isso porque, por exemplo, math.exp(1000) gera um erro
   if x < -100: return -1
   elif x > 100: return 1

   em2x = math.exp(-2 * x)
   return ( 1 - em2x) / (1 + em2x)

class Tanh(Layer):
   def forward(self, input: Tensor) -> Tensor:
      # Salve a saída de tanh para usar na retrotransmissão
      self.tanh = tensor_apply(tanh, input)
      return self.tanh
   
   def backward(self, gradient: Tensor) -> Tensor:
      return tensor_combine(
         lambda tanh, grad: (1 - tanh ** 2) * grad,
         self.tanh,
         gradient
      )
   
class Relu(Layer):
   def forward(self, input: Tensor) -> Tensor:
      self.input = input
      return tensor_apply(lambda x: max(x,0), input)
   
   def backward(self, gradient: Tensor) -> Tensor:
      return tensor_combine(lambda x, grad: grad if x > 0 else 0,
                            self.input,
                            gradient)

from cap04 import Vector

def fizz_buzz_encode(x: int) -> Vector:
   if x % 15 == 0:
      return [0, 0, 0, 1]
   elif x % 5 == 0:
      return [0, 0, 1, 0]
   elif x % 3 == 0:
      return [0, 1, 0, 0]
   else:
      return [1, 0, 0, 0]
   
assert fizz_buzz_encode(2) == [1, 0, 0, 0]
assert fizz_buzz_encode(6) == [0, 1, 0, 0]
assert fizz_buzz_encode(10) == [0, 0, 1, 0]
assert fizz_buzz_encode(30) == [0, 0, 0, 1]

from cap18 import binary_encode, fizz_buzz_encode, argmax

def fizz_buzz_accuracy(low: int, hi: int, net: Layer) -> float:
   num_correct = 0
   for n in range(low, hi):
      x = binary_encode(n)
      predicted = argmax(net.forward(x))
      actual = argmax(fizz_buzz_encode(n))
      if predicted == actual:
         num_correct += 1

   return num_correct / (hi - low)

def softmax(tensor: Tensor) -> Tensor:
   """Softmax na última dimensão"""
   if is_1d(tensor):
      # Subtraia o maior valor para fins de estabilidade numérica.
      largest = max(tensor)
      exps = [math.exp(x - largest) for x in tensor]

      sum_of_exps = sum(exps)             # Este é o peso total
      return [exp_i / sum_of_exps         # A probabilidade é a fração
              for exp_i in exps]          # do peso total
   else:
      return [softmax(tensor_i) for tensor_i in tensor]


class SoftmaxCrossEntropy(Loss):
   """
   Esse é o log de verossimilhança negativa dos valores observados neste modelo da rede neural.
   Portanto, se escolhermos os pesos para minimizá-lo, o modelo maximizará a verossimilhanca dos dados observados.
   """
   def loss(self, predicted: Tensor, actual: Tensor) -> float:
      # Aplique o softmax para obter as probabilidades
      probabilities = softmax(predicted)

      # Este será o log p_i para a classe i real e 0 para as outras classes
      # Adicionamos uma pequena quantidade a p para evitar o log(0).
      likelihoods = tensor_combine(lambda p, act: math.log(p + 1e-30) * act,
                                    probabilities,
                                    actual)
      
      # Em seguida, somamos os negativos
      return -tensor_sum(likelihoods)
   
   def gradient(self, predicted: Tensor, actual: Tensor) -> Tensor:
      probabilities = softmax(predicted)

      return tensor_combine(lambda p, actual: p - actual,
                            probabilities,
                            actual)


class Dropout(Layer):
   def __init__(self, p: float) -> None:
      self.p = p
      self.train = True

   def forward(self, input: Tensor) -> Tensor:
      if self.train:
         # Crie uma máscara de 0s e 1s com o formato da entrada usando a probabilidade especificada
         self.mask = tensor_apply(
            lambda _: 0 if random.random() < self.p else 1,
            input
         )
         # Multiplique pela máscara para desligar as entradas
         return tensor_combine(
            operator.mul,
            input,
            self.mask
         )
      else:
         # Durante a avaliação, reduza as saídas uniformemente
         return tensor_apply(lambda x: x * (1 - self.p), input)
   
   def backward(self, gradient: Tensor) -> Tensor:
      if self.train:
         # Propague apenas os gradientes em que a máscara == 1
         return tensor_combine(operator.mul, gradient, self.mask)
      else:
         raise RuntimeError("don t call backward when not in train mode")
      
def one_hot_encode(i: int, num_labels: int = 10) -> List[float]:
   return [1.0 if j == i else 0.0 for j in range(num_labels)]

assert one_hot_encode(3) == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
assert one_hot_encode(2, num_labels=5) == [0, 0, 1, 0, 0]

import tqdm

def loop(model: Layer,
         images: List[Tensor],
         labels: List[Tensor],
         loss: Loss,
         optimizer: Optimizer = None) -> None:
   correct = 0                                     # Monitore o número de previsões corretas
   total_loss = 0.0                                # Monitore a perda total

   with tqdm.trange(len(images)) as t:
      for i in t:
         predicted = model.forward(images[i])               # Preveja
         if argmax(predicted) == argmax(labels[i]):         # Verifique a correção
            correct += 1
         total_loss += loss.loss(predicted, labels[i])      # Compute a perda

         # Se estiver treinando, faça a retropropagação do gradiente e atualize os pesos
         if optimizer is not None:
            gradient = loss.gradient(predicted, labels[i])
            model.backward(gradient)
            optimizer.step(model)

         # E atualize as métricas na barra de progresso
            avg_loss = total_loss / (i + 1)
            acc = correct / (i + 1)
            t.set_description(f"mnist loss: {avg_loss:.3f} acc: {acc:.3f}")



def main():

   # Voltando ao XOR

   # dados de treinamento
   xs = [[0., 0], [0., 1], [1., 0], [1., 1]]
   ys = [[0.], [1.], [1.], [0.]]

   random.seed(0)

   net = Sequential([
      Linear(input_dim=2, output_dim=2),
      Sigmoid(),
      Linear(input_dim=2, output_dim=1)
   ])

   import tqdm

   optimizer = GradientDescendent(learning_rate=0.1)
   loss = SSE()

   with tqdm.trange(3000) as t:
      for epoch in t:
         epoch_loss = 0.0

         for x, y in zip(xs, ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)
         
         t.set_description(f"xor loss {epoch_loss:.3f}")

   for param in net.params():
      print(param)

   xs = [binary_encode(n) for n in range (101, 1024)]
   ys = [fizz_buzz_encode(n) for n in range(101, 1024)]

   NUM_HIDDEN = 25

   random.seed(0)

   net = Sequential([
      Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
      Tanh(),
      Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform'),
      Sigmoid()
   ])

   optimizer = Momentum(learning_rate=0.1, momentum=0.9)
   loss = SSE()

   with tqdm.trange(1000) as t:
      for epoch in t:
         epoch_loss = 0.0

         for x, y in zip(xs,ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

         accuracy = fizz_buzz_accuracy(101, 1024, net)
         t.set_description(f"fb loss: {epoch_loss:.2f} acc: {accuracy:.2f}")

   print("test results", fizz_buzz_accuracy(1, 101, net))

   net = Sequential([
      Linear(input_dim=10, output_dim=NUM_HIDDEN, init='uniform'),
      Tanh(),
      Linear(input_dim=NUM_HIDDEN, output_dim=4, init='uniform')
      # Sem camada sigmoide no final
   ])

   optimizer = Momentum(learning_rate=0.1, momentum=0.9)
   loss = SoftmaxCrossEntropy()

   with tqdm.trange(100) as t:
      for epoch in t:
         epoch_loss = 0.0

         for x, y in zip(xs,ys):
            predicted = net.forward(x)
            epoch_loss += loss.loss(predicted, y)
            gradient = loss.gradient(predicted, y)
            net.backward(gradient)

            optimizer.step(net)

         accuracy = fizz_buzz_accuracy(101, 1024, net)
         t.set_description(f"fb loss: {epoch_loss:.2f} acc: {accuracy:.2f}")

   print("test results", fizz_buzz_accuracy(1, 101, net))


   import mnist

   mnist.temporary_dir = lambda: 'C:\\Users\\User\\Documents\\data_sience_do_zero\\exemplos\\tmp'

   # Cada uma dessas funções baixa os dados e retorna um arrau numpy
   # Chamamos .tolist() porque os tensores são listas
   train_images = mnist.train_images().tolist()
   train_labels = mnist.train_labels().tolist()

   assert shape(train_images) == [60_000, 28, 28]
   assert shape(train_labels) == [60_000]

   import matplotlib.pyplot as plt
   
   fig, ax = plt.subplots(10,10)

   for i in range(10):
      for j in range(10):
         # Plote cada imagem em preto e branco e oculte os eixos
         ax[i][j].imshow(train_images[10 * i + j] , cmap='Greys')
         ax[i][j].xaxis.set_visible(False)
         ax[i][j].yaxis.set_visible(False)

   plt.show()

   test_images = mnist.test_images().tolist()
   test_labels = mnist.test_labels().tolist()

   assert shape(test_images) == [10_000, 28, 28]
   assert shape(test_labels) == [10_000]


   # Compute o valor médio do pixel
   avg = tensor_sum(train_images) / 60000 / 28 / 28

   # Centralize novamente, redimensiona e mescle
   train_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                   for image in train_images]
   test_images = [[(pixel - avg) / 256 for row in image for pixel in row]
                   for image in test_images]

   assert shape(train_images) == [60_000, 784], "images should be flattened"
   assert shape(test_images) == [10_000, 784], "images should be flattened"

   train_labels = [one_hot_encode(label) for label in train_labels]
   test_labels = [one_hot_encode(label) for label in test_labels]

   assert shape(train_labels) == [60000, 10]
   assert shape(test_labels) == [10000, 10]

if __name__ == "__main__": main()