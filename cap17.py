from typing import List
import math


def entropy(class_probabilities: List[float]) -> float:
   """Caso haja uma lista de probabilidades de classe, calcula a entropia"""
   return sum(-p * math.log(p,2)
              for p in class_probabilities
              if p > 0)    # ignore probabilidades zero


from typing import Any
from collections import Counter


def class_probabilities(labels: List[Any]) -> List[float]:
   total_count =  len(labels)
   return [count / total_count
           for count in Counter(labels).values()]


def data_entropy(labels: List[Any]) -> float:
   return entropy(class_probabilities(labels))


def partition_entropy(subsets: List[List[Any]]) -> float:
   """Retorna a entropia dessa partição dos dados em subconjuntos"""
   total_count = sum(len(subset) for subset in subsets)

   return sum(data_entropy(subset) * len(subset) / total_count
              for subset in subsets)


from typing import NamedTuple, Optional

class Candidate(NamedTuple):
   level: str
   lang: str
   tweets: bool
   phd: bool
   did_well: Optional[bool] = None  # permita dados não rotulados


from typing import Dict, TypeVar
from collections import defaultdict

T = TypeVar('T')  # Tipo genérico para entradas

def partition_by(inputs: List[T], attribute: str) -> Dict[Any, List[T]]:
   """Particione as entradas em listas com base no atributo especificado"""
   partitions: Dict[Any, List[T]] = defaultdict(list)
   for input in inputs:
      key = getattr(input, attribute)     # valor do atributo especificado
      partitions[key].append(input)       # adicione a entrada à partição correta

   return partitions


def partition_entropy_by(inputs: List[Any],
                         attribute: str,
                         label_attribute: str) -> float:
   """Compute a entropia correspondente à partição especificada"""
   # as partições contêm as entradas
   partitions = partition_by(inputs, attribute)

   # mas o partition_entropy só precisa dos rótulos das classes
   labels = [[getattr(input, label_attribute) for input in partition]
             for partition in partitions.values()]
   
   return partition_entropy(labels)


# Definimos uma árvore como:
   #  Um leaf (que prevê um só valor); ou
   #  Um Split (que contém um atributo para orientar a divisão, sub-árvores
   #  para valores específicos desse atributo e, possivelmente, um valor padrão
   #  para indicar valores desconhecidos)


from typing import Union

class Leaf(NamedTuple):
   value: Any

class Split(NamedTuple):
   attribute: str
   subtrees: dict
   default_value: Any = None

DecisionTree = Union[Leaf, Split]

# Com a definição acima, a árvore de contratação fica da seguinte forma:

hiring_tree = Split('level', {   # primeiro considera 'level'
   'Junior': Split('phd', {      # se 'level' for 'Junior', analise 'phd'
   False: Leaf(True),            # se "phd" for False, preveja True
      True: Leaf(False)          # se for True, preveja False
   }),
   'Mid': Leaf(True),            # se "level" for "Mid", sempre preveja True
   'Senior': Split('tweets', {   # se "level" for "Senior", analise "tweets"
      False: Leaf(False),        # se "tweets" for False, preveja False
      True: Leaf(True)           # se "Tweets" for True, preveja True
   })
})


def classify(tree: DecisionTree, input: Any) -> Any:
   """Classifique a entrada usando a árvore de decisão indicada"""

   # Se for um nó folha, retorne seu valor
   if isinstance(tree, Leaf):
      return tree.value
   
   # Caso contrário, a árvore consiste em um atributo de divisão
   # e um dicionário cujas chaves são valores desse atributo
   # e cujos valores são sub-árvores que serão consideradas em seguida
   subtree_key = getattr(input, tree.attribute)

   if subtree_key not in tree.subtrees:         # se não houver sub-=arvore para a chave
      return tree.default_value                 # retorne o valor padrão
   
   subtree = tree.subtrees[subtree_key]         # Escolha a sub-árvore adequada
   return classify(subtree, input)              # e use-a para classificar a entrada


# CONSTRUIR A REPRESENTAÇÃO DA ÁRVORE A PARTIR DOS DADOS DE TREINAMENTO

def build_tree_id3(inputs: List[Any],
                   split_attributes: List[str],
                   target_attribute: str) -> DecisionTree:
   # Conte os rótulos especificados
   label_counts = Counter(getattr(input, target_attribute)
                          for input in inputs)
   most_commom_label = label_counts.most_common(1)[0][0]

   # Se houver só um rótulo, preveja esse rótulo
   if len(label_counts) == 1:
      return Leaf(most_commom_label)
   
   # Se não restar nenhum atributo de divisão, retorne o rótulo majoritário
   if not split_attributes:
      return Leaf(most_commom_label)
   
   # Caso contrário, divida pelo melhor atributo
   def split_entropy(attribute: str) -> float:
      """A função auxiliar para encontrar o melhor atributo"""
      return partition_entropy_by(inputs, attribute, target_attribute)
   
   best_attribute = min(split_attributes, key=split_entropy)

   partitions = partition_by(inputs, best_attribute)
   new_attributes = [a for a in split_attributes if a != best_attribute]

   # Construa recursivamente as sub-árvores
   subtrees = {attribute_value : build_tree_id3(subset,
                                                new_attributes,
                                                target_attribute)
               for attribute_value, subset in partitions.items()}
   
   return Split(best_attribute, subtrees, default_value=most_commom_label)


def main():

   assert entropy([1.0]) == 0
   assert entropy([0.5, 0.5]) == 1
   assert 0.81 < entropy([0.25, 0.75]) < 0.82, f"{entropy([0.25, 0.75])}"

   assert data_entropy(['a']) ==0
   assert data_entropy([True, False]) == 1
   assert data_entropy([3, 4, 4, 4]) == entropy([0.25, 0.75])

                     #  level     lang     tweets  phd  did_well
   inputs = [Candidate('Senior', 'Java',   False, False, False),
             Candidate('Senior', 'Java',   False, True,  False),
             Candidate('Mid',    'Python', False, False, True),
             Candidate('Junior', 'Python', False, False, True),
             Candidate('Junior', 'R',      True,  False, True),
             Candidate('Junior', 'R',      True,  True,  False),
             Candidate('Mid',    'R',      True,  True,  True),
             Candidate('Senior', 'Python', False, False, False),
             Candidate('Senior', 'R',      True,  False, True),
             Candidate('Junior', 'Python', True,  False, True),
             Candidate('Senior', 'Python', True,  True,  True),
             Candidate('Mid',    'Python', False, True,  True),
             Candidate('Mid',    'Java',   True,  False, True),
             Candidate('Junior', 'Python', False, True,  False)
            ]


   for key in ['level', 'lang', 'tweets', 'phd']:
      print(key, partition_entropy_by(inputs, key, 'did_well'))

   # level 0.6935361388961919       como a entropia mais baixa está em level, vamos criar uma sub-árvore para cada valor de level
   # lang 0.8601317128547441
   # tweets 0.7884504573082896
   # phd 0.8921589282623617
      
   # Todo candidato Mid(Intermediário) é True em did_well, a sub-árvore Mid é um nó folha que prevê True
   # já para Sêniors temos Trues e Falses, dividiremos novamente:
      
   senior_inputs = [input for input in inputs if input.level == 'Senior']

   for key in ['lang', 'tweets', 'phd']:
      print(key, partition_entropy_by(senior_inputs, key, 'did_well'))

   # lang 0.4
   # tweets 0.0   a próxima divisão deve ser por tweets pois quando True, sempre resultam em did_well True, enquanto tweets False
                  # resultam sempre em did_well False
   # phd 0.9509775004326938
      
   # Para Juniors(Novatos):
   junior_inputs = [input for input in inputs if input.level == 'Junior']

   for key in ['lang', 'tweets', 'phd']:
      print(key, partition_entropy_by(junior_inputs, key, 'did_well'))

   # lang 0.9509775004326938
   # tweets 0.9509775004326938
   # phd 0.0   True PhD resulta sempre em True e False PhD resulta sempre em false
      
   tree = build_tree_id3(inputs, ['level', 'lang', 'tweets', 'phd'], 'did_well')

   assert classify(tree, Candidate("Junior", "Java", True, False))      # Deve prever True
   assert not classify(tree, Candidate("Junior", "Java", True, True))   # Deve prever False

   # Também é possível aplicar a dados com valores inesperados

   # Deve prever True
   assert classify(tree, Candidate("Intern", "Java", True, True))

if __name__ == "__main__": main()