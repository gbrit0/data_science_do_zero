users_interests = [
    ["Hadoop", "Big Data", "HBase", "Java", "Spark", "Storm", "Cassandra"],
    ["NoSQL", "MongoDB", "Cassandra", "HBase", "Postgres"],
    ["Python", "scikit-learn", "scipy", "numpy", "statsmodels", "pandas"],
    ["R", "Python", "statistics", "regression", "probability"],
    ["machine learning", "regression", "decision trees", "libsvm"],
    ["Python", "R", "Java", "C++", "Haskell", "programming languages"],
    ["statistics", "probability", "mathematics", "theory"],
    ["machine learning", "scikit-learn", "Mahout", "neural networks"],
    ["neural networks", "deep learning", "Big Data", "artificial intelligence"],
    ["Hadoop", "Java", "MapReduce", "Big Data"],
    ["statistics", "R", "statsmodels"],
    ["C++", "deep learning", "artificial intelligence", "probability"],
    ["pandas", "R", "Python"],
    ["databases", "HBase", "Postgres", "MySQL", "MongoDB"],
    ["libsvm", "regression", "support vector machines"]
]

# Recomendando as Opções Mais Populares
from collections import Counter

popular_interests = Counter(interest
                            for user_interests in users_interests
                            for interest in user_interests)

from typing import List, Tuple

def most_popular_new_interests(
      user_interests: List[str],
      max_results: int = 5 ) -> List[Tuple[str, int]]:
   suggestions = [(interest, frequency)
                  for interest, frequency in popular_interests.most_common()
                  if interest not in user_interests]
   return suggestions[:max_results]

# Filtragem Colaborativa Baseada no Usuário

# print(most_popular_new_interests(users_interests[1]))

unique_interests = sorted({interest
                           for user_interests in users_interests
                           for interest in user_interests})

# print(unique_interests)

def make_user_interest_vector(user_interests: List[str]) -> List[int]:
   """
   Ao receber uma lista de interesses, produza um vetor cujo elemento i seja 1,
   se unique_interests[i] estiver na lista, ou 0, nos outros casos
   """
   return [1 if interest in user_interests else 0
         for interest in unique_interests]

user_interests_vectors = [make_user_interest_vector(user_interests)
                          for user_interests in users_interests]

from cap21 import cosine_similarity

user_similarities = [[cosine_similarity(interest_vector_i, interest_vector_j)
                    for interest_vector_j in user_interests_vectors]
                    for interest_vector_i in user_interests_vectors]

# for row in user_similarity:
#    print(row)

assert 0.56 < user_similarities[0][9] < 0.58, "vários interesses compartilhados"
assert 0.18 < user_similarities[0][8] < 0.20, "apenas um interesse compartilhado"

def most_similar_users_to(user_id: int) -> List[Tuple[int, float]]:
   pairs = [(other_user_id, similarity)                                 # Encontre outros usuários
            for other_user_id, similarity in                            # com semelhança diferente de zero
            enumerate(user_similarities[user_id]) 
            if user_id != other_user_id and similarity > 0]
   return sorted(pairs,
                 key=lambda pair: pair[-1],                             # Classifique-os a partir do mais semelhante
                 reverse=True)

from collections import defaultdict
from typing import Dict

def user_based_suggestions(user_id: int,
                           include_current_interests: bool = False):
   # Some as semelhanças
   suggestions: Dict[str, float] = defaultdict(float)
   for other_user_id, similarity in most_similar_users_to(user_id):
      for interest in users_interests[other_user_id]:
         suggestions[interest] += similarity

   # Converta-as em uma lista classificada
   suggestions = sorted(suggestions.items(),
                        key=lambda pair: pair[-1], # weight
                        reverse=True)
   
   # E (talvez) exclua interesses existentes
   if include_current_interests:
      return suggestions
   else:
      return [(suggestion, weight)
              for suggestion, weight in suggestions
              if suggestion not in users_interests[user_id]]
   
# Filtragem Colaborativa Baseada em Itens

interest_user_matrix = [[user_interest_vector[j]
                         for user_interest_vector in user_interests_vectors]
                         for j, _ in enumerate(unique_interests)]

interest_similarities = [[cosine_similarity(user_vector_i, user_vector_j)
                          for user_vector_j in interest_user_matrix]
                          for user_vector_i in interest_user_matrix]

def most_similar_interests_to(interest_id: int):
   similarities = interest_similarities[interest_id]
   pairs = [(unique_interests[other_interest_id], similarity)
            for other_interest_id, similarity in enumerate(similarities)
            if interest_id != other_interest_id and similarity > 0]
   return sorted(pairs,
                 key=lambda pair: pair[-1],
                 reverse=True)

def item_based_suggestions(user_id: int,
                           include_current_interests: bool = False):
   # Some os interesses semelhantes
   suggestions = defaultdict(float)
   user_interest_vector = user_interests_vectors[user_id]
   for interest_id, is_interested in enumerate(user_interest_vector):
      if is_interested == 1:
         similar_interests = most_similar_interests_to(interest_id)
         for interest, similarity in similar_interests:
            suggestions[interest] += similarity

   # Classifique-os pelo peso
   suggestions = sorted(suggestions.items(),
                        key=lambda pair: pair[-1], reverse=True)
   
   if include_current_interests:
      return suggestions
   else:
      return [(suggestion, weight)
              for suggestion, weight in suggestions
              if suggestion not in users_interests[user_id]]