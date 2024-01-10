users = [ # data dump em que cada user é representado por um dict com seu id e name
    {"id":0, "name": "Hero"},
    {"id":1, "name": "Dunn"},
    {"id":2, "name": "Sue"},
    {"id":3, "name": "Chi"},
    {"id":4, "name": "Thor"},
    {"id":5, "name": "Clive"},
    {"id":6, "name": "Hicks"},
    {"id":7, "name": "Devin"},
    {"id":8, "name": "Kate"},
    {"id":9, "name": "Klein"},
]

friendships_pairs = [ # pares de id's representando as conexões, amizades.
    (0,1),(0,2),(1,2),(1,3),(2,3),(3,4),(4,5),(5,6),(5,7),(6,8),(7,8),(8,9)
]

# inicializando o dict(dicionário) com uma lista vazia para cada id de usuário
friendships = {
    user["id"]: [] for user in users
}
# loop pelos pares de amigo para preencher friednships:
for i, j in friendships_pairs:
    friendships[i].append(j) # Adiciona j como amigo de i
    friendships[j].append(i) # Adiciona i como amigo de j

# print(friendships)
def number_of_friends(user):
    """Quantos amigos tem o _user_?"""
    user_id = user["id"]
    friend_ids = friendships[user_id]
    return len(friend_ids)


total_connections = sum(number_of_friends(user) for user in users)

# print(total_connections)

num_users = len(users)
avg_connections = total_connections / num_users # conexões médias

# print(avg_connections)

# Criar uma lista (user_id, number_of_friends)
num_friends_by_id = [(user["id"], number_of_friends(user))
                     for user in users]

num_friends_by_id.sort(key=lambda id_and_friends: id_and_friends[1], # ordena a lista decrescentemente pela quantidade de conexões
                       reverse=True)

from collections import Counter # não é carregado por padrão

def friends_of_friends(user): # contagem de amigos em comum do user. Retorna o id do não amigo e a contagem de conexões em comum
        # """foaf significa 'friend of a friend' [amigo de um amigo]"""
    user_id = user["id"]
    return Counter(
        foaf_id
        for friend_id in friendships[user_id]    # Para cada amigo meu
        for foaf_id in friendships[friend_id]    # encontre os amigos deles
        if foaf_id != user_id                    # que não sejam eu
        and foaf_id not in friendships[user_id]  # e não sejam meus amigos
    )

# print(friends_of_friends(users[3])) # Counter({0: 2, 5: 1})

#CIENTISTAS DE DADOS QUE TALVEZ VOCÊ CONHEÇA:

interests = [
    (0,"Hadoop"),(0,"Big Data"),(0,"HBase"),(0,"Java"),
    (0,"Spark"),(0,"Storm"),(0,"Cassandra"),
    (1,"NoSQL"),(1,"MongoDB"),(1,"Cassandra"),(1,"HBase"),
    (1, "Postgres"),(2,"Python"),(2,"scikit-learn"),(2,"scipy"),
    (2,"numpy"),(2,"statsmodels"),(2,"pandas"),(3,"R"),(3,"Python"),
    (3,"statistics"),(3,"regression"),(3,"probability"),
    (4,"machine learning"),(4,"regression"),(4,"decision trees"),
    (4,"libsvm"),(5,"Python"),(5,"R"),(5,"Java"),(5,"C++"),
    (5,"Haskell"),(5,"programming languages"),(6,"statistics"),
    (6,"probability"),(6,"mathematics"),(6,"theory"),
    (7,"machine learning"),(7,"scikit-learn"),(7,"Mahout"),
    (7,"neural networks"),(8,"neural networks"),(8,"deep learning"),
    (8,"Big Data"),(8,"artificial intelligence"),(9,"Hadoop"),
    (9,"Java"),(9,"MapReduce"),(9,"Big Data")
]

# construir uma função para encontrar usuários com o mesmo interesse
def data_scientists_who_like(target_interest):
    """Encontre os ids dos usuários com o mesmo interesse"""
    return [user_id
    for user_id, user_interest in interests
    if user_interest == target_interest]
# funciona mas tem que examinar a lista a cada busca

from collections import defaultdict

# As chaves são interesses, os valores são listas de user_ids com o interesse em questão
user_ids_by_interest = defaultdict(list) # função usada pra criar valores dict a partir de uma lista

for user_id, interest in interests:
    user_ids_by_interest[interest].append(user_id)


# As chaves são user_ids, os valores são listas de interesses do user_ids em questão
interests_by_user_id = defaultdict(list) # função usada pra criar valores dict a partir de uma lista

for user_id, interest in interests:
    interests_by_user_id[user_id].append(interest)

def most_common_interests_with(user):
    return Counter(
        interested_user_id
        for interest in interests_by_user_id[user["id"]]
        for interested_user_id in user_ids_by_interest[interest]
        if interested_user_id != user["id"]
    )

# print(most_common_interests_with(users[3]))

#SALÁRIOS E EXPERIÊNCIA

salaries_and_tenures = [
    (83000, 8.7), (88000, 8.1),
    (48000, 0.7), (76000, 6),
    (69000, 6.5), (76000, 7.5),
    (60000, 2.5), (83000, 10),
    (48000, 1.9), (63000, 4.2)
]

# # IMPRESSÃO DO GRÁFICO:
# salaries_and_tenures.sort(key=lambda salaries_and_tenures: salaries_and_tenures[1], # ordena a lista decrescentemente pelo tenure
#                        reverse=True)

# # Descompacte as tuplas em duas listas
# salaries, tenures = zip(*salaries_and_tenures)

# import matplotlib.pyplot as plt
# plt.plot(tenures, salaries, 'o')  # Adicionando marcadores " marker='o' " para melhor visualização
# plt.xlabel('Experiência (Anos)') # Rótulo eixo x
# plt.ylabel('Salário ($/Ano)') # Rótulo eixo y
# plt.title('Salário vs. Experiência') # Título do gráfico
# plt.show() # Exibir o gráfico

# As chaves são anos, os valores são listas de salários por anos de experiência.
salary_by_tenure = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    salary_by_tenure[tenure].append(salary)

# print(salary_by_tenure)

# As chaves são anos, cada valor é o salário médio associado ao número de anos de experiência.
average_salary_by_tenure = { # -> Essa informação não parece muito útil já que os funcionários não tem os mesmos anos de experiência. Talvez seja melhor fazer buckets(agrupamentos) de experiências
    tenure: sum(salaries) / len(salaries)
    for tenure, salaries in salary_by_tenure.items()
}

# print(average_salary_by_tenure)

def tenure_bucket(tenure):
    if tenure < 2:
        return "less than two"
    elif tenure < 5:
        return "between two and five"
    else:
        return "more than five"
    
# As chaves são buckets de anos de experiência, os valores são as listas de salary_by_tenure associadas ao bucket em questão
salary_by_tenure_bucket = defaultdict(list)

for salary, tenure in salaries_and_tenures:
    bucket = tenure_bucket(tenure)
    salary_by_tenure_bucket[bucket].append(salary)

# print(salary_by_tenure_bucket)

# E finalmente computamos a média salarial de cada bucket:

# As chaves são buckets de anos de experiência, os valores são a média salarial do bucket em questão
average_salary_by_bucket = {
    tenure_bucket: sum(salaries) / len(salaries)
    for tenure_bucket, salaries in salary_by_tenure_bucket.items()
}
# print(average_salary_by_bucket)

def predict_paid_or_unpaid(years_experience):
    if years_experience <3.0:
        return "paid"
    elif years_experience <8.5:
        return "unpaid"
    else:
        return "paid"

#encontrar os interesses mais populares:    
words_and_counts = Counter(word
                           for user, interest in interests
                           for word in interest.lower().split()) #escrever os intereseses em letras minúsculas e diviví-los em palavras

for word, count in words_and_counts.most_common(): #contagem de resultados
    if count > 1:
        print(word, count)