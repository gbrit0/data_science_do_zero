from matplotlib import pyplot as plt 

anos = [1950, 1960, 1970, 1980, 1990, 2000, 2010]
pib = [300.2, 543.3, 1075.9, 2862.5, 5979.6, 10289.7, 14958.3]

#crie um gráfico de linhas, anos no eixo x, pib no eixo y
plt.plot(anos, pib, color='green', marker='o', linestyle='solid')

# adicionando título
plt.title("PIB Nominal")

# adicionando rótulo ao eixo y e x
plt.ylabel("Bilhões de Dólares")
plt.xlabel("Anos")
plt.show()

#GRÁFICO DE BARRAS PARA REPRESENTAR UM CONJUNTO DISCRETO DE ITENS
movies = ["Annie Hall", "Ben-Hur", "Casablanca", "Gandhi", "West Side Story"]
num_oscars = [5, 11, 3, 8, 10]

plt.bar(range(len(movies)), num_oscars)
plt.title("Meus Filmes Favoritos")        #adicionando um título
plt.ylabel("Nº de Oscars")                # rótulo do eixo y
plt.xticks(range(len(movies)), movies)    # rótulo do eixo x com os nomes dos filmes nos centros das barras

plt.show()

#GRÁFICO DE BARRAS TAMBÉM É BOM ARA REPRESENTAR DISTRIBUIÇÕES DE VALORES
from collections import Counter
grades = [83, 95, 91, 87, 70, 0, 85, 82, 100, 67, 73, 77, 0]

#agrupando por decil, mas com o 100 e o 90 agrupados
histogram = Counter(min(grade // 10 * 10, 90) for grade in grades)

plt.bar([x + 5 for x in histogram.keys()],      #Mova as barras para a direita em 5 un
        histogram.values(),                     #Atribua a altura correta a cada barra
        10,                                     #Atribua a largura 10 a cada barra
        edgecolor=(0, 0, 0))                    #Escureça as bordas das barras


plt.axis([-5, 105, 0, 5])     #-5 <= x <= 105 e 0 <= y <= 5

plt.xticks([10 * i for i in range(11)])         #rótulos do eixo x em 0, 10, ..., 100
plt.xlabel("Decil")
plt.ylabel("Nº de Alunos")
plt.title("Distribuição das Notas da 1ª Avaliação")
plt.show()

#CRITÉRIOS AO USAR O PLT.AXIS()
mentions = [500, 505]
years = [2017, 2018]

plt.bar(years, mentions, 0.8)
plt.xticks(years)
plt.ylabel("Nº de vezes que ouvi alguém dizer 'data science'")
plt.axis([2016.5, 2018.5, 499, 506]) #eixo "malandro"
plt.title("Observe Esse Aumento 'Imenso'")
plt.show()

plt.axis([2016.5, 2018.5, 0, 550]) # eixo correto
plt.title("O Aumento Diminuiu Bastante")
plt.show()

#GRÁFICOS DE LINHAS
variance= [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [2 ** x for x in [8, 7, 6, 5, 4, 3, 2, 1, 0]]
total_error = [ x + y for x, y in zip(variance, bias_squared)]
xs = [i for i, _ in enumerate(variance)]


#podemos fazer várias chamadas ao plt.plot para mostrar múltipas séries no mesmo gráfico
plt.plot(xs, variance, 'g-', label='variância')       #linha verde sólida
plt.plot(xs, bias_squared, 'r-.', label='viés^2')     #linha vermelha de ponto tracejado
plt.plot(xs, total_error, 'b:', label='erro total')   #linha pontilhada azul

# Como atribuímos rótulos a cada série (label=""), podemos criar uma legenda 'de graça'(loc=9 significa 'top center')
plt.legend(loc=9)
plt.xlabel("Complexidade do Modelo")
plt.xticks([])
plt.title("O Dilema Viés-Variância")
plt.show()

#GRÁFICOS DE DISPERSÃO
# melhor opção para representar as relações entre pares de conjunto de dados.
friends = [70, 65, 72, 63, 71, 64, 60, 64, 67]
minutes = [175, 170, 205, 120, 220, 130, 105, 145, 190]
labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

plt.scatter(friends, minutes)

# rotule cada ponto
for label, friend_count, minute_count in zip(labels, friends, minutes):
   plt.annotate(label,
               xy=(friend_count, minute_count),    #Coloque o rótulo no respectvivo
               xytext=(5, -5),                     # Mas levemente deslocado
               textcoords='offset points')
   
plt.title("Minutos DIários x Número de Amigos")
plt.xlabel("Nº de Amigos")
plt.ylabel("Minutos Diários Dedicados ao Site")
plt.show()

# se permitir o matplotlib selecionar a escala ao dispersas variáveis comparáveis, talvez obtenha um gráfico equivocado como a seguir:
test_1_grades = [99, 90, 85, 97, 80]
test_2_grades = [100, 85, 60, 90, 70]

plt.scatter(test_1_grades, test_2_grades)
plt.title("Eixos Não Comparáveis")
plt.xlabel("Notas do teste 1")
plt.ylabel("Notas do teste 2")
plt.show()

# quando incluímos uma chamada para plt.axis("equal") o gráfico mostra com mais precisão que a maior parte da variação acontece no teste 2
plt.scatter(test_1_grades, test_2_grades)
plt.title("Eixos Comparáveis")
plt.xlabel("Notas do teste 1")
plt.ylabel("Notas do teste 2")
plt.axis("equal")
plt.show()