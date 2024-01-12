# Data Science do Zero - Noções Fundamentais com Python

###  Exemplos práticos do livro "Data Science do Zero - Noções Fundamentais com Python 2ª ed." da editora O'REILLY

## Capítulo 1 - Introdução
"Dados! Dados! Dados!", espravejou, impaciente."Não posso fazer tijolos sem barro." - Arthur Conan Doyle

## Capítulo 2 - Um Curso Intensivo de Python

### The Zen of Python
    Beautiful is better than ugly.
    Explicit is better than implicit.
    Simple is better than complex.
    Complex is better than complicated.
    Flat is better than nested.
    Sparse is better than dense.
    Readability counts.
    Special cases aren't special enough to break the rules.
    Although practicality beats purity.
    Errors should never pass silently.
    Unless explicitly silenced.
    In the face of ambiguity, refuse the temptation to guess.
    There should be one-- and preferably only one --obvious way to do it.
    Although that way may not be obvious at first unless you're Dutch.
    Now is better than never.
    Although never is often better than *right* now.
    If the implementation is hard to explain, it's a bad idea.
    If the implementation is easy to explain, it may be a good idea.
    Namespaces are one honking great idea -- let's do more of those!

## Capítulo 3 - Visualizando Dados
"Acredito que a visualização seja uma das formas mais poderossas de atingir metas pessoais." - Harvey Mackay

### Embora seja muito fácil criá-las, é bem difícil produzir boas visualizações!

matplotlib - para gráficos simples de barras, de linhas e de dispersão, funciona muito bem.
"Em gráficos de barra é um grande vacilo não iniciar o eixo y em 0 pois pode induzir a desorientação nos usuários"

## Capítulo 4 - Álgebra Linear
#### "Existe algo mais inútil ou menos útil do Álgebra?" - Billy Connolly

"Embora você não pense nos dados como vetores, essa é uma ótima forma de representar dados numéricos."

Matrizes podem representar relações binárias como as extremidades de uma rede como coleção de pares do caítulo 1. É possível criar uma matriz que será igual a 1 se os nós i e j estiverem conectados e 0 nos demais casos (binário)

### Dicas Aleatórias:
1 - Para limpar a tela dentro do IPython usar !CLS

2 - "Usar listas como vetores é bom como apresentação, mas terrível para o desempenho. No código de produção, use a bibliotreca NumPy, que contém uma classe array de alto desempenho com diversas operações aritméticas." 
[How to.](https://www.geeksforgeeks.org/how-to-create-a-vector-in-python-using-numpy/)

## Capítulo 5 - Estatística
#### "Os fatos são teimosos, mas as estatísticas são mais maleáveis." - Mark Twain

*Outliers* são o problema da estatística kkk

## Capítulo 6 - Probabilidade
#### "As leis da probabilidade, tão verdadeiras no plano geral, tão enganosas em casos específicos" - Edward Gibbon


Ocasionalmente é preciso inverter a normal_cdf para obter o valor correspondente à probabilidade especificada. Não existe uma forma simples de calcular mas como a normal_cdf é contínua e cresce estritamente, podemos usar uma [busca binária](https://pt.khanacademy.org/computing/computer-science/algorithms/binary-search/a/binary-search#:~:text=A%20busca%20bin%C3%A1ria%20%C3%A9%20um,de%20adivinha%C3%A7%C3%A3o%20no%20tutorial%20introdut%C3%B3rio.)

[Recomendação de Livro de Probabilidade](https://math.dartmouth.edu/~prob/prob/prob.pdf)

## Capítulo 7 - Hipótese e Inferência
#### "É um traço da pessoa muito inteligente o dar crédito a estatísticas" - Georde Bernard Shaw

A ciência da Ciência de Dados muitas vezes formula e testa *hipóteses* sobre os dados e seus processos geradores.

Com base em diversas premissas, as estatísticas expressam observações de variáveis aleatórias em distribuições conhecidas que viabilizam declarações sobre a probabilidade de exatidão das premissas em questão.
