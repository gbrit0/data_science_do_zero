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
### "Existe algo mais inútil ou menos útil do Álgebra?" - Billy Connolly

"Embora você não pense nos dados como vetores, essa é uma ótima forma de representar dados numéricos."

Matrizes podem representar relações binárias como as extremidades de uma rede como coleção de pares do caítulo 1. É possível criar uma matriz que será igual a 1 se os nós i e j estiverem conectados e 0 nos demais casos (binário)

### Dicas Aleatórias:
1 - Para limpar a tela dentro do IPython usar !CLS

2 - "Usar listas como vetores é bom como apresentação, mas terrível para o desempenho. No código de produção, use a bibliotreca NumPy, que contém uma classe array de alto desempenho com diversas operações aritméticas." 
[How to.](https://www.geeksforgeeks.org/how-to-create-a-vector-in-python-using-numpy/)

## Capítulo 5 - Estatística
### "Os fatos são teimosos, mas as estatísticas são mais maleáveis." - Mark Twain

*Outliers* são o problema da estatística kkk

## Capítulo 6 - Probabilidade
### "As leis da probabilidade, tão verdadeiras no plano geral, tão enganosas em casos específicos" - Edward Gibbon

Ocasionalmente é preciso inverter a normal_cdf para obter o valor correspondente à probabilidade especificada. Não existe uma forma simples de calcular mas como a normal_cdf é contínua e cresce estritamente, podemos usar uma [busca binária](https://pt.khanacademy.org/computing/computer-science/algorithms/binary-search/a/binary-search#:~:text=A%20busca%20bin%C3%A1ria%20%C3%A9%20um,de%20adivinha%C3%A7%C3%A3o%20no%20tutorial%20introdut%C3%B3rio.)

[Recomendação de Livro de Probabilidade](https://math.dartmouth.edu/~prob/prob/prob.pdf)

## Capítulo 7 - Hipótese e Inferência
### "É um traço da pessoa muito inteligente o dar crédito a estatísticas" - Georde Bernard Shaw

A ciência da Ciência de Dados muitas vezes formula e testa *hipóteses* sobre os dados e seus processos geradores.

Com base em diversas premissas, as estatísticas expressam observações de variáveis aleatórias em distribuições conhecidas que viabilizam declarações sobre a probabilidade de exatidão das premissas em questão.

## Capítulo 8 - Gradiente Descendente
### "Aqueles que se gabam das suas origens, enaltecem suas dívidas para com os outros." - Sêneca

Na maior parte das vezes que praticamos o data science queremos definir o melhor modelo para uma determinada situação. E, geralmente, o "melhor" é o que "minimiza o erro das previsões" ou "maximiza a probabilidade dos dados". Ou seja, é a solução para um tipo de problema de otimização.

Com o:
```python
from typing import TypeVar
T = TypeVar('T')
```
criamos uma função "genérica". Ele indica que o conjunto de dados pode ser uma lista qualquer tipo - str's, int's, list's etc - mas, em todos os casos, as saídas serão batches.

## Capítulo 9 - Obtendo Dados
### "Demorei três meses para escrever, três minutos para criar e a vida inteira para coletar os dados." - F. Scott Fitzgerald

...Em caso de emergência, você pode inserir os dados pessoalmente **(ou, melhor ainda, pedir para os estagiários)**...

Ao executar scripts na linha de comando é possível canalizar (pipe) os dados usando sys.stdin e sys.stdout

Nunca analise um arquivo separado por vírgulas por conta própria. Voc~e estragará os casos extremos!

Para extrair dados HTML, usare-mos a biblioteca [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/) v: 4.6.0 que constrói uma árvore com os vários elementos da página da web e fornece uma interface simples de acesso a eles. E usaremos também a biblioteca [Requests](https://requests.readthedocs.io/en/latest/) para fazer solicitações ao HTTP. e o analisador **html5lib**.

JSON são bem parecidos com os dicts do Python.

Às vezes, encontramos um provedor API de mau humor, que só fornece respostas em XML. A Beautiful Soup obtém dados do XML da mesma forma como extrai co HTML; mais informações na documentação.

#### Tópico "Usando o Twython" seria um ótimo tópico se a API do xwitter não fosse paga

O [Pandas](https://pandas.pydata.org/) é a biblioteca primária dos cientistas de dados - especialmente para importá-los;
O [Scrapy](https://scrapy.org/_) é um framework open source para extração de dados de sites de maneira rápida, simples e extensível;
O [Kaggle](https://www.kaggle.com/datasets/) hospeda muitos conjuntos de dados úteis para estudos.

## Capítulo 10 - Trabalhando com Dados
### "Em geral, os especialistas têm mais dados que discernimento" - Colin Powell

1. Primeiro *explore* os dados.

[Esse Data Set é legal.](https://www.kaggle.com/datasets/mdismielhossenabir/psychosocial-dimensions-of-student-life)

O primeiro passo é computar algumas estatísticas sumárias, como o número de pontos dados, o menor, o maior, a média e o desvio-padrão. **Mas isso não fornece uma boa compreensão, uma boa ideia é criar um histograma para agrupar dados em *buckets* discretos e contar os pontos em cada um deles**

[**Dataclasses**](https://docs.python.org/3/library/dataclasses.html) são úteis pois possibilitam modificar os valores em uma instância, ao contrário de tuples, [*namedtuples*](https://docs.python.org/3/library/collections.html#collections.namedtuple) ou [*NamedTuples*](https://docs.python.org/3/library/typing.html) porém não usaremos por conta da possibilidade de erros que causam.

Quando as dimensões não são comparáveis entre si, às vezes *redimensionamos* os dados para que cada dimensão tenha média 0 e desvio-padrão 1. Isso efetivamente elimina as unidades, pois converte as dimensões em "desvios-padrão da média"

A bilioteca *tqdm* gera barras de progresso personalizadas

## Capítulo 11 - Aprendizado de Máquina
### "Estou sempre disposto a aprender, mas nem sempre gosto de ser ensinado" - Winston Churchill

Um modelo é uma especificação de uma relação matemática (ou probabilística) entre diferentes variáveis.

*Aprendizado de Máquina* se refere à criação e ao uso de modelos que *aprendem com dados*.

Nosso objetivo é usar dados existentes para o desenvolvimento modelos e prever vários resultados para novos dados

### Sobreajuste e Subajuste

*Sobreajuste* - produzir um modelo que tem bom desempenho com os dados de treinamento, mas que generaliza mal os novos dados.

*Subajuste* - produzir um modelo que não tem um bom desempenho nem com os dados de treinamento.

Como criar modelos sem complexidade excessiva? R= usar diferentes dados para treinar e testar o modelo.

Escolher o modelo com o melhor desempenho no conjunto de testes é um metatreinamento que define o conjunto de testes como um segundo conjunto de treinamento. Nessa situação, divida os dados em três partes: um conjunto de treinamento para modelos em desenvolvimento, um conjunto de *validação* para escolher entre modelos treinados em um conjunto de testes para avaliar o modelos final.

### Correção

Não costumamos usar a "precisão" como critério para mediar a qualidade de um modelo.

### O Dilema Viés-Variância

Se o modelo não tem recursos suficientes para capturar irregularidades, incluir mais dados não resolverá essa situação.

### Extração e Seleção de Recursos

**Os recursos são as entradas que inserimos no modelo**

# Capítulo 12 - k-Nearest Neighbors
### "Se você quer irritar seus vizinhos, conte a verdade sobre eles." - Pietro Aretino

A classificação baseada nos vizinhos mais próximos é um dos modelos preditivos mais simples que existem. Os únicos necessários são:
    
* Alguma noção de distância;

* Uma hipótese sobre a semelhança entre pontos próximos.

O algoritmo k-Nearest Neighbors tem dificuldades com muitas dimensões devido à vastidão dos espações com muitas dimensões. Em espaços de alta dimensionalidade os pontos tendem a não estar próximos um do outro.

À medida que o número de dimensões aumenta, a distância média entre os pontos também aumenta. Entretanto, o fator mais problemático é a relação entre a distância mais próxima e a distância média.

Portanto, antes de usar a classificação baseada nos vizinhos mais próximos em muitas dimensões, é uma boa ideia fazer algum tipo de redução de dimensionalidade.

[Módulo do Scikit Learn para Vizinhos Próximos.](https://scikit-learn.org/stable/modules/neighbors.html)

# Capítulo 13 - Naive Bayes
### "É bom para o coração ser ingênuo e para mente não o ser" - Anatole France

 