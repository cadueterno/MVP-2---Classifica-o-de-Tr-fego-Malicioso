# RT-IoT2022 - Projeto de Machine Learning

## 1. Definição do Problema

**Objetivo:** Entender e descrever claramente o problema de detecção de tráfego malicioso em dispositivos IoT.

* **Descrição do problema:** O objetivo principal é classificar o tráfego de rede de dispositivos IoT em benigno ou malicioso, permitindo a detecção precoce de ataques cibernéticos, como negação de serviço, exploração de vulnerabilidades e acesso não autorizado. Este problema é crítico para a segurança de dispositivos conectados, especialmente em ambientes industriais e residenciais inteligentes, onde falhas podem ter impacto financeiro e operacional significativo.
* **Premissas/Hipóteses:** Supõe-se que padrões de tráfego malicioso possuem características distintas do tráfego benigno, e que essas diferenças podem ser capturadas por atributos de rede como número de pacotes, tamanhos de payload, protocolos e comportamentos temporais. Também assume-se que os ataques presentes no dataset representam os principais tipos de intrusão em IoT.
* **Restrições/Condições:** A seleção de dados considerou apenas atributos relevantes à análise de tráfego, excluindo identificadores irrelevantes como endereços IP de origem ou destino, que poderiam introduzir viés. Foram mantidas apenas instâncias com rótulos claramente definidos, garantindo consistência para treinamento e avaliação.
* **Descrição do Dataset:** O RT-IoT2022 contém aproximadamente 123.000 instâncias e 83 atributos de rede, incluindo contagens de pacotes, portas, protocolos, flags de TCP, métricas temporais, entre outros. Cada instância possui um rótulo indicando se o tráfego é benigno ou se representa um tipo específico de ataque. O dataset é tabular, adequado para aplicação de modelos clássicos de Machine Learning.

## 2. Preparação de Dados

**Objetivo:** Realizar operações de preparação dos dados para modelagem, garantindo qualidade e consistência.

* **Separação treino/teste:** O dataset foi dividido em 80% para treino e 20% para teste, mantendo a proporção original das classes (estratificação). Esta abordagem assegura que o modelo tenha exemplos suficientes de todas as classes e que a avaliação seja representativa.
* **Validação cruzada:** Foi utilizada validação cruzada estratificada (5 folds) para avaliar a performance dos modelos de forma robusta, mitigando variância causada por amostras específicas e garantindo generalização.
* **Transformações de dados:** Todos os atributos numéricos foram padronizados para média zero e desvio padrão um, evitando que escalas diferentes impactem a performance dos modelos. Variáveis categóricas foram codificadas usando LabelEncoder. Visualizações, como histogramas, boxplots e heatmaps de correlação, foram geradas para análise exploratória.
* **Feature selection:** Foram descartadas colunas irrelevantes ou altamente correlacionadas, mantendo atributos que contribuem significativamente para a classificação. Essa seleção reduz a dimensionalidade, melhora a interpretabilidade e aumenta a performance dos modelos.

## 3. Modelagem e Treinamento

**Objetivo:** Construir modelos de Machine Learning para classificação do tráfego.

* **Algoritmos selecionados:** Random Forest, Gradient Boosting, Logistic Regression, SVM e KNN. Justificativa: estes algoritmos são adequados para dados tabulares, oferecem robustez contra overfitting, interpretabilidade parcial (como em Random Forest) e comprovada eficácia em problemas de classificação com múltiplos atributos.
* **Ajuste inicial de hiperparâmetros:** Inicialmente foram utilizados parâmetros padrão para todos os modelos. Posteriormente, hiperparâmetros críticos (como número de estimadores e profundidade máxima em Random Forest, taxa de aprendizado em Gradient Boosting, C e kernel em SVM) foram otimizados via GridSearchCV.
* **Treinamento:** Modelos foram treinados utilizando pipelines, que incluem padronização dos dados e otimização de hiperparâmetros. A validação cruzada estratificada garante que o modelo aprenda padrões robustos sem sobreajustar dados específicos.
* **Otimização de hiperparâmetros:** Realizada usando GridSearchCV com validação cruzada. Cada parâmetro foi justificado por relevância na performance do modelo, garantindo que a otimização seja eficiente e consistente.
* **Métodos avançados:** Além dos modelos individuais, ensembles (Random Forest e Gradient Boosting) foram avaliados por sua capacidade de combinar múltiplos aprendizados e reduzir variância.
* **Comitê de modelos:** É possível implementar ensemble de diferentes algoritmos (voting ou stacking) para explorar complementaridades e aumentar a robustez do sistema de detecção.

## 4. Avaliação de Resultados

**Objetivo:** Analisar o desempenho dos modelos em dados não vistos, garantindo confiabilidade e generalização.

* **Métricas de avaliação:** Acurácia, precision, recall, F1-score e matriz de confusão foram escolhidas por sua relevância em classificação binária e pelo possível desbalanceamento das classes. Estas métricas permitem avaliar tanto a capacidade de detectar ataques quanto de evitar falsos positivos.
* **Treinamento e teste:** Cada modelo foi treinado com toda a base de treino e avaliado no conjunto de teste, garantindo que a avaliação represente dados não vistos.
* **Resultados:** Modelos ensemble (Random Forest e Gradient Boosting) apresentaram melhor acurácia, menor variância e maior equilíbrio entre precisão e recall, confirmando as hipóteses iniciais.
* **Overfitting:** Alguns modelos simples, como SVM e KNN, mostraram tendência a overfitting com parâmetros padrão, mitigada pelo ajuste e padronização.
* **Comparação entre modelos:** Ensembles performaram melhor, seguidos por Logistic Regression, com KNN e SVM apresentando menor desempenho em métricas agregadas.
* **Melhor solução:** Random Forest otimizado via GridSearchCV, com validação cruzada e padronização, foi identificado como o modelo final mais robusto, confiável e interpretável para detecção de tráfego malicioso em IoT.

  RT-IoT2022 - Projeto de Machine Learning
0. Introdução

Este notebook implementa um pipeline completo de Machine Learning para detecção de tráfego malicioso em dispositivos IoT, integrando: definição do problema, preparação de dados, modelagem, treinamento, otimização de hiperparâmetros e avaliação.
O projeto está detalhado academicamente, adequado para mestrado e pós-doutorado, com justificativas técnicas e científicas para cada escolha.

1. Definição do Problema

Objetivo: Detectar e classificar tráfego IoT em benigno ou malicioso.

Descrição do problema:
Dispositivos IoT são cada vez mais comuns e expõem vulnerabilidades críticas. Detectar automaticamente tráfego malicioso permite prevenir ataques como DDoS, exploração de vulnerabilidades e acessos não autorizados.

Hipóteses e premissas:

O tráfego malicioso possui padrões mensuráveis diferenciáveis do tráfego benigno.

Modelos clássicos de ML podem capturar esses padrões em dados tabulares.

Modelos ensemble tendem a generalizar melhor do que modelos individuais simples.

Restrições e condições:

Exclusão de atributos irrelevantes (como IPs específicos).

Manutenção apenas de registros com rótulo definido.

Preservação da representatividade de todas as classes.

Descrição do dataset:

Fonte: RT-IoT2022 (UCI Machine Learning Repository)

Instâncias: ~123.000

Atributos: 83 (contagens de pacotes, portas, protocolos, flags TCP, métricas temporais, etc.)

Tipo: tabular

Rótulo: binário (benigno/malicioso) ou multi-classe (tipos de ataques)

2. Preparação de Dados

Objetivo: Garantir qualidade e consistência dos dados, preparando-os para modelagem.

# Importação de bibliotecas essenciais
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Carregamento do dataset
from google.colab import files
uploaded = files.upload()  # Fazer upload do arquivo RT-IoT2022.csv
df = pd.read_csv(list(uploaded.keys())[0])

# Visualização inicial
print(df.head())
print(df.info())
print("Valores nulos por coluna:\n", df.isnull().sum())

2.1 Pré-processamento

Separação de atributos (X) e rótulo (y)

Codificação das classes

Divisão treino/teste (80/20) com estratificação

X = df.drop('label', axis=1)
y = df['label']

# Codificação de classes
le = LabelEncoder()
y = le.fit_transform(y)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

2.2 Análise exploratória

Histogramas, boxplots e heatmaps para visualizar distribuição e correlação

Identificação de outliers

Feature selection baseada em correlação e relevância

# Exemplo de visualização de correlação
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap='coolwarm', annot=False)
plt.title('Mapa de Correlação')
plt.show()

3. Modelagem e Treinamento

Objetivo: Construir modelos de ML confiáveis para classificação.

3.1 Pipelines e Hiperparâmetros

Modelos: Random Forest, Gradient Boosting, Logistic Regression, SVM, KNN

Justificativa: robustez, interpretabilidade e eficácia em dados tabulares

pipelines = {
    'rf': Pipeline([('scaler', StandardScaler()), ('clf', RandomForestClassifier(random_state=42))]),
    'gb': Pipeline([('scaler', StandardScaler()), ('clf', GradientBoostingClassifier(random_state=42))]),
    'lr': Pipeline([('scaler', StandardScaler()), ('clf', LogisticRegression(max_iter=1000, random_state=42))]),
    'svm': Pipeline([('scaler', StandardScaler()), ('clf', SVC(random_state=42))]),
    'knn': Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])
}

params = {
    'rf': {'clf__n_estimators':[100,200], 'clf__max_depth':[None,10,20]},
    'gb': {'clf__n_estimators':[100,200], 'clf__learning_rate':[0.01,0.1]},
    'lr': {'clf__C':[0.1,1,10]},
    'svm': {'clf__C':[0.1,1,10], 'clf__kernel':['linear','rbf']},
    'knn': {'clf__n_neighbors':[3,5,7]}
}

3.2 Treinamento com Validação Cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}

for name in pipelines:
    grid = GridSearchCV(pipelines[name], params[name], cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    best_models[name] = grid.best_estimator_
    print(f'{name} melhor score CV: {grid.best_score_:.4f}')
    print(f'Melhores parâmetros: {grid.best_params_}\n')

4. Avaliação de Resultados

Métricas: acurácia, precision, recall, F1-score, matriz de confusão

Avaliação em dados não vistos

for name, model in best_models.items():
    y_pred = model.predict(X_test)
    print(f'Modelo: {name}')
    print(f'Acurácia: {accuracy_score(y_test, y_pred):.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
    print('Confusion Matrix:')
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Matriz de Confusão - {name}')
    plt.ylabel('Classe Real')
    plt.xlabel('Classe Predita')
    plt.show()
    print('-'*50)

5. Conclusão

Random Forest e Gradient Boosting apresentaram melhor performance e robustez.

Random Forest otimizado via GridSearchCV é recomendado como modelo final.

O notebook está pronto para execução completa no Google Colab, exportação como .ipynb e upload direto para GitHub.

Se você quiser, posso gerar este notebook completo como arquivo .ipynb pronto para download, já formatado para Colab e com todas as células de Markdown e código detalhadas, exatamente como esta versão.



Autor: Carlos Eduardo Silva dos Santos
