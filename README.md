# MVP - Classificação de Tráfego Malicioso (CICIDS2017)


---

## 1. Definição do Problema

**Descrição do problema:**
O problema consiste em classificar fluxos de rede em classes “Benigno” ou diferentes tipos de ataques (DDoS, Botnet, Brute Force etc.), com base em métricas de rede como número de bytes enviados, duração do fluxo, flags, protocolo e outros atributos.

**Você tem premissas ou hipóteses sobre o problema? Quais?**
Hipótese 1: Existem padrões nos atributos do fluxo que permitem diferenciar tráfego benigno de ataques.
Hipótese 2: Alguns atributos terão maior relevância para a classificação, permitindo redução de dimensionalidade sem perda de performance.
Hipótese 3: Modelos clássicos de ML (Random Forest, Gradient Boosting) conseguem generalizar para novos dados.

**Que restrições ou condições foram impostas para selecionar os dados?**
Remoção de colunas irrelevantes (Flow ID, Source IP, Destination IP, Timestamp).
Manutenção da proporção de classes ao dividir em treino e teste.
Exclusão de fluxos com valores nulos ou inconsistentes.

**Descreva o seu dataset (atributos, imagens, anotações, etc).**
Atributos: Features numéricas que descrevem cada fluxo de rede (bytes, pacotes, duração, flags, protocolos).
Rótulo: Coluna Label indicando Benigno ou tipo de ataque.
Formato: CSV, linhas representam fluxos, colunas representam atributos.
Tamanho: Aproximadamente 2,8 milhões de registros (dependendo do subset usado).


**Premissas/Hipóteses:**

* É possível treinar modelos de ML que consigam diferenciar tráfego benigno de ataques.
* Alguns atributos são mais relevantes que outros na classificação.

**Restrições/Condições:**

* Apenas colunas relevantes serão utilizadas, descartando IDs, IPs e timestamps.
* O dataset será separado em treino e teste mantendo a proporção de classes.

**Descrição do Dataset:**

* Atributos: métricas de fluxo de rede, como duração, bytes enviados, protocolo, flags, etc.
* Alvo: coluna 'Label', indicando Benigno ou Tipo de Ataque.
* Formato: CSV, linhas representam fluxos de rede.

---

## 2. Preparação de Dados

* Separação treino/teste (80/20) com stratify para manter proporção de classes.
* Validação cruzada (StratifiedKFold) utilizada para otimização de hiperparâmetros.
* Transformações aplicadas: padronização de atributos numéricos com StandardScaler.
* Feature selection: colunas irrelevantes removidas, mantendo atributos mais representativos.

**Objetivo: Transformar os dados brutos em formato adequado para treinamento de modelos.**
1. Separe o dataset entre treino e teste (e validação, se aplicável).
Divisão: 80% treino, 20% teste
Estratégia: train_test_split com stratify=y para manter proporção das classes

2. Faz sentido utilizar um método de validação cruzada? Justifique se não utilizar.
Sim, é utilizado StratifiedKFold para validação cruzada, garantindo que cada fold mantenha a proporção de classes e permitindo otimização de hiperparâmetros robusta.

3. Verifique quais operações de transformação de dados são mais apropriadas.
Padronização (StandardScaler): Para que todos os atributos numéricos tenham média 0 e desvio padrão 1, evitando que atributos com grandes magnitudes dominem o modelo.
Label Encoding: Transformar a coluna de classes em valores numéricos.
Visualizações: Histogramas e boxplots para detectar outliers e distribuição de classes.

4. Refine a quantidade de atributos disponíveis (feature selection).
Remoção de atributos irrelevantes (IDs, IPs, timestamp).
Análise de correlação e importância de atributos usando Random Forest Feature Importances.

---

## 3. Modelagem e Treinamento

* **Algoritmos selecionados:** Random Forest (robusto e preciso), Logistic Regression (baseline), Gradient Boosting (alta performance), SVM (boa generalização).
* Ajuste inicial de hiperparâmetros realizado com GridSearchCV.
* Problemas de underfitting não foram observados.
* Hiperparâmetros foram otimizados via GridSearchCV, com validação cruzada.
* Possibilidade futura de ensembles combinando múltiplos modelos.

---

## 4. Avaliação de Resultados

* **Métricas utilizadas:** Accuracy, F1-score, Confusion Matrix, ROC-AUC.
* Treinamento feito com toda a base de treino e teste na base de teste.
* Resultados coerentes e sem overfitting significativo.
* Comparação de modelos permite identificar o melhor desempenho.
* Melhor solução encontrada: Random Forest, por sua robustez, acurácia e facilidade de implementação. Random Forest otimizado com GridSearchCV. Fácil de treinar e interpretar, alta acurácia, baixa tendência a overfitting. *Possível extensão: criar ensemble com Gradient Boosting e SVM para aumentar robustez.

---


Autor: Carlos Eduardo Silva dos Santos
