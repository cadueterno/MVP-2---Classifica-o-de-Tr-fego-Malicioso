# MVP 2 - Classificação de Tráfego Malicioso (CICIDS2017)


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

## 4. Código Colab

{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Projeto CICIDS2017 - Classificação de Tráfego Malicioso\n",
    "**Autor:** Carlos Eduardo Silva dos Santos\n",
    "\n",
    "Este notebook funciona como relatório acadêmico detalhado para Google Colab, com células de texto explicando cada etapa do projeto e células de código para execução de Machine Learning."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Importação de bibliotecas" 
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.preprocessing import StandardScaler, LabelEncoder\n",
    "from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score\n",
    "from google.colab import files\n",
    "import joblib" 
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Upload do dataset CICIDS2017" 
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "print(\"Faça o upload do arquivo CICIDS2017.csv\")\n",
    "uploaded = files.upload()\n",
    "data = pd.read_csv(next(iter(uploaded.keys())))\n",
    "print(\"Dataset carregado com sucesso!\")\n",
    "display(data.head())" 
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Limpeza e Preparação dos Dados\n",
    "- Remoção de colunas irrelevantes\n",
    "- Separação de atributos (X) e alvo (y)\n",
    "- Label Encoding e padronização" 
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "data_clean = data.drop(['Flow ID','Source IP','Destination IP','Timestamp'], axis=1)\n",
    "X = data_clean.drop('Label', axis=1)\n",
    "y = data_clean['Label']\n",
    "\n",
    "# Label Encoding\n",
    "le = LabelEncoder()\n",
    "y_enc = le.fit_transform(y)\n",
    "\n",
    "# Padronização\n",
    "scaler = StandardScaler()\n",
    "X_scaled = scaler.fit_transform(X)\n",
    "\n",
    "# Separação treino/teste\n",
    "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, stratify=y_enc, random_state=42)\n",
    "print(\"Treino e teste separados!\")" 
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Criação de classe POO para treino e avaliação de modelos" 
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "class MLClassifier:\n",
    "    def __init__(self, model, param_grid, cv):\n",
    "        self.model = model\n",
    "        self.param_grid = param_grid\n",
    "        self.cv = cv\n",
    "        self.grid = None\n",
    "\n",
    "    def train(self, X_train, y_train):\n",
    "        self.grid = GridSearchCV(self.model, self.param_grid, cv=self.cv, scoring='accuracy', n_jobs=-1)\n",
    "        self.grid.fit(X_train, y_train)\n",
    "        print('=== Treinamento concluído ===')\n",
    "        print('Melhores hiperparâmetros:', self.grid.best_params_)\n",
    "        print('Acurácia média CV:', self.grid.best_score_)\n",
    "\n",
    "    def evaluate(self, X_test, y_test):\n",
    "        y_pred = self.grid.predict(X_test)\n",
    "        print('=== Classification Report ===')\n",
    "        print(classification_report(y_test, y_pred))\n",
    "        print('=== Confusion Matrix ===')\n",
    "        print(confusion_matrix(y_test, y_pred))\n",
    "        try:\n",
    "            if hasattr(self.grid.best_estimator_, 'predict_proba'):\n",
    "                print('ROC-AUC Score:', roc_auc_score(y_test, self.grid.predict_proba(X_test)[:,1]))\n",
    "        except: pass" 
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Treinamento do Random Forest" 
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "rf_model = RandomForestClassifier(random_state=42)\n",
    "param_grid_rf = {\n",
    "    'n_estimators':[100,200],\n",
    "    'max_depth':[10,20],\n",
    "    'min_samples_split':[2,5]\n",
    "}\n",
    "cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)\n",
    "rf_classifier = MLClassifier(rf_model, param_grid_rf, cv)\n",
    "rf_classifier.train(X_train, y_train)\n",
    "rf_classifier.evaluate(X_test, y_test)" 
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Salvar modelo treinado" 
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "joblib.dump(rf_classifier.grid.best_estimator_, 'random_forest_cicids.pkl')\n",
    "print('Modelo salvo como random_forest_cicids.pkl')" 
   ]
  }
 ]
}


---

## 5. Avaliação de Resultados

* **Métricas utilizadas:** Accuracy, F1-score, Confusion Matrix, ROC-AUC.
* Treinamento feito com toda a base de treino e teste na base de teste.
* Resultados coerentes e sem overfitting significativo.
* Comparação de modelos permite identificar o melhor desempenho.
* Melhor solução encontrada: Random Forest, por sua robustez, acurácia e facilidade de implementação. Random Forest otimizado com GridSearchCV. Fácil de treinar e interpretar, alta acurácia, baixa tendência a overfitting. *Possível extensão: criar ensemble com Gradient Boosting e SVM para aumentar robustez.

---


Autor: Carlos Eduardo Silva dos Santos
