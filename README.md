# MVP - Classificação de Tráfego Malicioso (CICIDS2017)

# Projeto CICIDS2017 - Estrutura para GitHub

Este arquivo contém toda a documentação e estrutura do projeto para ser usado diretamente no GitHub, seguindo o checklist sugerido.

---

## 1. Definição do Problema

**Descrição do problema:**
Classificar o tráfego de rede como benigno ou malicioso usando o dataset CICIDS2017, que contém dados de fluxos de rede normais e ataques.

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
* Melhor solução encontrada: Random Forest, por sua robustez, acurácia e facilidade de implementação.

---


Autor: Carlos Eduardo Silva dos Santos
