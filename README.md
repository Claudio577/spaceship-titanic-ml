# 🚀 Spaceship Titanic – Machine Learning

Projeto de **Machine Learning** desenvolvido a partir do desafio *Spaceship Titanic* do Kaggle, com o objetivo de prever quais passageiros foram transportados para uma dimensão alternativa após uma anomalia espaço-temporal.

O projeto combina **análise de dados**, **modelagem preditiva** e **deploy interativo** utilizando Streamlit.

---

## 🧠 Objetivo

Construir um modelo de classificação capaz de prever a variável **`Transported`** (`True` ou `False`) com base em características numéricas dos passageiros, como idade e gastos em serviços da nave.

---

## 📊 Dataset

- Fonte: Kaggle – *Spaceship Titanic*
- Link: https://www.kaggle.com/competitions/spaceship-titanic/data
- Registros: ~8.700 passageiros
- Tipo de problema: **Classificação binária**

### Variáveis utilizadas no modelo:
- `Age`
- `RoomService`
- `FoodCourt`
- `ShoppingMall`
- `Spa`
- `VRDeck`

> Observação: o projeto foca apenas em variáveis numéricas para simplificar o pipeline e tornar o modelo mais interpretável.

---

## ⚙️ Tecnologias Utilizadas

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- Streamlit  
- Git & GitHub  

---

## 🧪 Metodologia

1. **Carregamento e limpeza dos dados**
   - Tratamento de valores ausentes
   - Seleção de variáveis relevantes

2. **Modelagem**
   - Algoritmo: **Random Forest Classifier**
   - Separação treino/teste (80/20)
   - Avaliação com **Accuracy**

3. **Deploy**
   - Aplicação interativa criada com **Streamlit**
   - Modelo treinado automaticamente ao iniciar o app

---

## 📈 Resultados

- Acurácia média: ~**70%**
- O modelo gera previsões **probabilísticas**, refletindo a natureza não determinística dos dados reais.

> Importante: passageiros com altos gastos podem ou não ser transportados. O modelo aprende padrões estatísticos, não regras fixas.

---

## 🖥️ Aplicação Interativa

A aplicação permite simular diferentes perfis de passageiros e visualizar a previsão em tempo real.

### Funcionalidades:
- Entrada manual de dados do passageiro
- Previsão instantânea
- Interface simples e intuitiva

🔗 **App online (Streamlit):**  
> *(adicione aqui o link do seu app)*

---

## 📂 Estrutura do Projeto
