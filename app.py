import streamlit as st
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# CONFIGURAÇÃO DA PÁGINA
# -----------------------------
st.set_page_config(
    page_title="Spaceship Titanic - ML",
    page_icon="🚀",
    layout="centered"
)

st.title("🚀 Spaceship Titanic")
st.subheader("Previsão de passageiros transportados para outra dimensão")

st.markdown(
    """
    Este app utiliza **Machine Learning** para prever se um passageiro foi
    transportado após a colisão da nave com uma anomalia do espaço-tempo.
    """
)

# -----------------------------
# CAMINHO DOS DADOS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "train.csv"

# -----------------------------
# CARREGAR E TREINAR MODELO
# -----------------------------
@st.cache_resource
def train_model():
    df = pd.read_csv(DATA_PATH)

    features = [
        "Pclass", "Age", "SibSp", "Parch",
        "Fare", "RoomService", "FoodCourt",
        "ShoppingMall", "Spa", "VRDeck"
    ]

    df = df[features + ["Transported"]].copy()

    # Preenchendo valores ausentes
    df.fillna(0, inplace=True)

    X = df[features]
    y = df["Transported"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )
    model.fit(X_train, y_train)

    accuracy = accuracy_score(y_test, model.predict(X_test))
    return model, accuracy, features

model, accuracy, features = train_model()

st.success(f"✅ Modelo treinado com sucesso | Acurácia: **{accuracy:.2%}**")

# -----------------------------
# INTERFACE DE ENTRADA
# -----------------------------
st.markdown("## 🧑‍🚀 Dados do Passageiro")

col1, col2 = st.columns(2)

with col1:
    pclass = st.selectbox("Classe do Passageiro", [1, 2, 3])
    age = st.slider("Idade", 0, 80, 30)
    sibsp = st.number_input("Irmãos / Cônjuges a bordo", 0, 10, 0)
    parch = st.number_input("Pais / Filhos a bordo", 0, 10, 0)

with col2:
    fare = st.number_input("Tarifa paga", 0.0, 10000.0, 1000.0)
    roomservice = st.number_input("Gastos com Room Service", 0.0, 5000.0, 0.0)
    foodcourt = st.number_input("Gastos com Food Court", 0.0, 5000.0, 0.0)
    shoppingmall = st.number_input("Gastos com Shopping Mall", 0.0, 5000.0, 0.0)
    spa = st.number_input("Gastos com Spa", 0.0, 5000.0, 0.0)
    vrdeck = st.number_input("Gastos com VR Deck", 0.0, 5000.0, 0.0)

# -----------------------------
# PREVISÃO
# -----------------------------
if st.button("🔮 Prever Transporte"):
    input_data = pd.DataFrame([[
        pclass, age, sibsp, parch, fare,
        roomservice, foodcourt, shoppingmall, spa, vrdeck
    ]], columns=features)

    prediction = model.predict(input_data)[0]

    if prediction:
        st.error("🚀 **Passageiro TRANSPORTADO para outra dimensão**")
    else:
        st.success("🛑 **Passageiro NÃO transportado**")

# -----------------------------
# RODAPÉ
# -----------------------------
st.markdown("---")
st.caption(
    "Projeto de Machine Learning • Dataset: Kaggle Spaceship Titanic • "
    "Autor: Claudio Hideki Yoshida"
)
