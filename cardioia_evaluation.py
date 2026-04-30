import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Carregar modelo e dados
model = joblib.load('modelo_cardioia.pkl')
df = pd.read_csv('base_cardioia.csv')

# 1. Avaliação Visual
def plot_evaluation(model, df):
    X = df.drop('pico_risco', axis=1)
    y = df['pico_risco']
    
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusão - CardioIA')
    plt.ylabel('Real')
    plt.xlabel('Predito')
    plt.savefig('conf_matrix.png')
    print("Matriz de confusão salva em 'conf_matrix.png'.")

# 2. Simulação de Novo Paciente
def simulate_new_patient(model):
    # Simulação de um paciente com alto risco
    new_patient = pd.DataFrame([{
        'idade': 75,
        'frequencia_cardiaca': 110,
        'spo2': 88,
        'carga_sistema': 0.8,
        'disponibilidade_recursos': 0.4,
        'historico_cardiaco': 1
    }])
    
    prob = model.predict_proba(new_patient)[0][1]
    pred = model.predict(new_patient)[0]
    
    print("\n--- Simulação de Novo Paciente ---")
    print(new_patient.iloc[0])
    print(f"\nProbabilidade de Pico de Risco: {prob:.2%}")
    print(f"Classificação: {'Alto Risco' if pred == 1 else 'Baixo Risco'}")
    
    return prob, pred

if __name__ == "__main__":
    plot_evaluation(model, df)
    simulate_new_patient(model)
