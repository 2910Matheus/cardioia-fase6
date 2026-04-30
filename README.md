# CardioIA - Fase 6: Sistema Preditivo Multiagente para Eventos Cardíacos

## Integrantes do Grupo
*   **Bruno Henrique Nielsen Conter** (RM560518)
*   **Fabio Santos Cardoso** (RM560479)
*   **Matheus Augusto Rodrigues Maia** (RM560683)

Este repositório contém a implementação da Fase 6 do projeto CardioIA, que foca no desenvolvimento de um sistema preditivo multiagente para eventos cardíacos. O projeto integra um modelo de Machine Learning para previsão de risco com uma arquitetura multiagente baseada no **OpenAI Agents SDK**, utilizando o endpoint compatível do Google Gemini.

## Estrutura do Projeto

*   `cardioia_ml.py`: Script Python para geração da base de dados sintética e treinamento do modelo de Machine Learning.
*   `cardioia_evaluation.py`: Script Python para avaliação do modelo treinado e simulação de previsão para um novo paciente.
*   `modelo_cardioia.pkl`: Modelo de Machine Learning treinado e serializado.
*   `base_cardioia.csv`: Base de dados sintética gerada.
*   `conf_matrix.png`: Imagem da matriz de confusão do modelo.
*   `cardioia_agents.py`: Implementação do sistema multiagente com **OpenAI Agents SDK**, incluindo os agentes Analista de Risco, Especialista em Protocolos e Orquestrador, com handoffs, tools, histórico de mensagens e validação de saída.
*   `log_sistema.txt`: Exemplo de saída do sistema multiagente.
*   `relatorio_tecnico_parte1.md`: Relatório técnico detalhando o modelo preditivo.
*   `arquitetura_multiagente_parte2.md`: Documento de arquitetura descrevendo o sistema multiagente.
*   `arquitetura_multiagente_diagram.png`: Diagrama da arquitetura multiagente.
*   `cardioia_colab_notebook.ipynb`: Notebook Google Colab com a implementação completa da Parte 1 (modelo preditivo).
*   `README.md`: Este arquivo.

## Dependências

Para executar os scripts Python, você precisará das seguintes bibliotecas:

*   `pandas`
*   `numpy`
*   `scikit-learn`
*   `joblib`
*   `matplotlib`
*   `seaborn`
*   `openai` (cliente OpenAI para Python)
*   `openai-agents` (OpenAI Agents SDK)
*   `pydantic`

Você pode instalá-las usando pip:

```bash
pip install pandas numpy scikit-learn joblib matplotlib seaborn openai openai-agents pydantic python-dotenv
```

## Configuração

### Chave de API do Google Gemini

O sistema multiagente utiliza o **OpenAI Agents SDK** apontando para o endpoint compatível do Google Gemini. Para isso, é necessária uma chave de API do Google AI Studio (gratuita).

1.  Acesse [Google AI Studio](https://aistudio.google.com/) e gere uma chave de API.
2.  Abra o arquivo `.env` na raiz do projeto.
3.  Substitua `SUA_CHAVE_AQUI` pela chave que você acabou de gerar:
    ```env
    GOOGLE_API_KEY="AIzaSySuaChaveGerada..."
    ```

## Instruções de Execução

1.  **Geração de Dados e Treinamento do Modelo:**
    ```bash
    python cardioia_ml.py
    ```
    Este script irá gerar a base de dados sintética (`base_cardioia.csv`) e treinar o modelo (`modelo_cardioia.pkl`).

2.  **Avaliação do Modelo e Simulação:**
    ```bash
    python cardioia_evaluation.py
    ```
    Este script irá gerar a matriz de confusão (`conf_matrix.png`) e simular a previsão para um novo paciente.

3.  **Execução do Sistema Multiagente:**
    ```bash
    python cardioia_agents.py
    ```
    Este script demonstrará o fluxo de trabalho completo do sistema multiagente:
    - Recebe os dados do novo paciente.
    - O **Agente Orquestrador** coordena o fluxo via **handoffs**.
    - O **Agente Analista de Risco** consulta o modelo preditivo via **tool** `predict_risk`.
    - O **Agente Especialista em Protocolos** consulta a base de protocolos via **tool** `get_protocols`.
    - A resposta final é gerada de forma estruturada.
    - O histórico de mensagens e o log completo são salvos em `log_sistema.txt`.

## Arquitetura do Sistema Multiagente

O sistema utiliza o **OpenAI Agents SDK** com as seguintes funcionalidades:

| Funcionalidade | Implementação |
|---|---|
| **Agentes** | 3 agentes definidos com `Agent()`: Orquestrador, Analista de Risco, Especialista em Protocolos |
| **Handoffs** | Uso de `handoff()` para transferir controle entre o Orquestrador e os agentes especializados |
| **Tools** | `@function_tool` para `predict_risk` (modelo ML) e `get_protocols` (base de protocolos) |
| **Histórico de Mensagens** | Registrado via `result.to_input_list()` após execução pelo `Runner` |
| **Validação de Saída** | Modelo Pydantic `CardioIAOutput` para garantir formato estruturado |
| **LLM Backend** | Google Gemini (via endpoint OpenAI-compatível) |

## Vídeo Demonstrativo

Um vídeo demonstrando o fluxo completo do sistema (entrada do novo paciente → acionamento dos agentes → geração da resposta final) está disponível no YouTube (não listado):

[Assista ao Vídeo no YouTube](https://youtu.be/KvHgcYQLBPk)

