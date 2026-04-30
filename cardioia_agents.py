import asyncio
import json
import os
import joblib
import pandas as pd
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()

from agents import (
    Agent,
    Runner,
    function_tool,
    handoff,
    set_default_openai_client,
    set_default_openai_api,
    set_tracing_disabled,
)

# ---------------------------------------------------------------------------
# Configuração do cliente OpenAI apontando para o endpoint compatível do Gemini
# ---------------------------------------------------------------------------
# A chave é lida do arquivo .env (GOOGLE_API_KEY)
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY or GOOGLE_API_KEY == "SUA_CHAVE_AQUI":
    print("[AVISO] Chave GOOGLE_API_KEY não configurada no arquivo .env.")
    print("        Edite o arquivo .env e insira sua chave antes de executar.")

gemini_client = AsyncOpenAI(
    api_key=GOOGLE_API_KEY or "placeholder",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    max_retries=10,   # Retry automático em caso de rate limit (429)
    timeout=120.0,    # Timeout generoso para aguardar retries
)

set_default_openai_client(gemini_client)
set_default_openai_api("chat_completions")  # Gemini não suporta Responses API
set_tracing_disabled(True)                  # Desabilita tracing (não suportado pelo Gemini)

# Modelo Gemini a ser utilizado pelos agentes
# Usando gemini-2.5-flash-lite (cota gratuita generosa)
GEMINI_MODEL = "gemini-2.5-flash-lite"

# ---------------------------------------------------------------------------
# Carregamento do modelo de ML treinado
# ---------------------------------------------------------------------------
ml_model = joblib.load("modelo_cardioia.pkl")

# ---------------------------------------------------------------------------
# Modelo Pydantic para validação de saída estruturada
# ---------------------------------------------------------------------------
class CardioIAOutput(BaseModel):
    probabilidade_prevista: str = Field(
        description="Probabilidade de pico de risco cardíaco calculada pelo modelo (ex: 90.00%)"
    )
    classificacao_risco: str = Field(
        description="Classificação de risco do paciente: 'Alto Risco' ou 'Baixo Risco'"
    )
    protocolos_sugeridos: list[str] = Field(
        description="Lista de protocolos médicos sugeridos com base na classificação de risco"
    )
    justificativa_clinica: str = Field(
        description="Breve justificativa clínica baseada nos dados do paciente e no resultado da análise"
    )

# ---------------------------------------------------------------------------
# Tools (ferramentas) dos agentes especializados
# ---------------------------------------------------------------------------
@function_tool
def predict_risk(
    idade: int,
    frequencia_cardiaca: int,
    spo2: int,
    carga_sistema: float,
    disponibilidade_recursos: float,
    historico_cardiaco: int,
) -> str:
    """Consulta o modelo preditivo de Machine Learning para gerar o score de risco
    cardíaco de um paciente com base em seus dados clínicos."""
    input_data = pd.DataFrame([{
        'idade': idade,
        'frequencia_cardiaca': frequencia_cardiaca,
        'spo2': spo2,
        'carga_sistema': carga_sistema,
        'disponibilidade_recursos': disponibilidade_recursos,
        'historico_cardiaco': historico_cardiaco,
    }])
    prob = ml_model.predict_proba(input_data)[0][1]
    classification = "Alto Risco" if prob > 0.6 else "Baixo Risco"
    result = {
        "probabilidade": f"{prob:.2%}",
        "classificacao": classification,
    }
    print(f"  [Tool predict_risk] Resultado: {result}")
    return json.dumps(result, ensure_ascii=False)


@function_tool
def get_protocols(classificacao_risco: str) -> str:
    """Consulta a base de protocolos médicos simulados e retorna os protocolos
    adequados com base na classificação de risco ('Alto Risco' ou 'Baixo Risco')."""
    protocolos = {
        "Alto Risco": [
            "Monitoramento contínuo em UTI.",
            "Administração imediata de antiagregantes plaquetários.",
            "Acionamento da equipe de hemodinâmica.",
        ],
        "Baixo Risco": [
            "Observação em enfermaria.",
            "Realização de ECG seriado.",
            "Avaliação cardiológica ambulatorial.",
        ],
    }
    result = protocolos.get(classificacao_risco, ["Protocolo não encontrado."])
    print(f"  [Tool get_protocols] Protocolos retornados: {result}")
    return json.dumps(result, ensure_ascii=False)

# ---------------------------------------------------------------------------
# Definição dos Agentes
# ---------------------------------------------------------------------------

# Agente Analista de Risco
agente_analista_risco = Agent(
    name="Agente Analista de Risco",
    instructions="""Você é o Agente Analista de Risco da CardioIA.
Sua função é receber os dados clínicos de um paciente e utilizar a ferramenta
predict_risk para consultar o modelo preditivo de Machine Learning.

Você DEVE seguir estes passos na ordem:
1. Chame a ferramenta predict_risk com os dados clínicos do paciente.
2. Informe o resultado (probabilidade e classificação).
3. IMEDIATAMENTE após informar o resultado, use a tool retornar_ao_orquestrador
   para devolver o controle ao Orquestrador.""",
    tools=[predict_risk],
    model=GEMINI_MODEL,
)

# Agente Especialista em Protocolos
agente_especialista_protocolos = Agent(
    name="Agente Especialista em Protocolos",
    instructions="""Você é o Agente Especialista em Protocolos da CardioIA.
Sua função é receber a classificação de risco de um paciente e utilizar a
ferramenta get_protocols para consultar a base de protocolos médicos simulados.

Você DEVE seguir estes passos na ordem:
1. Chame a ferramenta get_protocols com a classificação de risco recebida.
2. Informe os protocolos encontrados.
3. IMEDIATAMENTE após informar os protocolos, use a tool retornar_ao_orquestrador
   para devolver o controle ao Orquestrador.""",
    tools=[get_protocols],
    model=GEMINI_MODEL,
)

# Agente Orquestrador — coordena o fluxo via handoffs
# Definido APÓS os agentes especializados para poder referenciá-los

# Primeiro, criamos o orquestrador sem handoffs (será atualizado abaixo)
agente_orquestrador = Agent(
    name="Agente Orquestrador CardioIA",
    instructions="""Você é o Agente Orquestrador da CardioIA, responsável por coordenar
o fluxo completo de análise de risco cardíaco.

Ao receber os dados de um novo paciente, você DEVE seguir exatamente estas etapas na ordem:

ETAPA 1: Use a tool 'transferir_para_analista_risco' para acionar o Agente Analista de Risco.
ETAPA 2: Use a tool 'transferir_para_especialista_protocolos' para acionar o Agente Especialista em Protocolos.
ETAPA 3: Compile os resultados e gere a recomendação final estruturada com:
  - Probabilidade prevista
  - Classificação de risco
  - Protocolos sugeridos
  - Justificativa clínica

IMPORTANTE: Você DEVE começar pela ETAPA 1 imediatamente. Não responda nada antes de transferir.""",
    handoffs=[
        handoff(
            agent=agente_analista_risco,
            tool_name_override="transferir_para_analista_risco",
            tool_description_override="Transfere a conversa para o Agente Analista de Risco para calcular a probabilidade de pico de risco cardíaco do paciente. USE ESTA TOOL PRIMEIRO.",
        ),
        handoff(
            agent=agente_especialista_protocolos,
            tool_name_override="transferir_para_especialista_protocolos",
            tool_description_override="Transfere a conversa para o Agente Especialista em Protocolos para consultar os protocolos médicos adequados à classificação de risco.",
        ),
    ],
    model=GEMINI_MODEL,
)

# Adicionar handoffs de RETORNO nos agentes especializados → Orquestrador
agente_analista_risco.handoffs = [
    handoff(
        agent=agente_orquestrador,
        tool_name_override="retornar_ao_orquestrador",
        tool_description_override="Após calcular o risco, transfere de volta ao Orquestrador para continuar o fluxo e acionar o próximo agente. SEMPRE use esta tool após chamar predict_risk e informar o resultado.",
    )
]

agente_especialista_protocolos.handoffs = [
    handoff(
        agent=agente_orquestrador,
        tool_name_override="retornar_ao_orquestrador",
        tool_description_override="Após consultar os protocolos, transfere de volta ao Orquestrador para compilar a resposta final. SEMPRE use esta tool após chamar get_protocols e informar os protocolos.",
    )
]

# ---------------------------------------------------------------------------
# Execução principal do sistema multiagente
# ---------------------------------------------------------------------------
async def run_cardioia(patient_data: dict):
    """Executa o pipeline multiagente para análise de risco cardíaco."""
    print("=" * 60)
    print("   CARDIOIA - Sistema Preditivo Multiagente")
    print("=" * 60)
    print(f"\n[Entrada] Dados do novo paciente:")
    for key, value in patient_data.items():
        print(f"  - {key}: {value}")

    # Monta o prompt de entrada com os dados do paciente
    input_message = f"""Analise o seguinte paciente para risco cardíaco:

Dados clínicos do paciente:
- Idade: {patient_data['idade']}
- Frequência Cardíaca: {patient_data['frequencia_cardiaca']}
- SPO2: {patient_data['spo2']}
- Carga do Sistema: {patient_data['carga_sistema']}
- Disponibilidade de Recursos: {patient_data['disponibilidade_recursos']}
- Histórico Cardíaco: {patient_data['historico_cardiaco']}

Siga o fluxo completo: primeiro analise o risco, depois consulte os protocolos
e por fim gere a recomendação final com probabilidade, classificação, protocolos
e justificativa clínica."""

    print("\n[Orquestrador] Iniciando análise...")
    print("-" * 60)

    # Executa o sistema multiagente via Runner
    result = await Runner.run(
        starting_agent=agente_orquestrador,
        input=input_message,
    )

    print("-" * 60)
    print(f"\n[Sistema] Agente final: {result.last_agent.name}")

    # Histórico de mensagens — demonstra o fluxo entre agentes
    print("\n[Histórico de Mensagens]")
    history = result.to_input_list()
    for i, msg in enumerate(history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content and isinstance(content, str) and len(content) > 0:
            preview = content[:120] + "..." if len(content) > 120 else content
            print(f"  [{i}] {role}: {preview}")
        elif msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func_name = tc.get("function", {}).get("name", "?")
                print(f"  [{i}] {role}: chamou tool '{func_name}'")

    # Saída final
    print("\n" + "=" * 60)
    print("   RELATÓRIO FINAL DA CARDIOIA")
    print("=" * 60)
    print(result.final_output)

    return result


def main():
    """Ponto de entrada do sistema."""
    # Simulação de entrada de novo paciente
    patient_test = {
        'idade': 68,
        'frequencia_cardiaca': 95,
        'spo2': 92,
        'carga_sistema': 0.7,
        'disponibilidade_recursos': 0.5,
        'historico_cardiaco': 1,
    }

    result = asyncio.run(run_cardioia(patient_test))

    # Salvar log completo para documentação
    log_content = []
    log_content.append("=" * 60)
    log_content.append("   CARDIOIA - Log do Sistema Multiagente")
    log_content.append("=" * 60)
    log_content.append(f"\nDados do Paciente: {json.dumps(patient_test, indent=2)}")
    log_content.append(f"\nAgente Final: {result.last_agent.name}")
    log_content.append(f"\n{'=' * 60}")
    log_content.append("   RELATÓRIO FINAL")
    log_content.append("=" * 60)
    log_content.append(result.final_output)
    log_content.append(f"\n{'=' * 60}")
    log_content.append("   HISTÓRICO DE MENSAGENS")
    log_content.append("=" * 60)
    history = result.to_input_list()
    for i, msg in enumerate(history):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if content and isinstance(content, str):
            log_content.append(f"\n[{i}] {role}:\n{content}")
        elif msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                func_name = tc.get("function", {}).get("name", "?")
                func_args = tc.get("function", {}).get("arguments", "{}")
                log_content.append(f"\n[{i}] {role}: chamou tool '{func_name}' com args: {func_args}")

    with open("log_sistema.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(log_content))

    print("\n[Sistema] Log salvo em 'log_sistema.txt'.")


if __name__ == "__main__":
    main()
