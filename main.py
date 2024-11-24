import simpy
import numpy as np
from scipy.stats import expon, uniform
import pandas as pd
import matplotlib.pyplot as plt

# Listas para registrar os tempos e dados de interesse
chegadas, saidas = [], []
in_queue_manual, in_queue_auto, in_system = [], [], []
desistencias = 0
veiculos_em_atendimento = 0
fila_temporal_manual, fila_temporal_auto = [], []  # Para registrar a fila ao longo do tempo
tempo_trabalho_manuais, tempo_trabalho_automaticas = [], []  # Tempo de trabalho dos servidores


def registrar_estado_filas(env):
    """Registra o número de carros nas filas manual e automática ao longo do tempo."""
    while True:
        fila_temporal_manual.append((env.now, len(cabines_manuais.queue)))
        fila_temporal_auto.append((env.now, len(cabines_automaticas.queue)))
        yield env.timeout(1)  # Atualiza a cada 1 segundo


# Parâmetros gerais
TEMPO_DE_SIMULACAO = 3600
TRAFEGO_INTENSO = 1200 / 3600  # Chegada de veiculos por segundo
TRAFEGO_MODERADO = 600 / 3600
TRAFEGO_LEVE = 300 / 3600
CHANCE_AUTOMATICO = 0.1
CHANCE_DESISTENCIA = 0.02

# Configurações de cabines
CABINES_MANUAIS = 3
CABINES_AUTOMATICAS = 1
TEMPO_MANUAL_MIN = 10
TEMPO_MANUAL_MAX = 30
TEMPO_AUTOMATICO_MIN = 5
TEMPO_AUTOMATICO_MAX = 10


def distribuicao_chegada(taxa_chegada):
    """Gera o tempo de chegada entre veículos usando distribuição exponencial."""
    return expon.rvs(scale=1 / taxa_chegada)


def tempo_manual():
    """Tempo para passar por uma cabine manual (uniforme)."""
    return uniform.rvs(loc=TEMPO_MANUAL_MIN, scale=(TEMPO_MANUAL_MAX - TEMPO_MANUAL_MIN))


def tempo_automatico():
    """Tempo para passar por uma cabine automática (uniforme)."""
    return uniform.rvs(loc=TEMPO_AUTOMATICO_MIN, scale=(TEMPO_AUTOMATICO_MAX - TEMPO_AUTOMATICO_MIN))


def calcula_tempo_no_sistema(env, horario_chegada):
    """Registra o tempo total no sistema."""
    horario_saida = env.now
    saidas.append(horario_saida)
    tempo_total = horario_saida - horario_chegada
    in_system.append(tempo_total)


def atendimento_manual(env, veiculo_id, horario_chegada):
    global desistencias, veiculos_em_atendimento

    veiculos_em_atendimento += 1

    with cabines_manuais.request() as req:
        if np.random.rand() < CHANCE_DESISTENCIA:
            desistencias += 1
            veiculos_em_atendimento -= 1
            print(f"Veículo {veiculo_id} desistiu na fila manual em {env.now:.2f}s")
            return

        yield req
        inicio_atendimento = env.now
        yield env.timeout(tempo_manual())
        fim_atendimento = env.now

        # Registro do tempo de trabalho da cabine manual
        tempo_trabalho_manuais.append((inicio_atendimento, fim_atendimento))

        tempo_na_fila = inicio_atendimento - horario_chegada
        in_queue_manual.append(tempo_na_fila)

        print(f"Veículo {veiculo_id} atendido na cabine manual em {env.now:.2f}s")
        calcula_tempo_no_sistema(env, horario_chegada)

    veiculos_em_atendimento -= 1


def atendimento_automatico(env, veiculo_id, horario_chegada):
    global veiculos_em_atendimento, desistencias

    veiculos_em_atendimento += 1

    with cabines_automaticas.request() as req:
        if np.random.rand() < CHANCE_DESISTENCIA:
            desistencias += 1
            veiculos_em_atendimento -= 1
            print(f"Veículo {veiculo_id} desistiu na fila automática em {env.now:.2f}s")
            return

        yield req
        inicio_atendimento = env.now
        yield env.timeout(tempo_automatico())
        fim_atendimento = env.now

        # Registro do tempo de trabalho da cabine automática
        tempo_trabalho_automaticas.append((inicio_atendimento, fim_atendimento))

        tempo_na_fila = inicio_atendimento - horario_chegada
        in_queue_auto.append(tempo_na_fila)

        print(f"Veículo {veiculo_id} atendido na cabine automática em {env.now:.2f}s")
        calcula_tempo_no_sistema(env, horario_chegada)

    veiculos_em_atendimento -= 1


def chegada_de_veiculos(env, taxa_chegada):
    """Gera veiculos que chegam ao pedágio."""
    veiculo_id = 0

    while True:
        yield env.timeout(distribuicao_chegada(taxa_chegada))
        veiculo_id += 1
        horario_chegada = env.now
        chegadas.append(horario_chegada)

        print(f"Veículo {veiculo_id} chegou ao pedágio em {horario_chegada:.2f}s")

        # Decide se o veiculo vai para cabine automática ou manual
        if np.random.rand() < CHANCE_AUTOMATICO:
            env.process(atendimento_automatico(env, veiculo_id, horario_chegada))
        else:
            env.process(atendimento_manual(env, veiculo_id, horario_chegada))


# Configuração inicial
np.random.seed()  # Seed para reprodutibilidade

# Configurações do ambiente
env = simpy.Environment()
cabines_manuais = simpy.Resource(env, capacity=CABINES_MANUAIS)
cabines_automaticas = simpy.Resource(env, capacity=CABINES_AUTOMATICAS)

# Definir o tráfego: escolha intenso, moderado ou leve
TAXA_CHEGADA = TRAFEGO_MODERADO  # Alterar para TRAFEGO_INTENOS, TRAFEGO_MODERADO ou TRAFEGO_LEVE conforme necessário

# Iniciar processos
env.process(chegada_de_veiculos(env, TAXA_CHEGADA))
env.process(registrar_estado_filas(env))

# Rodar simulação
env.run(until=TEMPO_DE_SIMULACAO)

# Estatísticas finais
print("\nResultados da Simulação:")
print(f"Total de veículos que chegaram: {len(chegadas)}")
print(f"Total de desistências: {desistencias}")
print(f"Tempo médio na fila manual: {np.mean(in_queue_manual):.2f}s")
print(f"Tempo médio na fila automática: {np.mean(in_queue_auto):.2f}s")
print(f"Tempo médio total no sistema: {np.mean(in_system):.2f}s")

# Garantir que ambas as listas tenham o mesmo tamanho para não dar erro
max_len = max(len(chegadas), len(saidas))

# Preencher com NaN para igualar os comprimentos
chegadas = chegadas + [float('nan')] * (max_len - len(chegadas))
saidas = saidas + [float('nan')] * (max_len - len(saidas))

# Criar DataFrame para organizar os dados
df_entrada_saida = pd.DataFrame({
    'chegadas': chegadas,
    'saidas': saidas
})

# Gráfico de pontos: Entradas e saídas
fig, ax = plt.subplots()
fig.set_size_inches(10, 5.4)

y1, x1 = list(range(len(df_entrada_saida['chegadas']))), df_entrada_saida['chegadas']
y2, x2 = list(range(len(df_entrada_saida['saidas']))), df_entrada_saida['saidas']

ax.plot(x1, y1, color='blue', marker="o", linewidth=0, label="Chegada")
ax.plot(x2, y2, color='red', marker="o", linewidth=0, label="Saída")

ax.set_xlabel('Tempo (s)')
ax.set_ylabel('Veículo ID')
ax.set_title("Chegadas e Saídas no Pedágio")
ax.legend()
ax.grid()

plt.show()

# Gráficos de linhas: Carros na fila
fig, ax = plt.subplots()
fig.set_size_inches(10, 5.4)

# Dados para as filas temporais
temporal_manual, fila_manual = zip(*fila_temporal_manual)
temporal_auto, fila_auto = zip(*fila_temporal_auto)

ax.plot(temporal_manual, fila_manual, label="Fila Manual", color="blue")
ax.plot(temporal_auto, fila_auto, label="Fila Automática", color="green")

ax.set_xlabel("Tempo (s)")
ax.set_ylabel("Número de veículos na fila")
ax.set_title("Número de veículos nas filas ao longo do tempo")
ax.legend()
plt.show()

# Gráfico de segmentos de linha: Tempo de trabalho dos pedágios (ajustado para o texto caber na tela)
fig, ax = plt.subplots()
fig.set_size_inches(12, 6)

for inicio, fim in tempo_trabalho_manuais:
    ax.plot([inicio, fim], [0, 0], color="blue", linewidth=2, label="Manual" if inicio == tempo_trabalho_manuais[0][0] else "")
for inicio, fim in tempo_trabalho_automaticas:
    ax.plot([inicio, fim], [1, 1], color="green", linewidth=2, label="Automático" if inicio == tempo_trabalho_automaticas[0][0] else "")

ax.set_xlabel("Tempo (s)")
ax.set_xlim(0, TEMPO_DE_SIMULACAO)
ax.set_yticks([0, 1])
ax.set_yticklabels(["Cabines Manuais", "Cabines Automáticas"])
ax.set_title("Tempo de Trabalho dos Servidores ao Longo do Tempo")

ax.legend(loc="center left", bbox_to_anchor=(-0.1, 0.5), fontsize=10)

plt.tight_layout()
plt.show()
