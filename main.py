import simpy
import numpy as np
import pandas as pd
from scipy.stats import expon
import matplotlib.pyplot as plt

# Dados para a simulação
TEMPO_DE_SIMULACAO = 100
QUANTIDADE_DE_CABINES = 1  # Quantidade de cabines de pedágio

# Parâmetros de distribuição
MEDIA_DE_CHEGADA_DE_VEICULOS = 2  # Média de chegada dos veículos

# Tempos médios de pagamento e desvios padrão por tipo de veículo
TEMPOS_DE_PAGAMENTO = {
    'moto': {'media': 0.5, 'desvio': 0.1},
    'carro': {'media': 1.5, 'desvio': 0.3},
    'caminhao': {'media': 3.0, 'desvio': 0.5}
}

# Lista de tipos de veículos e suas probabilidades
TIPOS_DE_VEICULOS = ['moto', 'carro', 'caminhao']
PROBABILIDADES_TIPOS = [0.3, 0.5, 0.2]

# Listas para armazenar as informações da simulação
chegadas, saidas, tipos = [], [], []
in_queue, in_system = [], []
horarios_nas_filas, tamanho_da_fila = [], []

# Funções de distribuição
def distribuicao_chegada_de_veiculos():
    tempo_do_proximo_veiculo = expon.rvs(scale=MEDIA_DE_CHEGADA_DE_VEICULOS, size=1)
    return tempo_do_proximo_veiculo[0]

def tempo_de_pagamento_veiculo(tipo):
    parametros = TEMPOS_DE_PAGAMENTO[tipo]
    return max(0, np.random.normal(parametros['media'], parametros['desvio']))

# Função para salvar o tempo na fila
def salva_info_da_fila(env, pedagio):
    horario_medicao = env.now
    tamanho_da_fila_agora = len(pedagio.queue)
    horarios_nas_filas.append(horario_medicao)
    tamanho_da_fila.append(tamanho_da_fila_agora)
    return horario_medicao

# Função que define o tempo no sistema
def calcula_tempo_no_sistema(env, horario_chegada):
    horario_saida = env.now
    saidas.append(horario_saida)
    tempo_total = horario_saida - horario_chegada
    in_system.append(tempo_total)

# Função que simula a chegada dos veículos
def chegada_dos_veiculos(env, cabines_de_pedagio):
    veiculo_id = 0

    while True:
        tempo_do_proximo_veiculo = distribuicao_chegada_de_veiculos()
        yield env.timeout(tempo_do_proximo_veiculo)

        tempo_de_chegada = env.now
        tipo_veiculo = np.random.choice(TIPOS_DE_VEICULOS, p=PROBABILIDADES_TIPOS)
        tipos.append(tipo_veiculo)

        chegadas.append(tempo_de_chegada)
        veiculo_id += 1
        print(f'Veículo {veiculo_id} ({tipo_veiculo}) chegou ao pedágio em {tempo_de_chegada:.2f}')

        env.process(pagamento(env, veiculo_id, tempo_de_chegada, tipo_veiculo, cabines_de_pedagio))

# Função que simula o pagamento do pedágio
def pagamento(env, veiculo_id, horario_chegada, tipo_veiculo, cabines_de_pedagio):
    with cabines_de_pedagio.request() as req:
        print(f'Veículo {veiculo_id} ({tipo_veiculo}) entrou na fila em {env.now:.2f}')
        horario_entrada_da_fila = salva_info_da_fila(env, cabines_de_pedagio)
        yield req  # Espera pela cabine de pedágio

        print(f'Veículo {veiculo_id} ({tipo_veiculo}) saiu da fila em {env.now:.2f}')
        horario_saida_da_fila = salva_info_da_fila(env, cabines_de_pedagio)

        tempo_na_fila = horario_saida_da_fila - horario_entrada_da_fila
        in_queue.append(tempo_na_fila)

        # Execução do pagamento
        tempo_pagamento = tempo_de_pagamento_veiculo(tipo_veiculo)
        yield env.timeout(tempo_pagamento)
        print(f'Veículo {veiculo_id} ({tipo_veiculo}) pagou o pedágio em {tempo_pagamento:.2f} minutos')

        calcula_tempo_no_sistema(env, horario_chegada)

# Função para calcular a média do tamanho da fila
def media_fila(df_tamanho_fila):
    df_tamanho_fila['delta'] = df_tamanho_fila['horario'].shift(-1) - df_tamanho_fila['horario']
    df_tamanho_fila = df_tamanho_fila[0:-1]  # Remove última linha que tem delta infinito
    return np.average(df_tamanho_fila['tamanho'], weights=df_tamanho_fila['delta'])

# Função para calcular a utilização do serviço
def utilizacao_servico(df_tamanho_fila):
    soma_servico_livre = df_tamanho_fila[df_tamanho_fila['tamanho'] == 0]['delta'].sum()
    primeiro_evento = df_tamanho_fila['horario'].iloc[0]
    soma_servico_livre += primeiro_evento
    return round((1 - soma_servico_livre / TEMPO_DE_SIMULACAO) * 100, 2)

# Função para calcular a porcentagem de veículos que não esperaram na fila
def porcentagem_de_nao_esperaram(df_tamanho_fila):
    soma_nao_esperaram = df_tamanho_fila[df_tamanho_fila['tamanho'] >= 1]['delta'].sum()
    return round((soma_nao_esperaram / TEMPO_DE_SIMULACAO) * 100, 2)

# Função para rodar a simulação
def rodar_simulacao():
    np.random.seed(1)

    # Prepara o ambiente
    env = simpy.Environment()

    # Definindo recursos (quantidade de cabines de pedágio)
    cabines_de_pedagio = simpy.Resource(env, capacity=QUANTIDADE_DE_CABINES)

    # Inicializa a chegada de veículos
    env.process(chegada_dos_veiculos(env, cabines_de_pedagio))

    # Executa a simulação
    env.run(until=TEMPO_DE_SIMULACAO)

    # Criando DataFrames com os dados da simulação
    df1 = pd.DataFrame(horarios_nas_filas, columns=['horario'])
    df2 = pd.DataFrame(tamanho_da_fila, columns=['tamanho'])
    df3 = pd.DataFrame(chegadas, columns=['chegadas'])
    df4 = pd.DataFrame(saidas, columns=['partidas'])

    df_tamanho_da_fila = pd.concat([df1, df2], axis=1)
    df_entrada_saida = pd.concat([df3, df4], axis=1)

    # Exibindo gráficos de chegada e saída
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5.4)
    x1, y1 = list(df_entrada_saida['chegadas'].keys()), df_entrada_saida['chegadas']
    x2, y2 = list(df_entrada_saida['partidas'].keys()), df_entrada_saida['partidas']

    ax.plot(x1, y1, color='blue', marker="o", linewidth=0, label="Chegada")
    ax.plot(x2, y2, color='red', marker="o", linewidth=0, label="Saída")
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Veículo ID')
    ax.set_title("Chegadas & Saídas no Pedágio")
    ax.legend()
    fig.show()

    # Exibindo gráfico de tamanho da fila
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 5.4)
    ax.plot(df_tamanho_da_fila['horario'], df_tamanho_da_fila['tamanho'], color='blue', linewidth=1)
    ax.set_xlabel('Tempo')
    ax.set_ylabel('Número de Veículos na Fila')
    ax.set_title('Número de veículos na fila')
    ax.grid()
    fig.show()

    # Exibindo gráfico da distribuição dos tipos de veículos
    tipos_contagem = pd.Series(tipos).value_counts()
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5)
    tipos_contagem.plot(kind='bar', color=['orange', 'green', 'blue'], ax=ax)
    ax.set_xlabel('Tipos de Veículos')
    ax.set_ylabel('Quantidade')
    ax.set_title('Distribuição dos Tipos de Veículos')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    fig.show()

    # Cálculos estatísticos
    print('O tempo médio na fila é de %.2f' % (np.mean(in_queue)))
    print('O tempo médio no sistema é %.2f' % (np.mean(in_system)))
    print('O número médio de veículos na fila é %.2f' % media_fila(df_tamanho_da_fila))
    print('A utilização do serviço é %.2f %%' % utilizacao_servico(df_tamanho_da_fila))
    print('A probabilidade de veículos que não podem esperar na fila é %.2f' % (porcentagem_de_nao_esperaram(df_tamanho_da_fila)))

# Rodar a simulação
rodar_simulacao()
