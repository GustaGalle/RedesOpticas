#!/usr/bin/env python
# coding: utf-8

# In[6]:


#fibra convencional

import numpy as np
import matplotlib.pyplot as plt

dados = [0, 1, 0, 1, 0, 1, 0, 1] #sinal aleátorio em binário

comprimento_fibra = 300  # km
atenuacao = 0.05  # dB/km

perda = -atenuacao * 300
conversao_dB = 10**(perda/ 20) #tem que converter, pq db é em log

sinal_recebido = [d * conversao_dB for d in dados]

print(sinal_recebido)

plt.stem(dados, label='Enviado')
plt.stem(sinal_recebido, linefmt='r--', markerfmt='ro', label='Recebido')
plt.legend()
plt.title("Sinal antes e depois da fibra")
plt.show()


# In[7]:


import numpy as np
import matplotlib.pyplot as plt

#SMD 2 nucleos 
dados_faixa1 = [0, 1, 0, 1, 0, 1, 0, 1]  
dados_faixa2 = [1, 0, 1, 0, 1, 0, 1, 0] 

comprimento_fibra = 300  # km
atenuacao = 0.05  # dB/km

crosstalk = 0.1 #interrferencia entre os nucleos 

perda = -atenuacao * 300
fator_perda = 10**(perda/ 20)

sinal_faixa1_recebido = [
    (d1 * fator_perda) + (d2 * fator_perda * crosstalk)  # Sinal principal + interferência
    for d1, d2 in zip(dados_faixa1, dados_faixa2)
]

sinal_faixa2_recebido = [
    (d2 * fator_perda) + (d1 * fator_perda * crosstalk)
    for d1, d2 in zip(dados_faixa1, dados_faixa2)
]


plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.stem(dados_faixa1, label="Faixa 1 (enviado)", basefmt=" ")
plt.stem(sinal_faixa1_recebido, linefmt='r--', label="Faixa 1 (recebido + interferência)")
plt.title("Faixa 1 - Efeito do Crosstalk")
plt.legend()

plt.subplot(1, 2, 2)
plt.stem(dados_faixa2, label="Faixa 2 (enviado)", basefmt=" ")
plt.stem(sinal_faixa2_recebido, linefmt='g--', label="Faixa 2 (recebido + interferência)")
plt.title("Faixa 2 - Efeito do Crosstalk")
plt.legend()
plt.show()


# ## O que é Dispersão Cromática?
# É quando diferentes frequências da luz viajam em velocidades diferentes dentro da fibra óptica.
# Acontece devido às diferentes frequências (luz/cores) na fibra, O vidro (ou plástico) da fibra desacelera algumas cores mais que outras. Após muitas distancias a diferença se acumula e estica o pulso.
# 
# - unidade: ps/(nm·km) (picossegundos por nanômetro por quilômetro).
# 
# - Exemplo: D = 17 ps/(nm·km) significa:
# Para cada 1 nm de largura espectral e 1 km de fibra, o pulso atrasa 17 ps.
# Em 300 km: 17 * 300 = 5100 ps = 5.1 ns de atraso entre cores!
# 
# - Como evitar:lasers mais puros(menos cores), amplificadores a cada 80km “reiniciaram o problema”  ou uso de Fibras "DCF" (Dispersion Compensating Fibers)
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt

#pulso de luz (um "1")
tempo = np.linspace(-10, 10, 1000)  # tempo em ns
pulso = np.exp(-(tempo**2)) 

#dispersão (alargamento)
D = 17  # ps/(nm·km)
distancia = 300  # km
alargamento = D * distancia * 0.1  # fator simplificado (0.1 = largura espectral)
pulso_disperso = np.exp(-(tempo**2)/(2*(alargamento)**2))

# 3. Plotar
plt.plot(tempo, pulso, label="Pulso original (1)")
plt.plot(tempo, pulso_disperso, label="Pulso após 300 km (disperso)")
plt.legend()
plt.title("Dispersão Cromática")
plt.xlabel("Tempo (ns)")
plt.ylabel("Intensidade")
plt.show()


# # com um compensador de dispersão
# 
# - Usa-se uma fibra com dispersão oposta à da fibra principal.
# Exemplo:
# Fibra normal: D = +17 ps/(nm·km) (estica o pulso).
# Fibra compensadora: D = -17 ps/(nm·km) (comprime o pulso).
# - Gratings de Bragg: Dispositivos que refletem as cores mais rápidas para que elas atrasem e realinhem com as lentas.
# - Pré-compensação em Transmissores: Antes de enviar o sinal, aplica-se um "pré-esticamento" inverso.
# 
# 
# 

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq

# parâmetros
taxa_amostragem = 100e9  # 100 GHz
num_amostras = 1024
duracao_bit = 1e-9  # 1 ns
comprimento_fibra = 300  # km
D = 17  # ps/(nm·km) (dispersão normal)
D_comp = -17  # ps/(nm·km) (dispersão compensadora)
largura_espectral = 0.1  # nm

# pulso (bit "1")
tempo = np.linspace(-num_amostras/2, num_amostras/2-1, num_amostras) * duracao_bit
pulso = np.exp(-(tempo**2)/(2*(duracao_bit/2)**2))  # Gaussiano

# transformada de Fourier
espectro = fft(np.fft.fftshift(pulso))
frequencias = fftfreq(num_amostras, d=1/taxa_amostragem)
omega = 2 * np.pi * frequencias

# dispersão normal na fibra principal
beta2 = -(D * largura_espectral * (1e-12)**2) / (2 * np.pi * 3e8 * 1e-12)  # ps²/km
dispersao = np.exp(1j * beta2 * (omega**2) * comprimento_fibra / 2)
espectro_disperso = espectro * dispersao

# compensação na fibra compensadora
beta2_comp = -(D_comp * largura_espectral * (1e-12)**2) / (2 * np.pi * 3e8 * 1e-12)
compensacao = np.exp(1j * beta2_comp * (omega**2) * comprimento_fibra / 2)
espectro_compensado = espectro_disperso * compensacao


pulso_disperso = np.fft.ifftshift(ifft(espectro_disperso))
pulso_compensado = np.fft.ifftshift(ifft(espectro_compensado))

plt.figure(figsize=(12, 6))
plt.plot(tempo, np.abs(pulso)**2, label='Pulso original')
plt.plot(tempo, np.abs(pulso_disperso)**2, label='Pulso após dispersão (300 km)')
plt.plot(tempo, np.abs(pulso_compensado)**2, label='Pulso compensado')
plt.title('Compensação de Dispersão Cromática')
plt.xlabel('Tempo (ns)')
plt.ylabel('Intensidade')
plt.legend()
plt.grid()
plt.show()


# In[13]:


from IPython.display import display, Markdown

markdown_text = """
### **Formato Matemático Matriz Acoplamento**

É uma matriz quadrada $M \\times M$ (onde $M$ = número de modos), em que:

- **Elementos diagonais** ($c_{ii}$): Representam a energia que permanece no mesmo modo.
- **Elementos não diagonais** ($c_{ij}, i \\neq j$): Representam o acoplamento entre modos diferentes.

---

### **Exemplo para 2 Modos**

A matriz de acoplamento $C$ é dada por:

$$
C = \\begin{bmatrix}
c_{11} & c_{12} \\\\
c_{21} & c_{22}
\\end{bmatrix}
$$

- $c_{12}$: Fração de energia que vaza do **Modo 1** para o **Modo 2**.
- $c_{21}$: Fração de energia que vaza do **Modo 2** para o **Modo 1**.
"""

display(Markdown(markdown_text))


# In[ ]:


import numpy as np

# matriz de acoplamento para 3 modos
matrix_acoplamento = np.array([
    [0.95, 0.03, 0.02],  # 95% do Modo 1 permanece, 3% vaza pro Modo 2, 2% pro Modo 3
    [0.04, 0.94, 0.02],
    [0.01, 0.01, 0.98]
])


# In[ ]:


from gnpy.core.network import Network
from gnpy.core.elements import Transceiver, Fiber, Amplifier
from gnpy.core.info import create_input_spectral_information
from gnpy.tools.json_io import load_json, save_json
import matplotlib.pyplot as plt
import numpy as np

# Carregar configurações padrão do GNPy
eqpt_config = load_json('examples/eqpt_config.json')  # Configuração dos equipamentos
network_config = load_json('examples/edfa_example_network.json')  # Topologia da rede

# Criar a rede
network = Network(network_config)

# Configurar os elementos da rede
network.reset_network()  # Reinicia a rede
network.set_nodes(eqpt_config)  # Define os equipamentos nos nós

# Parâmetros: frequência central, largura de banda total, número de canais, etc.
spectral_info = create_input_spectral_information(
    f_min=191.3e12,  # Frequência mínima (Hz)
    f_max=196.1e12,  # Frequência máxima (Hz)
    roll_off=0.15,   # Roll-off dos filtros
    baud_rate=32e9,  # Taxa de símbolo (baud rate)
    power_dbm=0      # Potência inicial (dBm)
)

# Propagar o sinal pela rede
network.propagate(spectral_info, path=['trx_A', 'trx_B'])  # Caminho entre transceptores

# Extrair resultados
results = network.results['trx_A']['trx_B']  # Resultados do caminho A -> B


frequencies = results.frequency / 1e12  # Frequências em THz
gains = results.gain  # Ganho ao longo do caminho
noises = results.nli + results.ase  # Ruído total (NLI + ASE)


plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(frequencies, gains, label="Ganho (dB)")
plt.title("Ganho ao Longo do Caminho")
plt.xlabel("Frequência (THz)")
plt.ylabel("Ganho (dB)")
plt.grid()
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(frequencies, noises, label="Ruído Total (dB)", color='orange')
plt.title("Ruído ao Longo do Caminho")
plt.xlabel("Frequência (THz)")
plt.ylabel("Ruído (dB)")
plt.grid()
plt.legend()

plt.tight_layout()
plt.show()


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # Para visualização de topologias

# Parâmetros da rede
numero_modos = 3
numero_nos = 2  # Topologia ponto a ponto tem 2 nós
banda_total = 400  # GHz
slot_size = 12.5  # GHz
numero_slots = int(banda_total / slot_size)

# Matriz de adjacência (topologia ponto a ponto)
matriz_adjacencia = np.array([[0, 1],
                              [1, 0]])

# Matriz de acoplamento entre modos
# Valoreo de coeficiente de acoplamento entre modos
matriz_acoplamento = np.array([[0.0, 0.3, 0.1],
                               [0.3, 0.0, 0.2],
                               [0.1, 0.2, 0.0]])


G = nx.from_numpy_array(matriz_adjacencia)
pos = nx.spring_layout(G)
plt.figure(figsize=(6, 3))
nx.draw(G, pos, with_labels=True, node_size=2000, node_color='lightblue', 
        font_size=14, font_weight='bold')
plt.title("Topologia da Rede (Ponto a Ponto)")
plt.show()

# Estado dos enlaces (matriz 3D: [nó_origem, nó_destino, modo, slot])
# Inicializa todos os slots como livres (0)
rede = np.zeros((numero_nos, numero_nos, numero_modos, numero_slots))

# Função para verificar disponibilidade considerando acoplamento
def verificar_disponibilidade(enlace, modo_alvo, inicio, quantidade, matriz_acoplamento):
    # Verifica o modo principal
    if np.any(rede[enlace][modo_alvo, inicio:inicio+quantidade] != 0):
        return False

    # Verifica modos acoplados com coeficiente > 0.2
    for modo in range(numero_modos):
        if matriz_acoplamento[modo_alvo, modo] > 0.2:
            if np.any(rede[enlace][modo, inicio:inicio+quantidade] != 0):
                return False
    return True

# Simulação de alocação
np.random.seed(42)
numero_conexoes = 5
tamanhos = np.random.randint(1, 5, numero_conexoes)

for i in range(numero_conexoes):
    # Seleciona enlace (no ponto a ponto só existe um: 0 -> 1)
    enlace_origem = 0
    enlace_destino = 1

    # Seleciona modo aleatório
    modo_atual = np.random.randint(0, numero_modos)

    # Tenta alocação com verificação de acoplamento
    alocado = False
    for inicio in range(numero_slots - tamanhos[i] + 1):
        if verificar_disponibilidade((enlace_origem, enlace_destino), 
                                    modo_atual, inicio, tamanhos[i], 
                                    matriz_acoplamento):
            # Aloca no modo principal
            rede[enlace_origem, enlace_destino, modo_atual, inicio:inicio+tamanhos[i]] = 1
            print(f"Conexão {i+1} alocada no modo {modo_atual+1}")
            alocado = True
            break
    if not alocado:
        print(f"Conexão {i+1} bloqueada")

# Visualização do espectro
plt.figure(figsize=(12, 6))
for modo in range(numero_modos):
    plt.subplot(numero_modos, 1, modo+1)
    plt.imshow(rede[0,1,modo,:,:], aspect='auto', cmap='binary', interpolation='none')
    plt.title(f"Modo {modo+1}")
    plt.xlabel("Slots de Frequência")
    plt.ylabel("Estado")
    plt.yticks([])
plt.tight_layout()
plt.show()


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import networkx as nx  # Para visualizar a topologia

# Parâmetros da rede
numero_nos = 4  # Número de nós na rede
numero_modos = 3  # Número de modos espaciais
banda_total = 400  # Largura de banda total por modo (GHz)
slot_size = 12.5  # Tamanho de cada slot (GHz)
numero_slots = int(banda_total / slot_size)  # Slots por modo

# Matriz de distâncias (pesos das arestas)
# Cada entrada representa o "peso" entre dois nós (não necessariamente distância física)
matriz_distancias = np.array([
    [0, 1, 2, 3],  # Nó 0 conectado a 1, 2, 3
    [1, 0, 1, 2],  # Nó 1 conectado a 0, 2, 3
    [2, 1, 0, 1],  # Nó 2 conectado a 0, 1, 3
    [3, 2, 1, 0]   # Nó 3 conectado a 0, 1, 2
])

# Estrutura 3D da rede: [nó_origem, nó_destino, modo, slot]
rede = np.zeros((numero_nos, numero_nos, numero_modos, numero_slots))

# Função para verificar disponibilidade de slots
def verificar_disponibilidade(origem, destino, modo, inicio, quantidade):
    return np.all(rede[origem, destino, modo, inicio:inicio+quantidade] == 0)

# Função para alocar slots
def alocar_slots(origem, destino, modo, inicio, quantidade):
    rede[origem, destino, modo, inicio:inicio+quantidade] = 1

# Simulação de alocação
np.random.seed(42)  # Semente para reproducibilidade
numero_conexoes = 5  # Número de requisições de conexão
tamanhos = np.random.randint(1, 5, numero_conexoes)  # Tamanho de cada requisição (slots)

for i in range(numero_conexoes):
    # Seleciona origem e destino aleatórios
    origem, destino = np.random.choice(numero_nos, size=2, replace=False)

    # Seleciona modo aleatório
    modo_atual = np.random.randint(0, numero_modos)

    # Considera o peso da aresta para priorizar caminhos mais curtos
    peso = matriz_distancias[origem, destino]

    # Tenta alocação com estratégia first-fit
    alocado = False
    for inicio in range(numero_slots - tamanhos[i] + 1):
        if verificar_disponibilidade(origem, destino, modo_atual, inicio, tamanhos[i]):
            alocar_slots(origem, destino, modo_atual, inicio, tamanhos[i])
            print(f"Conexão {i+1}: Origem={origem}, Destino={destino}, Modo={modo_atual+1}, Peso={peso}")
            alocado = True
            break

    if not alocado:
        print(f"Conexão {i+1} bloqueada (Origem={origem}, Destino={destino})")

# Visualização da topologia
G = nx.from_numpy_array(matriz_distancias)
pos = nx.spring_layout(G)  
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=12, font_weight='bold')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Topologia da Rede (Matriz de Distâncias)")
plt.show()

# Visualização do espectro
plt.figure(figsize=(12, 8))
for origem in range(numero_nos):
    for destino in range(numero_nos):
        if origem != destino:
            for modo in range(numero_modos):
                plt.subplot(numero_nos, numero_nos, origem * numero_nos + destino + 1)
                plt.imshow(rede[origem, destino, modo, :], aspect='auto', cmap='binary', interpolation='none')
                plt.title(f"Origem={origem}, Destino={destino}, Modo={modo+1}")
                plt.xlabel("Slots de Frequência")
                plt.ylabel("Estado")
                plt.yticks([])
plt.tight_layout()
plt.show()


# In[ ]:




