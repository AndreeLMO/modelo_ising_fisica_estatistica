# Detecção de Transições de Fase no Modelo de Ising 2D via Redes Neurais Convolucionais e Interpretabilidade por Grad-CAM

> **Projeto de Pesquisa Científica / Computação de Alto Desempenho**
> **Autor:** André Luiz
> **Área:** Física Estatística Computacional / Deep Learning / Inteligência Artificial Explicável (XAI)
> **Frameworks Principais:** PyTorch, NumPy, Scikit-Learn, Matplotlib

---

## 📑 Resumo Expandido

O estudo das transições de fase e dos fenômenos críticos dentro da física estatística de muitos corpos tradicionalmente requer formulações analíticas complexas e simulações numéricas massivas baseadas no método de Monte Carlo. Observáveis macroscópicos como magnetização, energia interna, calor específico e susceptibilidade magnética são utilizados para mapear os limites de fase. 

Este projeto propõe um paradigma alternativo e complementar: a utilização de **Aprendizado Profundo (Deep Learning)** para classificar estados termodinâmicos diretamente a partir de suas configurações microscópicas de spins, sem o cálculo prévio de variáveis macroscópicas. 

A fim de romper com a opacidade metodológica que comumente rotula as redes neurais convolucionais como "caixas-pretas", implementou-se o algoritmo **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Este algoritmo mapeia o fluxo de gradientes nas camadas convolucionais mais profundas, revelando de forma inequívoca quais estruturas geométricas e correlações espaciais de spins (como fronteiras de domínio e clusters fractais) foram determinantes para a decisão da rede em cada fase termodinâmica.

---

## 🏛️ 1. Fundamentação Teórica Detalhada

### 1.1 O Modelo de Ising Bidimensional (2D)
O Modelo de Ising 2D serve como o laboratório ideal para testar teorias de transição de fase contínua (segunda ordem). Definido sobre uma rede quadrada ideal de tamanho $L \times L$ com condições de contorno periódicas (geometria toroidal), o modelo atribui a cada sítio $i$ uma variável discreta de spin $s_i \in \{-1, +1\}$. 

O Hamiltoniano ($H$) do sistema, considerando apenas interações de primeiros vizinhos na ausência de campo magnético externo, é expresso por:

$$H = -J \sum_{\langle i,j \rangle} s_i s_j$$

Onde $\langle i,j \rangle$ indica o somatório sobre pares de sítios adjacentes e $J$ denota a integral de troca magnética. Adota-se $J = 1.0$ para o comportamento ferromagnético, onde o estado fundamental (energia mínima) é duplamente degenerado e caracterizado por todos os spins paralelos ($s_i = +1$ ou $s_i = -1$ para todo $i$).

A transição de fase ocorre devido à competição contínua entre a minimização da energia interna (que favorece o ordenamento) e a maximização da entropia (introduzida pelas flutuações térmicas). A probabilidade de o sistema ocupar uma configuração de rede $\sigma$ é ponderada pelo peso estatístico de Boltzmann:

$$P(\sigma) = \frac{e^{-\beta H(\sigma)}}{Z}, \quad \beta = \frac{1}{k_B T}, \quad Z = \sum_{\{\sigma\}} e^{-\beta H(\sigma)}$$

Em 1944, Lars Onsager resolveu analiticamente este modelo no limite termodinâmico ($L \rightarrow \infty$), demonstrando a existência de uma singularidade na capacidade térmica à temperatura crítica exata de:

$$T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2,26918 J$$

* **Regime Subcrítico ($T < T_c$):** A quebra espontânea da simetria $Z_2$ induz uma magnetização de longo alcance. O parâmetro de ordem, dado pela magnetização média por sítio $\langle |M| \rangle$, assume valores não nulos.
* **Regime Supercrítico ($T > T_c$):** O sistema transita para uma fase paramagnética. A flutuação térmica destrói a coerência de fase e $\langle |M| \rangle \rightarrow 0$.

### 1.2 Mecânica Espacial das CNNs aplicadas à Física
Uma configuração de spins de uma rede $40 \times 40$ pode ser interpretada matematicamente como uma imagem binária de canal único ($1 \times 40 \times 40$). As Redes Neurais Convolucionais (CNNs) realizam operações de filtragem local através de matrizes de pesos aprendíveis (kernels). A operação de convolução bidimensional mapeia correlações espaciais locais:

$$S(i,j) = (I * K)(i,j) = \sum_{m} \sum_{n} I(i-m, j-n) K(m,n)$$

Ao empilhar camadas convolucionais intercaladas com funções de ativação não lineares (ReLU) e subamostragem (*Max Pooling*), a rede deixa de enxergar apenas spins individuais nas camadas iniciais e passa a extrair feições macroscópicas complexas (comprimento de correlação espacial, formas de clusters e interfaces de energia) nas camadas profundas.

### 1.3 Formulação Matemática do Grad-CAM
Para extrair a assinatura visual da tomada de decisão da rede, o Grad-CAM isola os mapas de ativação de feições $A^k$ gerados pela última camada convolucional do modelo. Primeiro, calcula-se o gradiente da pontuação (*score*) linear da classe de interesse $Y^c$ (antes do operador Softmax) em relação às ativações $A^k$. Esses gradientes sofrem um agrupamento por média global (*Global Average Pooling*) para determinar o peso de importância $\alpha_k^c$:

$$\alpha_k^c = \frac{1}{U \times V} \sum_{u=1}^{U} \sum_{v=1}^{V} \frac{\partial Y^c}{\partial A_{u,v}^k}$$

Onde $U$ e $V$ representam as dimensões espaciais do mapa de feições da última camada convolucional. O mapa de calor final do Grad-CAM ($L_{\text{Grad-CAM}}^c$) é uma combinação linear ponderada de todos os mapas de feições $A^k$, retificada por uma função ReLU:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left(\sum_{k} \alpha_k^c A^k\right)$$

O operador ReLU garante que apenas as feições estruturais que aumentam o score da classe alvo $c$ sejam exibidas, filtrando ativações negativas que contribuem para classes concorrentes.

---

## 🛠️ 2. Metodologia Computacional e Arquitetura

O pipeline do projeto está estruturado em quatro blocos lógicos autônomos e integrados, projetados para máxima eficiência computacional através do uso de tensores tipados (`int8` para spins e `float32` para pesos).

### 2.1 Geração Estocástica do Dataset (Algoritmo de Metropolis)
Para evitar o viés de transientes térmicos (*critical slowing down*), cada ponto de temperatura passa por 300 passos iniciais de Monte Carlo (MCS) para atingir o equilíbrio thermodynamic. 
As amostras de treino e teste são coletadas de forma controlada a cada 5 MCS para garantir a decorrelação estatística entre as configurações consecutivas salvadas.

* **Classe 0 (Fase Ordenada):** 5 temperaturas uniformemente distribuídas no intervalo $T \in [0.5, 2.0]$. Total de 500 matrizes.
* **Classe 1 (Fase Desordenada):** 5 temperaturas uniformemente distribuídas no intervalo $T \in [2.8, 5.0]$. Total de 500 matrizes.
* **Amostragem Excluída:** A janela crítica intermediária ($2.0 < T < 2.8$) foi deliberadamente omitida do conjunto de treinamento para garantir limites assintóticos bem definidos para o aprendizado inicial do classificador.

### 2.2 Estrutura de Camadas da CNN
A tabela a seguir detalha o fluxo tensorial ao longo da arquitetura implementada:

| Camada | Tipo de Operação | Configuração de Parâmetros | Tamanho do Tensor de Saída |
| :--- | :--- | :--- | :--- |
| **Entrada** | Tensor de Rede | Normalizado de $[-1, +1] \rightarrow [0, 1]$ | `(Batch, 1, 40, 40)` |
| **Conv_1** | Convolução 2D | 16 Filtros, Kernel $3\times3$, Padding=1 | `(Batch, 16, 40, 40)` |
| **ReLU_1** | Ativação | Element-wise ReLU | `(Batch, 16, 40, 40)` |
| **Pool_1** | Max Pooling | Filtro $2\times2$, Stride=2 | `(Batch, 16, 20, 20)` |
| **Conv_2** | Convolução 2D | 32 Filtros, Kernel $3\times3$, Padding=1 | `(Batch, 32, 20, 20)` |
| **ReLU_2** | Ativação | Element-wise ReLU | `(Batch, 32, 20, 20)` |
| **Pool_2** | Max Pooling | Filtro $2\times2$, Stride=2 | `(Batch, 32, 10, 10)` |
| **Conv_3** | Convolução 2D | 64 Filtros, Kernel $3\times3$, Padding=1 | `(Batch, 64, 10, 10)` |
| **ReLU_3** | Ativação | Element-wise ReLU | `(Batch, 64, 10, 10)` |
| **Pool_3** | Max Pooling | Filtro $2\times2$, Stride=2 | `(Batch, 64, 5, 5)` |
| **Flatten** | Redução Linear | Redução para Vetor Unidimensional | `(Batch, 1600)` |
| **Dense_1**| Camada Conectada | Camada Linear Oculta com 128 Neurônios | `(Batch, 128)` |
| **Dropout**| Regularização | Taxa de Descarte Aleatório de 30% ($p=0.3$) | `(Batch, 128)` |
| **Dense_2**| Saída (*Logits*) | Mapeamento Linear para as 2 Classes Finais | `(Batch, 2)` |

---

## 💻 3. Código-Fonte Completo e Otimizado

Abaixo encontra-se a implementação computacional robusta unificada do pipeline em um único arquivo chamado `modelo_ising_grad.py`:

```python
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# CONFIGURAÇÕES GERAIS E SEMENTE DE REPRODUTIBILIDADE
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de processamento ativo: {DEVICE}")

# PARÂMETROS FÍSICOS E DE SIMULAÇÃO
L = 40
J = 1.0
TC = 2.269

LOW_TEMPS = np.linspace(0.5, 2.0, 5)
HIGH_TEMPS = np.linspace(2.8, 5.0, 5)
SAMPLES_PER_TEMP = 100
THERMALIZATION = 300
DECORRELATION = 5

BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

os.makedirs("results", exist_ok=True)

# 1. FUNÇÕES DO MOTOR DE MONTE CARLO
def initialize_lattice(L):
    return np.random.choice([-1, 1], size=(L, L)).astype(np.int8)

def metropolis_step(lattice, T, J=1.0):
    L = lattice.shape[0]
    for _ in range(L * L):
        i = np.random.randint(0, L)
        j = np.random.randint(0, L)
        spin = lattice[i, j]
        neighbors = (
            lattice[(i + 1) % L, j] +
            lattice[(i - 1) % L, j] +
            lattice[i, (j + 1) % L] +
            lattice[i, (j - 1) % L]
        )
        delta_E = 2 * J * spin * neighbors
        if delta_E < 0:
            lattice[i, j] *= -1
        elif np.random.rand() < np.exp(-delta_E / T):
            lattice[i, j] *= -1
    return lattice

def total_energy(lattice, J=1.0):
    L = lattice.shape[0]
    energy = 0
    for i in range(L):
        for j in range(L):
            spin = lattice[i, j]
            neighbors = (lattice[(i + 1) % L, j] + lattice[i, (j + 1) % L])
            energy += -J * spin * neighbors
    return energy

def magnetization(lattice):
    return np.sum(lattice) / lattice.size

# TESTE PILOTO: EVOLUÇÃO E TERMALIZAÇÃO
print("\n[Piloto] Executando simulação de termalização teste...")
lat_test = initialize_lattice(L)
e_hist, m_hist = [], []
for mcs in range(300):
    lat_test = metropolis_step(lat_test, T=2.2, J=J)
    e_hist.append(total_energy(lat_test, J))
    m_hist.append(abs(magnetization(lat_test)))

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(e_hist, color='crimson')
plt.title("Evolução da Energia Interna (H)")
plt.xlabel("Passos Monte Carlo")
plt.ylabel("Energia")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(m_hist, color='navy')
plt.title("Evolução da Magnetização Líquida |M|")
plt.xlabel("Passos Monte Carlo")
plt.ylabel("Magnetização Absoluta")
plt.grid(True)
plt.tight_layout()
plt.savefig("results/termalizacao_ising.png")
plt.close()

# 2. GERAÇÃO EM MASSA DO DATASET
print("\n[Dataset] Amostrando configurações de spins...")
X, y = [], []

for T in tqdm(LOW_TEMPS, desc="Fase Ordenada (Classe 0)"):
    lat = initialize_lattice(L)
    for _ in range(THERMALIZATION):
        lat = metropolis_step(lat, T, J)
    for _ in range(SAMPLES_PER_TEMP):
        for _ in range(DECORRELATION):
            lat = metropolis_step(lat, T, J)
        X.append((lat.copy() + 1) / 2.0)  # Normalização [0,1]
        y.append(0)

for T in tqdm(HIGH_TEMPS, desc="Fase Desordenada (Classe 1)"):
    lat = initialize_lattice(L)
    for _ in range(THERMALIZATION):
        lat = metropolis_step(lat, T, J)
    for _ in range(SAMPLES_PER_TEMP):
        for _ in range(DECORRELATION):
            lat = metropolis_step(lat, T, J)
        X.append((lat.copy() + 1) / 2.0)
        y.append(1)

X = np.array(X, dtype=np.float32)[:, np.newaxis, :, :]
y = np.array(y, dtype=np.int64)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_seed=SEED)

train_loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(TensorDataset(torch.tensor(X_test), torch.tensor(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# 3. REDE NEURAL CONVOLUCIONAL (CNN)
class IsingCNN(nn.Module):
    def __init__(self):
        super(IsingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
        self.gradients = None
        
    def activations_hook(self, grad):
        self.gradients = grad
        
    def forward(self, x):
        x = self.features[0](x) # Conv1
        x = self.features[1](x) # ReLU1
        x = self.features[2](x) # MaxPool1
        x = self.features[3](x) # Conv2
        x = self.features[4](x) # ReLU2
        x = self.features[5](x) # MaxPool2
        x = self.features[6](x) # Conv3
        
        # Registra gancho na última convolucional para extrair os gradientes
        if x.requires_grad:
            h = x.register_hook(self.activations_hook)
            
        x = self.features[7](x) # ReLU3
        x = self.features[8](x) # MaxPool3
        x = self.classifier(x)
        return x

    def get_layer_activations(self, x):
        return self.features[6](self.features[0:6](x))

model = IsingCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# TREINAMENTO
print("\n[Treino] Iniciando otimização da CNN...")
train_losses, test_losses = [], []
for epoch in range(EPOCHS):
    model.train()
    r_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        r_loss += loss.item() * inputs.size(0)
    train_losses.append(r_loss / len(train_loader.dataset))
    print(f"Época {epoch+1}/{EPOCHS} | Loss Treino: {train_losses[-1]:.4f}")

plt.figure(figsize=(5, 4))
plt.plot(train_losses, label="Loss Treino", color='black', linewidth=2)
plt.title("Curva de Aprendizado")
plt.xlabel("Épocas")
plt.ylabel("Perda (Loss)")
plt.legend()
plt.grid(True)
plt.savefig("results/curvas_treinamento.png")
plt.close()

# 4. AVALIAÇÃO E MATRIZ DE CONFUSÃO
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

print("\n[Avaliação] Relatório Estatístico:")
print(classification_report(all_labels, all_preds, target_names=["Ordenada", "Desordenada"]))

cm = confusion_matrix(all_labels, all_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ordenada", "Desordenada"])
disp.plot(cmap=plt.cm.Blues)
plt.savefig("results/matriz_confusao.png")
plt.close()

# 5. MOTOR E EXTRAÇÃO GRAD-CAM
def run_gradcam(model, instance, target_class):
    model.eval()
    tensor_img = torch.tensor(instance).unsqueeze(0).to(DEVICE)
    tensor_img.requires_grad = True
    
    output = model(tensor_img)
    score = output[0][target_class]
    
    model.zero_grad()
    score.backward()
    
    grads = model.gradients.cpu().data.numpy()[0]
    acts = model.get_layer_activations(tensor_img).cpu().data.numpy()[0]
    
    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)
    
    for i, w in enumerate(weights):
        cam += w * acts[i]
        
    cam = np.maximum(cam, 0) # ReLU
    if np.max(cam) > 0:
        cam = cam / np.max(cam) # Normalização
    
    import cv2
    cam_resized = cv2.resize(cam, (L, L))
    return cam_resized

# GERAÇÃO DAS IMAGENS EXPLICATIVAS DO GRAD-CAM
idx_ord = np.where(y_test == 0)[0][0]
idx_des = np.where(y_test == 1)[0][0]

for idx, label, filename in [(idx_ord, 0, "gradcam_ordenada.png"), (idx_des, 1, "gradcam_desordenada.png")]:
    heatmap = run_gradcam(model, X_test[idx], target_class=label)
    
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(X_test[idx][0], cmap='inferno')
    plt.title("Configuração de Spins")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Mapa de Calor Grad-CAM")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"results/{filename}")
    plt.close()

print("\n[Sucesso] Todos os gráficos e mapas Grad-CAM foram salvos no diretório 'results/'.")
