# 🧲 Detecção de Transições de Fase no Modelo de Ising 2D com CNN e Grad-CAM
 
> **Autor:** André Luiz Magalhães de Oliveira

---

## 📋 Sumário

- [Visão Geral](#-visão-geral)
- [Fundamentação Teórica](#-fundamentação-teórica)
  - [Modelo de Ising 2D](#modelo-de-ising-2d)
  - [Simulação de Monte Carlo — Algoritmo de Metropolis](#simulação-de-monte-carlo--algoritmo-de-metropolis)
  - [Redes Neurais Convolucionais (CNNs)](#redes-neurais-convolucionais-cnns)
  - [Interpretabilidade via Grad-CAM](#interpretabilidade-via-grad-cam)
- [Pipeline do Projeto](#-pipeline-do-projeto)
- [Metodologia e Implementação](#-metodologia-e-implementação)
  - [Etapa 1 — Simulação do Modelo de Ising 2D](#etapa-1--simulação-do-modelo-de-ising-2d)
  - [Etapa 2 — Geração do Dataset](#etapa-2--geração-do-dataset)
  - [Etapa 3 — Arquitetura CNN em PyTorch](#etapa-3--arquitetura-cnn-em-pytorch)
  - [Etapa 4 — Treinamento e Avaliação](#etapa-4--treinamento-e-avaliação)
  - [Etapa 5 — Mapas de Ativação Grad-CAM](#etapa-5--mapas-de-ativação-grad-cam)
- [Resultados](#-resultados)
- [Requisitos e Instalação](#-requisitos-e-instalação)
- [Como Executar](#-como-executar)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Referências](#-referências)

---

## 🔭 Visão Geral

Este projeto implementa um pipeline completo de **simulação estocástica + aprendizado profundo + inteligência artificial explicável (XAI)**, aplicado à detecção automática de transições de fase no **Modelo de Ising bidimensional**.

A abordagem integra três frentes:

1. **Simulação de Monte Carlo** (Algoritmo de Metropolis) para geração de configurações microscópicas de spins nas fases ordenada e desordenada.
2. **Rede Neural Convolucional (CNN)** treinada para classificar as fases termodinâmicas diretamente a partir das imagens de rede de spins, sem cálculo explícito de observáveis físicos.
3. **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) para decodificar quais regiões espaciais da rede de spins a CNN utiliza em sua tomada de decisão, estabelecendo um vínculo direto entre ativações neurais e física de clusters de spin.

O modelo atingiu **100% de acurácia** na classificação das fases, e os mapas Grad-CAM revelaram que os filtros convolucionais aprendem espontaneamente conceitos físicos como **coerência de domínio ferromagnético** e **entropia de interface paramagnética** — sem supervisão explícita sobre esses conceitos.

---

## 📐 Fundamentação Teórica

### Modelo de Ising 2D

O Modelo de Ising 2D consiste em uma rede quadrada L × L de variáveis discretas de spin $s_i \in \{-1, +1\}$. O Hamiltoniano do sistema na ausência de campo externo é:

$$\mathcal{H} = -J \sum_{\langle i,j \rangle} s_i s_j$$

onde $J > 0$ é a constante de acoplamento ferromagnético e $\langle i,j \rangle$ indica a soma sobre pares de primeiros vizinhos. A probabilidade canônica de cada microestado $\sigma$ é dada pela distribuição de Boltzmann:

$$P(\sigma) = \frac{e^{-\beta \mathcal{H}(\sigma)}}{Z}, \quad \beta = \frac{1}{k_B T}$$

**Solução exata de Onsager (1944):** O modelo 2D exibe uma transição de fase contínua de segunda ordem na temperatura crítica:

$$T_c = \frac{2J}{k_B \ln(1 + \sqrt{2})} \approx 2{,}269 \; J/k_B$$

| Regime | Temperatura | Comportamento |
|--------|------------|---------------|
| **Fase Ordenada** (Ferromagnética) | $T < T_c$ | Quebra espontânea de simetria; magnetização $\langle M \rangle \neq 0$ |
| **Ponto Crítico** | $T = T_c \approx 2{,}269$ | Flutuações de escala infinita; clusters fractais |
| **Fase Desordenada** (Paramagnética) | $T > T_c$ | Dominância de flutuações térmicas; $\langle M \rangle = 0$ |

---

### Simulação de Monte Carlo — Algoritmo de Metropolis

Dado o espaço de estados de escala $2^{L \times L}$, a função de partição $Z$ é computacionalmente intratável de forma direta. O **Algoritmo de Metropolis** realiza amostragem de importância no ensemble canônico, obedecendo à condição de balanço detalhado.

**Regra de aceitação de Metropolis:**

$$W(s_i \to -s_i) = \min\!\left(1,\; e^{-\beta \Delta E}\right)$$

onde $\Delta E = E_\text{nova} - E_\text{atual} = 2J \, s_i \sum_{\delta} s_{i+\delta}$ é a variação de energia da inversão local do spin $i$.

Cada **Passo Monte Carlo (MCS)** consiste em $L^2$ tentativas de inversão, garantindo que cada spin tenha, em média, uma oportunidade de atualização. A termalização é atingida após um número suficiente de MCS, a partir do qual o sistema flutua ao redor do equilíbrio termodinâmico.

---

### Redes Neurais Convolucionais (CNNs)

CNNs são arquiteturas profundas otimizadas para extração hierárquica de feições espaciais em dados bidimensionais. A operação de convolução discreta é definida por:

$$S(i,j) = (I * K)(i,j) = \sum_m \sum_n I(i-m,\, j-n)\, K(m,n)$$

onde $I$ é a matriz de entrada (rede de spins) e $K$ é o kernel de pesos aprendíveis.

**Componentes da arquitetura utilizada:**

| Componente | Função |
|-----------|--------|
| `Conv2d` | Extração de feições locais via kernels 3×3 |
| `ReLU` | Não-linearidade: $f(x) = \max(0, x)$ |
| `MaxPool2d` | Redução de dimensionalidade e invariância local |
| `Dropout` | Regularização para mitigação de overfitting |
| `Linear` | Classificador final nas representações latentes |

---

### Interpretabilidade via Grad-CAM

O **Grad-CAM** (*Gradient-weighted Class Activation Mapping*) produz mapas de relevância espacial retroprojetando os gradientes de uma classe-alvo até a última camada convolucional.

**Peso de importância da k-ésima feature map para a classe c:**

$$\alpha_k^c = \frac{1}{Z} \sum_u \sum_v \frac{\partial Y^c}{\partial A_{uv}^k}$$

**Mapa de ativação final:**

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\!\left(\sum_k \alpha_k^c \, A^k\right)$$

A aplicação da ReLU garante que o mapa capture exclusivamente as feições com influência positiva direta no score da classe analisada. O resultado é um **mapa térmico** sobreposto à rede de spins original, revelando as regiões espaciais decisivas para a classificação.

---

## 🔄 Pipeline do Projeto

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PIPELINE COMPLETO                            │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │  Simulação   │    │  Dataset     │    │   Rede Neural CNN      │ │
│  │  Metropolis  │───▶│  L=40, 1000  │───▶│   3 blocos conv.       │ │
│  │  Monte Carlo │    │  amostras    │    │   PyTorch              │ │
│  └──────────────┘    └──────────────┘    └────────────┬───────────┘ │
│         │                                             │             │
│         ▼                                             ▼             │
│  ┌──────────────┐                        ┌────────────────────────┐ │
│  │  Observáveis │                        │      Grad-CAM          │ │
│  │  E(t), M(t)  │                        │  Mapas de Ativação     │ │
│  │  Termalização│                        │  Interpretabilidade    │ │
│  └──────────────┘                        └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🛠 Metodologia e Implementação

### Etapa 1 — Simulação do Modelo de Ising 2D

**Parâmetros físicos e computacionais:**

```python
L = 40          # Tamanho linear da rede quadrada
J = 1.0         # Constante de acoplamento ferromagnético
TC = 2.269      # Temperatura crítica de Onsager
SEED = 42       # Semente de reprodutibilidade
```

O sistema é simulado em `T = 2.2` (próximo à criticalidade), partindo de um estado totalmente desordenado (aleatório). A evolução temporal dos observáveis macroscópicos — energia interna $\mathcal{H}$ e magnetização absoluta $|M|$ — é monitorada ao longo de **300 passos de Monte Carlo**.

**Configuração da rede de spins em T = 2.2:**

> *Figura 1 — Snapshot da rede de spins 40×40 após termalização em T = 2.2. Vermelho: spin +1, Azul: spin -1. Observa-se a formação de domínios ferromagnéticos com flutuações características da criticalidade.*

![Modelo de Ising 2D | T=2.2](figures/ising_snapshot_T2.2.png)

**Evolução dos observáveis durante a termalização:**

> *Figura 2 — (Superior) Curva de decaimento da energia interna do sistema ao longo dos 300 passos de Monte Carlo. (Inferior) Crescimento da magnetização absoluta indicando a formação de ordem ferromagnética de longo alcance.*

![Energia e Magnetização vs Passos MC](figures/energia_magnetizacao_mc.png)

**Valores típicos durante a evolução estocástica:**

| Métrica Termodinâmica | Estado Inicial (Passo 0) | Fase Intermediária (Passo 50) | Equilíbrio (Passo 300) |
|----------------------|--------------------------|-------------------------------|------------------------|
| Energia Interna $\mathcal{H}$ | ≈ −100 J | ≈ −1.800 J | ≈ −2.500 J |
| Magnetização Absoluta $|M|$ | ≈ 0,05 | ≈ 0,35 | ≈ 0,82 |

A queda monotônica da energia nos primeiros 100 MCS evidencia a eficiência do algoritmo Metropolis na minimização da energia livre de Helmholtz. O platô de $|M| \approx 0{,}82$ confirma a **quebra espontânea de simetria** e a estabilização de domínios macroscópicos ferromagnéticos.

---

### Etapa 2 — Geração do Dataset

O dataset é composto por **1.000 amostras** de configurações de spins 40×40, geradas a partir de dois subconjuntos de temperatura correspondentes às duas fases termodinâmicas bem separadas da criticalidade:

| Classe | Label | Intervalo de Temperatura | Nº de Pontos | Amostras por Ponto |
|--------|-------|--------------------------|-------------|-------------------|
| **Fase Ordenada** (Ferromagnética) | 0 | $T \in [0{,}5;\, 2{,}0]$ | 5 | 100 |
| **Fase Desordenada** (Paramagnética) | 1 | $T \in [2{,}8;\, 5{,}0]$ | 5 | 100 |

> ⚠️ **Nota metodológica:** A janela crítica $2{,}0 < T < 2{,}8$ foi **intencionalmente excluída** do dataset de treinamento, eliminando o ruído gerado pelas flutuações de escala infinita e pelo *finite-size scaling effect*. Isso permite que a rede mapeie as características geométricas extremas de cada fase de forma limpa.

**Protocolo de amostragem:**
- **Termalização:** 300 MCS iniciais descartados (burn-in)
- **Decorrelação:** amostras salvas a cada 5 MCS para contornar o tempo de autocorrelação crítica
- **Formato final do tensor:** `(1000, 1, 40, 40)` — 1000 amostras, 1 canal, 40×40 pixels

**Exemplos de configurações por temperatura e classe:**

> *Figura 3 — Mosaico de 10 configurações de spins amostradas. Linhas superiores: fases desordenadas (alta temperatura, mistura caótica de domínios vermelhos e azuis). Linhas inferiores: fases ordenadas (baixa temperatura, dominância de uma cor indicando alinhamento ferromagnético).*

![Amostras do Dataset](figures/dataset_amostras.png)

**Distribuição balanceada das classes:**

> *Figura 4 — Histograma de distribuição das classes no dataset. 500 amostras da fase ordenada (Classe 0) e 500 amostras da fase desordenada (Classe 1), garantindo balanceamento perfeito para o treinamento.*

![Distribuição das Classes](figures/distribuicao_classes.png)

---

### Etapa 3 — Arquitetura CNN em PyTorch

As configurações de spin são normalizadas do intervalo $[-1, +1]$ para $[0, 1]$ via transformação linear $x \mapsto (x+1)/2$ e redimensionadas para o formato de entrada `(1, 40, 40)`.

**Arquitetura `IsingCNN`:**

```python
class IsingCNN(nn.Module):
    def __init__(self):
        super(IsingCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 40×40 → 20×20

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 20×20 → 10×10

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)           # 10×10 → 5×5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )
```

**Fluxo dimensional pelo backbone convolucional:**

```
Entrada:  (B, 1,  40, 40)
Block 1:  (B, 16, 20, 20)   ← Conv(1→16) + ReLU + MaxPool2d
Block 2:  (B, 32, 10, 10)   ← Conv(16→32) + ReLU + MaxPool2d
Block 3:  (B, 64, 5,  5 )   ← Conv(32→64) + ReLU + MaxPool2d
Flatten:  (B, 1600)
FC-128:   (B, 128)           ← Linear + ReLU + Dropout(0.3)
Saída:    (B, 2)             ← Linear (logits das 2 classes)
```

**Hiperparâmetros de treinamento:**

| Hiperparâmetro | Valor |
|---------------|-------|
| Otimizador | Adam |
| Taxa de aprendizado | 0.001 |
| Função de perda | CrossEntropyLoss |
| Batch size | 64 |
| Épocas | 10 |
| Split treino/teste | 80% / 20% |
| Dropout | 0.3 |

---

### Etapa 4 — Treinamento e Avaliação

**Evolução do treinamento por época:**

| Época | Loss | Acurácia |
|-------|------|----------|
| 1/10 | 0.5473 | 72,00% |
| 2/10 | 0.0722 | 99,88% |
| 3/10 | 0.0001 | 100,00% |
| 4–10/10 | 0.0000 | **100,00%** |

A convergência rápida a partir da 2ª época reflete a alta separabilidade topológica das duas fases no espaço de feições convolucionais.

**Curva de Loss de Treinamento:**

> *Figura 5 — Decaimento da função de perda (Cross-Entropy Loss) ao longo das 10 épocas de treinamento. A queda abrupta entre as épocas 1 e 3 indica que a rede aprendeu os padrões discriminativos fundamentais nas primeiras iterações.*

![Loss de Treinamento](figures/loss_treinamento.png)

**Curva de Acurácia de Treinamento:**

> *Figura 6 — Evolução da acurácia de treinamento por época. A rede atinge 100% já na 3ª época e mantém essa performance até o final do treinamento.*

![Accuracy de Treinamento](figures/accuracy_treinamento.png)

**Matriz de Confusão no conjunto de teste (200 amostras):**

> *Figura 7 — Matriz de confusão no conjunto de teste independente. Todos os 100 exemplos da fase ordenada e todos os 100 da fase desordenada foram classificados corretamente, resultando em zero erros.*

![Matriz de Confusão](figures/matriz_confusao.png)

**Relatório estatístico de classificação:**

| Fase Alvo | Precisão | Recall | F1-Score | Suporte |
|-----------|----------|--------|----------|---------|
| Classe 0 — Ordenada (Ferromagnética) | 1,00 | 1,00 | 1,00 | 100 |
| Classe 1 — Desordenada (Paramagnética) | 1,00 | 1,00 | 1,00 | 100 |
| **Média Global (Macro Average)** | **1,00** | **1,00** | **1,00** | **200** |

**Predições sobre exemplos individuais do conjunto de teste:**

> *Figura 8 — Grade de 10 configurações de spins do conjunto de teste com seus rótulos reais (`Real`) e predições da CNN (`Pred`). Todas as classificações estão corretas.*

![Predições no Conjunto de Teste](figures/predicoes_teste.png)

---

### Etapa 5 — Mapas de Ativação Grad-CAM

Para cada amostra analisada, são exibidas três representações:

1. **Lattice original** — configuração de spins 40×40 (vermelho: +1, azul: −1)
2. **Mapa Grad-CAM** — mapa térmico de relevância espacial (vermelho: alta ativação, azul: baixa ativação)
3. **Sobreposição** — fusão do mapa Grad-CAM sobre a rede de spins para localização visual das regiões decisivas

**Mapas Grad-CAM por classe:**

> *Figura 9 — Resultados Grad-CAM para 4 amostras (1 ordenada + 3 desordenadas). **Linha 1 (Real=0, Ordenada):** o mapa de ativação concentra-se no interior do bulk ferromagnético homogêneo, ignorando as bordas periódicas — a rede reconhece a uniformidade de alinhamento como assinatura da fase de baixa temperatura. **Linhas 2–4 (Real=1, Desordenadas):** a atenção distribui-se de forma difusa e fragmentada, com picos nas regiões de alta frequência de transição entre spins vizinhos — a rede detecta a entropia geométrica local característica da fase paramagnética.*

![Mapas Grad-CAM](figures/gradcam_resultados.png)

**Interpretação física dos mapas Grad-CAM:**

| Fase | Padrão de Ativação Grad-CAM | Interpretação Física |
|------|----------------------------|----------------------|
| **Ordenada** ($T < T_c$) | Alta ativação no bulk ferromagnético; baixa nas bordas | A CNN reconhece a **coerência espacial de spins** como assinatura da ordem de longo alcance |
| **Desordenada** ($T > T_c$) | Ativação difusa e granular, com picos nas interfaces de domínio | A CNN detecta a **alta densidade de fronteiras de domínio** e a **entropia geométrica local** |

Este resultado demonstra que a CNN **aprendeu espontaneamente** conceitos físicos reais — sem que esses conceitos tenham sido explicitamente fornecidos como rótulos durante o treinamento.

---

## 📊 Resultados

### Resumo Consolidado

| Métrica | Resultado |
|---------|-----------|
| Acurácia no conjunto de teste | **100,00%** |
| F1-Score (macro average) | **1,00** |
| Épocas até convergência total | **3** |
| Total de amostras de treino | 800 |
| Total de amostras de teste | 200 |
| Arquitetura | CNN 3 blocos conv. + FC |
| Parâmetros treináveis | ~210k |

### Conclusões Físicas

1. **Validação da separabilidade de fases:** A CNN aprendeu a discriminar perfeitamente as fases termodinâmicas do Modelo de Ising 2D a partir de imagens brutas de spins, sem cálculo explícito de observáveis como magnetização ou susceptibilidade.

2. **Física emergente nos filtros convolucionais:** Os mapas Grad-CAM revelaram que os filtros da última camada convolucional sintonizam sua resposta com:
   - **Fase ordenada:** Coerência e homogeneidade espacial dos domínios ferromagnéticos.
   - **Fase desordenada:** Densidade e fragmentação das fronteiras de domínio.

3. **XAI como ponte física:** A metodologia Grad-CAM estabelece um vínculo verificável entre representações latentes neurais e conceitos físicos concretos, desmistificando a natureza de "caixa-preta" dos modelos de aprendizado profundo.

4. **Perspectivas futuras:** Esta abordagem é diretamente extensível a sistemas mais complexos como *spin glasses*, modelos frustrados e sistemas quânticos, onde os parâmetros de ordem não são conhecidos a priori.

---

## 💻 Requisitos e Instalação

### Dependências

```bash
Python >= 3.9
torch >= 2.0.0
numpy >= 1.24.0
matplotlib >= 3.7.0
scikit-learn >= 1.2.0
tqdm >= 4.65.0
```

### Instalação

```bash
# Clone o repositório
git clone https://github.com/<seu-usuario>/ising-cnn-gradcam.git
cd ising-cnn-gradcam

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
source venv/bin/activate        # Linux/macOS
# venv\Scripts\activate         # Windows

# Instale as dependências
pip install -r requirements.txt
```

### Suporte a GPU (opcional)

O código detecta automaticamente a disponibilidade de CUDA:

```python
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

Para treinamento em GPU, instale a versão CUDA do PyTorch conforme as instruções em [pytorch.org](https://pytorch.org).

---

## ▶️ Como Executar

### Execução completa (pipeline integrado)

```bash
python Modelo_ising_grad.py
```

O script executa sequencialmente as 4 etapas:

```
============================================================
ETAPA 1 — SIMULAÇÃO DO MODELO DE ISING
============================================================
[Gera snapshot em T=2.2, plota energia e magnetização vs MCS]

============================================================
ETAPA 2 — GERAÇÃO DO DATASET
============================================================
[Amostra 1000 configurações nas fases ordenada e desordenada]
Dataset final: (1000, 1, 40, 40)

============================================================
ETAPA 3 — CNN
============================================================
Epoch 1/10  | Loss=0.5473 | Acc=72.00%
Epoch 2/10  | Loss=0.0722 | Acc=99.88%
...
Epoch 10/10 | Loss=0.0000 | Acc=100.00%

============================================================
ETAPA 4 — GRAD-CAM
============================================================
[Gera mapas de ativação para amostras selecionadas]
```

Todos os gráficos são salvos automaticamente na pasta `results/`.

### Execução via Jupyter Notebook

```bash
jupyter notebook Modelo_ising_grad.ipynb
```

---

## 📁 Estrutura do Repositório

```
ising-cnn-gradcam/
│
├── Modelo_ising_grad.py          # Script principal (pipeline completo)
├── Modelo_ising_grad.ipynb       # Notebook Jupyter com outputs
│
├── requirements.txt              # Dependências Python
├── README.md                     # Este arquivo
│
├── results/                      # Saídas geradas automaticamente
│   ├── ising_snapshot_T2.2.png   # Snapshot da rede em T=2.2
│   ├── energia_magnetizacao.png  # Curvas de termalização
│   ├── dataset_amostras.png      # Mosaico de amostras do dataset
│   ├── distribuicao_classes.png  # Histograma de classes
│   ├── loss_treinamento.png      # Curva de loss por época
│   ├── accuracy_treinamento.png  # Curva de acurácia por época
│   ├── matriz_confusao.png       # Matriz de confusão no teste
│   ├── predicoes_teste.png       # Grade de predições individuais
│   └── gradcam_resultados.png    # Mapas de ativação Grad-CAM
│
└── figures/                      # Figuras para o README
    └── ...
```

---

## 📚 Referências

@article{ising1925beitrag,
  author  = {Ising, Ernst},
  title   = {Beitrag zur Theorie des Ferromagnetismus},
  journal = {Zeitschrift für Physik},
  volume  = {31},
  number  = {1},
  pages   = {253--258},
  year    = {1925}
}

@article{onsager1944crystal,
  author  = {Onsager, Lars},
  title   = {Crystal statistics. {I}. A two-dimensional {Ising} model with an order-disorder transition},
  journal = {Physical Review},
  volume  = {65},
  number  = {3--4},
  pages   = {117},
  year    = {1944}
}

@inproceedings{selvaraju2017grad,
  author    = {Selvaraju, Ramprasaath R. and others},
  title     = {{Grad-CAM}: Visual explanations from deep networks via gradient-based localization},
  booktitle = {Proceedings of the IEEE International Conference on Computer Vision (ICCV)},
  pages     = {618--626},
  year      = {2017}
}

@article{paszke2019pytorch,
  author  = {Paszke, Adam and others},
  title   = {{PyTorch}: An imperative style, high-performance deep learning library},
  journal = {Advances in Neural Information Processing Systems},
  volume  = {32},
  pages   = {8026--8037},
  year    = {2019}
}
```

<div align="center">

**André Luiz Magalhães de Oliveira - Físico Médico**

</div>
