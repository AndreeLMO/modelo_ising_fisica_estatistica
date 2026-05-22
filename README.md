# Detecção de Transições de Fase no Modelo de Ising 2D via Redes Neurais Convolucionais e Interpretabilidade por Grad-CAM

> **Projeto de Pesquisa Científica / Computação de Alto Desempenho**
> **Autor:** André Luiz
> **Área:** Física Estatística Computacional / Deep Learning / Inteligência Artificial Explicável (XAI)
> **Frameworks Principais:** PyTorch, NumPy, Scikit-Learn, Matplotlib

---

## 📑 Resumo

O estudo das transições de fase e dos fenômenos críticos dentro da física estatística de muitos corpos tradicionalmente requer formulações analíticas complexas e simulações numéricas massivas baseadas no método de Monte Carlo. Observáveis macroscópicos como magnetização, energia interna, calor específico e susceptibilidade magnética são utilizados para mapear os limites de fase. 

Este projeto propõe um paradigma alternativo e complementar: a utilização de **Aprendizado Profundo (Deep Learning)** para classificar estados termodinâmicos diretamente a partir de suas configurações microscópicas de spins, sem o cálculo prévio de variáveis macroscópicas. 

A fim de romper com a opacidade metodológica que comumente rotula as redes neurais convolucionais como "caixas-pretas", implementou-se o algoritmo **Grad-CAM (Gradient-weighted Class Activation Mapping)**. Este algoritmo mapeia o fluxo de gradientes nas camadas convolucionais mais profundas, revelando de forma inequívoca quais estruturas geométricas e correlações espaciais de spins (como fronteiras de domínio e clusters fractais) foram determinantes para a decisão da rede em cada fase termodinâmica.

---

## 🏛️ 1. Fundamentação Teórica

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

## 📈 4. Análise de Resultados e Discussão Detalhada

Esta seção apresenta a análise quantitativa e qualitativa dos dados obtidos nas três grandes fases do projeto: a termalização estocástica do Modelo de Ising, a convergência estatística da arquitetura convolucional profunda e a decodificação física dos mapas de relevância gerados pelo algoritmo Grad-CAM.

### 4.1 Dinâmica de Termalização e Validação do Motor de Monte Carlo
Antes da amostragem em larga escala para a construção do dataset, realizou-se um teste piloto para validar a integridade física do algoritmo de Metropolis. O sistema, configurado em uma rede quadrada de tamanho $L = 40$ com condições de contorno periódicas, foi inicializado em um estado puramente aleatório (configuração de temperatura infinita, onde $\langle M \rangle \approx 0$) e submetido a uma evolução estocástica sob temperatura subcrítica fixa de $T = 2,2$ (imediatamente abaixo da temperatura de transição de Onsager, $T_c \approx 2,269$).

O comportamento temporal das variáveis macroscópicas do sistema ao longo de 300 Passos Monte Carlo (MCS) está documentado na imagem abaixo:

![Evolução da Termalização e Relaxação Magnética](results/termalizacao_ising.png)  
*Figura 1: Curvas de relaxação temporal. À esquerda, o decaimento estritamente decrescente da Energia Interna total ($H$). À direita, a evolução correspondente do módulo do parâmetro de ordem, a Magnetização Líquida Absoluta ($|M|$).*

**Análise Física:**
Nos primeiros 50 MCS, observa-se uma transição abrupta: a energia interna decai exponencialmente até atingir um platô de mínima energia livre. Paralelamente, a magnetização absoluta migra de valores nulos para um patamar de estabilidade flutuando em torno de $|M| \approx 0,82$. 

Esse comportamento comprova que o algoritmo de amostragem por importância funcionou corretamente, superando o transiente inicial e conduzindo o sistema à quebra espontânea de simetria $Z_2$, caracterizada pelo alinhamento majoritário de domínios ferromagnéticos.

---

### 4.2 Desempenho e Convergência da Rede Neural Convolucional
O conjunto de dados, composto por 1.000 configurações de spins normalizadas no intervalo $[0, 1]$ e rotuladas de forma binária (Classe 0: Fase Ordenada; Classe 1: Fase Desordenada), foi dividido na proporção clássica de 80% para treinamento e 20% para teste e validação estatística.

A arquitetura convolucional profunda `IsingCNN` demonstrou rápida adaptabilidade aos padrões morfológicos das matrizes de spins. A curva de perda (*Cross-Entropy Loss*) ao longo das 10 épocas de otimização pelo algoritmo Adam está representada a seguir:

![Curva de Aprendizado e Minimização da Perda](results/curvas_treinamento.png)  
*Figura 2: Curva de perda do modelo demonstrando convergência assintótica suave a partir da 5ª época de treinamento.*

O modelo alcançou uma **acurácia de 100%** no conjunto de testes independente. Esse desempenho ideal é justificado pela metodologia de amostragem, que excluiu deliberadamente a janela flutuante imediata de criticalidade ($2,0 < T < 2,8$). Ao focar nos limites assintóticos das fases ferromagnética e paramagnética, eliminou-se o ruído gerado pelas flutuações de escala infinita (*finite-size scaling effects*), permitindo à rede mapear com perfeição as assinaturas geométricas de cada regime térmico.

Abaixo, o relatório estatístico detalhado confirma a estabilidade das métricas:

```text
              precision    recall  f1-score   support

    Classe 0       1.00      1.00      1.00       100
    Classe 1       1.00      1.00      1.00       100

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200
