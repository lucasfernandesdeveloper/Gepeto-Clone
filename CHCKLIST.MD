### Preparação do Ambiente e Dados
- [ ] Ambiente
- [ ] Dataset
- [ ] Tokenização
- [ ] Data Loader  

### Arquitetura do Transformer 
- [ ] Embeddings 
- [ ] Residual Connections
- [ ] Layer Norm 
- [ ] Self-Attention 
- [ ] Causal Masking 
- [ ] Multi-Head Attention 
- [ ] MLP Layer 

### Loop de Treinamento 
- [ ] Forward Pass
- [ ] Otimizador 
- [ ] Geração Inicial 
- [ ] Loop Principal 

### Otimização
- [ ] Mixed Precision (AMP) 
- [ ] PyTorch Complie 
- [ ] Flash Attention 

### Escalando 
- [ ] Byte Pair Enconding 
- [ ] Pesos Pré-treinados 


### 1. O Modelo "Bigram" e Pipeline de Dados
Para este estágio, o repositório principal do nanoGPT é "avançado demais". O Karpathy mantém a versão educativa em um script separado que serve como a base perfeita:
* **Arquivo:** `v2_shakespeare.py` (Disponível no histórico ou no vídeo "Let's build GPT").
* **O que olhar:** As primeiras 50 linhas mostram a tokenização de caracteres e o `BigramLanguageModel`. No repo principal, veja o `data/shakespeare/prepare.py` para entender como os dados são preparados.

### 2 e 3. Self-Attention, Multi-Head e Resíduos
Aqui você já entra no núcleo do modelo.
* **Arquivo:** `model.py`
* **O que olhar:** * Procure pela classe `CausalSelfAttention`. Lá você verá as projeções para **Query, Key e Value**.
    * Observe o uso de `split` para criar as **Múltiplas Cabeças (Multi-Head)**.
    * A **Máscara Causal** é implementada via `register_buffer("bias", ...)` (uma matriz triangular de zeros e uns).
    * As **Conexões Residuais** aparecem no método `forward` como `x = x + self.attn(self.ln_1(x))`.

### 4. O Bloco Transformer (A Unidade Repetível)
Toda a lógica de "embrulhar" a atenção e o feed-forward em uma peça única está aqui.
* **Arquivo:** `model.py`
* **O que olhar:** Procure pela classe `Block`. Ela é o "tijolo" que empilha a `LayerNorm`, a `CausalSelfAttention` e o `MLP` (Multi-Layer Perceptron).


### 5. Escalonamento e Otimização
Para ver como tudo isso se torna um sistema de alto desempenho:
* **Arquivo:** `train.py` (O motor de treino).
* **O que olhar:** * **Hiperparâmetros:** O dicionário no topo do arquivo com `learning_rate`, `min_lr`, `weight_decay`.
    * **AdamW:** A configuração do otimizador para ignorar certos parâmetros (como biases) no weight decay.
    * **Geração:** O arquivo `sample.py` contém a lógica de carregar o modelo treinado e gerar texto novo.

### Resumo de Navegação:
| Etapa | Arquivo Principal | Classe/Função Chave |
| :--- | :--- | :--- |
| **Pipeline/Data** | `data/shakespeare/prepare.py` | `train_ids`, `val_ids` |
| **Arquitetura** | `model.py` | `GPT`, `Block`, `CausalSelfAttention` |
| **Treinamento** | `train.py` | `optimizer.step()`, `loss.backward()` |
| **Geração** | `sample.py` | `model.generate()` |


### Resumo do Fluxo de Dados:
> **Dados Brutos** $\rightarrow$ **Tokens** $\rightarrow$ **Embeddings** $\rightarrow$ **Blocos de Atenção (xN)** $\rightarrow$ **Layer Final** $\rightarrow$ **Probabilidades (Softmax)**.
