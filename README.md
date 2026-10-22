# machado-gpt — Gerador de texto no estilo de Machado de Assis

Um transformer GPT minimalista, escrito do zero em PyTorch, treinado em 11 obras
de Machado de Assis para gerar novos textos no estilo do autor.

> Capitu n'um companheiro de palavras cabeças, e, antes de apenas alheias.
> Pedia esquecer o padre Luiz Garcia que se deixasse ao meu pae, mas tinha um
> olhar de sentimento, alguma cousa que os tres annos, um manuscripto…
>
> — texto gerado pelo modelo de 10,8 M de parâmetros (`--temperature 0.7 --top_k 20`)

O projeto é **didático**: a ideia é entender na prática cada componente de um
transformer, seguindo a aula
["Let's build GPT from scratch"](https://www.youtube.com/watch?v=kCc8FmEb1nY)
de Andrej Karpathy. Não usamos frameworks de alto nível (HuggingFace, Keras etc.).

## Arquitetura

- Tokenização character-level (um token = um caractere)
- Embedding de tokens + positional encoding
- N blocos de transformer, cada um com:
  - Multi-head self-attention
  - Feed-forward (MLP)
  - Layer normalization
- Head de linguagem (projeção para o vocabulário)

## Estrutura

```
machado-gpt/
├── data/
│   ├── raw/                    # cache dos .txt baixados do Gutenberg (ignorado pelo git)
│   └── dataset.txt             # corpus limpo usado no treino
├── src/
│   ├── collect.py              # download e limpeza das obras
│   ├── data.py                 # tokenizador, split treino/validação e batches
│   ├── model.py                # definição do modelo GPT
│   ├── train.py                # loop de treinamento
│   ├── generate.py             # geração de texto a partir de um prompt
│   ├── plot_loss.py            # curvas de loss
│   └── visualize_attention.py  # heatmaps dos pesos de atenção
├── figures/                    # figuras geradas pelos scripts de análise
├── checkpoints/                # modelos treinados (ignorado pelo git)
├── requirements.txt
└── README.md
```

## Instalação

Requer Python 3.11+.

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Como usar

Todos os comandos são rodados a partir da raiz do projeto.

### 1. Coleta

```bash
python src/collect.py
```

Baixa as obras do Gutenberg, remove cabeçalho e rodapé de licença, junta as
linhas de cada parágrafo e salva `data/dataset.txt`. O texto bruto fica em cache
em `data/raw/`; rodando de novo, as obras já baixadas não são baixadas outra vez.

Para inspecionar o tokenizador, a divisão treino/validação e um batch de exemplo:

```bash
python src/data.py
```

### 2. Treino

```bash
python src/train.py
```

Com os padrões (5000 passos), leva cerca de 80 minutos numa CPU i5 de 13ª
geração. A cada 250 passos o script mostra a loss de treino e validação, o tempo
restante e uma amostra de texto. Tudo é salvo em `checkpoints/`:

| Arquivo          | Conteúdo                                              |
|------------------|-------------------------------------------------------|
| `best.pt`        | modelo com a menor val loss (usado na geração)        |
| `last.pt`        | estado mais recente, para retomar o treino            |
| `tokenizer.json` | vocabulário usado no treino                           |
| `train_log.csv`  | loss de treino/validação a cada avaliação             |

Opções úteis:

```bash
# teste rápido
python src/train.py --max_iters 200 --eval_interval 100

# mudar a arquitetura ou o treino
python src/train.py --n_layer 6 --n_embd 192 --n_head 6 --learning_rate 1e-3
```

Rodar `python src/train.py` sem `--resume` começa do zero e sobrescreve os
checkpoints existentes.

Para o modelo maior (configuração final da aula, ~10,8 M de parâmetros), use
uma GPU. Num notebook do Kaggle ou do Colab, a partir da raiz do projeto:

```bash
python -u src/train.py --n_embd 384 --n_head 6 --n_layer 6 --dropout 0.2 \
    --batch_size 64 --max_iters 5000 --eval_interval 500 --eval_iters 100
```

Numa GPU P100 do Kaggle, isso levou cerca de 50 minutos.

### 3. Retomada

Aperte **Ctrl+C** a qualquer momento: o estado atual é salvo em `last.pt`. Para
continuar de onde parou:

```bash
python src/train.py --resume
```

A arquitetura vem sempre do checkpoint; `--max_iters` pode ser aumentado para
treinar por mais passos.

### 4. Geração

```bash
python src/generate.py --prompt "Capitu"
```

O texto aparece caractere por caractere (streaming). Ctrl+C interrompe a
geração. Parâmetros de amostragem:

| Parâmetro          | Padrão          | Efeito                                                   |
|--------------------|-----------------|----------------------------------------------------------|
| `--prompt`         | `"\n"`          | texto inicial (`\n` vira quebra de linha)                |
| `--max_new_tokens` | 500             | quantos caracteres gerar                                 |
| `--temperature`    | 0.8             | < 1: mais conservador; > 1: mais criativo e com erros    |
| `--top_k`          | nenhum          | sorteia só entre os k caracteres mais prováveis          |
| `--num_samples`    | 1               | quantas gerações independentes                           |
| `--seed`           | nenhum          | fixa o resultado (reprodutível)                          |
| `--checkpoint`     | `checkpoints/best.pt` | checkpoint a usar                                  |

Exemplos:

```bash
# conservador e focado
python src/generate.py --prompt "Capitu" --temperature 0.5 --top_k 10

# mais ousado
python src/generate.py --prompt "Capitu" --temperature 1.3

# três variações do mesmo início, reprodutíveis
python src/generate.py --prompt "Era uma vez" --num_samples 3 --seed 42

# texto longo a partir do estado do último passo de treino
python src/generate.py --checkpoint checkpoints/last.pt --max_new_tokens 2000
```

O prompt só pode conter caracteres que aparecem nas obras (emojis, por exemplo,
não funcionam).

### 5. Análise dos modelos

Dois scripts geram figuras em `figures/` (requerem `matplotlib`):

```bash
# curvas de loss (treino, validação e a diferença entre elas)
python src/plot_loss.py
python src/plot_loss.py --runs checkpoints_pequeno checkpoints --output figures/loss_curves.png

# pesos de atenção de cada cabeça (padrão: modelo em checkpoints/, última camada)
python src/visualize_attention.py
python src/visualize_attention.py --layer 0
python src/visualize_attention.py --checkpoint checkpoints_pequeno/best.pt --text "Era uma vez"
```

- **`plot_loss.py`** lê o `train_log.csv` e o `best.pt` de cada pasta, plota
  train e val loss por passo (com a loss de chute aleatório, ln(vocab_size),
  como referência) e a diferença val − train, que indica overfitting. No
  terminal, mostra uma tabela com parâmetros, passos, caracteres vistos no
  treino e as losses finais.
- **`visualize_attention.py`** roda o modelo sobre um texto curto (até ~60
  caracteres, para os rótulos ficarem legíveis) e salva um heatmap por cabeça
  da camada escolhida (`figures/attention_layer{N}.png`) e uma visão geral de
  todas as camadas × cabeças (`figures/attention_overview.png`, sobrescrita a
  cada execução). No terminal, lista os caracteres que mais recebem atenção em
  algumas posições do texto.

## Resultados

Foram treinados dois modelos, ambos por 5000 passos:

| Modelo  | Parâmetros | Arquitetura (n_embd / cabeças / camadas) | Dados    | Hardware   | Melhor val loss |
|---------|-----------:|------------------------------------------|----------|------------|----------------:|
| Pequeno |     0,85 M | 128 / 4 / 4, dropout 0.1, batch 32       | 5 obras  | CPU, ~75 min | 1,4304 |
| Grande  |    10,8 M  | 384 / 6 / 6, dropout 0.2, batch 64       | 11 obras | GPU, ~50 min | 1,2188 |

Para comparação, um modelo que chuta ao acaso tem loss ln(vocab_size) ≈ 4,8.
O modelo grande escreve quase só palavras reais, na grafia da época, e frases
curtas gramaticais, mas ainda sem sentido entre as frases: é o limite esperado
de um modelo character-level desse tamanho.

![Curvas de loss](figures/loss_curves.png)

Pesos de atenção do modelo grande para o texto
`--Capitu, que é isso? perguntou José Dias.` Na primeira camada, as cabeças
capturam padrões locais (como o caractere anterior); nas últimas, relações mais
longas (como o início da palavra atual):

![Atenção em todas as camadas](figures/attention_overview.png)

| Figura | Conteúdo |
|--------|----------|
| [`attention_layer0.png`](figures/attention_layer0.png) | modelo grande, primeira camada |
| [`attention_layer5.png`](figures/attention_layer5.png) | modelo grande, última camada |
| [`attention_layer3.png`](figures/attention_layer3.png) | modelo pequeno, última camada |

Os checkpoints não estão no repositório (o `best.pt` do modelo grande tem
~139 MB). Para gerar texto, treine um modelo com `src/train.py` ou baixe o
`best.pt` e o `tokenizer.json` da página de Releases e coloque-os em
`checkpoints/`.

## Hiperparâmetros padrão

| Parâmetro       | Valor | Descrição                              |
|-----------------|-------|----------------------------------------|
| `block_size`    | 256   | Tamanho do contexto (em caracteres)    |
| `n_embd`        | 128   | Dimensão dos embeddings                |
| `n_head`        | 4     | Número de cabeças de atenção           |
| `n_layer`       | 4     | Número de blocos de transformer        |
| `dropout`       | 0.1   | Taxa de dropout                        |
| `batch_size`    | 32    | Tamanho do batch                       |
| `learning_rate` | 3e-4  | Taxa de aprendizado                    |

## Dataset

Obras de Machado de Assis em domínio público, do
[Project Gutenberg](https://www.gutenberg.org):

| Obra                             | Caracteres |
|----------------------------------|-----------:|
| Dom Casmurro                     |    378.928 |
| Memórias Póstumas de Brás Cubas  |    364.903 |
| Quincas Borba                    |    453.033 |
| Esaú e Jacó                      |    414.921 |
| Papéis Avulsos (contém O Alienista) | 322.576 |
| Helena                           |    336.180 |
| Memorial de Aires                |    289.033 |
| Iaiá Garcia                      |    330.227 |
| A Mão e a Luva                   |    203.671 |
| Histórias sem Data               |    299.536 |
| Relíquias de Casa Velha          |    278.349 |

No total são cerca de 3,67 milhões de caracteres, com um vocabulário de 122
caracteres únicos. A validação usa os últimos 10% de cada obra.

Para adicionar outra obra, inclua o título e a URL do `.txt` na constante
`WORKS`, no topo de `src/collect.py`.

O `collect.py` remove o cabeçalho e o rodapé de licença do Gutenberg e junta as
linhas de cada parágrafo numa só. O travessão "—", usado em algumas edições, é
convertido para "--", a forma usada no resto do corpus. As edições são antigas e usam a grafia da época
(ex.: "Braz", "theoria"), que o modelo vai aprender junto com o estilo.
