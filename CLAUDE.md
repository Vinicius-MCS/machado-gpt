# machado-gpt — Gerador de texto no estilo de Machado de Assis

## Objetivo
Implementar um transformer GPT minimalista (estilo Andrej Karpathy) treinado
em obras de Machado de Assis, baixadas do Project Gutenberg.
O foco é didático: entender cada componente do transformer na prática.

## Stack
- Python 3.11+
- PyTorch (principal framework)
- requests (download dos textos)
- Sem frameworks de alto nível (sem HuggingFace, sem Keras)

## Arquitetura alvo
- Tokenização: character-level (um token = um caractere, padrão) ou BPE com
  caracteres como base (src/bpe.py, --tokenizer bpe), para comparação
- Métrica comparável entre tokenizadores: bits por caractere
- Embedding + positional encoding
- N blocos de self-attention (multi-head)
- Feed-forward MLP por bloco
- Layer normalization
- Head de linguagem (projeção → vocabulário)

## Dataset
- Fonte: Project Gutenberg (https://www.gutenberg.org), domínio público
- Obras (11; lista em WORKS, no topo de src/collect.py):
  Dom Casmurro, Memórias Póstumas de Brás Cubas, Quincas Borba, Esaú e Jacó,
  Papéis Avulsos, Helena, Memorial de Aires, Iaiá Garcia, A Mão e a Luva,
  Histórias sem Data, Relíquias de Casa Velha
- Tamanho: ~3,67 milhões de caracteres, vocabulário de 122 caracteres
- Limpeza: travessão "—" normalizado para "--" (padrão do corpus nos diálogos)
- Cache bruto: data/raw/{id}.txt
- Saída: data/dataset.txt (texto puro, um parágrafo por linha, parágrafos
  separados por \n\n e obras separadas por \n\n---\n\n)

## Convenções de código
- snake_case para funções e variáveis
- PascalCase para classes
- Type hints em todas as funções
- Comentários explicativos em português (pt-BR)
- Cada arquivo deve ter docstring explicando seu propósito

## Como rodar
- Coleta: python src/collect.py
- Treino: python src/train.py
- Treino com BPE: python src/train.py --tokenizer bpe --vocab_size 1000 --out_dir checkpoints_bpe1000
- Geração: python src/generate.py --prompt "Capitu"
- Testes: python -m pytest (dependências em requirements-dev.txt)

## Hiperparâmetros padrão (ajustáveis)
- block_size: 256 (contexto de 256 tokens; 256 caracteres no tokenizador de caractere)
- n_embd: 128
- n_head: 4
- n_layer: 4
- dropout: 0.1
- batch_size: 32
- learning_rate: 3e-4

## Referência principal
Aula "Let's build GPT from scratch" de Andrej Karpathy (YouTube)