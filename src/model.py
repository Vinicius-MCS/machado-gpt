"""Arquitetura do GPT (transformer decoder-only), construída do zero.

Segue a aula "Let's build GPT from scratch", de Andrej Karpathy. As classes
aparecem de baixo para cima, das peças menores para o modelo completo:

- Head:               uma cabeça de self-attention causal.
- MultiHeadAttention: várias cabeças em paralelo.
- FeedForward:        MLP aplicado a cada token isoladamente.
- Block:              atenção + feed-forward, com residual e LayerNorm.
- GPTLanguageModel:   embeddings + N blocos + cabeça de linguagem.

Convenção dos comentários de shape:
    B = batch, T = tempo/posição, C = canais (n_embd), hs = head_size,
    V = vocab_size.

Uso (demonstração com o modelo ainda não treinado):
    python src/model.py
"""

import math
import sys
from collections.abc import Iterator
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    """Hiperparâmetros da arquitetura (padrões do CLAUDE.md)."""

    vocab_size: int
    block_size: int = 256  # tamanho máximo do contexto, em tokens
    n_embd: int = 128  # dimensão dos embeddings (C)
    n_head: int = 4  # número de cabeças de atenção por bloco
    n_layer: int = 4  # número de blocos
    dropout: float = 0.1

    def __post_init__(self) -> None:
        # Cada cabeça recebe uma fatia de n_embd / n_head canais; a divisão
        # precisa ser exata para as saídas concatenadas voltarem a ter n_embd.
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) precisa ser divisível por n_head ({self.n_head})."
            )


class Head(nn.Module):
    """Uma cabeça de self-attention causal."""

    def __init__(self, config: GPTConfig, head_size: int) -> None:
        super().__init__()
        # Cada token produz três vetores a partir do seu embedding:
        # - query: "o que eu estou procurando?"
        # - key:   "o que eu contenho?" (o que ofereço a quem procura)
        # - value: "o que eu comunico" se alguém prestar atenção em mim
        self.key = nn.Linear(config.n_embd, head_size, bias=False)
        self.query = nn.Linear(config.n_embd, head_size, bias=False)
        self.value = nn.Linear(config.n_embd, head_size, bias=False)
        # Máscara triangular inferior: a linha t tem 1 nas colunas 0..t.
        # register_buffer faz ela acompanhar o modelo (.to(device), state_dict)
        # sem ser tratada como parâmetro treinável.
        self.register_buffer("tril", torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)
        self.head_size = head_size
        # Gancho para visualização (visualize_attention.py): com
        # record_attention=True, o forward guarda os pesos da atenção em
        # last_attention. O padrão é False, e aí nada é guardado: treino e
        # geração funcionam exatamente como antes, sem gastar memória extra.
        self.record_attention: bool = False
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        k = self.key(x)  # (B, T, C) -> (B, T, hs)
        q = self.query(x)  # (B, T, C) -> (B, T, hs)

        # Afinidade entre cada par de posições: o quanto a query de t "casa"
        # com a key de cada outra posição.
        # Dividimos por sqrt(hs) porque o produto escalar de vetores com
        # variância 1 tem variância hs. Sem a escala, os scores ficam grandes,
        # o softmax satura e vira quase one-hot: cada token prestaria atenção
        # em um único outro, e os gradientes ficariam minúsculos.
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5  # (B, T, hs) @ (B, hs, T) -> (B, T, T)

        # Máscara causal: a posição t só pode olhar para 0..t. Se pudesse ver
        # o futuro, o modelo "colaria" a resposta no treino e não aprenderia
        # a prever. Recortamos a máscara para T x T porque a sequência pode
        # ser menor que block_size (por exemplo, no início da geração).
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        # -inf vira peso 0 depois do softmax; cada linha soma 1.
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        if self.record_attention:
            # Guardamos antes do dropout, para ver os pesos "puros" do modelo.
            # detach() desliga o tensor do grafo de gradientes.
            self.last_attention = wei.detach()  # (B, T, T)
        wei = self.dropout(wei)

        # Média ponderada dos values, com os pesos da atenção.
        v = self.value(x)  # (B, T, C) -> (B, T, hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    """Várias cabeças de self-attention em paralelo.

    Intuição: cada cabeça pode aprender a prestar atenção em um tipo
    diferente de relação (a letra anterior, o início da palavra, a abertura
    de aspas...), em vez de uma única cabeça ter que misturar tudo.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        head_size = config.n_embd // config.n_head
        self.heads = nn.ModuleList([Head(config, head_size) for _ in range(config.n_head)])
        # Projeção que mistura as saídas das cabeças e volta para o caminho residual.
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # n_head tensores (B, T, hs) concatenados -> (B, T, n_head * hs) = (B, T, C)
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.dropout(self.proj(out))  # (B, T, C) -> (B, T, C)
        return out


class FeedForward(nn.Module):
    """MLP simples aplicado a cada posição de forma independente.

    A atenção é a "comunicação": os tokens trocam informação entre si. O
    feed-forward é a "computação": cada token processa sozinho o que recebeu.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # A camada interna é 4x maior, como no paper "Attention Is All You Need".
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.ReLU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # (B, T, C) -> (B, T, 4C) -> (B, T, C)


class Block(nn.Module):
    """Bloco do transformer: comunicação (atenção) seguida de computação (MLP).

    Duas ideias tornam viável empilhar muitos blocos:
    - Conexões residuais (x = x + f(x)): a soma deixa o gradiente fluir direto
      da loss até as primeiras camadas, como uma "via expressa". Cada bloco só
      precisa aprender um ajuste sobre o que já existe.
    - LayerNorm: normaliza cada token para média 0 e variância 1 antes de
      entrar na subcamada, mantendo as ativações em uma escala estável.
      Usamos "pre-norm" (normalizar antes de sa/ffwd), como na aula.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.sa(self.ln1(x))  # (B, T, C)
        x = x + self.ffwd(self.ln2(x))  # (B, T, C)
        return x


class GPTLanguageModel(nn.Module):
    """Modelo de linguagem GPT completo."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        # Embedding de token: "qual caractere é este?"
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        # Embedding de posição (aprendido): "em que posição ele está?".
        # Sem ele, a atenção trataria a entrada como um conjunto: "amor" e
        # "roma" teriam a mesma representação, pois a atenção é só uma média
        # ponderada e não sabe a ordem dos caracteres.
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)  # LayerNorm final
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Inicialização da versão final da aula: pesos pequenos, bias zero."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Calcula os logits do próximo token e, se houver targets, a loss."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(
                f"Sequência de {T} tokens excede o block_size ({self.config.block_size})."
            )

        tok_emb = self.token_embedding(idx)  # (B, T) -> (B, T, C)
        pos = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.position_embedding(pos)  # (T,) -> (T, C)
        x = tok_emb + pos_emb  # (B, T, C) + (T, C) -> (B, T, C), via broadcasting
        x = self.blocks(x)  # (B, T, C)
        x = self.ln_f(x)  # (B, T, C)
        logits = self.lm_head(x)  # (B, T, C) -> (B, T, V)

        if targets is None:
            return logits, None
        # cross_entropy espera (N, V) e (N,): achatamos batch e tempo juntos.
        loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate_tokens(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> Iterator[int]:
        """Gera tokens um a um, devolvendo cada id assim que é sorteado.

        É um gerador (yield) para permitir streaming: quem chama pode imprimir
        cada caractere sem esperar a geração inteira. idx tem shape (1, T).
        """
        for _ in range(max_new_tokens):
            # O modelo só conhece block_size posições: usamos os últimos tokens.
            idx_cond = idx[:, -self.config.block_size :]  # (1, T') com T' <= block_size
            logits, _ = self(idx_cond)  # (1, T', V)
            # Só interessa a previsão da última posição.
            # temperature < 1 deixa a distribuição mais "confiante";
            # temperature > 1, mais aleatória.
            logits = logits[:, -1, :] / temperature  # (1, V)
            if top_k is not None:
                # Descarta tudo fora dos k tokens mais prováveis.
                kth_value = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth_value, float("-inf"))
            probs = F.softmax(logits, dim=-1)  # (1, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (1, 1)
            idx = torch.cat([idx, idx_next], dim=1)  # (1, T + 1)
            yield idx_next.item()

    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Versão sem streaming: devolve o contexto inicial + tokens gerados."""
        new_ids = list(self.generate_tokens(idx, max_new_tokens, temperature, top_k))
        new_tensor = torch.tensor([new_ids], dtype=torch.long, device=idx.device)  # (1, max_new_tokens)
        return torch.cat([idx, new_tensor], dim=1)  # (1, T + max_new_tokens)

    def num_params(self) -> int:
        """Número total de parâmetros treináveis."""
        return sum(p.numel() for p in self.parameters())


def main() -> None:
    """Demonstra o modelo ainda não treinado num batch real do dataset."""
    # Reaproveitamos a preparação de dados do data.py.
    from data import DATASET_PATH, Tokenizer, get_batch, load_text, split_data

    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(1337)  # resultado reprodutível

    text = load_text(DATASET_PATH)
    tokenizer = Tokenizer(text)
    train_data, _ = split_data(text, tokenizer)

    config = GPTConfig(vocab_size=tokenizer.vocab_size)
    model = GPTLanguageModel(config)
    print(f"Configuração: {config}")
    print(f"Parâmetros: {model.num_params():,}")

    x, y = get_batch(train_data, config.block_size, batch_size=4, device="cpu")
    logits, loss = model(x, y)
    # Com pesos aleatórios pequenos, os logits saem quase todos iguais e o
    # softmax fica perto do uniforme: probabilidade 1/V para cada caractere.
    # A cross-entropy de chutar uniformemente é -ln(1/V) = ln(V). Se a loss
    # inicial estiver bem acima disso, a inicialização está "confiante demais"
    # em respostas erradas.
    print(f"\nShape dos logits: {tuple(logits.shape)}")
    print(f"Loss inicial:              {loss.item():.4f}")
    print(f"Esperada (ln(vocab_size)): {math.log(tokenizer.vocab_size):.4f}")

    # Geração com o modelo NÃO treinado: o resultado deve ser aleatório.
    model.eval()  # desliga o dropout
    start = torch.tensor([tokenizer.encode("\n")], dtype=torch.long)  # (1, 1)
    generated = model.generate(start, max_new_tokens=200)
    print("\n--- Geração sem treino (200 caracteres) ---")
    print(tokenizer.decode(generated[0].tolist()))


if __name__ == "__main__":
    main()
