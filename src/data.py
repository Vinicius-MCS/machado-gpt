"""Preparação dos dados para o treino do GPT.

Este módulo transforma o texto bruto de data/dataset.txt em tensores que o
modelo consegue consumir. Ele é usado pelo train.py e pelo generate.py:

- load_text:      lê o dataset do disco.
- Tokenizer:      converte texto ↔ lista de inteiros, um inteiro por caractere.
- load_tokenizer: recria o tokenizador salvo (caractere ou BPE, ver bpe.py).
- split_text:     separa treino e validação, pegando um pedaço de cada obra.
- split_data:     faz o mesmo e já devolve os tensores de ids.
- get_batch:      sorteia trechos aleatórios e monta os pares (entrada, alvo).

Segue a abordagem da aula "Let's build GPT from scratch", de Andrej Karpathy.

Uso (demonstração):
    python src/data.py
"""

import json
import sys
from pathlib import Path

import torch

from bpe import BPETokenizer

# Raiz do projeto (pasta acima de src/), para os caminhos funcionarem
# independentemente do diretório de onde o script é executado.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_PATH = PROJECT_ROOT / "data" / "dataset.txt"

# Separador entre obras, o mesmo usado pelo collect.py ao gerar o dataset.
WORK_SEPARATOR = "\n\n---\n\n"


def load_text(path: Path) -> str:
    """Lê o dataset inteiro como uma única string."""
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Dataset não encontrado ou vazio em {path}.\n"
            "Gere-o primeiro com: python src/collect.py"
        )
    return path.read_text(encoding="utf-8")


class Tokenizer:
    """Tokenizador character-level: cada caractere é um token.

    Por que character-level? É o tokenizador mais simples possível, o que é
    ótimo para fins didáticos. A troca em relação a tokenizadores subword
    (como o BPE do GPT-2/GPT-4) é:

    - Character-level: vocabulário minúsculo (~100 tokens aqui), mas cada
      palavra vira vários tokens. As sequências ficam longas, e o modelo
      precisa "gastar" contexto e capacidade só para aprender a soletrar.
    - Subword (BPE): vocabulário grande (dezenas de milhares de tokens), mas
      palavras comuns viram um único token. As sequências ficam curtas, e o
      mesmo block_size enxerga muito mais texto.
    """

    def __init__(self, text: str) -> None:
        # O vocabulário é o conjunto de caracteres distintos do texto,
        # ordenado para que o mapeamento seja sempre o mesmo.
        chars = sorted(set(text))
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        self.itos: dict[int, str] = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        """Número de tokens distintos (tamanho da saída do modelo)."""
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        """Converte texto em lista de ids."""
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            raise ValueError(
                f"Caracteres fora do vocabulário: {unknown!r}. "
                "O modelo só conhece os caracteres presentes nas obras do dataset."
            )
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        """Converte lista de ids de volta em texto."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        """Salva o vocabulário em JSON.

        O generate.py precisa do mesmo vocabulário usado no treino: se a ordem
        dos caracteres mudar, o id 42 passa a significar outra letra e o
        modelo treinado vira lixo.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        chars = [self.itos[i] for i in range(self.vocab_size)]
        # "type" diz ao load_tokenizer qual classe recriar.
        path.write_text(json.dumps({"type": "char", "chars": chars}, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Tokenizer":
        """Reconstrói o tokenizador a partir do JSON salvo por save()."""
        chars = json.loads(path.read_text(encoding="utf-8"))["chars"]
        # Os caracteres já estão em ordem, então passá-los como "texto" para o
        # construtor recria exatamente os mesmos ids.
        return cls("".join(chars))


# Qualquer um dos dois tokenizadores: os dois têm a mesma interface.
AnyTokenizer = Tokenizer | BPETokenizer


def load_tokenizer(path: Path) -> AnyTokenizer:
    """Recria o tokenizador salvo em `path`, seja de caractere ou BPE.

    Arquivos antigos (antes do BPE) não têm o campo "type" e são de caractere.
    """
    kind = json.loads(path.read_text(encoding="utf-8")).get("type", "char")
    if kind == "bpe":
        return BPETokenizer.load(path)
    return Tokenizer.load(path)


def split_text(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Separa o texto em treino e validação, ainda como texto.

    Em vez de cortar só os últimos 10% do texto concatenado, cortamos os
    últimos 10% de CADA obra. Como as obras estão em sequência no dataset, o
    corte simples deixaria a validação quase toda com Papéis Avulsos (contos),
    e a loss de validação mediria só o quanto o modelo generaliza para aquela
    obra. Pegando um pedaço de cada uma, a validação representa o corpus todo.

    O corte é feito no TEXTO, antes de tokenizar. Assim, tokenizadores
    diferentes são avaliados exatamente nos mesmos caracteres, e o BPE pode
    ser treinado só com a parte de treino, sem "espiar" a validação.
    """
    train_parts: list[str] = []
    val_parts: list[str] = []
    for work in text.split(WORK_SEPARATOR):
        cut = int(len(work) * (1 - val_fraction))
        train_parts.append(work[:cut])
        val_parts.append(work[cut:])
    # Reunimos os pedaços com o mesmo separador, para manter o padrão do dataset.
    return WORK_SEPARATOR.join(train_parts), WORK_SEPARATOR.join(val_parts)


def split_data(
    text: str, tokenizer: AnyTokenizer, val_fraction: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Separa o texto (ver split_text) e devolve tensores de ids de treino e validação."""
    train_text, val_text = split_text(text, val_fraction)
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    return train_data, val_data


def get_batch(
    data: torch.Tensor, block_size: int, batch_size: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sorteia um batch de trechos aleatórios.

    Retorna x e y com formato (batch_size, block_size), em que y é x deslocado
    uma posição à frente. Cada linha contém, na verdade, block_size exemplos
    de treino: na posição t, o modelo vê os caracteres 0..t de x e precisa
    prever y[t] (o caractere t+1). Assim ele aprende a prever com contextos
    de tamanho 1 até block_size.
    """
    # Posições iniciais aleatórias; o "- block_size" garante que o trecho
    # (e o alvo deslocado) caiba dentro do tensor.
    starts = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in starts])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in starts])
    return x.to(device), y.to(device)


def main() -> None:
    """Demonstra cada etapa da preparação dos dados."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    torch.manual_seed(1337)  # resultado reprodutível

    text = load_text(DATASET_PATH)
    tokenizer = Tokenizer(text)
    print(f"Tamanho do texto: {len(text):,} caracteres")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"Vocabulário: {''.join(tokenizer.itos[i] for i in range(tokenizer.vocab_size))!r}")

    sample = "Capitu"
    ids = tokenizer.encode(sample)
    print(f"\nencode({sample!r}) = {ids}")
    print(f"decode(encode({sample!r})) == {sample!r}: {tokenizer.decode(ids) == sample}")

    train_data, val_data = split_data(text, tokenizer)
    print(f"\nTreino:    {len(train_data):,} tokens")
    print(f"Validação: {len(val_data):,} tokens")

    block_size, batch_size = 8, 4
    x, y = get_batch(train_data, block_size, batch_size, device="cpu")
    print(f"\nx.shape = {tuple(x.shape)}, y.shape = {tuple(y.shape)}")
    print(f"x[0] = {x[0].tolist()} → {tokenizer.decode(x[0].tolist())!r}")
    print(f"y[0] = {y[0].tolist()} → {tokenizer.decode(y[0].tolist())!r}")

    print("\nExemplos de treino contidos na primeira linha do batch:")
    for t in range(block_size):
        context = tokenizer.decode(x[0, : t + 1].tolist())
        target = tokenizer.decode([y[0, t].item()])
        print(f"  quando o contexto é {context!r}, o alvo é {target!r}")


if __name__ == "__main__":
    main()
