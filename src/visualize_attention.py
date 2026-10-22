"""Visualização da atenção: para onde cada cabeça "olha".

Roda o modelo treinado sobre um texto curto e desenha, para cada cabeça, a
matriz de pesos da atenção: a linha t mostra como a posição t (a "query", que
pergunta) distribui sua atenção entre as posições 0..t (as "keys",
consultadas). Cada linha soma 1.

Gera duas figuras:
- --output: um heatmap por cabeça da camada escolhida, com os caracteres
  como rótulos;
- figures/attention_overview.png: miniaturas de todas as camadas × cabeças.

Como ler os heatmaps:
- O triângulo superior é sempre zero por causa da máscara causal: a posição
  t não pode olhar para o futuro (t+1, t+2...), senão o modelo "colaria" a
  resposta durante o treino.
- Padrões que costumam aparecer: cabeças que olham para o caractere
  anterior (uma diagonal logo abaixo da principal), para o início da palavra
  atual, para espaços ou pontuação, ou que espalham a atenção por todo o
  contexto (uma espécie de "média geral").
- Camadas iniciais tendem a capturar padrões locais, como soletração; as
  finais, relações mais longas.

Essas interpretações são tendências observadas em vários modelos, não
regras: nem toda cabeça tem um padrão legível, e uma mesma cabeça pode
misturar comportamentos.

Uso:
    python src/visualize_attention.py
    python src/visualize_attention.py --layer 0
    python src/visualize_attention.py --checkpoint checkpoints_pequeno/best.pt
"""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera PNG sem precisar de janela gráfica
import matplotlib.pyplot as plt
import torch

from generate import load_model
from model import GPTLanguageModel
from train import BEST_PATH, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "figures"
OVERVIEW_PATH = FIGURES_DIR / "attention_overview.png"
DEFAULT_TEXT = "--Capitu, que é isso? perguntou José Dias."
MAX_READABLE_LENGTH = 60  # acima disso, os rótulos dos eixos ficam ilegíveis
TOP_N = 3  # quantos caracteres mais atendidos mostrar no resumo textual


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Visualiza os pesos de atenção do GPT.")
    parser.add_argument("--checkpoint", type=Path, default=BEST_PATH)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    parser.add_argument("--layer", type=int, default=None, help="camada (padrão: a última)")
    parser.add_argument("--output", type=Path, default=None,
                        help="padrão: figures/attention_layer{N}.png")
    return parser.parse_args()


def visible(ch: str) -> str:
    """Troca espaço e quebra de linha por símbolos visíveis nos rótulos."""
    return {" ": "␣", "\n": "↵"}.get(ch, ch)


def collect_attention(model: GPTLanguageModel, ids: list[int]) -> list[list[torch.Tensor]]:
    """Roda um forward e devolve attention[camada][cabeça], cada uma (T, T)."""
    heads_per_layer = [block.sa.heads for block in model.blocks]
    for heads in heads_per_layer:
        for head in heads:
            head.record_attention = True
    with torch.no_grad():
        model(torch.tensor([ids], dtype=torch.long))  # (1, T)
    # last_attention tem shape (1, T, T); tiramos a dimensão do batch.
    return [[head.last_attention[0] for head in heads] for heads in heads_per_layer]


def pick_positions(text: str) -> list[int]:
    """Escolhe algumas posições interessantes do texto para o resumo textual.

    Pegamos a última letra de uma palavra, um caractere logo depois de um
    espaço (início de palavra), uma pontuação e a última posição. Preferimos
    exemplos do meio do texto, onde há mais contexto para olhar.
    """
    middle = len(text) // 2

    def closest_to_middle(candidates: list[int]) -> list[int]:
        return [min(candidates, key=lambda i: abs(i - middle))] if candidates else []

    word_ends = [i for i in range(1, len(text) - 1) if text[i].isalpha() and not text[i + 1].isalpha()]
    word_starts = [i for i in range(1, len(text)) if text[i - 1] == " " and text[i] != " "]
    punctuation = [i for i in range(1, len(text) - 1) if text[i] in ",.;:?!"]
    positions = (
        closest_to_middle(word_ends) + closest_to_middle(word_starts)
        + closest_to_middle(punctuation) + [len(text) - 1]
    )
    return sorted(set(positions))


def plot_layer(attention: list[torch.Tensor], text: str, layer: int, output: Path) -> None:
    """Grade de heatmaps, uma por cabeça da camada escolhida."""
    n_head = len(attention)
    ncols = math.ceil(math.sqrt(n_head))
    nrows = math.ceil(n_head / ncols)
    labels = [visible(ch) for ch in text]
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.5 * ncols + 1, 5.5 * nrows + 0.5), squeeze=False, layout="constrained"
    )

    for h, ax in enumerate(axes.flat):
        if h >= n_head:
            ax.axis("off")
            continue
        image = ax.imshow(attention[h].numpy(), cmap="viridis", vmin=0, vmax=1)
        ax.set_title(f"cabeça {h}")
        ax.set_xticks(range(len(text)), labels, fontsize=7)
        ax.set_yticks(range(len(text)), labels, fontsize=7)
        ax.set_xlabel("posição consultada (key)")
        ax.set_ylabel("posição que pergunta (query)")

    fig.suptitle(f"Atenção na camada {layer}: \"{text}\"", fontsize=13)
    fig.colorbar(image, ax=axes, shrink=0.6, label="peso da atenção")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110)
    plt.close(fig)


def plot_overview(attention: list[list[torch.Tensor]], output: Path) -> None:
    """Miniaturas de todas as camadas × cabeças, para comparar o padrão geral."""
    n_layer, n_head = len(attention), len(attention[0])
    fig, axes = plt.subplots(n_layer, n_head, figsize=(2 * n_head, 2 * n_layer), squeeze=False)
    for layer in range(n_layer):
        for h in range(n_head):
            ax = axes[layer][h]
            ax.imshow(attention[layer][h].numpy(), cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if layer == 0:
                ax.set_title(f"cabeça {h}", fontsize=9)
            if h == 0:
                ax.set_ylabel(f"camada {layer}", fontsize=9)
    fig.suptitle("Atenção em todas as camadas (linhas) e cabeças (colunas)")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=110)
    plt.close(fig)


def print_summary(attention: list[torch.Tensor], text: str, layer: int) -> None:
    """Para algumas posições, mostra os caracteres que mais recebem atenção."""
    positions = pick_positions(text)
    print(f"Resumo da camada {layer} (os {TOP_N} caracteres mais atendidos por posição):")
    for h, weights in enumerate(attention):
        print(f"\ncabeça {h}")
        for t in positions:
            # A posição t só enxerga 0..t, então pode haver menos de TOP_N opções.
            values, indices = torch.topk(weights[t, : t + 1], min(TOP_N, t + 1))
            ranked = " | ".join(
                f"'{visible(text[i])}'({i}) {v:.2f}" for v, i in zip(values.tolist(), indices.tolist())
            )
            print(f"  cabeça {h}, posição {t} ('{visible(text[t])}'): {ranked}")


def main() -> None:
    """Valida a entrada, coleta a atenção e gera as figuras e o resumo."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    model, checkpoint, tokenizer = load_model(args.checkpoint, "cpu")
    config = model.config
    print(f"[{args.checkpoint} | {model.num_params():,} parâmetros | "
          f"{config.n_layer} camadas × {config.n_head} cabeças]\n")

    text = args.text.replace("\\n", "\n")
    unknown = sorted({ch for ch in text if ch not in tokenizer.stoi})
    if unknown:
        sys.exit(
            f"O texto tem caracteres que o modelo nunca viu: {' '.join(unknown)}\n"
            "Remova esses caracteres e tente de novo."
        )
    if len(text) > config.block_size:
        sys.exit(f"Erro: o texto tem {len(text)} caracteres; o máximo é o block_size ({config.block_size}).")
    if len(text) > MAX_READABLE_LENGTH:
        print(f"Aviso: com {len(text)} caracteres, os rótulos dos eixos ficam difíceis de ler "
              f"(recomendado: até ~{MAX_READABLE_LENGTH}).\n")

    layer = config.n_layer - 1 if args.layer is None else args.layer
    if not 0 <= layer < config.n_layer:
        sys.exit(f"Erro: --layer deve estar entre 0 e {config.n_layer - 1}.")
    output = args.output or FIGURES_DIR / f"attention_layer{layer}.png"

    attention = collect_attention(model, tokenizer.encode(text))
    plot_layer(attention[layer], text, layer, output)
    plot_overview(attention, OVERVIEW_PATH)
    print_summary(attention[layer], text, layer)

    print(f"\nFiguras salvas em:\n  {output}\n  {OVERVIEW_PATH}")


if __name__ == "__main__":
    main()
