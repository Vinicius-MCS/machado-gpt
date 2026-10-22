"""Curvas de aprendizado: compara o treino de um ou mais modelos.

Lê o train_log.csv e o best.pt de cada pasta de checkpoint e gera uma figura
com dois gráficos lado a lado:

1. loss de treino (tracejada) e de validação (cheia) por passo, com a "loss
   de chute aleatório" ln(vocab_size) como referência;
2. a diferença val − train ao longo do treino.

Também imprime no terminal uma tabela resumo e um guia de leitura.

Uso:
    python src/plot_loss.py
    python src/plot_loss.py --runs checkpoints --output figures/so_o_grande.png
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # gera PNG sem precisar de janela gráfica
import matplotlib.pyplot as plt

from generate import load_model
from train import BEST_PATH, LOG_PATH, PROJECT_ROOT

FIGURES_DIR = PROJECT_ROOT / "figures"

READING_GUIDE = """
Como ler os gráficos:
- As curvas caem rápido no começo e depois achatam: os primeiros passos
  ensinam o básico (letras comuns, espaços, sílabas), e cada melhoria
  seguinte custa muito mais passos (retornos decrescentes).
- A diferença val − train mede o quanto o modelo está decorando o texto de
  treino. Uma diferença pequena e estável é normal. Se ela cresce enquanto
  a val loss estaciona, é overfitting: o modelo melhora no que já viu, mas
  não no texto novo.
- Comparar por "passo" não é totalmente justo: cada passo processa
  batch_size × block_size caracteres. Com batch sizes diferentes, um passo
  de um modelo pode ver o dobro de texto de um passo do outro (veja a coluna
  "caracteres vistos").
"""


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Plota as curvas de loss dos treinos.")
    parser.add_argument(
        "--runs",
        type=Path,
        nargs="+",
        default=[PROJECT_ROOT / "checkpoints_pequeno", PROJECT_ROOT / "checkpoints"],
        help="pastas de checkpoint (cada uma com train_log.csv, best.pt e tokenizer.json)",
    )
    parser.add_argument("--output", type=Path, default=FIGURES_DIR / "loss_curves.png")
    return parser.parse_args()


def read_log(path: Path) -> dict[str, list[float]]:
    """Lê o train_log.csv e devolve uma lista de valores por coluna."""
    if not path.exists():
        sys.exit(f"Erro: {path} não encontrado.")
    columns: dict[str, list[float]] = {"iter": [], "train_loss": [], "val_loss": []}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            for name in columns:
                columns[name].append(float(row[name]))
    return columns


def load_run(run_dir: Path) -> dict:
    """Junta o log e os metadados do best.pt de uma pasta de checkpoint."""
    # load_model (generate.py) já carrega o tokenizador da mesma pasta e
    # confere que o vocabulário bate com o checkpoint.
    model, checkpoint, tokenizer = load_model(run_dir / BEST_PATH.name, "cpu")
    train_args = checkpoint["train_args"]
    log = read_log(run_dir / LOG_PATH.name)
    params = model.num_params()
    best_val = checkpoint["best_val_loss"]
    return {
        "label": f"{run_dir.name} ({params / 1e6:.2f} M, melhor val {best_val:.4f})",
        "name": run_dir.name,
        "params": params,
        "best_val": best_val,
        "vocab_size": tokenizer.vocab_size,
        "log": log,
        # Caracteres processados no treino: cada passo vê batch_size trechos
        # de block_size caracteres.
        "chars_seen": int(log["iter"][-1]) * train_args["batch_size"] * train_args["block_size"],
        "batch_size": train_args["batch_size"],
    }


def plot_runs(runs: list[dict], output: Path) -> None:
    """Desenha os dois gráficos e salva a figura em PNG."""
    fig, (ax_loss, ax_gap) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run, color in zip(runs, colors):
        log = run["log"]
        # Mesma cor por modelo: cheia = validação, tracejada = treino.
        ax_loss.plot(log["iter"], log["val_loss"], color=color, label=f"{run['label']} — val")
        ax_loss.plot(log["iter"], log["train_loss"], color=color, linestyle="--", label=f"{run['name']} — train")
        # Loss de quem chuta uniformemente entre os caracteres: ln(vocab_size).
        # Qualquer modelo que aprendeu algo fica bem abaixo dessa linha.
        ax_loss.axhline(
            math.log(run["vocab_size"]), color=color, linestyle=":", linewidth=1,
            label=f"{run['name']} — chute aleatório ln({run['vocab_size']})",
        )
        gap = [val - train for val, train in zip(log["val_loss"], log["train_loss"])]
        ax_gap.plot(log["iter"], gap, color=color, marker="o", markersize=3, label=run["name"])

    ax_loss.set_title("Loss de treino e validação")
    ax_loss.set_xlabel("passo")
    ax_loss.set_ylabel("loss (cross-entropy)")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    ax_gap.set_title("Diferença val − train (quanto o modelo decora)")
    ax_gap.set_xlabel("passo")
    ax_gap.set_ylabel("val loss − train loss")
    ax_gap.axhline(0, color="gray", linewidth=0.8)
    ax_gap.legend(fontsize=8)
    ax_gap.grid(alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def print_summary(runs: list[dict]) -> None:
    """Imprime a tabela resumo de cada modelo."""
    header = (
        f"{'modelo':<22}{'parâmetros':>12}{'passos':>8}{'batch':>7}{'caracteres vistos':>19}"
        f"{'melhor val':>12}{'train final':>13}{'val−train final':>17}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        log = run["log"]
        gap = log["val_loss"][-1] - log["train_loss"][-1]
        print(
            f"{run['name']:<22}{run['params']:>12,}{int(log['iter'][-1]):>8}{run['batch_size']:>7}"
            f"{run['chars_seen'] / 1e6:>17.1f} M{run['best_val']:>12.4f}"
            f"{log['train_loss'][-1]:>13.4f}{gap:>17.4f}"
        )


def main() -> None:
    """Carrega os treinos, desenha as curvas e imprime o resumo."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()

    runs = [load_run(run_dir) for run_dir in args.runs]
    plot_runs(runs, args.output)
    print_summary(runs)
    print(READING_GUIDE)
    print(f"Figura salva em: {args.output}")


if __name__ == "__main__":
    main()
