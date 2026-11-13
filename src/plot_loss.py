"""Curvas de aprendizado: compara o treino de um ou mais modelos.

Lê o train_log.csv e o best.pt de cada pasta de checkpoint e gera uma figura
com dois gráficos lado a lado:

1. erro de treino (tracejado) e de validação (cheio) por passo, com o erro de
   "chute aleatório" como referência;
2. a diferença val − train ao longo do treino.

Tudo é medido em bits por caractere, e não em loss. A loss é o erro por
TOKEN, e tokenizadores diferentes (caractere ou BPE) cortam o texto em
pedaços de tamanhos diferentes: a loss de um não é comparável com a do
outro. Bits por caractere mede o erro na mesma unidade para todos.

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

from bpe import BPETokenizer
from generate import load_model
from train import BEST_PATH, LOG_PATH, PROJECT_ROOT, bits_per_char

FIGURES_DIR = PROJECT_ROOT / "figures"

READING_GUIDE = """
Como ler os gráficos (em bits por caractere: quanto menor, melhor):
- As curvas caem rápido no começo e depois achatam: os primeiros passos
  ensinam o básico (letras comuns, espaços, sílabas), e cada melhoria
  seguinte custa muito mais passos (retornos decrescentes).
- A diferença val − train mede o quanto o modelo está decorando o texto de
  treino. Uma diferença pequena e estável é normal. Se ela cresce enquanto
  a validação estaciona, é overfitting: o modelo melhora no que já viu, mas
  não no texto novo.
- Comparar por "passo" não é totalmente justo: cada passo processa
  batch_size × block_size TOKENS. Com batch sizes diferentes, ou com BPE
  (em que cada token vale vários caracteres), um passo de um modelo pode
  ver muito mais texto que um passo do outro (veja "caracteres vistos").
"""


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Plota as curvas de aprendizado dos treinos.")
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
    # Checkpoints de antes do BPE não guardam chars_per_token, mas todos eles
    # são de caractere: 1 caractere por token.
    cpt = checkpoint.get("chars_per_token", {"train": 1.0, "val": 1.0})
    log = read_log(run_dir / LOG_PATH.name)
    # Uma única conversão para todos os modelos: loss -> bits por caractere.
    train_bpc = [bits_per_char(loss, cpt["train"]) for loss in log["train_loss"]]
    val_bpc = [bits_per_char(loss, cpt["val"]) for loss in log["val_loss"]]
    params = model.num_params()
    best_bpc = bits_per_char(checkpoint["best_val_loss"], cpt["val"])
    kind = "BPE" if isinstance(tokenizer, BPETokenizer) else "caractere"
    return {
        "label": f"{run_dir.name} ({params / 1e6:.2f} M, {kind} {tokenizer.vocab_size}, "
                 f"melhor {best_bpc:.3f})",
        "name": run_dir.name,
        "kind": kind,
        "params": params,
        "vocab_size": tokenizer.vocab_size,
        "chars_per_token": cpt["val"],
        "best_bpc": best_bpc,
        "iters": log["iter"],
        "train_bpc": train_bpc,
        "val_bpc": val_bpc,
        # Chute aleatório: ln(vocab_size) nats por token, na mesma unidade.
        "random_bpc": bits_per_char(math.log(tokenizer.vocab_size), cpt["val"]),
        # Caracteres processados no treino: cada passo vê batch_size trechos
        # de block_size tokens, e cada token vale chars_per_token caracteres.
        "chars_seen": int(log["iter"][-1]) * train_args["batch_size"]
                      * train_args["block_size"] * cpt["train"],
        "batch_size": train_args["batch_size"],
    }


def plot_runs(runs: list[dict], output: Path) -> None:
    """Desenha os dois gráficos e salva a figura em PNG."""
    fig, (ax_bpc, ax_gap) = plt.subplots(1, 2, figsize=(14, 5.5))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for run, color in zip(runs, colors):
        # Mesma cor por modelo: cheia = validação, tracejada = treino.
        ax_bpc.plot(run["iters"], run["val_bpc"], color=color, label=f"{run['label']} — val")
        ax_bpc.plot(run["iters"], run["train_bpc"], color=color, linestyle="--",
                    label=f"{run['name']} — train")
        # Erro de quem chuta uniformemente entre os tokens. Qualquer modelo
        # que aprendeu algo fica bem abaixo dessa linha.
        ax_bpc.axhline(run["random_bpc"], color=color, linestyle=":", linewidth=1,
                       label=f"{run['name']} — chute aleatório")
        gap = [val - train for val, train in zip(run["val_bpc"], run["train_bpc"])]
        ax_gap.plot(run["iters"], gap, color=color, marker="o", markersize=3, label=run["name"])

    ax_bpc.set_title("Erro de treino e validação")
    ax_bpc.set_xlabel("passo")
    ax_bpc.set_ylabel("bits por caractere")
    ax_bpc.legend(fontsize=8)
    ax_bpc.grid(alpha=0.3)

    ax_gap.set_title("Diferença val − train (quanto o modelo decora)")
    ax_gap.set_xlabel("passo")
    ax_gap.set_ylabel("val − train (bits por caractere)")
    ax_gap.axhline(0, color="gray", linewidth=0.8)
    ax_gap.legend(fontsize=8)
    ax_gap.grid(alpha=0.3)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def print_summary(runs: list[dict]) -> None:
    """Imprime a tabela resumo de cada modelo (erros em bits por caractere)."""
    header = (
        f"{'modelo':<22}{'parâmetros':>12}{'tokenizador':>17}{'chars/token':>13}{'passos':>8}"
        f"{'batch':>7}{'caracteres vistos':>19}{'melhor val':>12}{'train final':>13}"
        f"{'val−train final':>17}"
    )
    print(header)
    print("-" * len(header))
    for run in runs:
        gap = run["val_bpc"][-1] - run["train_bpc"][-1]
        tokenizer = f"{run['kind']} {run['vocab_size']}"
        print(
            f"{run['name']:<22}{run['params']:>12,}{tokenizer:>17}{run['chars_per_token']:>13.2f}"
            f"{int(run['iters'][-1]):>8}{run['batch_size']:>7}{run['chars_seen'] / 1e6:>17.1f} M"
            f"{run['best_bpc']:>12.4f}{run['train_bpc'][-1]:>13.4f}{gap:>17.4f}"
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
