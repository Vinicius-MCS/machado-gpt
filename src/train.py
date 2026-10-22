"""Loop de treinamento do GPT.

Junta as peças do data.py (dados) e do model.py (arquitetura) e treina o
modelo para prever o próximo caractere. Ao longo do treino, o script:

- avalia a loss de treino e validação a cada eval_interval passos;
- gera uma amostra curta de texto para acompanhar o aprendizado;
- salva checkpoints em checkpoints/ (best.pt e last.pt) e o tokenizador;
- registra as métricas em checkpoints/train_log.csv.

O treino em CPU é longo, então o script é robusto a interrupções: com Ctrl+C
ele salva o estado atual, e --resume continua de onde parou.

Uso:
    python src/train.py
    python src/train.py --max_iters 200 --eval_interval 100   # teste rápido
    python src/train.py --resume                              # retomar
"""

import argparse
import csv
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from data import DATASET_PATH, Tokenizer, get_batch, load_text, split_data
from model import GPTConfig, GPTLanguageModel

# Raiz do projeto (pasta acima de src/), para os caminhos funcionarem
# independentemente do diretório de onde o script é executado.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_PATH = CHECKPOINT_DIR / "best.pt"
LAST_PATH = CHECKPOINT_DIR / "last.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
LOG_PATH = CHECKPOINT_DIR / "train_log.csv"

# Hiperparâmetros que definem a arquitetura (devem bater com o checkpoint).
ARCH_FIELDS = ("block_size", "n_embd", "n_head", "n_layer", "dropout")

SAMPLE_LENGTH = 200  # caracteres gerados em cada amostra durante o treino


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treina o GPT character-level.")
    # Arquitetura (padrões do CLAUDE.md)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Treino
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou mps")
    parser.add_argument("--resume", action="store_true", help=f"continua de {LAST_PATH.name}")
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    """Traduz "auto" para o melhor dispositivo disponível."""
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():  # GPU dos Macs com Apple Silicon
        return "mps"
    return "cpu"


def format_duration(seconds: float) -> str:
    """Formata segundos como "1h05m", "3m12s" ou "42s"."""
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


@torch.no_grad()
def estimate_loss(
    model: GPTLanguageModel,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    device: str,
) -> dict[str, float]:
    """Estima a loss média de treino e validação.

    Tiramos a média de eval_iters batches porque a loss de um único batch é
    ruidosa demais: depende muito de quais trechos foram sorteados, e
    poderia sugerir uma melhora (ou piora) que não aconteceu.
    """
    # Em modo eval o dropout fica desligado. Com ele ligado, parte dos
    # neurônios seria zerada ao acaso e a loss medida seria pior (e mais
    # ruidosa) do que a do modelo de verdade, que usamos sem dropout.
    model.eval()
    losses: dict[str, float] = {}
    for split, data in (("train", train_data), ("val", val_data)):
        total = 0.0
        for _ in range(eval_iters):
            x, y = get_batch(data, block_size, batch_size, device)
            _, loss = model(x, y)
            total += loss.item()
        losses[split] = total / eval_iters
    model.train()  # volta ao modo de treino (dropout ligado)
    return losses


def generate_sample(model: GPTLanguageModel, tokenizer: Tokenizer, device: str) -> str:
    """Gera um trecho curto a partir de "\n", para ver o que o modelo aprendeu."""
    model.eval()  # sem dropout na geração
    start = torch.tensor([tokenizer.encode("\n")], dtype=torch.long, device=device)
    ids = list(model.generate_tokens(start, SAMPLE_LENGTH))
    model.train()
    return tokenizer.decode(ids)


def save_checkpoint(path: Path, state: dict) -> None:
    """Salva o checkpoint de forma atômica.

    Escrevemos num arquivo temporário e só então o renomeamos. Se o processo
    morrer no meio da escrita, o checkpoint anterior continua intacto;
    os.replace troca um arquivo pelo outro numa única operação.
    """
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def append_log(row: dict[str, float]) -> None:
    """Acrescenta uma linha ao CSV de métricas (cria o cabeçalho se preciso)."""
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["iter", "train_loss", "val_loss", "tempo_decorrido"])
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Prepara dados e modelo, roda o loop de treino e salva os checkpoints."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    text = load_text(DATASET_PATH)
    checkpoint = None
    if args.resume:
        if not LAST_PATH.exists():
            sys.exit(f"Nada para retomar: {LAST_PATH} não existe. Rode sem --resume.")
        checkpoint = torch.load(LAST_PATH, map_location=device)
        # O tokenizador do treino original garante que os ids continuem os mesmos.
        tokenizer = Tokenizer.load(TOKENIZER_PATH)
        config = GPTConfig(**checkpoint["config"])
        # A arquitetura não pode mudar no meio do treino: os pesos salvos só
        # cabem num modelo com as mesmas dimensões.
        for name in ARCH_FIELDS:
            if getattr(args, name) != getattr(config, name):
                print(
                    f"Aviso: --{name} {getattr(args, name)} ignorado; "
                    f"usando {getattr(config, name)} do checkpoint."
                )
    else:
        if LAST_PATH.exists():
            print(f"Aviso: começando do zero; os checkpoints em {CHECKPOINT_DIR} serão sobrescritos.")
            LOG_PATH.unlink(missing_ok=True)
        tokenizer = Tokenizer(text)
        tokenizer.save(TOKENIZER_PATH)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, **{name: getattr(args, name) for name in ARCH_FIELDS}
        )

    train_data, val_data = split_data(text, tokenizer)
    model = GPTLanguageModel(config).to(device)
    # AdamW: Adam com weight decay "desacoplado", o otimizador padrão para transformers.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_iter = 0
    best_val_loss = math.inf
    elapsed_before = 0.0  # tempo gasto em execuções anteriores (ao retomar)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["iter"]
        best_val_loss = checkpoint["best_val_loss"]
        elapsed_before = checkpoint.get("elapsed", 0.0)
        print(f"Retomando do passo {start_iter} (melhor val loss até agora: {best_val_loss:.4f})")

    print(f"Device: {device}")
    print(f"Parâmetros: {model.num_params():,}")
    print(f"vocab_size: {tokenizer.vocab_size}")
    print(f"Treino: {len(train_data):,} tokens | Validação: {len(val_data):,} tokens")
    # Sanidade: antes de treinar, o modelo chuta quase uniformemente entre os
    # caracteres, então a loss no passo 0 deve ficar perto de ln(vocab_size).
    print(f"Loss esperada no passo 0: ~{math.log(tokenizer.vocab_size):.4f}")
    print(f"Treinando do passo {start_iter} até {args.max_iters}...\n")

    def make_state(iteration: int, elapsed: float) -> dict:
        """Monta o dicionário salvo em cada checkpoint."""
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "iter": iteration,  # número de passos já concluídos
            "best_val_loss": best_val_loss,
            "elapsed": elapsed,
            "train_args": vars(args),
        }

    model.train()
    session_start = time.time()
    iteration = start_iter  # passos concluídos até agora
    try:
        while iteration < args.max_iters:
            # Um passo de otimização: sorteia um batch, calcula a loss,
            # obtém os gradientes por backpropagation e atualiza os pesos.
            x, y = get_batch(train_data, config.block_size, args.batch_size, device)
            _, loss = model(x, y)
            # set_to_none=True libera os gradientes em vez de preenchê-los com
            # zeros: um pouco mais rápido e econômico em memória.
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            iteration += 1

            if iteration % args.eval_interval == 0 or iteration == args.max_iters:
                losses = estimate_loss(
                    model, train_data, val_data, config.block_size,
                    args.batch_size, args.eval_iters, device,
                )
                session_elapsed = time.time() - session_start
                elapsed = elapsed_before + session_elapsed
                # Estimativa do tempo restante pelo ritmo desta execução.
                per_step = session_elapsed / (iteration - start_iter)
                remaining = per_step * (args.max_iters - iteration)
                # Como ler train x val: no começo as duas caem juntas. Se a val
                # parar de cair (ou subir) enquanto a train continua caindo, o
                # modelo está decorando o texto de treino em vez de aprender
                # padrões gerais (overfitting).
                print(
                    f"passo {iteration}/{args.max_iters} | train {losses['train']:.4f} | "
                    f"val {losses['val']:.4f} | {format_duration(elapsed)} decorridos | "
                    f"~{format_duration(remaining)} restantes"
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(BEST_PATH, make_state(iteration, elapsed))
                    print(f"  ↳ nova melhor val loss; {BEST_PATH.name} salvo")
                save_checkpoint(LAST_PATH, make_state(iteration, elapsed))
                append_log({
                    "iter": iteration,
                    "train_loss": round(losses["train"], 4),
                    "val_loss": round(losses["val"], 4),
                    "tempo_decorrido": round(elapsed, 1),
                })

                print("--- amostra ---")
                print(generate_sample(model, tokenizer, device))
                print("---------------\n")
    except KeyboardInterrupt:
        # Ctrl+C: salvamos o estado atual para não perder o progresso.
        elapsed = elapsed_before + time.time() - session_start
        save_checkpoint(LAST_PATH, make_state(iteration, elapsed))
        print(f"\nInterrompido no passo {iteration}. Estado salvo em {LAST_PATH}.")
        print("Para continuar: python src/train.py --resume")
        sys.exit(130)

    print("Treino concluído.")
    print(f"Melhor val loss: {best_val_loss:.4f}")
    print(f"Melhor modelo: {BEST_PATH}")
    print('Para gerar texto: python src/generate.py --prompt "Capitu"')


if __name__ == "__main__":
    main()
