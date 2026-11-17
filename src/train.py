"""Loop de treinamento do GPT.

Junta as peças do data.py (dados) e do model.py (arquitetura) e treina o
modelo para prever o próximo token. O token pode ser um caractere (padrão)
ou um pedaço de palavra aprendido pelo BPE (--tokenizer bpe, ver bpe.py).
Ao longo do treino, o script:

- avalia a loss de treino e validação a cada eval_interval passos, também
  convertida para bits por caractere, a métrica comparável entre tokenizadores;
- gera uma amostra curta de texto para acompanhar o aprendizado;
- salva checkpoints em --out_dir (best.pt e last.pt) e o tokenizador;
- registra as métricas em train_log.csv, na mesma pasta.

O treino em CPU é longo, então o script é robusto a interrupções: com Ctrl+C
ele salva o estado atual, e --resume continua de onde parou.

Uso:
    python src/train.py
    python src/train.py --max_iters 200 --eval_interval 100   # teste rápido
    python src/train.py --resume                              # retomar
    python src/train.py --tokenizer bpe --vocab_size 1000 --out_dir checkpoints_bpe1000
"""

import argparse
import csv
import hashlib
import math
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch

from bpe import BPETokenizer
from data import (DATASET_PATH, AnyTokenizer, Tokenizer, get_batch, load_text,
                  load_tokenizer, split_text)
from model import GPTConfig, GPTLanguageModel

# Raiz do projeto (pasta acima de src/), para os caminhos funcionarem
# independentemente do diretório de onde o script é executado.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Pasta padrão dos checkpoints. Os nomes abaixo valem dentro de qualquer
# --out_dir; generate.py, plot_loss.py e visualize_attention.py usam estes
# caminhos como padrão.
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
BEST_PATH = CHECKPOINT_DIR / "best.pt"
LAST_PATH = CHECKPOINT_DIR / "last.pt"
TOKENIZER_PATH = CHECKPOINT_DIR / "tokenizer.json"
LOG_PATH = CHECKPOINT_DIR / "train_log.csv"
# Cache dos BPEs já treinados, para não retreinar a cada execução.
TOKENIZER_CACHE_DIR = PROJECT_ROOT / "data" / "tokenizers"

# Hiperparâmetros que definem a arquitetura (devem bater com o checkpoint).
ARCH_FIELDS = ("block_size", "n_embd", "n_head", "n_layer", "dropout", "tie_weights")

LOG_FIELDS = ["iter", "train_loss", "val_loss", "train_bpc", "val_bpc", "tempo_decorrido"]

SAMPLE_LENGTH = 200  # tokens gerados em cada amostra durante o treino


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Treina o GPT (tokenizador de caractere ou BPE).")
    # Arquitetura (padrões do CLAUDE.md)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--tie_weights", action="store_true",
                        help="compartilha a matriz do embedding com a lm_head")
    # Tokenizador
    parser.add_argument("--tokenizer", choices=["char", "bpe"], default="char")
    parser.add_argument("--vocab_size", type=int, default=1000,
                        help="tamanho total do vocabulário (só para --tokenizer bpe)")
    # Treino
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou mps")
    parser.add_argument("--out_dir", type=Path, default=CHECKPOINT_DIR,
                        help="pasta dos checkpoints, do tokenizador e do log")
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


def get_bpe_tokenizer(train_text: str, vocab_size: int, alphabet: str) -> BPETokenizer:
    """Devolve um BPE treinado no texto de treino, usando o cache se possível.

    As fusões vêm só do texto de treino; o alfabeto (os caracteres-base) vem
    do dataset inteiro, para a validação poder ser codificada mesmo com
    caracteres raros que só aparecem nela (ver BPETokenizer.train).

    O nome do arquivo inclui um hash (uma "impressão digital") do texto de
    treino e do alfabeto: se o dataset mudar, o hash muda e o BPE é
    retreinado, em vez de reaproveitarmos por engano um vocabulário antigo.
    """
    digest = hashlib.sha256((alphabet + "\0" + train_text).encode("utf-8")).hexdigest()[:12]
    cache_path = TOKENIZER_CACHE_DIR / f"bpe_{vocab_size}_{digest}.json"
    if cache_path.exists():
        print(f"BPE carregado do cache: {cache_path.name}")
        return BPETokenizer.load(cache_path)
    print(f"Treinando BPE com vocab_size={vocab_size} (só com o texto de treino)...")
    start = time.time()
    tokenizer = BPETokenizer.train(train_text, vocab_size, alphabet=alphabet)
    tokenizer.save(cache_path)
    print(f"BPE treinado em {format_duration(time.time() - start)} e salvo em {cache_path.name}")
    return tokenizer


def bits_per_char(loss: float, chars_per_token: float) -> float:
    """Converte a loss (nats por token) em bits por caractere.

    A loss é a cross-entropy média por TOKEN, em nats (logaritmo natural).
    Tokenizadores diferentes cortam o texto em pedaços de tamanhos
    diferentes, então a loss de um não é comparável com a do outro. Dividindo
    pelo número médio de caracteres por token, obtemos o erro por CARACTERE,
    que é a mesma unidade para todos; dividindo por ln 2, passamos de nats
    para bits. Para o tokenizador de caractere, chars_per_token é 1.
    """
    return loss / chars_per_token / math.log(2)


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


def generate_sample(model: GPTLanguageModel, tokenizer: AnyTokenizer, device: str) -> str:
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


def append_log(path: Path, row: dict[str, float]) -> None:
    """Acrescenta uma linha ao CSV de métricas (cria o cabeçalho se preciso).

    Ao retomar um treino antigo, o CSV pode ter sido criado antes das colunas
    de bits por caractere existirem. Nesse caso mantemos o cabeçalho antigo e
    gravamos só as colunas que ele tem, para não desalinhar o arquivo.
    """
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            fieldnames = next(csv.reader(file))
    else:
        fieldnames = LOG_FIELDS
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        if not path.stat().st_size:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    """Prepara dados e modelo, roda o loop de treino e salva os checkpoints."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    out_dir = args.out_dir
    best_path = out_dir / BEST_PATH.name
    last_path = out_dir / LAST_PATH.name
    tokenizer_path = out_dir / TOKENIZER_PATH.name
    log_path = out_dir / LOG_PATH.name
    out_dir.mkdir(parents=True, exist_ok=True)
    # Comando para retomar, com o --out_dir quando ele não é o padrão.
    resume_command = "python src/train.py --resume"
    if out_dir.resolve() != CHECKPOINT_DIR:
        resume_command += f" --out_dir {out_dir}"

    text = load_text(DATASET_PATH)
    # Separamos treino e validação no texto, ANTES de tokenizar: o BPE é
    # treinado só com o texto de treino, sem "espiar" a validação.
    train_text, val_text = split_text(text)

    checkpoint = None
    if args.resume:
        if not last_path.exists():
            sys.exit(f"Nada para retomar: {last_path} não existe. Rode sem --resume.")
        checkpoint = torch.load(last_path, map_location=device)
        # O tokenizador do treino original garante que os ids continuem os mesmos.
        tokenizer = load_tokenizer(tokenizer_path)
        kind = "bpe" if isinstance(tokenizer, BPETokenizer) else "char"
        if args.tokenizer != kind:
            print(f"Aviso: --tokenizer {args.tokenizer} ignorado; usando {kind} do checkpoint.")
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
        if last_path.exists():
            print(f"Aviso: começando do zero; os checkpoints em {out_dir} serão sobrescritos.")
            log_path.unlink(missing_ok=True)
        if args.tokenizer == "bpe":
            alphabet = "".join(sorted(set(text)))
            tokenizer = get_bpe_tokenizer(train_text, args.vocab_size, alphabet)
        else:
            tokenizer = Tokenizer(text)
        tokenizer.save(tokenizer_path)
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size, **{name: getattr(args, name) for name in ARCH_FIELDS}
        )

    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)
    # Quantos caracteres cada token representa, em média. É o que converte a
    # loss por token em bits por caractere (ver bits_per_char).
    chars_per_token = {
        "train": len(train_text) / len(train_data),
        "val": len(val_text) / len(val_data),
    }

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
    print(f"Pasta de saída: {out_dir}")
    print(f"Parâmetros: {model.num_params():,}")
    print(f"Tokenizador: {type(tokenizer).__name__} | vocab_size: {tokenizer.vocab_size}"
          f" | {chars_per_token['val']:.2f} caracteres por token")
    print(f"Treino: {len(train_data):,} tokens | Validação: {len(val_data):,} tokens")
    # Sanidade: antes de treinar, o modelo chuta quase uniformemente entre os
    # tokens, então a loss no passo 0 deve ficar perto de ln(vocab_size).
    print(f"Loss esperada no passo 0: ~{math.log(tokenizer.vocab_size):.4f}")
    print(f"Treinando do passo {start_iter} até {args.max_iters}...\n")

    # Os argumentos vão para o checkpoint como tipos simples: o torch.load
    # padrão (weights_only=True) recusa objetos como Path, por segurança.
    train_args = {key: str(value) if isinstance(value, Path) else value
                  for key, value in vars(args).items()}

    def make_state(iteration: int, elapsed: float) -> dict:
        """Monta o dicionário salvo em cada checkpoint."""
        return {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "config": asdict(config),
            "iter": iteration,  # número de passos já concluídos
            "best_val_loss": best_val_loss,
            "elapsed": elapsed,
            "train_args": train_args,
            "chars_per_token": chars_per_token,
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
                bpc = {split: bits_per_char(losses[split], chars_per_token[split])
                       for split in ("train", "val")}
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
                    f"val {losses['val']:.4f} ({bpc['val']:.4f} bits/caractere) | "
                    f"{format_duration(elapsed)} decorridos | ~{format_duration(remaining)} restantes"
                )

                if losses["val"] < best_val_loss:
                    best_val_loss = losses["val"]
                    save_checkpoint(best_path, make_state(iteration, elapsed))
                    print(f"  ↳ nova melhor val loss; {best_path.name} salvo")
                save_checkpoint(last_path, make_state(iteration, elapsed))
                append_log(log_path, {
                    "iter": iteration,
                    "train_loss": round(losses["train"], 4),
                    "val_loss": round(losses["val"], 4),
                    "train_bpc": round(bpc["train"], 4),
                    "val_bpc": round(bpc["val"], 4),
                    "tempo_decorrido": round(elapsed, 1),
                })

                print("--- amostra ---")
                print(generate_sample(model, tokenizer, device))
                print("---------------\n")
    except KeyboardInterrupt:
        # Ctrl+C: salvamos o estado atual para não perder o progresso.
        elapsed = elapsed_before + time.time() - session_start
        save_checkpoint(last_path, make_state(iteration, elapsed))
        print(f"\nInterrompido no passo {iteration}. Estado salvo em {last_path}.")
        print(f"Para continuar: {resume_command}")
        sys.exit(130)

    best_bpc = bits_per_char(best_val_loss, chars_per_token["val"])
    print("Treino concluído.")
    print(f"Melhor val loss: {best_val_loss:.4f} ({best_bpc:.4f} bits por caractere)")
    print(f"Melhor modelo: {best_path}")
    print(f'Para gerar texto: python src/generate.py --checkpoint {best_path} --prompt "Capitu"')


if __name__ == "__main__":
    main()
