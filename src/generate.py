"""Geração de texto com o GPT treinado.

Carrega um checkpoint salvo pelo train.py e gera texto a partir de um prompt,
imprimindo cada caractere assim que ele é sorteado (streaming).

Sobre a amostragem: a cada passo o modelo devolve uma distribuição de
probabilidade sobre o próximo caractere, e nós SORTEAMOS um caractere dessa
distribuição. Por que não pegar sempre o mais provável (argmax)? Porque a
geração vira determinística e o modelo tende a entrar em loop, repetindo o
mesmo trecho ("que não era que não era que não era..."). O sorteio mantém a
variedade, e dois parâmetros controlam o quanto arriscamos:

- temperature: divide os logits antes do softmax.
    0.5 → conservador: texto mais "seguro", mas repetitivo
    0.8 → bom equilíbrio (padrão)
    1.0 → a distribuição exatamente como o modelo aprendeu
    1.3 → mais criativo, com mais palavras inventadas e erros
- top_k: só sorteia entre os k caracteres mais prováveis, cortando a cauda
  de opções improváveis que geram "palavras" absurdas.

Uso:
    python src/generate.py --prompt "Capitu"
    python src/generate.py --prompt "Capitu" --temperature 0.5 --top_k 10
    python src/generate.py --num_samples 3 --seed 42
"""

import argparse
import sys
from pathlib import Path

import torch

from data import Tokenizer
from model import GPTConfig, GPTLanguageModel
from train import BEST_PATH, TOKENIZER_PATH, resolve_device

# O tokenizador é salvo na mesma pasta do checkpoint (train.py), com este nome.
TOKENIZER_FILENAME = TOKENIZER_PATH.name


def parse_args() -> argparse.Namespace:
    """Define e lê os argumentos de linha de comando."""
    parser = argparse.ArgumentParser(description="Gera texto com o GPT treinado.")
    parser.add_argument("--prompt", default="\n", help="texto inicial (\\n vira quebra de linha)")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--num_samples", type=int, default=1, help="gerações independentes")
    parser.add_argument("--checkpoint", type=Path, default=BEST_PATH)
    parser.add_argument("--seed", type=int, default=None, help="fixa o resultado (reprodutível)")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda ou mps")
    return parser.parse_args()


def load_model(checkpoint_path: Path, device: str) -> tuple[GPTLanguageModel, dict, Tokenizer]:
    """Reconstrói o modelo e o tokenizador a partir do checkpoint."""
    # Cada checkpoint tem o seu próprio vocabulário: o tokenizador precisa vir
    # da mesma pasta, senão os ids não correspondem aos caracteres certos.
    tokenizer_path = checkpoint_path.parent / TOKENIZER_FILENAME
    for path in (checkpoint_path, tokenizer_path):
        if not path.exists():
            sys.exit(f"Erro: {path} não encontrado. Treine o modelo primeiro com: python src/train.py")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    tokenizer = Tokenizer.load(tokenizer_path)
    if tokenizer.vocab_size != checkpoint["config"]["vocab_size"]:
        sys.exit(
            f"Erro: {tokenizer_path} tem {tokenizer.vocab_size} caracteres, mas o checkpoint "
            f"foi treinado com {checkpoint['config']['vocab_size']}. Use o tokenizer.json "
            "salvo junto com esse checkpoint."
        )
    # A arquitetura vem do checkpoint, para os pesos caberem exatamente.
    model = GPTLanguageModel(GPTConfig(**checkpoint["config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    # model.eval() desliga o dropout. Ele só serve para regularizar o treino;
    # na geração, queremos usar a rede inteira.
    model.eval()
    return model, checkpoint, tokenizer


def main() -> None:
    """Carrega o modelo, valida o prompt e gera texto em streaming."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    if args.seed is not None:
        torch.manual_seed(args.seed)
    device = resolve_device(args.device)

    model, checkpoint, tokenizer = load_model(args.checkpoint, device)
    print(
        f"[checkpoint: passo {checkpoint['iter']} | val loss {checkpoint['best_val_loss']:.4f} | "
        f"{model.num_params():,} parâmetros]\n"
    )

    # No terminal é difícil digitar uma quebra de linha; aceitamos "\n" escrito.
    prompt = args.prompt.replace("\\n", "\n")
    if not prompt:
        sys.exit("Erro: o prompt não pode ser vazio.")
    try:
        prompt_ids = tokenizer.encode(prompt)
    except ValueError:
        unknown = sorted({ch for ch in prompt if ch not in tokenizer.stoi})
        sys.exit(
            f"O prompt tem caracteres que o modelo nunca viu: {' '.join(unknown)}\n"
            "Ele só conhece os caracteres presentes nas obras de Machado de Assis. "
            "Remova esses caracteres e tente de novo."
        )

    block_size = model.config.block_size
    if len(prompt_ids) > block_size:
        # generate_tokens já recorta o contexto; aqui só avisamos o usuário.
        print(f"Aviso: o prompt tem {len(prompt_ids)} caracteres; só os últimos {block_size} serão usados como contexto.\n")

    try:
        for sample in range(args.num_samples):
            if sample > 0:
                print("\n---")
            idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)  # (1, T)
            # Imprimimos o prompt antes, para o texto gerado continuar dele.
            print(prompt, end="", flush=True)
            # Streaming: cada caractere aparece assim que é sorteado.
            for token_id in model.generate_tokens(idx, args.max_new_tokens, args.temperature, args.top_k):
                print(tokenizer.decode([token_id]), end="", flush=True)
        print()
    except KeyboardInterrupt:
        # Ctrl+C encerra a geração sem mostrar traceback.
        print("\n[geração interrompida]")
        sys.exit(130)


if __name__ == "__main__":
    main()
