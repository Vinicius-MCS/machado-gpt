"""Tokenizador BPE (Byte Pair Encoding) com caracteres como base.

O tokenizador do data.py usa um token por caractere. O BPE parte do mesmo
vocabulário (os caracteres do corpus) e aprende "fusões": repetidamente, junta
o par de tokens vizinhos mais frequente num token novo. Assim, pedaços comuns
como " que", "ão" e " perguntou" viram um único token, e palavras raras
continuam podendo ser montadas a partir de pedaços menores.

Por que isso ajuda o modelo:
- as sequências ficam mais curtas, então o mesmo block_size enxerga mais texto;
- cada token já carrega um pedaço de significado, e o modelo não precisa
  gastar capacidade "soletrando" palavras comuns.

A interface (encode, decode, vocab_size, save, load) é a mesma da classe
Tokenizer do data.py, para os dois serem intercambiáveis no treino.

Uso (demonstração):
    python src/bpe.py
    python src/bpe.py --vocab_size 1000
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Pré-divisão do texto, inspirada no GPT-2: cada palavra leva o espaço que a
# precede (" que"), sequências de pontuação ficam juntas ("--", "?"), e
# espaços/quebras de linha restantes formam pedaços próprios ("\n\n").
# As fusões só acontecem DENTRO de cada pedaço; sem isso, o BPE gastaria
# vocabulário com combinações acidentais entre palavras, como "a de".
# O \w do Python já reconhece letras acentuadas.
PRETOKENIZE_PATTERN = re.compile(r" ?\w+| ?[^\w\s]+|\s+")

# Tipo usado no JSON salvo, para o carregador saber qual tokenizador recriar.
TOKENIZER_TYPE = "bpe"


def merge_pair(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Substitui toda ocorrência de `pair` em `ids` pelo token `new_id`.

    Exemplo: merge_pair([5, 6, 5, 6, 7], (5, 6), 99) -> [99, 99, 7]
    """
    merged = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == pair:
            merged.append(new_id)
            i += 2  # pulamos os dois tokens que acabaram de ser fundidos
        else:
            merged.append(ids[i])
            i += 1
    return merged


class BPETokenizer:
    """Tokenizador BPE: caracteres + uma lista ordenada de fusões.

    Os ids seguem esta regra:
    - 0 .. len(chars)-1: os caracteres, em ordem, exatamente como no Tokenizer;
    - a partir daí, um id por fusão, na ordem em que foram aprendidas.

    Internamente tudo é feito com ids, não com textos. Duas fusões diferentes
    podem gerar o mesmo texto (por exemplo "a"+"bc" e "ab"+"c" dão "abc");
    com ids, cada fusão tem identidade própria e esse conflito não existe.
    """

    def __init__(self, chars: list[str], merges: list[tuple[int, int]]) -> None:
        self.chars = chars
        self.merges = merges
        # stoi só contém os caracteres-base: é o que generate.py e
        # visualize_attention.py usam para avisar sobre caracteres desconhecidos.
        self.stoi: dict[str, int] = {ch: i for i, ch in enumerate(chars)}
        # itos dá o texto de qualquer token. O texto de uma fusão é a
        # concatenação dos textos dos dois tokens que a formaram.
        self.itos: dict[int, str] = dict(enumerate(chars))
        # merge_ranks: par -> id do token criado. Como os ids das fusões
        # crescem na ordem do treino, o id também indica QUANDO a fusão foi
        # aprendida (id menor = aprendida antes). O encode usa isso.
        self.merge_ranks: dict[tuple[int, int], int] = {}
        for new_id, (a, b) in enumerate(merges, start=len(chars)):
            self.itos[new_id] = self.itos[a] + self.itos[b]
            self.merge_ranks[(a, b)] = new_id
        # Cache do encode: palavras se repetem muito (" que" aparece dezenas de
        # milhares de vezes), então cada pedaço distinto é codificado uma vez só.
        self._cache: dict[str, list[int]] = {}

    @classmethod
    def train(
        cls, text: str, vocab_size: int, alphabet: str = "", verbose: bool = False
    ) -> "BPETokenizer":
        """Aprende as fusões a partir de um texto.

        vocab_size é o tamanho TOTAL do vocabulário (caracteres + fusões), que
        é o que define o tamanho das camadas de embedding e de saída do modelo.

        alphabet: caracteres extras que devem existir no vocabulário-base mesmo
        sem aparecer em `text`. O train.py passa os caracteres do dataset
        inteiro: as fusões são aprendidas só com o treino, mas um caractere
        raro que só aparece na validação (como o "Í" de um "ÍNDICE" no fim do
        livro) precisa existir, senão a validação nem pode ser codificada.
        """
        chars = sorted(set(text) | set(alphabet))
        num_merges = vocab_size - len(chars)
        if num_merges < 0:
            raise ValueError(
                f"vocab_size ({vocab_size}) é menor que o número de caracteres "
                f"distintos do texto ({len(chars)})."
            )
        stoi = {ch: i for i, ch in enumerate(chars)}

        # Em vez de percorrer o texto inteiro a cada fusão, trabalhamos com os
        # pedaços DISTINTOS e quantas vezes cada um aparece. " que" aparece
        # ~24 mil vezes, mas é processado uma vez só, com peso 24 mil.
        chunk_counts = Counter(PRETOKENIZE_PATTERN.findall(text))
        words = [[stoi[ch] for ch in chunk] for chunk in chunk_counts]
        counts = list(chunk_counts.values())

        tokenizer = cls(chars, [])
        start = time.time()
        for step in range(num_merges):
            # 1. Conta todos os pares vizinhos, pesando pela frequência do pedaço.
            pair_counts: Counter[tuple[int, int]] = Counter()
            for ids, count in zip(words, counts):
                for pair in zip(ids, ids[1:]):
                    pair_counts[pair] += count
            if not pair_counts:
                # Todo pedaço já virou um único token: não há mais o que fundir.
                print(f"Aviso: só foi possível aprender {step} fusões "
                      f"(vocabulário final: {len(chars) + step}).")
                break

            # 2. Escolhe o par mais frequente. Em caso de empate, o menor par
            # de ids, para o resultado ser sempre o mesmo (reprodutível).
            best_pair = min(pair_counts, key=lambda pair: (-pair_counts[pair], pair))
            new_id = len(chars) + step

            # 3. Funde esse par em todos os pedaços.
            words = [merge_pair(ids, best_pair, new_id) for ids in words]
            tokenizer = cls(chars, tokenizer.merges + [best_pair])

            if verbose and (step < 10 or (step + 1) % 100 == 0):
                a, b = best_pair
                print(f"  fusão {step + 1:4d}/{num_merges}: {tokenizer.itos[a]!r} + {tokenizer.itos[b]!r}"
                      f" -> {tokenizer.itos[new_id]!r} ({pair_counts[best_pair]:,} ocorrências)"
                      f" [{time.time() - start:.0f}s]")
        return tokenizer

    @property
    def vocab_size(self) -> int:
        """Número de tokens distintos (tamanho da saída do modelo)."""
        return len(self.itos)

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Codifica um único pedaço da pré-divisão (ex.: " perguntou")."""
        if chunk in self._cache:
            return self._cache[chunk]
        ids = [self.stoi[ch] for ch in chunk]
        # Aplicar as fusões uma a uma, na ordem do treino, daria o resultado
        # certo, mas percorreria o pedaço centenas de vezes. Equivalente e bem
        # mais rápido: entre os pares presentes, fundir sempre o que foi
        # aprendido mais cedo (menor id), até não restar par conhecido.
        while len(ids) >= 2:
            pairs = set(zip(ids, ids[1:]))
            pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float("inf")))
            if pair not in self.merge_ranks:
                break  # nenhum par do pedaço está na lista de fusões
            ids = merge_pair(ids, pair, self.merge_ranks[pair])
        self._cache[chunk] = ids
        return ids

    def encode(self, text: str) -> list[int]:
        """Converte texto em lista de ids."""
        unknown = sorted({ch for ch in text if ch not in self.stoi})
        if unknown:
            raise ValueError(
                f"Caracteres fora do vocabulário: {unknown!r}. "
                "O modelo só conhece os caracteres presentes nas obras do dataset."
            )
        ids: list[int] = []
        for chunk in PRETOKENIZE_PATTERN.findall(text):
            ids.extend(self._encode_chunk(chunk))
        return ids

    def decode(self, ids: list[int]) -> str:
        """Converte lista de ids de volta em texto (basta concatenar os pedaços)."""
        return "".join(self.itos[i] for i in ids)

    def save(self, path: Path) -> None:
        """Salva caracteres e fusões em JSON.

        Assim como no Tokenizer, o generate.py precisa reconstruir exatamente o
        mesmo vocabulário do treino, senão os ids não correspondem aos tokens.
        A lista "tokens" é só para inspeção humana; o load não precisa dela.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "type": TOKENIZER_TYPE,
            "chars": self.chars,
            "merges": [list(pair) for pair in self.merges],
            "tokens": [self.itos[i] for i in range(self.vocab_size)],
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Reconstrói o tokenizador a partir do JSON salvo por save()."""
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("type") != TOKENIZER_TYPE:
            raise ValueError(f"{path} não é um tokenizador BPE.")
        return cls(data["chars"], [tuple(pair) for pair in data["merges"]])


def show_tokens(tokenizer: BPETokenizer, text: str) -> str:
    """Mostra a divisão de um texto em tokens, separados por "|"."""
    return "|".join(tokenizer.itos[i] for i in tokenizer.encode(text))


def main() -> None:
    """Demonstra o BPE num exemplo à mão e no corpus de Machado."""
    # Importado aqui para o bpe.py não depender do torch (via data.py) ao ser importado.
    from data import DATASET_PATH, load_text

    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Demonstração do tokenizador BPE.")
    parser.add_argument("--vocab_size", type=int, default=500)
    args = parser.parse_args()

    # 1. O exemplo calculado à mão: casa ×3, casas ×2, caso ×1. As palavras
    # ficam em linhas separadas para o espaço não entrar nas contagens.
    toy_text = "\n".join(["casa"] * 3 + ["casas"] * 2 + ["caso"])
    toy = BPETokenizer.train(toy_text, vocab_size=len(set(toy_text)) + 3)
    print("=== Exemplo à mão (casa ×3, casas ×2, caso ×1) ===")
    for step, (a, b) in enumerate(toy.merges, start=1):
        print(f"  fusão {step}: {toy.itos[a]!r} + {toy.itos[b]!r} -> {toy.itos[a] + toy.itos[b]!r}")
    # "casaco" não está no mini-corpus, mas é montada com pedaços conhecidos.
    for word in ("casa", "casas", "caso", "casaco"):
        print(f"  {word!r:10} -> {show_tokens(toy, word)}")

    # 2. O corpus de Machado. Aqui treinamos no texto inteiro só para
    # demonstrar; no treino do modelo, o BPE vai usar só a parte de treino.
    text = load_text(DATASET_PATH)
    print(f"\n=== Corpus de Machado ({len(text):,} caracteres), vocab_size = {args.vocab_size} ===")
    tokenizer = BPETokenizer.train(text, args.vocab_size, verbose=True)

    sample = "--Capitu, que é isso? perguntou José Dias."
    print(f"\n{show_tokens(tokenizer, sample)}")
    ids = tokenizer.encode(sample)
    print(f"{len(sample)} caracteres -> {len(ids)} tokens")
    print(f"decode(encode(texto)) == texto: {tokenizer.decode(ids) == sample}")


if __name__ == "__main__":
    main()
