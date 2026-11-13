"""Testes do tokenizador BPE (src/bpe.py).

Garantem que o tokenizador está correto ANTES de usá-lo para treinar modelos:
um bug aqui (um caractere perdido, um id trocado) estragaria o treino em
silêncio, sem nenhuma mensagem de erro.

Uso:
    pytest
    pytest -v          # mostra o nome de cada teste
"""

from pathlib import Path

import pytest

from bpe import PRETOKENIZE_PATTERN, BPETokenizer, merge_pair
from data import DATASET_PATH, load_text

CORPUS_VOCAB_SIZE = 300  # pequeno, para os testes rodarem rápido


@pytest.fixture(scope="module")
def corpus() -> str:
    """O dataset completo; os testes que dependem dele são pulados se não existir."""
    if not DATASET_PATH.exists():
        pytest.skip("data/dataset.txt não existe; rode python src/collect.py")
    return load_text(DATASET_PATH)


@pytest.fixture(scope="module")
def corpus_tokenizer(corpus: str) -> BPETokenizer:
    """Um tokenizador treinado UMA vez no corpus e compartilhado entre os testes."""
    return BPETokenizer.train(corpus, CORPUS_VOCAB_SIZE)


def test_merge_pair_substitui_todas_as_ocorrencias() -> None:
    assert merge_pair([5, 6, 5, 6, 7], (5, 6), 99) == [99, 99, 7]
    # Sobreposição: em "a a a", só o primeiro par é fundido, e o terceiro "a" sobra.
    assert merge_pair([1, 1, 1], (1, 1), 9) == [9, 1]


def test_exemplo_calculado_a_mao() -> None:
    """casa ×3, casas ×2, caso ×1 devem gerar as fusões as, cas, casa."""
    text = "\n".join(["casa"] * 3 + ["casas"] * 2 + ["caso"])
    tokenizer = BPETokenizer.train(text, vocab_size=len(set(text)) + 3)
    learned = [tokenizer.itos[a] + tokenizer.itos[b] for a, b in tokenizer.merges]
    assert learned == ["as", "cas", "casa"]
    assert tokenizer.decode(tokenizer.encode("casas")) == "casas"
    assert [tokenizer.itos[i] for i in tokenizer.encode("casas")] == ["cas", "as"]


def test_vocab_size_e_o_total(corpus_tokenizer: BPETokenizer) -> None:
    """Decisão 4: vocab_size conta caracteres + fusões."""
    assert corpus_tokenizer.vocab_size == CORPUS_VOCAB_SIZE
    assert len(corpus_tokenizer.chars) + len(corpus_tokenizer.merges) == CORPUS_VOCAB_SIZE


def test_pre_divisao_nao_perde_caracteres(corpus: str) -> None:
    """Juntar os pedaços da regex tem que devolver o texto original."""
    assert "".join(PRETOKENIZE_PATTERN.findall(corpus)) == corpus


def test_ida_e_volta_no_corpus_inteiro(corpus: str, corpus_tokenizer: BPETokenizer) -> None:
    ids = corpus_tokenizer.encode(corpus)
    assert corpus_tokenizer.decode(ids) == corpus
    # E o BPE de fato comprime: menos tokens que caracteres.
    assert len(ids) < len(corpus)


def test_encode_rapido_igual_a_aplicar_fusoes_em_ordem(
    corpus: str, corpus_tokenizer: BPETokenizer
) -> None:
    """O atalho do encode ("fundir sempre o par de menor id") dá o mesmo
    resultado que o método ingênuo: aplicar cada fusão, na ordem do treino."""
    chunks = set(PRETOKENIZE_PATTERN.findall(corpus[:200_000]))
    for chunk in chunks:
        ids = [corpus_tokenizer.stoi[ch] for ch in chunk]
        for new_id, pair in enumerate(corpus_tokenizer.merges, start=len(corpus_tokenizer.chars)):
            ids = merge_pair(ids, pair, new_id)
        assert corpus_tokenizer.encode(chunk) == ids, chunk


def test_salvar_e_carregar_preserva_a_codificacao(
    corpus_tokenizer: BPETokenizer, tmp_path: Path
) -> None:
    path = tmp_path / "tokenizer.json"
    corpus_tokenizer.save(path)
    loaded = BPETokenizer.load(path)
    sample = "--Capitu, que é isso? perguntou José Dias."
    assert loaded.encode(sample) == corpus_tokenizer.encode(sample)
    assert loaded.vocab_size == corpus_tokenizer.vocab_size


def test_caractere_desconhecido_da_erro_amigavel(corpus_tokenizer: BPETokenizer) -> None:
    with pytest.raises(ValueError, match="fora do vocabulário"):
        corpus_tokenizer.encode("Capitu 😀")


def test_load_recusa_arquivo_de_outro_tipo(tmp_path: Path) -> None:
    """Um tokenizer.json do modelo por caractere não pode ser lido como BPE."""
    path = tmp_path / "tokenizer.json"
    path.write_text('{"chars": ["a", "b"]}', encoding="utf-8")
    with pytest.raises(ValueError, match="não é um tokenizador BPE"):
        BPETokenizer.load(path)
