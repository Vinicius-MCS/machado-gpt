"""Testes da preparação dos dados (src/data.py).

Cobrem o que muda com a chegada do BPE: a separação treino/validação feita
no texto (antes de tokenizar) e o carregamento de qualquer tokenizador salvo.

Uso:
    pytest
"""

from pathlib import Path

from bpe import BPETokenizer
from data import WORK_SEPARATOR, Tokenizer, load_tokenizer, split_data, split_text

# Três "obras" de tamanhos diferentes, separadas como no dataset real.
WORKS = ["a" * 100, "b" * 50, "c" * 30]
TEXT = WORK_SEPARATOR.join(WORKS)


def test_split_text_corta_os_ultimos_10_por_cento_de_cada_obra() -> None:
    train_text, val_text = split_text(TEXT)
    assert train_text.split(WORK_SEPARATOR) == ["a" * 90, "b" * 45, "c" * 27]
    assert val_text.split(WORK_SEPARATOR) == ["a" * 10, "b" * 5, "c" * 3]


def test_split_data_equivale_a_tokenizar_o_resultado_do_split_text() -> None:
    """O refactor não muda os tensores: split_data = split_text + encode."""
    tokenizer = Tokenizer(TEXT)
    train_data, val_data = split_data(TEXT, tokenizer)
    train_text, val_text = split_text(TEXT)
    assert train_data.tolist() == tokenizer.encode(train_text)
    assert val_data.tolist() == tokenizer.encode(val_text)


def test_load_tokenizer_reconhece_os_dois_tipos(tmp_path: Path) -> None:
    char_path = tmp_path / "char.json"
    Tokenizer(TEXT).save(char_path)
    assert isinstance(load_tokenizer(char_path), Tokenizer)

    bpe_path = tmp_path / "bpe.json"
    BPETokenizer.train(TEXT, vocab_size=10).save(bpe_path)
    assert isinstance(load_tokenizer(bpe_path), BPETokenizer)


def test_load_tokenizer_aceita_arquivo_antigo_sem_tipo(tmp_path: Path) -> None:
    """Os tokenizer.json dos modelos já treinados não têm o campo "type"."""
    path = tmp_path / "tokenizer.json"
    path.write_text('{"chars": ["\\n", "a", "b"]}', encoding="utf-8")
    tokenizer = load_tokenizer(path)
    assert isinstance(tokenizer, Tokenizer)
    assert tokenizer.vocab_size == 3
