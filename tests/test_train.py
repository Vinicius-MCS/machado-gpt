"""Testes das funções auxiliares do treino (src/train.py).

Cobrem a conversão para bits por caractere (a métrica que compara
tokenizadores diferentes) e a compatibilidade do log com treinos antigos.

Uso:
    pytest
"""

import csv
import math
from pathlib import Path

import pytest

from train import LOG_FIELDS, append_log, bits_per_char

ROW = {"iter": 10, "train_loss": 1.5, "val_loss": 1.6, "train_bpc": 2.1,
       "val_bpc": 2.3, "tempo_decorrido": 12.0}


def test_bits_por_caractere_no_tokenizador_de_caractere_e_loss_sobre_ln2() -> None:
    """Com 1 caractere por token, só muda a unidade: nats -> bits."""
    assert bits_per_char(1.2188, 1.0) == pytest.approx(1.2188 / math.log(2))
    assert bits_per_char(1.2188, 1.0) == pytest.approx(1.758, abs=1e-3)


def test_bits_por_caractere_com_bpe() -> None:
    """O exemplo da explicação: loss 3,0 por token, 2,7 caracteres por token."""
    assert bits_per_char(3.0, 2.7) == pytest.approx(1.603, abs=1e-3)


def test_log_novo_tem_todas_as_colunas(tmp_path: Path) -> None:
    path = tmp_path / "train_log.csv"
    append_log(path, ROW)
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert list(rows[0]) == LOG_FIELDS
    assert float(rows[0]["val_bpc"]) == 2.3


def test_log_antigo_mantem_o_cabecalho_antigo(tmp_path: Path) -> None:
    """Retomar um treino de antes do BPE não pode desalinhar o CSV."""
    path = tmp_path / "train_log.csv"
    path.write_text("iter,train_loss,val_loss,tempo_decorrido\n5,1.9,2.0,3.0\n", encoding="utf-8")
    append_log(path, ROW)
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.reader(file))
    assert rows[0] == ["iter", "train_loss", "val_loss", "tempo_decorrido"]
    assert rows[-1] == ["10", "1.5", "1.6", "12.0"]
