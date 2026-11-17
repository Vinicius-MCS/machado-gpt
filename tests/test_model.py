"""Testes da arquitetura (src/model.py).

Por enquanto cobrem a opção de weight tying: ela tem que economizar
exatamente vocab_size × n_embd parâmetros e não pode mudar nada quando
está desligada (o padrão, usado pelos modelos já treinados).

Uso:
    pytest
"""

import torch

from model import GPTConfig, GPTLanguageModel

VOCAB_SIZE, N_EMBD = 50, 32


def make_model(tie_weights: bool) -> GPTLanguageModel:
    config = GPTConfig(vocab_size=VOCAB_SIZE, block_size=16, n_embd=N_EMBD,
                       n_head=4, n_layer=2, tie_weights=tie_weights)
    return GPTLanguageModel(config)


def test_tying_economiza_vocab_size_vezes_n_embd_parametros() -> None:
    untied, tied = make_model(False), make_model(True)
    assert untied.num_params() - tied.num_params() == VOCAB_SIZE * N_EMBD


def test_tying_usa_a_mesma_matriz_nas_duas_pontas() -> None:
    model = make_model(True)
    assert model.lm_head.weight is model.token_embedding.weight


def test_sem_tying_as_matrizes_sao_independentes() -> None:
    model = make_model(False)
    assert model.lm_head.weight is not model.token_embedding.weight


def test_config_antiga_sem_o_campo_carrega_sem_tying() -> None:
    """Os checkpoints já treinados guardam a config sem "tie_weights"."""
    old_config = {"vocab_size": VOCAB_SIZE, "block_size": 16, "n_embd": N_EMBD,
                  "n_head": 4, "n_layer": 2, "dropout": 0.1}
    assert GPTConfig(**old_config).tie_weights is False


def test_modelo_com_tying_treina_e_salva() -> None:
    """Um passo de treino e um ciclo state_dict -> load_state_dict funcionam."""
    torch.manual_seed(0)
    model = make_model(True)
    idx = torch.randint(VOCAB_SIZE, (2, 16))
    _, loss = model(idx, idx)
    loss.backward()
    restored = make_model(True)
    restored.load_state_dict(model.state_dict())
    assert restored.lm_head.weight is restored.token_embedding.weight
    assert torch.equal(restored.token_embedding.weight, model.token_embedding.weight)
