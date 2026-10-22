"""Coleta de obras de Machado de Assis no Project Gutenberg.

Este script é a primeira etapa do pipeline: ele monta o corpus de texto que o
modelo GPT vai aprender a imitar. O fluxo é:

1. Baixar o .txt de cada obra listada em WORKS (com cache em data/raw/).
2. Remover o cabeçalho e o rodapé de licença que o Gutenberg coloca em
   todo livro, mantendo só o texto da obra.
3. Limpar o texto: normalizar quebras de linha e juntar cada parágrafo numa
   linha só.
4. Concatenar as obras, separadas por "\n\n---\n\n", em data/dataset.txt.

Uso:
    python src/collect.py
"""

import re
import sys
import time
from pathlib import Path

import requests

# Obras a coletar (título → URL do .txt no Gutenberg).
# Para adicionar uma obra, basta incluir uma nova linha aqui.
WORKS: dict[str, str] = {
    "Dom Casmurro": "https://www.gutenberg.org/cache/epub/55752/pg55752.txt",
    "Memórias Póstumas de Brás Cubas": "https://www.gutenberg.org/cache/epub/54829/pg54829.txt",
    "Quincas Borba": "https://www.gutenberg.org/cache/epub/55682/pg55682.txt",
    "Esaú e Jacó": "https://www.gutenberg.org/cache/epub/56737/pg56737.txt",
    "Papéis Avulsos": "https://www.gutenberg.org/cache/epub/57001/pg57001.txt",
    "Helena": "https://www.gutenberg.org/cache/epub/67162/pg67162.txt",
    "Memorial de Aires": "https://www.gutenberg.org/cache/epub/55797/pg55797.txt",
    "Iaiá Garcia": "https://www.gutenberg.org/cache/epub/67780/pg67780.txt",
    "A Mão e a Luva": "https://www.gutenberg.org/cache/epub/53101/pg53101.txt",
    "Histórias sem Data": "https://www.gutenberg.org/cache/epub/33056/pg33056.txt",
    "Relíquias de Casa Velha": "https://www.gutenberg.org/cache/epub/67935/pg67935.txt",
}

# Raiz do projeto (pasta acima de src/), para os caminhos funcionarem
# independentemente do diretório de onde o script é executado.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUTPUT_PATH = DATA_DIR / "dataset.txt"

WORK_SEPARATOR = "\n\n---\n\n"

REQUEST_TIMEOUT = 30  # segundos
MAX_ATTEMPTS = 3  # tentativas por download antes de desistir
BACKOFF_BASE = 1.0  # espera 1s, 2s, 4s... entre tentativas

# Códigos HTTP que indicam problema temporário: vale a pena tentar de novo.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}

# Marcações que delimitam o texto da obra. Elas incluem o título e variam
# entre livros ("THE"/"THIS", espaços), por isso o regex é tolerante.
START_MARKER = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*")
END_MARKER = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*")

# Uma única sessão reaproveita a conexão TCP/TLS entre os downloads.
session = requests.Session()


def get_ebook_id(url: str) -> str:
    """Extrai o número do ebook da URL (ex.: ".../pg55752.txt" → "55752")."""
    match = re.search(r"pg(\d+)\.txt$", url)
    if not match:
        raise ValueError(f"URL fora do padrão do Gutenberg: {url}")
    return match.group(1)


def download_text(url: str) -> str:
    """Baixa um arquivo de texto, com retry e backoff exponencial."""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            if response.status_code in RETRYABLE_STATUS:
                raise requests.HTTPError(f"HTTP {response.status_code}")
            response.raise_for_status()
            # Decodificamos os bytes nós mesmos: response.text depende do
            # palpite de encoding do requests, que pode errar e quebrar os
            # acentos. O "-sig" descarta o BOM do início, se houver.
            return response.content.decode("utf-8-sig")
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError):
            if attempt == MAX_ATTEMPTS:
                raise
            time.sleep(BACKOFF_BASE * 2 ** (attempt - 1))
    raise AssertionError("inalcançável")


def fetch_work(url: str) -> str:
    """Retorna o texto bruto da obra, usando o cache em data/raw/ se existir."""
    cache_path = RAW_DIR / f"{get_ebook_id(url)}.txt"
    # Lemos e gravamos bytes, não texto: em modo texto o Windows trocaria cada
    # "\n" por "\r\n", e como o Gutenberg já usa "\r\n", o cache ficaria com
    # "\r\r\n", que na leitura vira uma linha em branco extra a cada linha.
    if cache_path.exists():
        return cache_path.read_bytes().decode("utf-8")
    text = download_text(url)
    cache_path.write_bytes(text.encode("utf-8"))
    return text


def strip_gutenberg_boilerplate(text: str, title: str) -> str:
    """Mantém só o texto entre as marcações START e END do Gutenberg."""
    start = START_MARKER.search(text)
    end = END_MARKER.search(text)
    # Melhor falhar do que treinar com o texto da licença no meio do dataset.
    if not start:
        raise ValueError(f"Marcação START do Gutenberg não encontrada em '{title}'")
    if not end:
        raise ValueError(f"Marcação END do Gutenberg não encontrada em '{title}'")
    return text[start.end() : end.start()]


def clean_text(text: str) -> str:
    """Normaliza o texto e junta as linhas de cada parágrafo numa só.

    O Gutenberg quebra as linhas em ~70 colunas por pura formatação. Num modelo
    character-level, cada "\n" é um token como outro qualquer: se mantivéssemos
    essas quebras fixas, o modelo aprenderia a pular linha a cada ~70
    caracteres, no meio da frase, em vez de aprender onde um parágrafo termina.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Algumas edições usam o travessão "—" nos diálogos; a maioria usa "--".
    # Unificamos em "--" para o modelo não gastar um token do vocabulário (e
    # exemplos de treino) com duas grafias da mesma coisa.
    text = text.replace("—", "--")
    # Parágrafos são separados por uma ou mais linhas em branco.
    paragraphs = re.split(r"\n\s*\n", text)
    joined = []
    for paragraph in paragraphs:
        # Junta as linhas do parágrafo com espaço, tirando espaços das pontas.
        lines = [line.strip() for line in paragraph.split("\n")]
        line = " ".join(part for part in lines if part)
        if line:
            joined.append(line)
    return "\n\n".join(joined).strip()


def save_dataset(works: list[str], output_path: Path) -> None:
    """Salva as obras num único arquivo, separadas por WORK_SEPARATOR."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # encoding explícito: no Windows o padrão não é UTF-8 e os acentos quebrariam.
    # newline="\n" evita que o Windows grave "\r\n" no lugar de cada "\n".
    output_path.write_text(WORK_SEPARATOR.join(works) + "\n", encoding="utf-8", newline="\n")


def main() -> None:
    """Executa a coleta completa e imprime um resumo do dataset gerado."""
    # Garante que os acentos apareçam corretamente no console do Windows.
    sys.stdout.reconfigure(encoding="utf-8")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_works: list[str] = []
    sizes: dict[str, int] = {}
    for index, (title, url) in enumerate(WORKS.items(), start=1):
        print(f"[{index}/{len(WORKS)}] {title}...")
        raw = fetch_work(url)
        text = clean_text(strip_gutenberg_boilerplate(raw, title))
        cleaned_works.append(text)
        sizes[title] = len(text)

    save_dataset(cleaned_works, OUTPUT_PATH)

    # Resumo: esses números mostram o que o modelo vai receber de entrada.
    dataset = OUTPUT_PATH.read_text(encoding="utf-8")
    print("\n=== Resumo ===")
    print(f"Obras: {len(cleaned_works)}")
    for title, size in sizes.items():
        print(f"  - {title}: {size:,} caracteres")
    print(f"Total de caracteres: {len(dataset):,}")
    # Na tokenização character-level, cada caractere único é um token.
    print(f"Vocabulário: {len(set(dataset))} caracteres únicos")
    print(f"Arquivo: {OUTPUT_PATH}")
    print("\n--- Prévia (300 primeiros caracteres) ---")
    print(dataset[:300])


if __name__ == "__main__":
    main()
