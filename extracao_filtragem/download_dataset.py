#!/usr/bin/env python3
"""
Script para baixar o dataset LDBC SNB (snapshot completo) do repositório SURF.
Suporta staging automático quando o arquivo está em tape.
"""

import argparse
import re
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Instale requests: pip install requests")
    sys.exit(1)


# URLs dos datasets (Datagen v1.0.0 - recomendado)
DATASETS = {
    "sf0.1": "https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/social_network-sf0.1-CsvBasic-LongDateFormatter.tar.zst",
    "sf0.3": "https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/social_network-sf0.3-CsvBasic-LongDateFormatter.tar.zst",
    "sf1": "https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/social_network-sf1-CsvBasic-LongDateFormatter.tar.zst",
    "sf3": "https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/social_network-sf3-CsvBasic-LongDateFormatter.tar.zst",
    "sf10": "https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/social_network-sf10-CsvBasic-LongDateFormatter.tar.zst",
    "sf30": "https://repository.surfsara.nl/datasets/cwi/ldbc-snb-interactive-v1-datagen-v100/files/social_network-sf30-CsvBasic-LongDateFormatter.tar.zst",
}

_SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = _SCRIPT_DIR / "dataset"


def log(msg: str) -> None:
    print(f"[Download] {msg}", flush=True)


# Desabilitar avisos de SSL quando verify=False
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def trigger_staging(url: str) -> bool:
    """Dispara staging quando o arquivo está em tape (409 Conflict)."""
    try:
        r = requests.get(url, timeout=30, verify=False)
        if r.status_code != 200:
            return False
        # Extrair URL de staging do HTML/JSON
        match = re.search(
            r'https://repository\.surfsara\.nl/api/objects/cwi/[A-Za-z0-9_-]+/stage/[0-9]+',
            r.text.replace("\\", "")
        )
        if match:
            stage_url = match.group(0)
            log(f"Disparando staging: {stage_url}")
            requests.post(stage_url, data={"share-token": ""}, timeout=30, verify=False)
            return True
    except Exception as e:
        log(f"Erro ao disparar staging: {e}")
    return False


def download(url: str, output_path: Path) -> bool:
    """Baixa o arquivo com suporte a staging."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(5):
        try:
            log(f"Tentativa {attempt + 1}: {url}")
            r = requests.get(url, stream=True, timeout=60, verify=False)

            if r.status_code == 409:
                log("Arquivo em tape. Disparando staging (aguarde 30-60s)...")
                if trigger_staging(url):
                    time.sleep(45)
                else:
                    log("Não foi possível obter URL de staging. Tente manualmente em:")
                    log(f"  {url}")
                    return False
                continue

            if r.status_code != 200:
                log(f"Erro HTTP {r.status_code}")
                return False

            total = int(r.headers.get("content-length", 0))
            log(f"Baixando ({total / (1024**3):.1f} GB) para {output_path}...")

            downloaded = 0
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = 100 * downloaded / total
                            mb = downloaded / (1024**2)
                            print(f"\r  {pct:.1f}% ({mb:.0f} MB)", end="", flush=True)

            print()
            log("Download concluído.")
            return True

        except requests.exceptions.RequestException as e:
            log(f"Erro de rede: {e}")
            if attempt < 4:
                time.sleep(10)
        except KeyboardInterrupt:
            log("Interrompido pelo usuário.")
            if output_path.exists():
                output_path.unlink()
            raise

    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Baixa dataset LDBC SNB do SURF")
    parser.add_argument(
        "--scale-factor", "-sf",
        choices=list(DATASETS.keys()),
        default="sf0.1",
        help="Scale factor (sf0.1 ~100MB, sf30 ~20GB). Padrão: sf0.1",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Pasta de destino",
    )
    args = parser.parse_args()

    url = DATASETS[args.scale_factor]
    filename = url.split("/")[-1]
    output_path = args.output / filename

    if output_path.exists():
        log(f"Arquivo já existe: {output_path}")
        log("Use --output para outro destino ou remova o arquivo.")
        return

    if not download(url, output_path):
        sys.exit(1)

    log(f"Próximo passo: python extracao_filtragem/pipeline.py")
    log(f"Saídas em: extracao_filtragem/output/")


if __name__ == "__main__":
    main()
