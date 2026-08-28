#!/usr/bin/env python3
from __future__ import annotations

import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path


MODEL_NAME = "vosk-model-small-ja-0.22"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DESTINATION = MODELS_DIR / MODEL_NAME


def _safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise RuntimeError("モデルZIPに不正なパスが含まれています")
    archive.extractall(destination)


def main() -> int:
    if DESTINATION.is_dir():
        print(f"Voskモデルは既にあります: {DESTINATION}")
        return 0

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="discord-bot-vosk-") as temp_dir:
        archive_path = Path(temp_dir) / f"{MODEL_NAME}.zip"
        print("公式配布元から日本語Voskモデルをダウンロードします（約48MB）")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as response:
            with archive_path.open("wb") as output:
                shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive_path) as archive:
            _safe_extract(archive, MODELS_DIR)

    if not DESTINATION.is_dir():
        raise RuntimeError("モデルを展開できませんでした")
    print(f"完了: {DESTINATION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
