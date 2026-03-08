"""
src/data/loader.py

Tanggung jawab:
- Load dataset AIGCodeSet dan CoDET-M4 dari HuggingFace
- Cache hasil ke data/raw/ sebagai Parquet supaya tidak re-download/re-convert
- Normalize ke unified schema
- Merge dan return satu DataFrame

Unified schema:
    code            (str)  : source code mentah
    label           (int)  : 0 = human, 1 = AI-generated
    language        (str)  : bahasa pemrograman (unknown jika tidak tersedia)
    generator       (str)  : model yang generate (human jika buatan manusia)
    source_dataset  (str)  : origin dataset (aigcodeset / codet_m4)

Cache behavior:
    Pertama kali  -> download dari HuggingFace, simpan ke data/raw/*.parquet
    Berikutnya    -> baca langsung dari data/raw/*.parquet, skip download

Tidak ada filtering, tidak ada preprocessing konten kode.
Semua itu tanggung jawab modul lain.
"""

import logging
from pathlib import Path
from typing import Literal

import pandas as pd
from datasets import load_dataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIGCODESET_HF_ID = "basakdemirok/AIGCodeSet"
CODET_M4_HF_ID   = "DaniilOr/CoDET-M4"

# __file__ = .../AICodeTrace/src/data/loader.py
# .parent.parent.parent = root project AICodeTrace/
RAW_DIR = Path(__file__).parent.parent.parent / "data" / "raw"

AIGCODESET_CACHE = RAW_DIR / "aigcodeset.parquet"
CODET_M4_CACHE   = RAW_DIR / "codet_m4.parquet"

# Kolom yang kita ambil dari masing-masing dataset sebelum normalize
AIGCODESET_KEEP_COLS = ["code", "label", "LLM"]
CODET_M4_KEEP_COLS   = ["code", "target", "language", "model"]

# Label mapping untuk CoDET-M4
CODET_M4_LABEL_MAP = {"human": 0, "machine": 1}


# ---------------------------------------------------------------------------
# Cache utilities
# ---------------------------------------------------------------------------

def _ensure_raw_dir() -> None:
    """Buat folder data/raw/ kalau belum ada."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def _save_cache(df: pd.DataFrame, path: Path) -> None:
    """Simpan DataFrame ke Parquet."""
    _ensure_raw_dir()
    df.to_parquet(path, index=False)
    logger.info("Cache disimpan ke: %s", path)


def _load_cache(path: Path) -> pd.DataFrame:
    """Baca DataFrame dari Parquet cache."""
    logger.info("Membaca dari cache lokal: %s", path)
    return pd.read_parquet(path)


# ---------------------------------------------------------------------------
# Private loaders
# ---------------------------------------------------------------------------

def _load_aigcodeset(force_reload: bool = False) -> pd.DataFrame:
    """
    Load AIGCodeSet. Gunakan cache Parquet kalau tersedia.

    Alur:
        1. Cek apakah data/raw/aigcodeset.parquet ada
        2. Kalau ada dan force_reload=False -> baca dari cache
        3. Kalau tidak ada atau force_reload=True -> download dari HF,
           normalize, simpan ke cache

    Catatan dataset:
        - label sudah int (0/1)
        - LLM kolom berisi nama model untuk AI, kosong/null untuk human
        - Tidak ada kolom language -> diisi 'unknown'
        - problem_id berformat AtCoder -> dominan CP code
    """
    # Cek cache dulu
    if AIGCODESET_CACHE.exists() and not force_reload:
        df = _load_cache(AIGCODESET_CACHE)
        logger.info("AIGCodeSet dari cache, shape: %s", df.shape)
        return df

    # Cache tidak ada atau force_reload -> download dari HuggingFace
    logger.info("Downloading AIGCodeSet dari HuggingFace...")
    raw = load_dataset(AIGCODESET_HF_ID)

    # Gabung semua split (train + test) jadi satu DataFrame
    # Splitting ulang dilakukan di preprocessing.py secara konsisten
    frames = []
    for split_name, split_data in raw.items():
        df_split = split_data.to_pandas()
        frames.append(df_split)
    df = pd.concat(frames, ignore_index=True)

    logger.info("AIGCodeSet raw shape: %s", df.shape)

    # Ambil kolom relevan saja
    df = df[AIGCODESET_KEEP_COLS].copy()

    # Normalize schema
    df.rename(columns={"LLM": "generator"}, inplace=True)

    # generator: null/kosong -> "human"
    df["generator"] = df["generator"].fillna("human")
    df["generator"] = df["generator"].replace("", "human")

    # Tidak ada kolom language di AIGCodeSet
    df["language"] = "unknown"

    # Origin marker
    df["source_dataset"] = "aigcodeset"

    # Pastikan label int
    df["label"] = df["label"].astype(int)

    # Reorder ke unified schema
    df = df[["code", "label", "language", "generator", "source_dataset"]]

    logger.info("AIGCodeSet normalized shape: %s", df.shape)
    logger.info("AIGCodeSet label distribution:\n%s", df["label"].value_counts())

    # Simpan ke cache
    _save_cache(df, AIGCODESET_CACHE)

    return df


def _load_codet_m4(force_reload: bool = False) -> pd.DataFrame:
    """
    Load CoDET-M4. Gunakan cache Parquet kalau tersedia.

    Alur:
        1. Cek apakah data/raw/codet_m4.parquet ada
        2. Kalau ada dan force_reload=False -> baca dari cache
        3. Kalau tidak ada atau force_reload=True -> download dari HF,
           normalize, simpan ke cache

    Catatan dataset:
        - target adalah string: 'human' atau 'machine' -> encode ke 0/1
        - language: python, java, cpp
        - model: 6 generator values
        - split kolom ada (train/val/test) tapi kita gabung semua ->
          splitting ulang dilakukan di preprocessing.py secara konsisten
        - features kolom (dict pre-computed stylistic) di-drop ->
          src/features/stylistic.py yang handle secara konsisten
    """
    # Cek cache dulu
    if CODET_M4_CACHE.exists() and not force_reload:
        df = _load_cache(CODET_M4_CACHE)
        logger.info("CoDET-M4 dari cache, shape: %s", df.shape)
        return df

    # Cache tidak ada atau force_reload -> download dari HuggingFace
    logger.info("Downloading CoDET-M4 dari HuggingFace... (dataset besar ~500k rows, sabar)")
    raw = load_dataset(CODET_M4_HF_ID)

    frames = []
    for split_name, split_data in raw.items():
        df_split = split_data.to_pandas()
        frames.append(df_split)
    df = pd.concat(frames, ignore_index=True)

    logger.info("CoDET-M4 raw shape: %s", df.shape)

    # Ambil kolom relevan saja
    df = df[CODET_M4_KEEP_COLS].copy()

    # Normalize schema
    df.rename(columns={"target": "label", "model": "generator"}, inplace=True)

    # Encode label string -> int
    df["label"] = df["label"].map(CODET_M4_LABEL_MAP)

    # Sanity check: ada label yang tidak ter-map?
    unmapped = df["label"].isna().sum()
    if unmapped > 0:
        logger.warning(
            "CoDET-M4: %d baris dengan label tidak dikenali, akan di-drop.", unmapped
        )
        df = df.dropna(subset=["label"])

    df["label"] = df["label"].astype(int)

    # generator: pastikan tidak ada null
    df["generator"] = df["generator"].fillna("unknown")

    # Origin marker
    df["source_dataset"] = "codet_m4"

    # Reorder ke unified schema
    df = df[["code", "label", "language", "generator", "source_dataset"]]

    logger.info("CoDET-M4 normalized shape: %s", df.shape)
    logger.info("CoDET-M4 label distribution:\n%s", df["label"].value_counts())
    logger.info("CoDET-M4 language distribution:\n%s", df["language"].value_counts())

    # Simpan ke cache
    _save_cache(df, CODET_M4_CACHE)

    return df


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_unified(
    source: Literal["both", "aigcodeset", "codet_m4"] = "both",
    force_reload: bool = False,
) -> pd.DataFrame:
    """
    Load dan merge dataset ke unified DataFrame.

    Pertama kali dipanggil akan download dari HuggingFace dan
    menyimpan cache ke data/raw/. Panggilan berikutnya baca dari
    cache lokal — jauh lebih cepat.

    Parameters
    ----------
    source : {"both", "aigcodeset", "codet_m4"}
        Dataset mana yang di-load. Default "both".
    force_reload : bool
        Kalau True, ignore cache dan re-download dari HuggingFace.
        Gunakan ini kalau cache corrupt atau dataset di HF diupdate.
        Default False.

    Returns
    -------
    pd.DataFrame
        Unified DataFrame dengan kolom:
        code, label, language, generator, source_dataset.

    Examples
    --------
    >>> from src.data.loader import load_unified

    >>> # Load semua dataset (dengan cache)
    >>> df = load_unified()

    >>> # Load CoDET-M4 saja
    >>> df = load_unified(source="codet_m4")

    >>> # Force re-download (ignore cache)
    >>> df = load_unified(force_reload=True)
    """
    frames = []

    if source in ("both", "aigcodeset"):
        frames.append(_load_aigcodeset(force_reload=force_reload))

    if source in ("both", "codet_m4"):
        frames.append(_load_codet_m4(force_reload=force_reload))

    if not frames:
        raise ValueError(f"source tidak valid: {source!r}")

    df = pd.concat(frames, ignore_index=True)

    # Drop baris dengan code null atau kosong
    before = len(df)
    df = df[df["code"].notna() & (df["code"].str.strip() != "")]
    after = len(df)
    if before != after:
        logger.warning(
            "Dropped %d baris dengan code null/kosong.", before - after
        )

    logger.info("Unified DataFrame final shape: %s", df.shape)
    logger.info("Label distribution (unified):\n%s", df["label"].value_counts())
    logger.info(
        "Source dataset distribution:\n%s", df["source_dataset"].value_counts()
    )

    return df


def clear_cache() -> None:
    """
    Hapus semua cache lokal di data/raw/.

    Gunakan ini kalau mau force fresh download pada pemanggilan
    load_unified() berikutnya tanpa perlu set force_reload=True,
    atau kalau cache corrupt.
    """
    for cache_path in [AIGCODESET_CACHE, CODET_M4_CACHE]:
        if cache_path.exists():
            cache_path.unlink()
            logger.info("Cache dihapus: %s", cache_path)
        else:
            logger.info("Cache tidak ditemukan (skip): %s", cache_path)


# ---------------------------------------------------------------------------
# CodeNet stub (future extension)
# ---------------------------------------------------------------------------

def _load_codenet() -> pd.DataFrame:
    """
    Stub untuk IBM Project CodeNet.

    CodeNet adalah human-only dataset. Untuk dipakai di AICodeTrace,
    butuh augmentasi: generate pasangan AI untuk setiap human sample.
    Ini non-trivial dan di-defer sampai dua dataset utama sudah solid.

    Raise NotImplementedError sampai diimplementasikan.
    """
    raise NotImplementedError(
        "CodeNet loader belum diimplementasikan. "
        "Butuh augmentasi pipeline (human -> AI pair generation) sebelum bisa dipakai."
    )