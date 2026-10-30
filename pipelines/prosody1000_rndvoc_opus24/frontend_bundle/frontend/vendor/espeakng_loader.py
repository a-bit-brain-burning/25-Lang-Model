"""Bundled compressed frontend supplies its own pinned eSpeak implementation."""
from pathlib import Path
def get_library_path():return str(Path(__file__).resolve().parent/'frontend/libs/libespeak_phonemes.so')
def get_data_path():raise RuntimeError('Instantiate TextFrontend first; compressed eSpeak data have not been extracted')
