import hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE
BASE=HERE.parent
LOCALES={'en': 'en-US', 'de': 'de-DE', 'nl': 'nl-NL', 'sv': 'sv-SE', 'no': 'nb-NO', 'da': 'da-DK', 'es': 'es-ES', 'pt': 'pt-PT', 'it': 'it-IT', 'fr': 'fr-FR', 'ro': 'ro-RO', 'cs': 'cs-CZ', 'sk': 'sk-SK', 'pl': 'pl-PL', 'hr': 'hr-HR', 'sl': 'sl-SI', 'bg': 'bg-BG', 'lt': 'lt-LT', 'lv': 'lv-LV', 'fi': 'fi-FI', 'et': 'et-EE', 'el': 'el-GR', 'hu': 'hu-HU', 'tr': 'tr-TR', 'zh': 'zh-CN'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
