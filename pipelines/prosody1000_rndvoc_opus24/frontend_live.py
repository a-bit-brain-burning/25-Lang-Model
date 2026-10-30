"""Keep the existing standalone/leading-zero identifier behavior at the CLI."""
import re
import frontend_repair as base
previous=base.normalize
def normalize(text,language):
    if language!='zh':return previous(text,language)
    saved=[]
    def protect(match):
        marker=chr(0xf000+len(saved));saved.append((marker,match[0]));return marker
    marked=re.sub(r'(?<![A-Za-z0-9])0\d+(?![A-Za-z0-9]|[ -]\d)',protect,text)
    if re.fullmatch(r'[0-9\s，,、]+',marked):
        marked=re.sub(r'\d+',protect,marked)
    normalized,items=previous(marked,language)
    for marker,value in saved:
        replacement=chr(0xe000+len(items));items.append(dict(marker=replacement,kind='digits',value=value));normalized=normalized.replace(marker,replacement)
    return normalized,items
base.normalize=normalize
build_zh=base.build_zh
