import re
LIMIT_PHONES=112
def protected_positions(text):
    patterns=[r'[A-Za-z0-9]+',r'[A-Za-z]{1,3}\s+\d[A-Za-z0-9]*',r'\d+(?::\d{2}){1,2}',r'\d[\d,]*(?:\.\d+)?(?:%?\s*[-~～]\s*\d[\d,]*(?:\.\d+)?%?)?']
    matches=[m for pattern in patterns for m in re.finditer(pattern,text)]
    return {i for m in matches for i in range(m.start()+1,m.end())}
