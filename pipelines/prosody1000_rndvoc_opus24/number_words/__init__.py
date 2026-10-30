"""English-only entry for the unchanged vendored num2words 0.5.14 converter.
Original module sources and LGPL license are included; other language modules
and the upstream CLI dispatcher are omitted. No external dependency is added.
"""
from .lang_EN import Num2Word_EN
_converter=Num2Word_EN()
def num2words(number,ordinal=False,to='cardinal'):
 if isinstance(number,str):number=_converter.str_to_number(number)
 if ordinal:to='ordinal'
 if to not in ('cardinal','ordinal','year'):raise ValueError('Unsupported number conversion')
 return getattr(_converter,'to_'+to)(number)
