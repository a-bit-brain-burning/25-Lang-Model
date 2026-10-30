"""Semantic regressions for contextual numbers and conservative lexical fallback."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from frontend_live import details
from reading_policy import ReadingPolicy
from english_lexicon import LexiconPolicy
class Numbers(unittest.TestCase):
 def test_function_words_are_not_identifier_prefixes(self):
  for text in ['My number is 831 920 9671.','The price is 64 pounds.','It was 25 degrees today.','There are 12 items.']:
   events=ReadingPolicy().scan(text,'en')[1];self.assertFalse(any(e['rule']=='alphanumeric_identifier' for e in events),text)
 def test_phone_and_money(self):
  self.assertEqual(details('My number is 831 920 9671.','en')[0],'My number is eight three one nine two zero nine six seven one.')
  self.assertEqual(details('I spent £20,000.00 on the phone.','en')[0],'I spent twenty thousand pounds on the phone.')
 def test_temperatures_survive_preceding_words(self):
  spoken=details('a high of 25°C and a low of 15°C.','en')[0]
  self.assertEqual(spoken.replace(' .','.'),'a high of twenty-five degrees Celsius and a low of fifteen degrees Celsius.')
 def test_date_range_and_percentage(self):
  self.assertEqual(details('The exam is held on 7 - 8 June.','en')[0],'The exam is held on seventh to eighth June.')
  self.assertEqual(details('The concentration is 10-15%.','en')[0],'The concentration is ten to fifteen percent.')
class Dictionary:
 def __init__(self):
  self.entries={'walk':'W AO K','search':'S ER CH','white':'W AY T','board':'B AO R D'}
  self.stress_entries={'walk':'W AO1 K','search':'S ER1 CH','white':'W AY1 T','board':'B AO1 R D'}
 def lookup_word(self,w):return self.entries.get(w)
 def stress_for(self,w,phones):return [-1]*len(phones),'original'
class Lexicon(unittest.TestCase):
 def test_productive_fallback_uses_actual_roots(self):
  d=Dictionary();before=dict(d.entries);p=LexiconPolicy(d)
  self.assertEqual(d.lookup_word('walked'),'W AO1 K T')
  self.assertEqual(d.lookup_word('searchable'),'S ER1 CH AH0 B AH0 L')
  self.assertEqual(d.lookup_word('whiteboard'),'W AY1 T B AO2 R D')
  self.assertIsNone(d.lookup_word('zzqplxx'))
  self.assertEqual(d.entries,before)
if __name__=='__main__':unittest.main()
