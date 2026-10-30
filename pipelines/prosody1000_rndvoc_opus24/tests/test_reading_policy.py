"""Ambiguity, false-positive, configuration and chunk-boundary regressions."""
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reading_policy import ReadingPolicy,spell_english

class PolicyTests(unittest.TestCase):
    def setUp(self):self.policy=ReadingPolicy()
    def readings(self,text,lang='en',policy=None,lexical=None):
        return [(e['source'],e['mode'],e['value'])for e in (policy or self.policy).scan(text,lang,lexical)[1]]
    def test_initialisms_and_word_acronyms(self):
        self.assertEqual(self.readings('NASA API cpu NATO SQL'),[('NASA','word','nasa'),('API','spell','API'),('cpu','spell','CPU'),('NATO','word','nato'),('SQL','spell','SQL')])
    def test_pronouns_and_articles(self):
        self.assertEqual(self.readings('I am in room 10 at 9:45. It helps us read a book.'),[])
        self.assertEqual(self.readings('US and IT'),[('US','spell','US'),('IT','spell','IT')])
        self.assertEqual(self.readings('Press A to continue.'),[('A','spell','A')])
    def test_punctuation_and_spaced_letters(self):
        self.assertEqual(self.readings('U.S.A. and U S B'),[('U.S.A.','spell','USA'),('U S B','spell','USB')])
        self.assertEqual(self.readings('USB-C GPT-4o H.264 HTTP/2'),[('USB-C','spell','USBC'),('GPT-4o','spell','GPT4o'),('H.264','spell','H264'),('HTTP/2','spell','HTTP2')])
        self.assertEqual(self.readings('read-only and mother-in-law'),[])
    def test_contractions_and_possessives_remain_whole(self):
        self.assertEqual(self.readings("I'm ready and NASA's CPU is working."),[("I'm",'word',"I'm"),("NASA's",'word',"NASA's"),('CPU','spell','CPU')])
        self.assertEqual(self.readings('I’ve checked it.'),[('I’ve','word',"I've")])
    def test_lexical_uppercase_words(self):
        lookup=lambda word:'word' if word=='HELLO' else None
        self.assertEqual(self.readings('HELLO XYZ',lexical=lookup),[('HELLO','word','hello'),('XYZ','spell','XYZ')])
    def test_word_abbreviations_and_case(self):
        self.assertEqual(self.readings('Dr.Smith and dr. Jones e.g. No.123'),[('Dr.','expand','doctor'),('dr.','expand','doctor'),('e.g.','expand','for example'),('No.','expand','number')])
    def test_units_and_product_versions(self):
        self.assertEqual(self.readings('1 GB 16GB 1,024 MB 2 Mb 24 kHz'),[('1 GB','expand','1 gigabyte'),('16GB','expand','16 gigabytes'),('1,024 MB','expand','1,024 megabytes'),('2 Mb','expand','2 megabits'),('24 kHz','expand','24 kilohertz')])
        self.assertEqual(self.readings('WiFi6 COVID-19'),[('WiFi6','expand','why fie 6'),('COVID-19','expand','covid 19')])
    def test_language_specific_custom_readings(self):
        policy=ReadingPolicy({'overrides':{'en':{'SQL':{'mode':'word','spoken':'sequel'}},'zh':{'AI':{'mode':'expand','spoken':'人工智能'}}}})
        self.assertEqual(self.readings('SQL and API',policy=policy),[('SQL','word','sequel'),('API','spell','API')])
        self.assertEqual(self.readings('请用AI和CPU','zh',policy),[('AI','expand','人工智能'),('CPU','spell','CPU')])
    def test_modes_do_not_rewrite_ordinary_words_or_identifiers(self):
        self.assertEqual(self.readings('NASA uses CPU',policy=ReadingPolicy(mode='spell')),[('NASA','spell','NASA'),('CPU','spell','CPU')])
        self.assertEqual(self.readings('NASA uses CPU CZ2671',policy=ReadingPolicy(mode='word')),[('NASA','word','nasa'),('CPU','word','cpu'),('CZ2671','spell','CZ2671')])
        self.assertEqual(spell_english('CPU'),'see pee you')
    def test_semantic_atoms_not_cut_by_long_text_planner(self):
        for text,needle in [('hello U.S.B. device here','U.S.B. device'),('hello Dr. Smith here','Dr. Smith'),('hello No.123 here','No.123'),('型号ＣＰＵ后面','ＣＰＵ')]:
            at=text.index(needle);self.assertIn((at,at+len(needle)),self.policy.atomic_spans(text,'en'))
    def test_invalid_configuration_is_rejected(self):
        for config in [{'version':2},{'unknown_key':{}},{'initialisms':'CPU'},{'overrides':{'en':{'SQL':{'mode':'phonemes'}}}},{'overrides':{'zh':{'AI':{'mode':'expand'}}}},{'overrides':{'zh':{'AI':{'mode':'word','spoken':'人工智能'}}}}]:
            with self.assertRaises(ValueError):ReadingPolicy(config)

if __name__=='__main__':unittest.main()
