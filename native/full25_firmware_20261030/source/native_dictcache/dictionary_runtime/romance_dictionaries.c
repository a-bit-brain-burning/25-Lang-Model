#include "romance_record.h"
static const char*const symbols_1[]={"a","b","c","d̪","e","f","i","j","k","l","m","n","o","p","r","s","tʃ","t̪","u","w","x","ç","ð","ŋ","ɟ","ɟʝ","ɡ","ɣ","ɲ","ɾ","ʃ","ʎ","ʝ","β","θ"};
static const uint32_t alphabet_1[]={97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,225,233,237,241,243,250,252};
static const unsigned char default_1[]={0,0,5,3,3,0,0,3,0};
static const char*const symbols_3[]={"a","aː","b","bː","d","dː","d͡ʒ","d͡ʒː","e","eː","f","fː","g","hː","i","j","k","kː","l","lː","m","mː","n","nː","o","oː","p","pː","r","rː","s","sː","t","tː","t͡s","t͡ʃ","t͡ʃː","u","v","vː","w","x","y","z","ã","æ","ð","ø","ħ","œ","ɔ","ə","ɛ","ɡ","ɡː","ɲ","ʃ","ʎ","ẽ"};
static const uint32_t alphabet_3[]={97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,224,227,228,229,230,232,233,234,236,237,238,240,242,243,244,248,249,250,251,257,269,295,333,339,353,382,601,720,7717,7877};
static const unsigned char default_3[]={0,0,5,3,3,0,0,3,0};
static const char*const symbols_5[]={"a","b","d","e","f","i","j","j̃","k","kʷ","l","m","n","o","p","s","t","u","v","w","w̃","z","õ","ĩ","ũ","ɐ","ɐ̃","ɔ","ɛ","ɛ̃","ɡ","ɡʷ","ɡ̃","ɾ","ʁ","ʃ","ʒ","ẽ"};
static const uint32_t alphabet_5[]={97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,224,225,226,227,231,232,233,234,237,243,244,245,250,252};
static const unsigned char default_5[]={0,0,5,3,3,0,0,3,0};
static const char*const symbols_2[]={"a","b","c","d","dʒ","e","f","i","j","k","l","m","mʲ","n","o","p","s","spn","t","ts","tʃ","u","v","w","y","z","ø","ŋ","œ","ɑ","ɑ̃","ɔ","ɔ̃","ə","ɛ","ɛ̃","ɟ","ɡ","ɥ","ɲ","ʁ","ʃ","ʎ","ʒ"};
static const uint32_t alphabet_2[]={97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,224,225,226,231,232,233,234,235,237,238,239,241,243,244,246,249,250,251,252,339};
static const unsigned char default_2[]={0,0,0};
static const char*const symbols_6[]={"a","b","d","dʒ","e","e̯","f","h","i","i̯","j","k","l","m","n","o","o̯","p","r","s","t","ts","tʃ","u","v","w","z","ə","ɡ","ɨ","ʃ","ʒ"};
static const uint32_t alphabet_6[]={};
static const unsigned char default_6[]={};
const RomanceDictionary romance_dictionaries[]={
{"es",1,6087,0,473591,35,symbols_1,alphabet_1,33,default_1,9},
{"it",3,6191,0,458603,59,symbols_3,alphabet_3,56,default_3,9},
{"pt",5,6216,0,359177,38,symbols_5,alphabet_5,40,default_5,9},
{"fr",2,47465,35,664098,44,symbols_2,alphabet_2,46,default_2,3},
{"ro",6,6419,0,581890,32,symbols_6,alphabet_6,0,default_6,0},
};
const unsigned romance_dictionary_count=sizeof(romance_dictionaries)/sizeof(romance_dictionaries[0]);
