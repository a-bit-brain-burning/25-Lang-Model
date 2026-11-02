int n25_dictionary_warm_es_probe(void);
/* Native text requests. No phone/features/audio fixtures are read by the device. */
__attribute__((noinline)) void tts_begin_language_text_request(unsigned request_id,unsigned language_id,const char*text){
 if(language_id>=25)fail("BAD_LANGUAGE_ID");
 n25_current_norm=n25_language_norms+language_id;
 adapter_input_op=adapter_input_ops[ACOUSTIC_DIM==39];acoustic_head_op=acoustic_head_ops[ACOUSTIC_DIM==39];
 tts_begin_text_request(request_id,text);
}
int main(void){
 tts_boot_start=bench_now();tts_runtime_init();n25_frontend_runtime_init();
 if(n25_dictionary_warm_es_probe())fail("RESIDENT_SPANISH_WARM_QUERY_PROBE");
 static const struct {unsigned language;const char *text;} cases[]={
{0,"Здравей, това е тест на гласа."},
{1,"Ahoj, toto je hlasový test."},
{2,"Dette er en test af stemmen."},
{3,"Hallo, dies ist ein Test der Stimme."},
{4,"Γεια σας, αυτή είναι μια δοκιμή φωνής."},
{5,"Hello, this is a test of the new voice model."},
{6,"Hola, esta es una prueba de voz."},
{7,"Tere, see on hääle test."},
{8,"Hei, tämä on puheen testi."},
{9,"Bonjour, ceci est un test de la voix."},
{10,"Pozdrav, ovo je test glasa."},
{11,"Szia, ez egy hangteszt."},
{12,"Ciao, questa è una prova della voce."},
{13,"Labas, tai yra balso testas."},
{14,"Sveiki, šis ir balss tests."},
{15,"Hallo, dit is een test van de stem."},
{16,"Hei, dette er en test av stemmen."},
{17,"Dzień dobry, to jest test głosu."},
{18,"Olá, este é um teste de voz."},
{19,"Salut, acesta este un test de voce."},
{20,"Ahoj, toto je skúška hlasu."},
{21,"Pozdravljeni, to je preizkus glasu."},
{22,"Hej, det här är ett test av rösten."},
{23,"Merhaba, bu bir ses testidir."},
{24,"你好，欢迎测试新的语音模型。"},
 };
 for(unsigned i=0;i<sizeof(cases)/sizeof(cases[0]);i++){
  unsigned language=cases[i].language;
  tts_begin_language_text_request(language,language,cases[i].text);
  int16_t packet[480];unsigned index=0,count;
  while((count=tts_pull20ms(packet))){
   char name[100];namefmt(name,sizeof(name),"request_%u_packet_%u.pcm",language,index++);
   if(bench_write_file(name,packet,count*2))fail("NATIVE_MULTILANG_PACKET_CAPTURE");
  }
  tts_verify_complete();tts_reset_request();
 }
 bench_metric("native_text_languages_completed",sizeof(cases)/sizeof(cases[0]));
 bench_memory_report();bench_puts("N25_NATIVE_MULTILANG_SUBSET_PASS_NOT_FULL25\n");return 0;
}
