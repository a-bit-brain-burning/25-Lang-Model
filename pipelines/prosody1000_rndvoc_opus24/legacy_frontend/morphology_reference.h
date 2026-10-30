/* Generic fallback for dictionary-unknown English inflections/compounds.
 * Contains only productive affix rules; no training/evaluation word list.
 */
static int qb_known(const char *text, Word *out) {
    memset(out, 0, sizeof(*out));
    if (strlen(text) >= sizeof(out->word)) return 0;
    strcpy(out->word, text);
    return !english_lookup(text, out->phone, out->stress, &out->length, &out->known) && out->known;
}
static int qb_join(Word *out, const Word *left, const Word *right) {
    if (left->length + right->length > ENGLISH_MAX_WORD_PHONES) return 0;
    *out = *left;
    memcpy(out->phone + left->length, right->phone, right->length);
    memcpy(out->stress + left->length, right->stress, right->length);
    out->length += right->length;
    out->known = 1; out->approximate = 1;
    return 1;
}
static int qb_voiceless(int phone) {
    return phone==26 || phone==19 || phone==30 || phone==28 || phone==13 || phone==29 || phone==7 || phone==31;
}
static int qb_morph(const char *text, Word *out, unsigned depth) {
    if (qb_known(text, out)) return 1;
    if (depth >= 3) return 0;
    size_t n = strlen(text);
    char stem[ENGLISH_MAX_WORD_BYTES];
    Word root, suffix, left, right;
    const char *endings[] = {"ed", "ing", "es", "s"};
    for (unsigned which=0; which<4; ++which) {
        size_t length = strlen(endings[which]);
        if (n <= length+2 || strcmp(text+n-length, endings[which])) continue;
        size_t base = n-length;
        memcpy(stem, text, base); stem[base]=0;
        int found = qb_morph(stem, &root, depth+1);
        if (!found && which < 2 && base+1 < sizeof(stem)) {
            stem[base]='e'; stem[base+1]=0;
            found = qb_morph(stem, &root, depth+1);
            stem[base]=0;
        }
        if (!found && which < 2 && base>2 && stem[base-1]==stem[base-2] && !strchr("aeiou",stem[base-1])) {
            stem[base-1]=0; found=qb_morph(stem,&root,depth+1);
        }
        if (!found) continue;
        memset(&suffix,0,sizeof(suffix));
        memset(suffix.stress,-1,sizeof(suffix.stress));
        int last=root.phone[root.length-1];
        if (which==0) {
            if (last==30 || last==8) {suffix.phone[0]=16;suffix.stress[0]=0;suffix.phone[1]=8;suffix.length=2;}
            else {suffix.phone[0]=qb_voiceless(last)?30:8;suffix.length=1;}
        } else if (which==1) {
            suffix.phone[0]=16;suffix.stress[0]=0;suffix.phone[1]=23;suffix.length=2;
        } else if (last==28 || last==37 || last==29 || last==38 || last==7 || last==18) {
            suffix.phone[0]=16;suffix.stress[0]=0;suffix.phone[1]=37;suffix.length=2;
        } else {suffix.phone[0]=qb_voiceless(last)?28:37;suffix.length=1;}
        if (qb_join(out,&root,&suffix)) return 1;
    }
    /* Unknown compounds use dictionary roots. Both sides must be known; no
     * guessed spelling pronunciation is promoted to a trusted dictionary hit. */
    for (size_t split=n>3?n-3:0; split>=3; --split) {
        memcpy(stem,text,split);stem[split]=0;
        if (qb_known(stem,&left) && qb_known(text+split,&right)) {
            for (int j=0;j<right.length;j++) if(right.stress[j]==1) right.stress[j]=2;
            if(qb_join(out,&left,&right)) return 1;
        }
    }
    return 0;
}
static void qb_repair_unknown_words(void) {
    for (int i=0;i<qb_front.words;i++) {
        if (words[i].known) continue;
        Word fixed;
        if (qb_morph(words[i].word,&fixed,0)) {
            memcpy(words[i].phone,fixed.phone,sizeof(fixed.phone));
            memcpy(words[i].stress,fixed.stress,sizeof(fixed.stress));
            words[i].length=fixed.length;words[i].known=1;words[i].approximate=1;
        }
    }
}
