/* Experimental generic Kokoro-token adapter over the preserved compact lexicon.
 * It does not contain test sentences, waveform data or prediction answers.
 * The original M4 frontend source is included without modifying the old copy.
 */
#include "zh_frontend.c"
#include "kokoro_roots.h"

static int append(char *out, int capacity, int *used, const char *text) {
    int count = (int)strlen(text);
    if (count >= capacity - *used) return -20;
    memcpy(out + *used, text, (size_t)count);
    *used += count;
    out[*used] = 0;
    return 0;
}

static const char *punctuation(uint32_t point) {
    switch (point) {
    case 0x3001: case 0xff0c: case ',': return ", ";
    case 0x3002: case 0xff0e: case '.': return ". ";
    case 0xff01: case '!': return "! ";
    case 0xff1f: case '?': return "? ";
    case 0xff1b: case ';': return "; ";
    case 0xff1a: case ':': return ": ";
    case 0xff08: case '(': return " (";
    case 0xff09: case ')': return ") ";
    case 0x300a: case 0x300c: case 0x3010: case 0x201c: return "“";
    case 0x300b: case 0x300d: case 0x3011: case 0x201d: return "”";
    case 0x2026: return "…";
    case 0x2014: return "—";
    default: return "";
    }
}

int qb_zh_kokoro(const char *text, char *out, int capacity) {
    if (!text || !out || capacity < 1) return -20;
    ready = 0;
    memset(memo_index, 0, sizeof(memo_index));
    memo_used = 0;
    int result = parse_text(text);
    if (result) return result;
    if ((result = load_primary()) || (result = build_context()) || (result = resolve_readings()) ||
        (result = fuse_erhua()) || (result = build_syllables()) || (result = neutral_policy()) ||
        (result = citation_policy())) return result;
    surface_policy();
    char boundary[MAX_HAN + 1][64];
    memset(boundary, 0, sizeof(boundary));
    int han = 0;
    for (const char *p = text; *p;) {
        uint32_t point;
        if (native_utf8_read(&p, &point) < 1) return -4;
        if (is_han(point)) {
            ++han;
            if (*p == '[') {
                const char *end;
                Reading hint;
                if (decode_hint(p + 1, &end, &hint)) p = end;
            }
        } else {
            if (point == '#' && *p >= '1' && *p <= '4') { ++p; continue; }
            const char *mapped = punctuation(point);
            int used = (int)strlen(boundary[han]);
            if (append(boundary[han], sizeof(boundary[han]), &used, mapped)) return -20;
        }
    }
    int used = 0;
    out[0] = 0;
    if (append(out, capacity, &used, boundary[0])) return -20;
    for (int index = 0; index < state.ns; ++index) {
        const Syllable *syllable = &state.syllables[index];
        if (!valid(syllable->surface) || !kokoro_roots[syllable->surface.root][0]) return -21;
        if (index && !boundary[syllable->start][0] && syllable->group != state.syllables[index - 1].group)
            if (append(out, capacity, &used, "/")) return -20;
        if (append(out, capacity, &used, kokoro_roots[syllable->surface.root])) return -20;
        char tone[2] = {(char)('0' + syllable->surface.tone), 0};
        if (append(out, capacity, &used, tone) || append(out, capacity, &used, boundary[syllable->end])) return -20;
    }
    while (used && out[used - 1] == ' ') out[--used] = 0;
    return used;
}
