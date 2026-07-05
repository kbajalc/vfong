/* Standalone `amap` for the OSEA qrs_detect extension.
 *
 * amap() lives in osea/bxbep.c, but that file is a full WFDB-I/O beat
 * evaluation program (needs libwfdb). amap itself is a pure switch over MIT
 * annotation codes returning an AAMI test-label character, so we provide it
 * here (verbatim from bxbep.c) plus the `fflag` global it reads, avoiding the
 * libwfdb dependency. qrs_detect calls amap on OSEA beat types, which are only
 * NORMAL/PVC/UNKNOWN, so fflag's value never affects the result; 0 is fine.
 */
#include <wfdb/ecgcodes.h>

int fflag = 0;

int amap(int a)  /* map MIT annotation code into AAMI test label */
{
    switch (a) {
        case NORMAL:
        case LBBB:
        case RBBB:
        case BBB:    return ('N');
        case NPC:
        case APC:
        case SVPB:
        case ABERR:
        case NESC:
        case AESC:
        case SVESC:  return (fflag > 3 ? 'S' : 'N');
        case PVC:
        case RONT:
        case VESC:   return ('V');
        case FUSION: return ('F');
        case UNKNOWN: return ('Q');
        case PACE:
        case PFUS:   return ('Q');
        case LEARN:  return ('Q');
        default:     return ('O');
    }
}
