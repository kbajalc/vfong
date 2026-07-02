/* Minimal stub of WFDB <wfdb/ecgcodes.h> for building the OSEA qrs_detect
 * extension without installing libwfdb. These are the canonical MIT-BIH/WFDB
 * annotation-code integer values (public, stable). OSEA's classifier was written
 * against these exact values, so they must match the real header. Only the codes
 * referenced by the compiled OSEA sources + amap are defined.
 */
#ifndef WFDB_ECGCODES_H_STUB
#define WFDB_ECGCODES_H_STUB

#define NOTQRS   0   /* not-QRS (not a getann/putann code) */
#define NORMAL   1   /* normal beat */
#define LBBB     2   /* left bundle branch block beat */
#define RBBB     3   /* right bundle branch block beat */
#define ABERR    4   /* aberrated atrial premature beat */
#define PVC      5   /* premature ventricular contraction */
#define FUSION   6   /* fusion of ventricular and normal beat */
#define NPC      7   /* nodal (junctional) premature beat */
#define APC      8   /* atrial premature contraction */
#define SVPB     9   /* premature or ectopic supraventricular beat */
#define VESC    10   /* ventricular escape beat */
#define NESC    11   /* nodal (junctional) escape beat */
#define PACE    12   /* paced beat */
#define UNKNOWN 13   /* unclassifiable beat */
#define NOISE   14   /* signal quality change */
#define BBB     25   /* left or right bundle branch block */
#define LEARN   30   /* learning */
#define FLWAV   31   /* ventricular flutter wave */
#define VFON    32   /* start of ventricular flutter/fibrillation */
#define VFOFF   33   /* end of ventricular flutter/fibrillation */
#define AESC    34   /* atrial escape beat */
#define SVESC   35   /* supraventricular escape beat */
#define PFUS    38   /* fusion of paced and normal beat */
#define RONT    41   /* R-on-T premature ventricular contraction */

#endif /* WFDB_ECGCODES_H_STUB */
