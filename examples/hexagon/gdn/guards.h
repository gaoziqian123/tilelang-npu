#ifndef GDN_GUARDS_H
#define GDN_GUARDS_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
/* Byte masks: 1=immutable input/alignment gap, 2=initial output NaN,
 * 3=output tail, which the current kernel explicitly writes as zero. */
static int gdn_guard(const unsigned char *s,const unsigned char *initial,
                     const unsigned char *mask,size_t n,int before){
 for(size_t i=0;i<n;i++){
  if(mask[i]==1 && s[i]!=initial[i])return 1;
  if(before && (mask[i]==2 || mask[i]==3) && s[i]!=initial[i])return 2;
  if(!before && mask[i]==3 && s[i]!=0)return 3;
 }
 return 0;
}
static uint64_t gdn_hash(const unsigned char*s,size_t n){
 uint64_t h=UINT64_C(14695981039346656037);
 for(size_t i=0;i<n;i++){h^=s[i];h*=UINT64_C(1099511628211);}return h;
}
static int gdn_ledger(const uint64_t*l,int hv,int nc,double rpc_ms,double *sum_ms){
 uint64_t total=0;
 if(l[1]!=0)return 1;
 for(int p=0;p<16;p++){
  uint64_t expected=p==0?1:(p==1||p==14?(uint64_t)hv:(p>=2&&p<=13?(uint64_t)hv*nc:0));
  if(l[3+2*p]!=expected || l[2+2*p]>UINT64_C(1000000000))return 2;
  total+=l[2+2*p];
 }
 *sum_ms=total*.001;
 return !isfinite(rpc_ms)||rpc_ms<0||*sum_ms>rpc_ms;
}
#endif
