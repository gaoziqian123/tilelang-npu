#include "guards.h"
#include <assert.h>
#include <stdio.h>
int main(void){
 unsigned char initial[]={42,43,0,126,0,126,99};
 unsigned char mask[]={1,1,2,2,3,3,0},s[7];
 memcpy(s,initial,7);assert(!gdn_guard(s,initial,mask,7,1));
 s[2]=1;assert(gdn_guard(s,initial,mask,7,1));
 memcpy(s,initial,7);s[4]=s[5]=0;assert(!gdn_guard(s,initial,mask,7,0));
 s[0]^=1;assert(gdn_guard(s,initial,mask,7,0));s[0]^=1;
 s[1]^=1;assert(gdn_guard(s,initial,mask,7,0));s[1]^=1;
 s[5]=1;assert(gdn_guard(s,initial,mask,7,0));
 uint64_t l[34]={0};double sum;
 for(int p=0;p<15;p++){l[2+2*p]=10;l[3+2*p]=p==0?1:(p==1||p==14?32:512);}
 assert(!gdn_ledger(l,32,16,1,&sum));assert(fabs(sum-.15)<1e-12);
 l[3+2*13]--;assert(gdn_ledger(l,32,16,1,&sum));l[3+2*13]++;
 l[2+2*13]=UINT64_MAX;assert(gdn_ledger(l,32,16,1,&sum));l[2+2*13]=10;
 assert(gdn_ledger(l,32,16,.1,&sum));
 l[1]=13;assert(gdn_ledger(l,32,16,1,&sum));
 assert(gdn_hash((const unsigned char*)"hello",5)==UINT64_C(0xa430d84680aabd0b));
 puts("PASS immutable/input-padding/initial-poison/output-tail corruption; ledger count/overflow/gap/active negatives; FNV known vector");
}
