#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include "remote.h"
#include "rpcmem.h"
#include "attnops.h"
#include "guards.h"
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static int check(unsigned char*s,const double*ref,size_t off,int hv,int nt,int tp,int state){
 double xy=0,xx=0,yy=0;size_t ix=0;
 for(int h=0;h<hv;h++)for(int t=0;t<(state?128:nt);t++)for(int d=0;d<128;d++){
  size_t at=((size_t)h*(state?128:tp)+t)*128+d;
  double x=state?((float*)(s+off))[at]:((__fp16*)(s+off))[at],y=ref[ix++];
  if(!isfinite(x)||!isfinite(y)){printf("CHECK nonfinite state=%d at=%zu\n",state,at);return 1;}
  xx+=x*x;xy+=x*y;yy+=y*y;
 }
 double cos=xy/sqrt(xx*yy);printf("CHECK %s full=%zu finite=1 cos=%.12f\n",state?"S1":"O",ix,cos);return !isfinite(cos)||cos<.999;
}
int main(int argc,char**argv){
  if(argc!=10)return 2;
 int bytes=atoi(argv[1]),iters=atoi(argv[2]);if(bytes<=0||iters<=0)return 2;
  int hv=atoi(argv[3]),nt=atoi(argv[4]),tpn=atoi(argv[5]);size_t oo=strtoull(argv[6],0,10),so=strtoull(argv[7],0,10);int profile=atoi(argv[8]);
  size_t logoff=strtoull(argv[9],0,10);
  if(hv<=0||nt<=0||tpn<nt||oo+(size_t)hv*tpn*256>(size_t)bytes||so+(size_t)hv*128*512>(size_t)bytes||logoff+272>(size_t)bytes)return 2;
  unsigned char *initial=malloc(bytes),*mask=malloc(bytes);
  if(!initial||!mask)return 4;
  const char*gf[]={"input.bin","guard.bin"};unsigned char*gb[]={initial,mask};
  for(int i=0;i<2;i++){FILE*f=fopen(gf[i],"rb");if(!f||fread(gb[i],1,bytes,f)!=(size_t)bytes||fgetc(f)!=EOF)return 6;fclose(f);}
  for(size_t i=0;i<(size_t)hv*tpn*128;i++)if(!isnan((float)((__fp16*)(initial+oo))[i]))return 12;
  for(size_t i=0;i<(size_t)hv*128*128;i++)if(!isnan(((float*)(initial+so))[i]))return 12;
  printf("INITIAL output_nan=1 input_fnv1a64=%016llx guard_fnv1a64=%016llx\n",(unsigned long long)gdn_hash(initial,bytes),(unsigned long long)gdn_hash(mask,bytes));
 double*gold[2];const char*files[]={"gold_o.bin","gold_s.bin"};
 for(int i=0;i<2;i++){size_t n=(size_t)hv*(i?128:nt)*128;gold[i]=malloc(n*8);FILE*f=fopen(files[i],"rb");if(!f||fread(gold[i],8,n,f)!=n)return 10;fclose(f);}
 setbuf(stdout,NULL);
 struct remote_rpc_control_unsigned_module u={.domain=CDSP_DOMAIN_ID,.enable=1};
 struct remote_rpc_thread_params tp={CDSP_DOMAIN_ID,-1,262144};
 int e=remote_session_control(FASTRPC_THREAD_PARAMS,&tp,sizeof(tp));printf("THREAD_PARAMS rc=%d\n",e);if(e)return 3;
 e=remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,&u,sizeof(u));printf("UNSIGNED rc=%d\n",e);if(e)return 3;
 unsigned char*s=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,bytes);if(!s)return 4;
 remote_register_buf_attr2(s,bytes,rpcmem_to_fd(s),FASTRPC_ATTR_COHERENT|FASTRPC_ATTR_KEEP_MAP|FASTRPC_ATTR_TRY_MAP_STATIC);
 remote_handle64 h;char uri[256];snprintf(uri,sizeof(uri),"%s&_dom=cdsp",attnops_URI);
 e=attnops_open(uri,&h);printf("OPEN rc=%d\n",e);if(e)return 5;
 for(int it=0;it<iters;it++){
   memcpy(s,initial,bytes);
   if(gdn_guard(s,initial,mask,bytes,1))return 12;
  double start=now(),ms[3];
  for(int k=0;k<3;k++){double t=now();e=attnops_tl_invoke(h,s,bytes,101+k,k==1?6:1,1);ms[k]=now()-t;if(e){printf("FAIL stage=%d rc=%d\n",k+1,e);return 7;}}
  printf("ITER %d host_ms=%.6f rpc1=%.6f rpc2=%.6f rpc3=%.6f\n",it,now()-start,ms[0],ms[1],ms[2]);
  if(check(s,gold[0],oo,hv,nt,tpn,0)||check(s,gold[1],so,hv,nt,tpn,1))return 11;
   int guard=gdn_guard(s,initial,mask,bytes,0);
   printf("GUARD iter=%d immutable_padding_rc=%d initial_nan=1 slab_fnv1a64=%016llx\n",it,guard,(unsigned long long)gdn_hash(s,bytes));if(guard)return 12;
   if(profile){
    uint64_t*l=(uint64_t*)(s+logoff);double sum=0;
    int rc=gdn_ledger(l,hv,(nt+63)/64,ms[2],&sum);
    const char*names[]={"entry","state_load","snapshot_pack","WS","residual_transpose","QK_pack","QS","QK","gate","PR","output","KR_prepare","state_decay","ordered_KR","state_store","unused"};
    for(int p=0;p<16;p++)printf("PHASE iter=%d id=%d name=%s us=%llu count=%llu\n",it,p,names[p],(unsigned long long)l[2+2*p],(unsigned long long)l[3+2*p]);
    printf("LEDGER iter=%d sum_ms=%.6f rpc3_ms=%.6f gap_ms=%.6f rc=%d\n",it,sum,ms[2],ms[2]-sum,rc);if(rc)return 13;
   }
   if(it==0||it==iters-1){char path[128];snprintf(path,sizeof(path),"output_%03d.bin",it);FILE*f=fopen(path,"wb");if(!f||fwrite(s,1,bytes,f)!=(size_t)bytes)return 8;fclose(f);}
 }
 e=attnops_close(h);remote_register_buf(s,bytes,-1);rpcmem_free(s);printf("CLOSE rc=%d\n",e);return e?9:0;
}
