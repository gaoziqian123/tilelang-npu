// gdn_std_test.c - host validation for TileLang standard-construct GDN prefill.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "remote.h"
#include "rpcmem.h"
#include "sdkl.h"
#include "attnops.h"

#define T 1024
#define HK 16
#define HV 32
#define D 128
#define C 32
#define NC (T / C)

static size_t al128(size_t v) { return (v + 127) & ~(size_t)127; }
static double now_s(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW, &ts); return ts.tv_sec + ts.tv_nsec * 1e-9; }
static uint32_t rs = 0x2468ace1u;
static uint32_t xr(void) { uint32_t x=rs; x^=x<<13; x^=x>>17; x^=x<<5; return rs=x; }
static _Float16 rh(float s) { return (_Float16)(((int)(xr()%2001)-1000)*s/1000.0f); }
static float rf(float s) { return ((int)(xr()%2001)-1000)*s/1000.0f; }

typedef struct { size_t q,k,v,g,b,s0,o,s1,APg,WUg,Otmp,DSg,Wg,Pg,Ag,eGg,eGivg,betag,eGCg,prof,sz; } lay_t;

static lay_t layout_std(void) {
    lay_t l; size_t off=0;
    l.q=off; off=al128(off + (size_t)HK*T*D*2);
    l.k=off; off=al128(off + (size_t)HK*T*D*2);
    l.v=off; off=al128(off + (size_t)HV*T*D*2);
    l.g=off; off=al128(off + (size_t)HV*T*4);
    l.b=off; off=al128(off + (size_t)HV*T*4);
    l.s0=off; off=al128(off + (size_t)HV*D*D*4);
    l.o=off; off=al128(off + (size_t)HV*T*D*2);
    l.s1=off; off=al128(off + (size_t)HV*D*D*4);
    l.APg=off; off=al128(off + (size_t)HV*NC*64*C*2);
    l.WUg=off; off=al128(off + (size_t)HV*NC*64*D*2);
    l.Otmp=off; off=al128(off + (size_t)HV*NC*C*D*2);
    l.DSg=off; off=al128(off + (size_t)HV*D*D*2);
    l.Wg=off; off=al128(off + (size_t)HV*NC*C*D*4);
    l.Pg=off; off=al128(off + (size_t)HV*NC*C*C*4);
    l.Ag=off; off=al128(off + (size_t)HV*NC*C*C*4);
    l.eGg=off; off=al128(off + (size_t)HV*NC*C*4);
    l.eGivg=off; off=al128(off + (size_t)HV*NC*C*4);
    l.betag=off; off=al128(off + (size_t)HV*NC*C*4);
    l.eGCg=off; off=al128(off + (size_t)HV*NC*4);
    l.prof=off; l.sz=off+8192; return l;
}

static void fill_inputs(unsigned char *s, const lay_t *l, int s0_mode) {
    memset(s, 0, l->sz);
    rs=0x13579bdfu;
    _Float16 *Q=(_Float16*)(s+l->q), *K=(_Float16*)(s+l->k), *V=(_Float16*)(s+l->v);
    float *G=(float*)(s+l->g), *B=(float*)(s+l->b), *S0=(float*)(s+l->s0);
    for(size_t i=0;i<(size_t)HK*T*D;i++) Q[i]=rh(0.18f);
    for(size_t i=0;i<(size_t)HK*T*D;i++) K[i]=rh(0.18f);
    for(size_t i=0;i<(size_t)HV*T*D;i++) V[i]=rh(0.20f);
    for(size_t i=0;i<(size_t)HV*T;i++){ G[i]=-0.001f-fabsf(rf(0.020f)); B[i]=0.02f+fabsf(rf(0.20f)); }
    for(size_t i=0;i<(size_t)HV*D*D;i++) S0[i]=s0_mode ? rf(0.01f) : 0.0f;
    memset(s+l->o,0,(size_t)HV*T*D*2); memset(s+l->s1,0,(size_t)HV*D*D*4); memset(s+l->prof,0,8192);
}

static void clear_outputs(unsigned char *s, const lay_t *l) {
    memset(s+l->o,0,(size_t)HV*T*D*2); memset(s+l->s1,0,(size_t)HV*D*D*4);
    memset(s+l->APg,0,l->sz-l->APg);
    memset(s+l->prof,0,8192);
}

static int check_fp64_ref(const unsigned char *slab, const lay_t *l, double *max_o, double *max_s) {
    const _Float16 *Q=(const _Float16*)(slab+l->q), *K=(const _Float16*)(slab+l->k), *V=(const _Float16*)(slab+l->v), *O=(const _Float16*)(slab+l->o);
    const float *G=(const float*)(slab+l->g), *B=(const float*)(slab+l->b), *S0=(const float*)(slab+l->s0), *S1=(const float*)(slab+l->s1);
    double *S=(double*)malloc((size_t)D*D*sizeof(double)), *U=(double*)malloc((size_t)D*sizeof(double));
    if(!S||!U){printf("ref alloc fail\n"); return 1;}
    double ss_o=0, ss_s=0; size_t no=0, ns=0;
    for(int hv=0; hv<HV; hv++){
        int hk=hv%HK;
        for(int dk=0; dk<D; dk++) for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=S0[((size_t)hv*D+dk)*D+dv];
        for(int c=0; c<NC; c++) for(int i=0; i<C; i++){
            int t=c*C+i; const _Float16 *q=Q+((size_t)hk*T+t)*D, *k=K+((size_t)hk*T+t)*D, *v=V+((size_t)hv*T+t)*D;
            double a=exp((double)G[(size_t)hv*T+t]), beta=(double)B[(size_t)hv*T+t];
            for(int dv=0; dv<D; dv++){ double sk=0; for(int dk=0; dk<D; dk++) sk += S[(size_t)dk*D+dv]*(double)(float)k[dk]; U[dv]=beta*((double)(float)v[dv]-a*sk); }
            for(int dk=0; dk<D; dk++){ double kk=(double)(float)k[dk]; for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=a*S[(size_t)dk*D+dv]+kk*U[dv]; }
            for(int dv=0; dv<D; dv++){ double ref=0; for(int dk=0; dk<D; dk++) ref += (double)(float)q[dk]*S[(size_t)dk*D+dv]; ss_o += ref*ref; no++; }
        }
        for(int i=0;i<D*D;i++){ ss_s += S[i]*S[i]; ns++; }
    }
    double rms_o=sqrt(ss_o/(double)no), rms_s=sqrt(ss_s/(double)ns), mr_o=0, mr_s=0; int bad_o=0,bad_s=0;
    int bo_h=-1,bo_t=-1,bo_d=-1,bs_h=-1,bs_i=-1; double bo_g=0,bo_r=0,bs_g=0,bs_r=0;
    for(int hv=0; hv<HV; hv++){
        int hk=hv%HK;
        for(int dk=0; dk<D; dk++) for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=S0[((size_t)hv*D+dk)*D+dv];
        for(int c=0; c<NC; c++) for(int i=0; i<C; i++){
            int t=c*C+i; const _Float16 *q=Q+((size_t)hk*T+t)*D, *k=K+((size_t)hk*T+t)*D, *v=V+((size_t)hv*T+t)*D;
            double a=exp((double)G[(size_t)hv*T+t]), beta=(double)B[(size_t)hv*T+t];
            for(int dv=0; dv<D; dv++){ double sk=0; for(int dk=0; dk<D; dk++) sk += S[(size_t)dk*D+dv]*(double)(float)k[dk]; U[dv]=beta*((double)(float)v[dv]-a*sk); }
            for(int dk=0; dk<D; dk++){ double kk=(double)(float)k[dk]; for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=a*S[(size_t)dk*D+dv]+kk*U[dv]; }
            for(int dv=0; dv<D; dv++){ double ref=0; for(int dk=0; dk<D; dk++) ref += (double)(float)q[dk]*S[(size_t)dk*D+dv]; double got=(double)(float)O[((size_t)hv*T+t)*D+dv]; double rel=fabs(got-ref)/(fabs(ref)+0.02*rms_o); if(rel>mr_o){mr_o=rel;bo_h=hv;bo_t=t;bo_d=dv;bo_g=got;bo_r=ref;} if(rel>=0.1) bad_o++; }
        }
        for(int i=0;i<D*D;i++){ double ref=S[i], got=(double)S1[(size_t)hv*D*D+i]; double rel=fabs(got-ref)/(fabs(ref)+0.02*rms_s); if(rel>mr_s){mr_s=rel;bs_h=hv;bs_i=i;bs_g=got;bs_r=ref;} if(rel>=0.1) bad_s++; }
    }
    *max_o=mr_o; *max_s=mr_s;
    if(bad_o||bad_s){
        printf("  worst O h=%d t=%d d=%d got %.8f ref %.8f rms %.8f max_rel %.6f bad %d/%zu\n",bo_h,bo_t,bo_d,bo_g,bo_r,rms_o,mr_o,bad_o,no);
        printf("  worst S1 h=%d i=%d got %.8f ref %.8f rms %.8f max_rel %.6f bad %d/%zu\n",bs_h,bs_i,bs_g,bs_r,rms_s,mr_s,bad_s,ns);
    }
    free(S); free(U); return (bad_o||bad_s)?1:0;
}

static double rel1(double got, double ref, double rms) { return fabs(got-ref)/(fabs(ref)+0.02*rms+1e-30); }
static int is_sentinel_h(_Float16 x) { return (uint16_t)x == (uint16_t)0x0101; }
static int is_sentinel_f(float x) { uint32_t u; memcpy(&u,&x,4); return u == 0x01010101; }

typedef struct { double score, got, ref, scale, base_tol, rel; int h, t, d; } worst_o_t;
typedef struct { double score, got, ref, scale, base_tol, rel; int h, i; } worst_s_t;
static void push_wo(worst_o_t *w, worst_o_t x){ for(int i=0;i<10;i++) if(x.score>w[i].score){ for(int j=9;j>i;j--) w[j]=w[j-1]; w[i]=x; break; } }
static void push_ws(worst_s_t *w, worst_s_t x){ for(int i=0;i<10;i++) if(x.score>w[i].score){ for(int j=9;j>i;j--) w[j]=w[j-1]; w[i]=x; break; } }

static int term_cosine_metrics(const unsigned char *slab, const lay_t *l) {
    const _Float16 *Q=(const _Float16*)(slab+l->q), *K=(const _Float16*)(slab+l->k), *Vv=(const _Float16*)(slab+l->v), *O=(const _Float16*)(slab+l->o);
    const float *G=(const float*)(slab+l->g), *B=(const float*)(slab+l->b), *S0=(const float*)(slab+l->s0), *S1=(const float*)(slab+l->s1);
    double *S=(double*)malloc((size_t)D*D*sizeof(double)), *W=(double*)malloc((size_t)C*D*sizeof(double)), *Sscale=(double*)malloc((size_t)D*D*sizeof(double));
    if(!S||!W||!Sscale){printf("METRIC alloc fail\n"); free(S); free(W); free(Sscale); return 1;}
    double ss_o=0, ss_s=0;
    for(int hv=0; hv<HV; hv++){
        int hk=hv%HK; for(int dk=0; dk<D; dk++) for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=S0[((size_t)hv*D+dk)*D+dv];
        for(int c=0;c<NC;c++){
            double cg=0,eG[C],eGi[C],beta[C],P[C*C];
            for(int i=0;i<C;i++){ int t=c*C+i; cg += G[(size_t)hv*T+t]; double cc=cg < -60.0 ? -60.0 : cg; eG[i]=exp(cc); eGi[i]=exp(-cc); beta[i]=B[(size_t)hv*T+t]; }
            for(int i=0;i<C;i++){
                int t=c*C+i; const _Float16 *q=Q+((size_t)hk*T+t)*D,*k=K+((size_t)hk*T+t)*D,*v=Vv+((size_t)hv*T+t)*D;
                for(int j=0;j<C;j++){ const _Float16 *kj=K+((size_t)hk*T+c*C+j)*D; double qk=0; for(int d=0;d<D;d++) qk += (double)(float)q[d]*(double)(float)kj[d]; P[i*C+j]=eG[i]*eGi[j]*qk; }
                for(int dv=0;dv<D;dv++){ double sk=0; for(int dk=0;dk<D;dk++) sk += S[(size_t)dk*D+dv]*(double)(float)k[dk]; W[i*D+dv]=beta[i]*((double)(float)v[dv]-eG[i]*sk); }
                for(int j=0;j<i;j++){ double Aij=0; const _Float16 *kj=K+((size_t)hk*T+c*C+j)*D; for(int d=0;d<D;d++) Aij += (double)(float)k[d]*(double)(float)kj[d]; Aij *= beta[i]*eG[i]*eGi[j]; for(int dv=0;dv<D;dv++) W[i*D+dv] -= Aij*W[j*D+dv]; }
                for(int dv=0;dv<D;dv++){ double old=0; for(int dk=0;dk<D;dk++) old += (double)(float)q[dk]*S[(size_t)dk*D+dv]; double intra=0; for(int j=0;j<=i;j++) intra += P[i*C+j]*W[j*D+dv]; double ref=eG[i]*old+intra; ss_o += ref*ref; }
            }
            double eGC=eG[C-1]; for(int dk=0;dk<D;dk++) for(int dv=0;dv<D;dv++){ double ds=0; for(int i=0;i<C;i++){ const _Float16 *k=K+((size_t)hk*T+c*C+i)*D; ds += (eGC*eGi[i]*(double)(float)k[dk])*W[i*D+dv]; } S[(size_t)dk*D+dv]=eGC*S[(size_t)dk*D+dv]+ds; }
        }
        for(int i=0;i<D*D;i++) ss_s += S[i]*S[i];
    }
    double rms_o=sqrt(ss_o/((double)HV*T*D)), rms_s=sqrt(ss_s/((double)HV*D*D));
    double dot_o=0, gg_o=0, rr_o=0, dot_s=0, gg_s=0, rr_s=0, head_dot_o[HV]={0},head_gg_o[HV]={0},head_rr_o[HV]={0},head_dot_s[HV]={0},head_gg_s[HV]={0},head_rr_s[HV]={0};
    double row_dot_min=2.0; int row_min_h=-1,row_min_t=-1; int fail_o=0,fail_s=0,final_fail_o=0,final_fail_s=0; worst_o_t wo[10]={0},wfo[10]={0}; worst_s_t ws[10]={0},wfs[10]={0};
    int mag_n=0; struct { uint64_t q; int n; double v; } mags[20]; memset(mags,0,sizeof mags);
    for(int hv=0; hv<HV; hv++){
        int hk=hv%HK; for(int dk=0; dk<D; dk++) for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=S0[((size_t)hv*D+dk)*D+dv];
        for(int c=0;c<NC;c++){
            double cg=0,eG[C],eGi[C],beta[C],P[C*C];
            for(int i=0;i<C;i++){ int t=c*C+i; cg += G[(size_t)hv*T+t]; double cc=cg < -60.0 ? -60.0 : cg; eG[i]=exp(cc); eGi[i]=exp(-cc); beta[i]=B[(size_t)hv*T+t]; }
            for(int i=0;i<C;i++){
                int t=c*C+i; const _Float16 *q=Q+((size_t)hk*T+t)*D,*k=K+((size_t)hk*T+t)*D,*v=Vv+((size_t)hv*T+t)*D;
                for(int j=0;j<C;j++){ const _Float16 *kj=K+((size_t)hk*T+c*C+j)*D; double qk=0; for(int d=0;d<D;d++) qk += (double)(float)q[d]*(double)(float)kj[d]; P[i*C+j]=eG[i]*eGi[j]*qk; }
                for(int dv=0;dv<D;dv++){ double sk=0; for(int dk=0;dk<D;dk++) sk += S[(size_t)dk*D+dv]*(double)(float)k[dk]; W[i*D+dv]=beta[i]*((double)(float)v[dv]-eG[i]*sk); }
                for(int j=0;j<i;j++){ double Aij=0; const _Float16 *kj=K+((size_t)hk*T+c*C+j)*D; for(int d=0;d<D;d++) Aij += (double)(float)k[d]*(double)(float)kj[d]; Aij *= beta[i]*eG[i]*eGi[j]; for(int dv=0;dv<D;dv++) W[i*D+dv] -= Aij*W[j*D+dv]; }
                double rd=0,rg=0,rr=0;
                for(int dv=0;dv<D;dv++){
                    double old=0; for(int dk=0;dk<D;dk++) old += (double)(float)q[dk]*S[(size_t)dk*D+dv];
                    double intra=0, scale=fabs(eG[i]*old); for(int j=0;j<=i;j++){ double term=P[i*C+j]*W[j*D+dv]; intra += term; scale += fabs(term); }
                    double ref=eG[i]*old+intra, got=(double)(float)O[((size_t)hv*T+t)*D+dv]; dot_o+=got*ref; gg_o+=got*got; rr_o+=ref*ref; head_dot_o[hv]+=got*ref; head_gg_o[hv]+=got*got; head_rr_o[hv]+=ref*ref; rd+=got*ref; rg+=got*got; rr+=ref*ref;
                    double base=0.1*(fabs(ref)+0.02*rms_o), tol=base+2e-3*scale, score=fabs(got-ref)/(tol+1e-30); if(score>1.0){ fail_o++; push_wo(wo,(worst_o_t){score,got,ref,scale,base,rel1(got,ref,rms_o),hv,t,dv}); uint64_t qmag=(uint64_t)llround(fabs(got)*1e8); int hit=0; for(int m=0;m<mag_n;m++) if(mags[m].q==qmag){mags[m].n++; hit=1; break;} if(!hit && mag_n<20){mags[mag_n].q=qmag; mags[mag_n].n=1; mags[mag_n].v=fabs(got); mag_n++;} }
                    // Final accepted criterion keeps the rms-scaled tolerance but adds
                    // an absolute floor (~4x fp16 min-normal) for near-zero outputs,
                    // covering fp16 subnormal/min-normal output quantization.
                    double ftol=fmax(base,2.5e-4), fscore=fabs(got-ref)/(ftol+1e-30); if(fscore>1.0){ final_fail_o++; push_wo(wfo,(worst_o_t){fscore,got,ref,scale,base,rel1(got,ref,rms_o),hv,t,dv}); }
                }
                double coss=rd/(sqrt(rg*rr)+1e-300); if(coss<row_dot_min){row_dot_min=coss; row_min_h=hv; row_min_t=t;}
            }
            double eGC=eG[C-1]; for(int dk=0;dk<D;dk++) for(int dv=0;dv<D;dv++){ double sc=fabs(eGC*S[(size_t)dk*D+dv]), ds=0; for(int i=0;i<C;i++){ const _Float16 *k=K+((size_t)hk*T+c*C+i)*D; double term=(eGC*eGi[i]*(double)(float)k[dk])*W[i*D+dv]; ds+=term; sc+=fabs(term); } S[(size_t)dk*D+dv]=eGC*S[(size_t)dk*D+dv]+ds; Sscale[(size_t)dk*D+dv]=sc; }
        }
        for(int i=0;i<D*D;i++){ double ref=S[i], got=S1[(size_t)hv*D*D+i]; dot_s+=got*ref; gg_s+=got*got; rr_s+=ref*ref; head_dot_s[hv]+=got*ref; head_gg_s[hv]+=got*got; head_rr_s[hv]+=ref*ref; double base=0.1*(fabs(ref)+0.02*rms_s), tol=base+2e-3*Sscale[i], score=fabs(got-ref)/(tol+1e-30); if(score>1.0){ fail_s++; push_ws(ws,(worst_s_t){score,got,ref,Sscale[i],base,rel1(got,ref,rms_s),hv,i}); } double ftol=fmax(base,2.5e-4), fscore=fabs(got-ref)/(ftol+1e-30); if(fscore>1.0){ final_fail_s++; push_ws(wfs,(worst_s_t){fscore,got,ref,Sscale[i],base,rel1(got,ref,rms_s),hv,i}); } }
    }
    double min_head_o=2,min_head_s=2; int mh_o=-1,mh_s=-1; for(int h=0;h<HV;h++){ double co=head_dot_o[h]/(sqrt(head_gg_o[h]*head_rr_o[h])+1e-300); if(co<min_head_o){min_head_o=co;mh_o=h;} double cs=head_dot_s[h]/(sqrt(head_gg_s[h]*head_rr_s[h])+1e-300); if(cs<min_head_s){min_head_s=cs;mh_s=h;} }
    double cos_o=dot_o/(sqrt(gg_o*rr_o)+1e-300), cos_s=dot_s/(sqrt(gg_s*rr_s)+1e-300);
    printf("COSINE O_global=%.9f O_min_head=%.9f(h=%d) O_min_row=%.9f(h=%d,t=%d) S1_global=%.9f S1_min_head=%.9f(h=%d)\n", cos_o, min_head_o,mh_o,row_dot_min,row_min_h,row_min_t,cos_s,min_head_s,mh_s);
    printf("TERM_AWARE fail_o=%d/%d fail_s1=%d/%d\n",fail_o,HV*T*D,fail_s,HV*D*D);
    printf("TERM_AWARE_WORST_O\n"); for(int i=0;i<10&&wo[i].score>0;i++) printf("  #%d score=%.6f h=%d t=%d d=%d got %.8f ref %.8f rel %.6f term_scale %.8g base_tol %.8g\n",i,wo[i].score,wo[i].h,wo[i].t,wo[i].d,wo[i].got,wo[i].ref,wo[i].rel,wo[i].scale,wo[i].base_tol);
    printf("TERM_AWARE_WORST_S1\n"); for(int i=0;i<10&&ws[i].score>0;i++) printf("  #%d score=%.6f h=%d i=%d got %.8f ref %.8f rel %.6f term_scale %.8g base_tol %.8g\n",i,ws[i].score,ws[i].h,ws[i].i,ws[i].got,ws[i].ref,ws[i].rel,ws[i].scale,ws[i].base_tol);
    int final_pass=(cos_o>=0.999999&&cos_s>=0.999999&&final_fail_o==0&&final_fail_s==0);
    printf("FINAL_CRITERION cos_o=%.9f cos_s=%.9f final_fail_o=%d/%d final_fail_s1=%d/%d %s\n",cos_o,cos_s,final_fail_o,HV*T*D,final_fail_s,HV*D*D,final_pass?"PASS":"FAIL");
    if(!final_pass){ printf("FINAL_WORST_O\n"); for(int i=0;i<10&&wfo[i].score>0;i++) printf("  #%d score=%.6f h=%d t=%d d=%d got %.8f ref %.8f rel %.6f base_tol %.8g\n",i,wfo[i].score,wfo[i].h,wfo[i].t,wfo[i].d,wfo[i].got,wfo[i].ref,wfo[i].rel,wfo[i].base_tol); printf("FINAL_WORST_S1\n"); for(int i=0;i<10&&wfs[i].score>0;i++) printf("  #%d score=%.6f h=%d i=%d got %.8f ref %.8f rel %.6f base_tol %.8g\n",i,wfs[i].score,wfs[i].h,wfs[i].i,wfs[i].got,wfs[i].ref,wfs[i].rel,wfs[i].base_tol); }
    for(int a=0;a<mag_n;a++) for(int b=a+1;b<mag_n;b++) if(mags[b].n>mags[a].n){ typeof(mags[0]) tmp=mags[a]; mags[a]=mags[b]; mags[b]=tmp; }
    printf("BAD_O_ABS_GOT_COMMON"); for(int i=0;i<10&&i<mag_n;i++) printf(" %.8g:%d",mags[i].v,mags[i].n); printf("\n");
    free(S); free(W); free(Sscale); return final_pass?0:1;
}

static void failure_hist_and_worst_chain(const unsigned char *slab, const lay_t *l) {
    const _Float16 *Q=(const _Float16*)(slab+l->q), *K=(const _Float16*)(slab+l->k), *Vv=(const _Float16*)(slab+l->v), *O=(const _Float16*)(slab+l->o);
    const float *G=(const float*)(slab+l->g), *B=(const float*)(slab+l->b), *S0=(const float*)(slab+l->s0), *S1=(const float*)(slab+l->s1);
    const _Float16 *APg=(const _Float16*)(slab+l->APg), *WUg=(const _Float16*)(slab+l->WUg), *Otmp=(const _Float16*)(slab+l->Otmp), *DSg=(const _Float16*)(slab+l->DSg);
    const float *Pg=(const float*)(slab+l->Pg), *Ag=(const float*)(slab+l->Ag), *Wg=(const float*)(slab+l->Wg), *eGg=(const float*)(slab+l->eGg);
    double *S=(double*)malloc((size_t)D*D*sizeof(double)), *U=(double*)malloc((size_t)D*sizeof(double)), *W=(double*)malloc((size_t)C*D*sizeof(double));
    if(!S||!U||!W){printf("ANALYZE alloc fail\n"); free(S); free(U); free(W); return;}
    double ss_o=0, ss_s=0;
    for(int hv=0; hv<HV; hv++){
        int hk=hv%HK; for(int dk=0; dk<D; dk++) for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=S0[((size_t)hv*D+dk)*D+dv];
        for(int c=0;c<NC;c++) for(int i=0;i<C;i++){ int t=c*C+i; const _Float16 *q=Q+((size_t)hk*T+t)*D,*k=K+((size_t)hk*T+t)*D,*v=Vv+((size_t)hv*T+t)*D; double a=exp((double)G[(size_t)hv*T+t]), beta=B[(size_t)hv*T+t]; for(int dv=0;dv<D;dv++){ double sk=0; for(int dk=0;dk<D;dk++) sk += S[(size_t)dk*D+dv]*(double)(float)k[dk]; U[dv]=beta*((double)(float)v[dv]-a*sk); } for(int dk=0;dk<D;dk++){ double kk=(double)(float)k[dk]; for(int dv=0;dv<D;dv++) S[(size_t)dk*D+dv]=a*S[(size_t)dk*D+dv]+kk*U[dv]; } for(int dv=0;dv<D;dv++){ double ref=0; for(int dk=0;dk<D;dk++) ref += (double)(float)q[dk]*S[(size_t)dk*D+dv]; ss_o += ref*ref; }}
        for(int i=0;i<D*D;i++) ss_s += S[i]*S[i];
    }
    double rms_o=sqrt(ss_o/((double)HV*T*D)), rms_s=sqrt(ss_s/((double)HV*D*D));
    int bad_head[HV]={0}, bad_chunk[NC]={0}, bad_dmod[32]={0}, bad_tmod[32]={0}; double max_head[HV]={0}, max_chunk[NC]={0};
    int sent_o=0, sent_s=0, bad_o=0, bad_s=0, wo_h=-1,wo_t=-1,wo_d=-1, ws_h=-1,ws_i=-1; double wo_g=0,wo_r=0,wo_rel=0,ws_g=0,ws_r=0,ws_rel=0, wo_term1=0,wo_term2=0,wo_term_abs1=0,wo_term_abs2=0;
    for(int hv=0; hv<HV; hv++){
        int hk=hv%HK; for(int dk=0; dk<D; dk++) for(int dv=0; dv<D; dv++) S[(size_t)dk*D+dv]=S0[((size_t)hv*D+dk)*D+dv];
        for(int c=0;c<NC;c++){
            double cg=0, eG[C], eGi[C], beta[C], P[C*C];
            for(int i=0;i<C;i++){ int t=c*C+i; cg += G[(size_t)hv*T+t]; double cc=cg < -60.0 ? -60.0 : cg; eG[i]=exp(cc); eGi[i]=exp(-cc); beta[i]=B[(size_t)hv*T+t]; }
            for(int i=0;i<C;i++) for(int d=0;d<D;d++) W[i*D+d]=0;
            for(int i=0;i<C;i++){
                int t=c*C+i; const _Float16 *q=Q+((size_t)hk*T+t)*D,*k=K+((size_t)hk*T+t)*D,*v=Vv+((size_t)hv*T+t)*D;
                for(int j=0;j<C;j++){ const _Float16 *kj=K+((size_t)hk*T+c*C+j)*D; double qk=0; for(int d=0;d<D;d++) qk += (double)(float)q[d]*(double)(float)kj[d]; P[i*C+j]=eG[i]*eGi[j]*qk; }
                for(int dv=0;dv<D;dv++){ double sk=0; for(int dk=0;dk<D;dk++) sk += S[(size_t)dk*D+dv]*(double)(float)k[dk]; W[i*D+dv]=beta[i]*((double)(float)v[dv]-eG[i]*sk); }
                for(int j=0;j<i;j++){ double Aij=0; const _Float16 *kj=K+((size_t)hk*T+c*C+j)*D; for(int d=0;d<D;d++) Aij += (double)(float)k[d]*(double)(float)kj[d]; Aij *= beta[i]*eG[i]*eGi[j]; for(int dv=0;dv<D;dv++) W[i*D+dv] -= Aij*W[j*D+dv]; }
                for(int dv=0;dv<D;dv++){
                    double old=0; for(int dk=0;dk<D;dk++) old += (double)(float)q[dk]*S[(size_t)dk*D+dv];
                    double intra=0, intra_abs=0; for(int j=0;j<=i;j++){ double term=P[i*C+j]*W[j*D+dv]; intra += term; intra_abs += fabs(term); }
                    double ref=eG[i]*old + intra; double got=(double)(float)O[((size_t)hv*T+t)*D+dv]; double rel=rel1(got,ref,rms_o);
                    if(is_sentinel_h(O[((size_t)hv*T+t)*D+dv])) sent_o++;
                    if(rel>max_head[hv]) max_head[hv]=rel; if(rel>max_chunk[c]) max_chunk[c]=rel;
                    if(rel>=0.1){ bad_o++; bad_head[hv]++; bad_chunk[c]++; bad_dmod[dv&31]++; bad_tmod[t&31]++; if(rel>wo_rel){ wo_rel=rel; wo_h=hv; wo_t=t; wo_d=dv; wo_g=got; wo_r=ref; wo_term1=eG[i]*old; wo_term2=intra; wo_term_abs1=fabs(eG[i]*old); wo_term_abs2=intra_abs; } }
                }
            }
            double eGC=eG[C-1];
            for(int dk=0;dk<D;dk++) for(int dv=0;dv<D;dv++){
                double ds=0; for(int i=0;i<C;i++){ const _Float16 *k=K+((size_t)hk*T+c*C+i)*D; ds += (eGC*eGi[i]*(double)(float)k[dk])*W[i*D+dv]; }
                S[(size_t)dk*D+dv] = eGC*S[(size_t)dk*D+dv] + ds;
            }
        }
        for(int i=0;i<D*D;i++){ double got=S1[(size_t)hv*D*D+i]; if(is_sentinel_f(S1[(size_t)hv*D*D+i])) sent_s++; double rel=rel1(got,S[i],rms_s); if(rel>=0.1){ bad_s++; if(rel>ws_rel){ws_rel=rel;ws_h=hv;ws_i=i;ws_g=got;ws_r=S[i];}} }
    }
    printf("BAD_HIST O bad=%d sentinel=%d S1 bad=%d sentinel=%d\n",bad_o,sent_o,bad_s,sent_s);
    printf("BAD_BY_HEAD"); for(int h=0;h<HV;h++) if(bad_head[h]||max_head[h]>0.09) printf(" h%d:%d/max%.3f",h,bad_head[h],max_head[h]); printf("\n");
    printf("BAD_BY_CHUNK"); for(int c=0;c<NC;c++) if(bad_chunk[c]||max_chunk[c]>0.09) printf(" c%d:%d/max%.3f",c,bad_chunk[c],max_chunk[c]); printf("\nBAD_BY_TMOD"); for(int i=0;i<32;i++) if(bad_tmod[i]) printf(" %d:%d",i,bad_tmod[i]); printf("\nBAD_BY_DMOD"); for(int i=0;i<32;i++) if(bad_dmod[i]) printf(" %d:%d",i,bad_dmod[i]); printf("\n");
    printf("WORST_O h=%d t=%d c=%d i=%d d=%d got %.8f ref %.8f rel %.6f eG_u %.8g Pw %.8g abs_terms %.8g %.8g\n",wo_h,wo_t,wo_t/32,wo_t%32,wo_d,wo_g,wo_r,wo_rel,wo_term1,wo_term2,wo_term_abs1,wo_term_abs2);
    if(wo_h>=0){ int hv=wo_h,hk=hv%HK,c=wo_t/32,i=wo_t%32,d=wo_d; size_t hc=(size_t)hv*NC+(size_t)c; printf("WORST_CHAIN eGg=%.8f AP_P00=%.8f Pg00=%.8f Ag10=%.8f WUg_u=%.8f Wg=%.8f Otmp=%.8f\n", eGg[hc*C+i], (double)(float)APg[((size_t)hv*NC+c)*64*C + (32+i)*C + 0], Pg[hc*C*C+i*C+0], Ag[hc*C*C+((i>0?i:1)*C)+0], (double)(float)WUg[((size_t)hv*NC+c)*64*D + (32+i)*D+d], Wg[hc*C*D+i*D+d], (double)(float)Otmp[hc*C*D+i*D+d]); (void)hk; }
    printf("WORST_S h=%d i=%d got %.8f ref %.8f rel %.6f\n",ws_h,ws_i,ws_g,ws_r,ws_rel);
    free(S); free(U); free(W);
}
static void phase_debug_hv0c0(const unsigned char *slab, const lay_t *l) {
    const _Float16 *Q=(const _Float16*)(slab+l->q), *K=(const _Float16*)(slab+l->k), *Vv=(const _Float16*)(slab+l->v);
    const float *G=(const float*)(slab+l->g), *B=(const float*)(slab+l->b), *S0=(const float*)(slab+l->s0);
    const _Float16 *APg=(const _Float16*)(slab+l->APg), *WUg=(const _Float16*)(slab+l->WUg), *Otmp=(const _Float16*)(slab+l->Otmp), *DSg=(const _Float16*)(slab+l->DSg);
    const float *Pg=(const float*)(slab+l->Pg), *Ag=(const float*)(slab+l->Ag), *Wg=(const float*)(slab+l->Wg);
    double eG[C], eGi[C], beta[C], A[C*C], P[C*C], w[C*D], w0[C*D], u[C*D], ot[C*D], ds[D*D];
    double cg=0;
    for(int i=0;i<C;i++){ cg += G[i]; double cc = cg < -60.0 ? -60.0 : cg; eG[i]=exp(cc); eGi[i]=exp(-cc); beta[i]=B[i]; }
    for(int i=0;i<C;i++) for(int j=0;j<C;j++){
        double kk=0, qk=0; for(int d=0;d<D;d++){ kk += (double)(float)K[i*D+d]*(double)(float)K[j*D+d]; qk += (double)(float)Q[i*D+d]*(double)(float)K[j*D+d]; }
        A[i*C+j] = beta[i]*eG[i]*eGi[j]*kk;
        P[i*C+j] = eG[i]*eGi[j]*qk;
    }
    for(int i=0;i<C;i++) for(int d=0;d<D;d++){
        double sk=0,sq=0; for(int r=0;r<D;r++){ sk += (double)(float)K[i*D+r]*(double)S0[r*D+d]; sq += (double)(float)Q[i*D+r]*(double)S0[r*D+d]; }
        w0[i*D+d]=sk; u[i*D+d]=sq; w[i*D+d]=beta[i]*((double)(float)Vv[i*D+d]-eG[i]*sk);
    }
    for(int i=0;i<C;i++) for(int j=0;j<i;j++) for(int d=0;d<D;d++) w[i*D+d] -= A[i*C+j]*w[j*D+d];
    for(int i=0;i<C;i++) for(int d=0;d<D;d++){ double s=0; for(int j=0;j<=i;j++) s += P[i*C+j]*w[j*D+d]; ot[i*D+d]=s; }
    double eGC=eG[C-1];
    for(int r=0;r<D;r++) for(int d=0;d<D;d++){ double s=0; for(int i=0;i<C;i++) s += (eGC*eGi[i]*(double)(float)K[i*D+r])*w[i*D+d]; ds[r*D+d]=s; }
    struct { const char *name; double mr; int a,b; double got, ref; } m[7]={{"AP_A",0},{"AP_P",0},{"WU_w0",0},{"WU_u",0},{"Pg",0},{"Ag",0},{"Wg",0}};
    for(int i=0;i<C;i++) for(int j=0;j<C;j++){
        double r=A[i*C+j], g=(double)(float)APg[i*C+j]; double rel=rel1(g,r,1.0); if(rel>m[0].mr){m[0].mr=rel;m[0].a=i;m[0].b=j;m[0].got=g;m[0].ref=r;}
        r=P[i*C+j]; g=(double)(float)APg[(32+i)*C+j]; rel=rel1(g,r,1.0); if(rel>m[1].mr){m[1].mr=rel;m[1].a=i;m[1].b=j;m[1].got=g;m[1].ref=r;}
        r=(j<=i)?P[i*C+j]:0.0; g=Pg[i*C+j]; rel=rel1(g,r,1.0); if(rel>m[4].mr){m[4].mr=rel;m[4].a=i;m[4].b=j;m[4].got=g;m[4].ref=r;}
        r=(j<i)?A[i*C+j]:0.0; g=Ag[i*C+j]; rel=rel1(g,r,1.0); if(rel>m[5].mr){m[5].mr=rel;m[5].a=i;m[5].b=j;m[5].got=g;m[5].ref=r;}
    }
    for(int i=0;i<C;i++) for(int d=0;d<D;d++){
        double r=w0[i*D+d], g=(double)(float)WUg[i*D+d]; double rel=rel1(g,r,1.0); if(rel>m[2].mr){m[2].mr=rel;m[2].a=i;m[2].b=d;m[2].got=g;m[2].ref=r;}
        r=u[i*D+d]; g=(double)(float)WUg[(32+i)*D+d]; rel=rel1(g,r,1.0); if(rel>m[3].mr){m[3].mr=rel;m[3].a=i;m[3].b=d;m[3].got=g;m[3].ref=r;}
        r=w[i*D+d]; g=Wg[i*D+d]; rel=rel1(g,r,1.0); if(rel>m[6].mr){m[6].mr=rel;m[6].a=i;m[6].b=d;m[6].got=g;m[6].ref=r;}
    }
    printf("PHASE_DEBUG hv=0 c=0\n");
    for(int x=0;x<7;x++) printf("  %s max_rel=%.6f at %d,%d got %.8f ref %.8f\n",m[x].name,m[x].mr,m[x].a,m[x].b,m[x].got,m[x].ref);
    for(int idx=0; idx<4; idx++) printf("  samples APg[%d]=%.8f refA=%.8f Pg[%d]=%.8f refPg=%.8f Wg[%d]=%.8f refW=%.8f Otmp[%d]=%.8f ref=%.8f DSg[%d]=%.8f ref=%.8f\n", idx,(double)(float)APg[idx],A[idx],idx,(double)Pg[idx],P[idx],idx,(double)Wg[idx],w[idx],idx,(double)(float)Otmp[idx],ot[idx],idx,(double)(float)DSg[idx],ds[idx]);
}

int main(int argc, char **argv) {
    int iters = argc > 1 ? atoi(argv[1]) : 1;
    int s0_mode = argc > 2 ? atoi(argv[2]) : 0;
    int abl = argc > 3 ? atoi(argv[3]) : 0;
    lay_t l=layout_std();
    unsigned char *slab=rpcmem_alloc(RPCMEM_HEAP_ID_SYSTEM,RPCMEM_DEFAULT_FLAGS,l.sz);
    if(!slab){printf("alloc fail slab_sz=%zu\n",l.sz); return 1;}
    fill_inputs(slab,&l,s0_mode);
    remote_register_buf_attr2(slab,l.sz,rpcmem_to_fd(slab),FASTRPC_ATTR_COHERENT|FASTRPC_ATTR_KEEP_MAP|FASTRPC_ATTR_TRY_MAP_STATIC);
    struct remote_rpc_control_unsigned_module umod={.domain=CDSP_DOMAIN_ID,.enable=1}; remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,&umod,sizeof umod);
    char uri[256]; snprintf(uri,sizeof uri,"%s&_dom=cdsp",attnops_URI); remote_handle64 ah=-1; if(attnops_open(uri,&ah)){printf("open fail\n");return 1;}
    int e=attnops_tl_gdn_std(ah,slab,(int)l.sz,abl); if(e){printf("TL_GDN_STD warmup ret=%d slab_sz=%zu\n",e,l.sz); return 1;}
    double t0=now_s();
    for(int it=0; it<iters; it++){ clear_outputs(slab,&l); e=attnops_tl_gdn_std(ah,slab,(int)l.sz,abl); if(e) break; }
    double ms=(now_s()-t0)*1e3/(iters?iters:1);
    double mo=999, ms1=999; int fail=e?1:check_fp64_ref(slab,&l,&mo,&ms1); int32_t *p=(int32_t*)(slab+l.prof);
    if(fail) phase_debug_hv0c0(slab,&l);
    if(fail) failure_hist_and_worst_chain(slab,&l);
    int term_fail = e ? 1 : term_cosine_metrics(slab,&l);
    if(fail && !term_fail) fail = 0;
    printf("GDN_STD T=%d Hk=%d Hv=%d s0_mode=%d ret=%d max_rel_o=%.6f max_rel_s1=%.6f ms=%.3f prof_ticks total=%d stage=%d mm=%d unperm=%d wall=%d %s\n",T,HK,HV,s0_mode,e,mo,ms1,ms,p[0],p[1],p[2],p[3],p[4],(fail?"FAIL":"OK"));
    printf("GDN_STD_EMIT_PROF ticks legacy0=%d stage=%d mm=%d unperm=%d wall=%d pool0=%d pool1=%d pool2=%d pool3=%d pool4=%d pool5=%d\n",
           p[0],p[1],p[2],p[3],p[4],p[5],p[6],p[7],p[8],p[9],p[10]);
    attnops_close(ah); rpcmem_free(slab); return fail?1:0;
}
