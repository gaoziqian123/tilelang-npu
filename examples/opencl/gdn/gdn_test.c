// Host validation for TileLang OpenCL GDN prefill kernels.
// Build (Android):
//   $ANDROID_NDK_ROOT/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang \
//     -target aarch64-linux-android26 -march=armv8.2-a+fp16 -std=c11 -O3 -fPIE -pie \
//     -I/root/project/backend/gpu/OpenCL-Headers gdn_test.c -ldl -lm -o gdn_test
// Run: LD_LIBRARY_PATH=.:/system/lib64:/vendor/lib64 ./gdn_test gdn_prep.cl gdn_seq.cl 1024 5

#include <dlfcn.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CL_TARGET_OPENCL_VERSION 300
#include "CL/cl.h"

#define HK 16
#define HV 32
#define D 128
#define CH 32
#define WG 128

typedef cl_int (*PFN_clGetPlatformIDs)(cl_uint, cl_platform_id*, cl_uint*);
typedef cl_int (*PFN_clGetDeviceIDs)(cl_platform_id, cl_device_type, cl_uint, cl_device_id*, cl_uint*);
typedef cl_context (*PFN_clCreateContext)(const cl_context_properties*, cl_uint, const cl_device_id*, void(CL_CALLBACK*)(const char*, const void*, size_t, void*), void*, cl_int*);
typedef cl_command_queue (*PFN_clCreateCommandQueueWithProperties)(cl_context, cl_device_id, const cl_queue_properties*, cl_int*);
typedef cl_program (*PFN_clCreateProgramWithSource)(cl_context, cl_uint, const char**, const size_t*, cl_int*);
typedef cl_int (*PFN_clBuildProgram)(cl_program, cl_uint, const cl_device_id*, const char*, void(CL_CALLBACK*)(cl_program, void*), void*);
typedef cl_int (*PFN_clGetProgramBuildInfo)(cl_program, cl_device_id, cl_program_build_info, size_t, void*, size_t*);
typedef cl_kernel (*PFN_clCreateKernel)(cl_program, const char*, cl_int*);
typedef cl_mem (*PFN_clCreateBuffer)(cl_context, cl_mem_flags, size_t, void*, cl_int*);
typedef cl_int (*PFN_clSetKernelArg)(cl_kernel, cl_uint, size_t, const void*);
typedef cl_int (*PFN_clEnqueueNDRangeKernel)(cl_command_queue, cl_kernel, cl_uint, const size_t*, const size_t*, const size_t*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clEnqueueWriteBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, const void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clEnqueueReadBuffer)(cl_command_queue, cl_mem, cl_bool, size_t, size_t, void*, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clEnqueueCopyBuffer)(cl_command_queue, cl_mem, cl_mem, size_t, size_t, size_t, cl_uint, const cl_event*, cl_event*);
typedef cl_int (*PFN_clFinish)(cl_command_queue);
#define DECL(n) static PFN_##n my_##n
DECL(clGetPlatformIDs); DECL(clGetDeviceIDs); DECL(clCreateContext); DECL(clCreateCommandQueueWithProperties);
DECL(clCreateProgramWithSource); DECL(clBuildProgram); DECL(clGetProgramBuildInfo); DECL(clCreateKernel); DECL(clCreateBuffer);
DECL(clSetKernelArg); DECL(clEnqueueNDRangeKernel); DECL(clEnqueueWriteBuffer); DECL(clEnqueueReadBuffer); DECL(clEnqueueCopyBuffer); DECL(clFinish);

#define CK(x) do { cl_int e__=(x); if(e__){fprintf(stderr,"CL error %d at %s:%d\n",e__,__FILE__,__LINE__); exit(2);} } while(0)

static void load_cl(void){
    const char *paths[]={"/vendor/lib64/libOpenCL.so","/system/vendor/lib64/libOpenCL.so","libOpenCL.so"}; void *lib=NULL;
    for(int i=0;i<3&&!lib;i++) lib=dlopen(paths[i],RTLD_NOW); if(!lib){fprintf(stderr,"dlopen OpenCL failed\n"); exit(2);} 
#define GET(n) do{ my_##n=(PFN_##n)dlsym(lib,#n); if(!my_##n){fprintf(stderr,"missing %s\n",#n); exit(2);} }while(0)
    GET(clGetPlatformIDs); GET(clGetDeviceIDs); GET(clCreateContext); GET(clCreateCommandQueueWithProperties); GET(clCreateProgramWithSource);
    GET(clBuildProgram); GET(clGetProgramBuildInfo); GET(clCreateKernel); GET(clCreateBuffer); GET(clSetKernelArg); GET(clEnqueueNDRangeKernel);
    GET(clEnqueueWriteBuffer); GET(clEnqueueReadBuffer); GET(clEnqueueCopyBuffer); GET(clFinish);
#undef GET
}
static char *read_file(const char *p, size_t *n){ FILE *f=fopen(p,"rb"); if(!f){perror(p); exit(2);} fseek(f,0,SEEK_END); long z=ftell(f); fseek(f,0,SEEK_SET); char *b=(char*)malloc((size_t)z+1); fread(b,1,(size_t)z,f); fclose(f); b[z]=0; *n=(size_t)z; return b; }
static double now_s(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC_RAW,&ts); return ts.tv_sec+ts.tv_nsec*1e-9; }
static unsigned short f2h(float f){ unsigned x; memcpy(&x,&f,4); unsigned sign=(x>>16)&0x8000; int e=((x>>23)&0xff)-127+15; unsigned m=x&0x7fffff; if(e<=0) return sign; if(e>=31) return sign|0x7bff; return sign|(e<<10)|(m>>13); }
static float h2f(unsigned short h){ unsigned sign=(h&0x8000)<<16; int e=(h>>10)&0x1f,m=h&0x3ff; if(e==0) return m*5.9604645e-8f; unsigned x=sign|((e-15+127)<<23)|(m<<13); float f; memcpy(&f,&x,4); return f; }
static cl_mem mkbuf(cl_context ctx, cl_mem_flags fl, size_t sz){ cl_int err; cl_mem m=my_clCreateBuffer(ctx,fl,sz,NULL,&err); CK(err); return m; }

int main(int argc,char**argv){
    const char *prep_path=argc>1?argv[1]:"gdn_prep.cl", *seq_path=argc>2?argv[2]:"gdn_seq.cl";
    int T=argc>3?atoi(argv[3]):1024, iters=argc>4?atoi(argv[4]):5, check=argc>5?atoi(argv[5]):1; int nc=T/CH;
    size_t nqk=(size_t)HK*T*D, nv=(size_t)HV*T*D, ns=(size_t)HV*D*D;
    unsigned short *Qh=malloc(nqk*2),*Kh=malloc(nqk*2),*Vh=malloc(nv*2),*Oh=malloc(nv*2);
    float *Gf=malloc((size_t)HV*T*4),*Bf=malloc((size_t)HV*T*4),*S0=calloc(ns,4),*Sf=malloc(ns*4);
    double *Qp=malloc(nqk*8),*Kp=malloc(nqk*8),*Vp=malloc(nv*8); srand(42);
    for(size_t i=0;i<nqk;i++){ Qp[i]=((rand()%1000)/1000.0-0.5)*0.4; Kp[i]=((rand()%1000)/1000.0-0.5)*0.4; Qh[i]=f2h((float)Qp[i]); Kh[i]=f2h((float)Kp[i]); }
    for(size_t i=0;i<nv;i++){ Vp[i]=((rand()%1000)/1000.0-0.5)*0.4; Vh[i]=f2h((float)Vp[i]); }
    for(size_t i=0;i<(size_t)HV*T;i++){ Gf[i]=-0.3f*(rand()%1000)/1000.0f; Bf[i]=0.05f+0.9f*(rand()%1000)/1000.0f; }
    load_cl(); cl_platform_id pf; cl_uint np,nd; CK(my_clGetPlatformIDs(1,&pf,&np)); cl_device_id dev; CK(my_clGetDeviceIDs(pf,CL_DEVICE_TYPE_GPU,1,&dev,&nd)); cl_int err;
    cl_context ctx=my_clCreateContext(NULL,1,&dev,NULL,NULL,&err); CK(err); cl_command_queue q=my_clCreateCommandQueueWithProperties(ctx,dev,NULL,&err); CK(err);
    size_t sz1,sz2; char *src1=read_file(prep_path,&sz1), *src2=read_file(seq_path,&sz2); const char* srcs[2]={src1,src2}; size_t szs[2]={sz1,sz2};
    cl_program prog=my_clCreateProgramWithSource(ctx,2,srcs,szs,&err); CK(err); err=my_clBuildProgram(prog,1,&dev,"-cl-fast-relaxed-math",NULL,NULL);
    if(err){ size_t ln=0; my_clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,0,NULL,&ln); char *log=calloc(ln+1,1); my_clGetProgramBuildInfo(prog,dev,CL_PROGRAM_BUILD_LOG,ln,log,NULL); fprintf(stderr,"build failed:\n%s\n",log); return 3; }
    cl_kernel kp=my_clCreateKernel(prog,"gdn_prep_kernel_kernel",&err); CK(err); cl_kernel ks=my_clCreateKernel(prog,"gdn_seq_kernel_kernel",&err); CK(err);
    cl_mem bQ=mkbuf(ctx,CL_MEM_READ_ONLY,nqk*2), bK=mkbuf(ctx,CL_MEM_READ_ONLY,nqk*2), bV=mkbuf(ctx,CL_MEM_READ_ONLY,nv*2), bG=mkbuf(ctx,CL_MEM_READ_ONLY,(size_t)HV*T*4), bB=mkbuf(ctx,CL_MEM_READ_ONLY,(size_t)HV*T*4), bO=mkbuf(ctx,CL_MEM_WRITE_ONLY,nv*2), bS=mkbuf(ctx,CL_MEM_READ_WRITE,ns*4), bS0=mkbuf(ctx,CL_MEM_READ_ONLY,ns*4);
    cl_mem bU=mkbuf(ctx,CL_MEM_READ_WRITE,(size_t)HV*nc*CH*D*2), bW=mkbuf(ctx,CL_MEM_READ_WRITE,(size_t)HV*nc*CH*D*2), bKD=mkbuf(ctx,CL_MEM_READ_WRITE,(size_t)HV*nc*CH*D*2), bA2=mkbuf(ctx,CL_MEM_READ_WRITE,(size_t)HV*nc*CH*CH*2), bEgc=mkbuf(ctx,CL_MEM_READ_WRITE,(size_t)HV*nc*CH*4), bEgl=mkbuf(ctx,CL_MEM_READ_WRITE,(size_t)HV*nc*4);
    CK(my_clEnqueueWriteBuffer(q,bQ,1,0,nqk*2,Qh,0,NULL,NULL)); CK(my_clEnqueueWriteBuffer(q,bK,1,0,nqk*2,Kh,0,NULL,NULL)); CK(my_clEnqueueWriteBuffer(q,bV,1,0,nv*2,Vh,0,NULL,NULL)); CK(my_clEnqueueWriteBuffer(q,bG,1,0,(size_t)HV*T*4,Gf,0,NULL,NULL)); CK(my_clEnqueueWriteBuffer(q,bB,1,0,(size_t)HV*T*4,Bf,0,NULL,NULL)); CK(my_clEnqueueWriteBuffer(q,bS0,1,0,ns*4,S0,0,NULL,NULL));
#define ARG(k,i,m) CK(my_clSetKernelArg((k),(i),sizeof(cl_mem),&(m)))
    ARG(kp,0,bA2); ARG(kp,1,bB); ARG(kp,2,bEgc); ARG(kp,3,bEgl); ARG(kp,4,bG); ARG(kp,5,bK); ARG(kp,6,bKD); ARG(kp,7,bQ); ARG(kp,8,bU); ARG(kp,9,bV); ARG(kp,10,bW);
    ARG(ks,0,bA2); ARG(ks,1,bEgc); ARG(ks,2,bEgl); ARG(ks,3,bKD); ARG(ks,4,bO); ARG(ks,5,bQ); ARG(ks,6,bS); ARG(ks,7,bU); ARG(ks,8,bW);
#undef ARG
    size_t gp[2]={(size_t)nc*WG,HV}, gs[2]={4*WG,HV}, l[2]={WG,1};
#define RUN() do{ CK(my_clEnqueueCopyBuffer(q,bS0,bS,0,0,ns*4,0,NULL,NULL)); CK(my_clEnqueueNDRangeKernel(q,kp,2,NULL,gp,l,0,NULL,NULL)); CK(my_clEnqueueNDRangeKernel(q,ks,2,NULL,gs,l,0,NULL,NULL)); }while(0)
    RUN(); CK(my_clFinish(q)); double t0=now_s(); for(int it=0;it<iters;it++) RUN(); CK(my_clFinish(q)); double ms=(now_s()-t0)*1e3/iters;
#undef RUN
    CK(my_clEnqueueReadBuffer(q,bO,1,0,nv*2,Oh,0,NULL,NULL)); CK(my_clEnqueueReadBuffer(q,bS,1,0,ns*4,Sf,0,NULL,NULL)); printf("TL GDN: T=%d Hk=%d Hv=%d %.3f ms/iter\n",T,HK,HV,ms);
    if(!check) return 0; double max_rel=0,max_rel_s=0, dot=0,ng=0,nr=0; long bad=0; 
    for(int h=0;h<HV;h++){ int hk=h%HK; double *St=calloc((size_t)D*D,8); const double *qh=Qp+(size_t)hk*T*D,*kh=Kp+(size_t)hk*T*D,*vh=Vp+(size_t)h*T*D; for(int t=0;t<T;t++){ double a=exp(Gf[(size_t)h*T+t]), b=Bf[(size_t)h*T+t]; const double *kt=kh+(size_t)t*D,*vt=vh+(size_t)t*D,*qt=qh+(size_t)t*D; for(int dv=0;dv<D;dv++){ double sk=0; for(int dk=0;dk<D;dk++) sk+=St[(size_t)dk*D+dv]*kt[dk]; double delta=(vt[dv]-a*sk)*b; for(int dk=0;dk<D;dk++) St[(size_t)dk*D+dv]=a*St[(size_t)dk*D+dv]+delta*kt[dk]; } for(int dv=0;dv<D;dv++){ double acc=0; for(int dk=0;dk<D;dk++) acc+=qt[dk]*St[(size_t)dk*D+dv]; double got=h2f(Oh[((size_t)h*T+t)*D+dv]); double rel=fabs(got-acc)/(fabs(acc)+1e-1); if(rel>max_rel)max_rel=rel; if(rel>0.1)bad++; dot+=got*acc; ng+=got*got; nr+=acc*acc; }} for(size_t i=0;i<(size_t)D*D;i++){ double rel=fabs(Sf[(size_t)h*D*D+i]-St[i])/(fabs(St[i])+1e-1); if(rel>max_rel_s)max_rel_s=rel; } free(St); }
    printf("%s out max_rel %.4f bad %ld state max_rel %.4f cosine %.9f\n",(max_rel<0.1&&max_rel_s<0.1)?"OK":"FAIL",max_rel,bad,max_rel_s,dot/sqrt(ng*nr));
    return (max_rel<0.1&&max_rel_s<0.1)?0:1;
}
