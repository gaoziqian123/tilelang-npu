#pragma once
#include <stdint.h>
#ifdef TL_HEX_REGION_PROFILE
extern "C" unsigned long long HAP_perf_get_time_us(void);
#endif
namespace tl {
// Explicit caller-owned DDR ledger, independent of kernel names and handles.
// [0]=last timestamp, [1]=active phase, [2+2*p]=us, [3+2*p]=entry count.
inline void profile_mark(uint64_t* ledger, int phase) {
#ifdef TL_HEX_REGION_PROFILE
  uint64_t now = HAP_perf_get_time_us();
  if (phase == -1) {
    ledger[0] = now; ledger[1] = 0;
    for (int i=2;i<34;i++) ledger[i]=0;
  } else {
    ledger[2+2*ledger[1]] += now-ledger[0];
    ledger[0]=now; ledger[1]=(uint64_t)phase;
    ledger[3+2*phase]++;
  }
#else
  (void)ledger; (void)phase;
#endif
}
}
#ifdef TL_HEX_REGION_PROFILE
extern "C" void tl_hex_region(int);
namespace tl { inline void profile_region(int region) { tl_hex_region(region); } }
#else
namespace tl { inline void profile_region(int) {} }
#endif
