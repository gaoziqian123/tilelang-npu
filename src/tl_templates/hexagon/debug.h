#pragma once
#include "common.h"
#include <stdio.h>

template <typename T> struct PrintTraits {
  static TL_DEVICE void print_var(const char *msg, T val) {
    printf("msg='%s' job=%d worker=%d value=%g\n", msg, tl_hex_job_id(), tl_hex_worker_id(), double(val));
  }
  static TL_DEVICE void print_buffer(const char *msg, const char *buf_name, int index, T val) {
    printf("msg='%s' job=%d worker=%d buffer=%s index=%d value=%g\n", msg, tl_hex_job_id(), tl_hex_worker_id(), buf_name, index, double(val));
  }
};
template <typename T> struct PrintTraits<T *> {
  static TL_DEVICE void print_var(const char *msg, T *val) {
    printf("msg='%s' job=%d worker=%d value=%p\n", msg, tl_hex_job_id(), tl_hex_worker_id(), (void *)val);
  }
  static TL_DEVICE void print_buffer(const char *msg, const char *buf_name, int index, T *val) {
    printf("msg='%s' buffer=%s index=%d value=%p\n", msg, buf_name, index, (void *)val);
  }
};
template <typename T> TL_DEVICE void debug_print_var(const char *msg, T var) { PrintTraits<T>::print_var(msg, var); }
template <typename T>
TL_DEVICE void debug_print_buffer_value(const char *msg, const char *buf_name, int index, T var) {
  PrintTraits<T>::print_buffer(msg, buf_name, index, var);
}
TL_DEVICE void debug_print_msg(const char *msg) {
  printf("msg='%s' job=%d worker=%d\n", msg, tl_hex_job_id(), tl_hex_worker_id());
}
