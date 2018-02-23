/*****************************************************************************
 *  CP2K: A general program to perform molecular dynamics simulations        *
 *  Copyright (C) 2000 - 2017  CP2K developers group                         *
 *****************************************************************************/

//! **************************************************************************
//!> \author Hans Pabst (Intel Corp.)
//! **************************************************************************

#if defined(__LIBXSMM) || (defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM))
#include "libxsmm_acc.h"

#if defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM)
# include <libxstream_begin.h>
#elif defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <iostream>
#include <cstdlib>
#include <cstring>
#if defined(_OPENMP)
# include <omp.h>
#endif
#if defined(CP2K_CONFIG_PREFIX)
# if (defined(__MKL) || defined(MKL_DIRECT_CALL_SEQ) || defined(MKL_DIRECT_CALL))
#   include <mkl_service.h>
# endif
# if defined(__TBBMALLOC)
#   include <tbb/scalable_allocator.h>
# endif
#endif
#if defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM)
# include <libxstream_end.h>
#elif defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif


#if defined(__LIBXSMM)
namespace libxsmm_acc_private {
  /** Internal type-agnostic call-forwarding to CP2K/intel stack processing; this is called by dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_[s|d]. */
  template<typename T>
  void process_mm_stack(const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
    const T* a, const T* b, T* c, LIBXSMM_ACC_FTYPE_LOGICAL* efficient)
  {
    int result = LIBXSMM_ACC_ERROR_CONDITION;
    if (0 != descriptor && 0 != params && 0 != myvalue && 0 != a && 0 != b && 0 != c) {
      result = libsmm_acc_process( // TODO: fix const-correctness in libsmm_acc.h
        params, *myvalue, LIBXSMM_ACC_NPARAMS, libxsmm_acc_elem<T,false>::type, const_cast<T*>(a), const_cast<T*>(b), c,
        descriptor->max_m, descriptor->max_n, descriptor->max_k, descriptor->defined_mnk, 0/*stream*/);
      if (efficient) *efficient = LIBXSMM_ACC_FTRUE;
    }
    switch (result) {
      case LIBXSMM_ACC_ERROR_CONDITION: LIBXSMM_ACC_ABORT("incorrect argument(s)"); break;
      default: if (LIBXSMM_ACC_ERROR_NONE != result) LIBXSMM_ACC_ABORT("unknown error");
    }
  }
  const char *const reconf_env = getenv("CP2K_RECONFIGURE");
  const bool reconfigure = ((reconf_env && *reconf_env) ? 0 != atoi(reconf_env) : true);
} // namespace libxsmm_acc_private


LIBXSMM_ACC_EXTERN void xsmm_acc_abort(const char* filename, int line_number, const char* message)
{
  if (filename && *filename) {
    std::cerr << filename << ':' << line_number << " - " << ((message && *message) ? message : "unknown error") << std::endl/*includes flush*/;
  }
  exit(-1);
}


#if defined(CP2K_CONFIG_PREFIX)
# if defined(__ELPA)
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(cp_fm_diag_mp_diag_init)(const char* diag_lib, LIBXSMM_ACC_FTYPE_LOGICAL* switched, const int* k_elpa,
  const LIBXSMM_ACC_FTYPE_LOGICAL* elpa_qr, const LIBXSMM_ACC_FTYPE_LOGICAL* elpa_print, const LIBXSMM_ACC_FTYPE_LOGICAL* elpa_qr_unsafe,
  int diag_lib_strlen);
# endif

void libxsmm_acc_reconfigure()
{
  extern libxsmm_acc_dbcsr_config_type LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(CP2K_CONFIG_PREFIX, dbcsr_cfg));
  libxsmm_acc_dbcsr_config_type& dbcsr_cfg = LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(CP2K_CONFIG_PREFIX, dbcsr_cfg));
  static bool once = false; // allow to check for multiple reconfigurations

#if defined(__TBBMALLOC)
  if (!once) {
    const char *const env_hugepages = getenv("CP2K_HUGEPAGES");
    if (0 == env_hugepages || 0 == *env_hugepages || 0 != atoi(env_hugepages)) {
      scalable_allocation_mode(TBBMALLOC_USE_HUGE_PAGES, 1);
    }
  }
#endif

#if defined(_OPENMP) && defined(KMP_VERSION_MAJOR) && 0/*disabled*/
  if (!once) {
    // setting the stacksize applies independent of nested parallelism (LIBXSMM_ACC_OPENMP)
    kmp_set_stacksize(52428800);
  }
#endif

#if defined(MKL_ENABLE_AVX512)
  if (!once) {
    mkl_enable_instructions(MKL_ENABLE_AVX512);
  }
#endif

#if defined(__LIBXSMM)
  if (!once) {
    libxsmm_init();
# if LIBXSMM_VERSION4(1, 6, 1, 83) <= LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH)
#   if LIBXSMM_VERSION4(1, 8, 3, 74) <= LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH)
    libxsmm_set_gemm_auto_prefetch(LIBXSMM_X86_AVX512_MIC != libxsmm_get_target_archid() ? LIBXSMM_GEMM_PREFETCH_AL2BL2_VIA_C : LIBXSMM_GEMM_PREFETCH_BL2_VIA_C);
#   else
    libxsmm_set_gemm_auto_prefetch(LIBXSMM_X86_AVX512_MIC != libxsmm_get_target_archid() ? LIBXSMM_PREFETCH_AL2BL2_VIA_C : LIBXSMM_PREFETCH_BL2_VIA_C);
#   endif
# endif
# if LIBXSMM_VERSION4(1, 6, 2, 4) <= LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH)
    libxsmm_set_dispatch_trylock(1);
# endif
  }
#endif

#if defined(__ELPA)
  if (!once) {
    const char *const env_elpa = getenv("CP2K_ELPA"), *const env_qr = getenv("CP2K_ELPA_QR");
    const int elpa = (0 == env_elpa || 0 == *env_elpa) ? 3/*block6*/ : atoi(env_elpa);
    const int qr = (0 == env_qr || 0 == *env_qr) ? 0/*disable*/ : atoi(env_qr);

    const LIBXSMM_ACC_FTYPE_LOGICAL elpa_qr_unsafe = ((0 != elpa && (1 < qr || -1 > qr)) ? LIBXSMM_ACC_FTRUE : LIBXSMM_ACC_FALSE);
    const LIBXSMM_ACC_FTYPE_LOGICAL elpa_print = ((0 != elpa && 0 <= qr) ? LIBXSMM_ACC_FALSE : LIBXSMM_ACC_FTRUE);
    const LIBXSMM_ACC_FTYPE_LOGICAL elpa_qr = ((0 != elpa && 0 != qr) ? LIBXSMM_ACC_FTRUE : LIBXSMM_ACC_FALSE);
    LIBXSMM_ACC_FTYPE_LOGICAL switched = LIBXSMM_ACC_FALSE;
    int k_elpa = 1; // auto

    if (0 != elpa) {
      static const char *const diag_lib = "ELPA";
# if LIBXSMM_VERSION4(1, 6, 3, 64) <= LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR, LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH)
      if (0 < elpa) {
#   if (201611 <= __ELPA)
        const int cpuid = LIBXSMM_MIN(libxsmm_cpuid(), LIBXSMM_X86_AVX512);
#   else // ensure to not request kernels beyond AVX2 with older ELPA
        const int cpuid = LIBXSMM_MIN(libxsmm_cpuid(), LIBXSMM_X86_AVX2);
#   endif
        if (LIBXSMM_X86_SSE3 <= cpuid) {
          const int k_elpa_base[] = { 6/*SSE3*/, 6/*SSE4*/, 9/*AVX*/, 12/*AVX2*/, 15/*AVX512*/ };
          const int block = LIBXSMM_MIN(elpa - 1, 2/*block6*/);
          k_elpa = k_elpa_base[cpuid-(LIBXSMM_X86_SSE3)] + block;
        }
      }
# endif
      LIBXSMM_ACC_FSYMBOL(cp_fm_diag_mp_diag_init)(diag_lib, &switched, &k_elpa,
        &elpa_qr, &elpa_print, &elpa_qr_unsafe,
        strlen(diag_lib));
    }
    else if (0 != isalpha(*env_elpa)) {
      LIBXSMM_ACC_FSYMBOL(cp_fm_diag_mp_diag_init)(env_elpa, &switched, &k_elpa,
        &elpa_qr, &elpa_print, &elpa_qr_unsafe,
        strlen(env_elpa));
    }
  }
#endif

  // better leave "CP2K_DRIVER" environment variable undocumented
  // variable takes the internal literal/number representing MM driver
  const char *const env_driver = getenv("CP2K_DRIVER");
  if (env_driver && *env_driver) {
    dbcsr_cfg.mm_driver = atoi(env_driver);
  }

  const char *const env_stacksize = getenv("CP2K_STACKSIZE");
  if (env_stacksize && *env_stacksize) {
    const int value = atoi(env_stacksize);
    if (0 < value) {
      dbcsr_cfg.mm_stack_size = value;
    }
#if defined(LIBXSMM_ACC_STACKSIZE)
    else {
      dbcsr_cfg.mm_stack_size = LIBXSMM_ACC_STACKSIZE;
    }
#endif
  }

  int rma = 0;
#if defined(__MPI_VERSION) && (3 <= __MPI_VERSION)
  const char *const env_rma = getenv("CP2K_RMA");
  if (env_rma && *env_rma) {
    rma = atoi(env_rma);
  }
  if (0 != rma) {
    dbcsr_cfg.use_mpi_rma = LIBXSMM_ACC_FTRUE;
    dbcsr_cfg.use_comm_thread = LIBXSMM_ACC_FALSE;
    dbcsr_cfg.comm_thread_load = 100;
  }
  else
#endif
  {
    const char *const env_commtload = getenv("CP2K_COMMTLOAD");
    if (env_commtload && *env_commtload) {
      const int value = atoi(env_commtload);
      if (0 != value) {
        if (0 < value) {
          dbcsr_cfg.comm_thread_load = value;
        }
#if defined(LIBXSMM_ACC_COMM_THREAD_LOAD)
        else {
# if (0 < LIBXSMM_ACC_COMM_THREAD_LOAD)
          dbcsr_cfg.comm_thread_load = LIBXSMM_ACC_COMM_THREAD_LOAD;
# else
          dbcsr_cfg.use_comm_thread = LIBXSMM_ACC_FALSE;
# endif
        }
#endif
        dbcsr_cfg.use_comm_thread = LIBXSMM_ACC_FTRUE;
      }
      else {
        dbcsr_cfg.use_comm_thread = LIBXSMM_ACC_FALSE;
      }
    }
#if defined(LIBXSMM_ACC_COMM_THREAD_LOAD) && 0/*disabled*/
    else {
# if (0 < LIBXSMM_ACC_COMM_THREAD_LOAD)
      dbcsr_cfg.comm_thread_load = LIBXSMM_ACC_COMM_THREAD_LOAD;
      dbcsr_cfg.use_comm_thread = LIBXSMM_ACC_FTRUE;
# else
      dbcsr_cfg.use_comm_thread = LIBXSMM_ACC_FALSE;
# endif
    }
#endif
  }

  const char *const env_multrec = getenv("CP2K_MULTREC");
  if (env_multrec && *env_multrec) {
    const int value = atoi(env_multrec);
    if (0 < value) {
      dbcsr_cfg.multrec_limit = value;
    }
#if defined(LIBXSMM_ACC_MULTREC_LIMIT) && (0 < LIBXSMM_ACC_MULTREC_LIMIT)
    else {
      dbcsr_cfg.multrec_limit = LIBXSMM_ACC_MULTREC_LIMIT;
    }
#endif
  }
#if defined(LIBXSMM_ACC_MULTREC_LIMIT) && (0 < LIBXSMM_ACC_MULTREC_LIMIT) && 0/*disabled*/
  else {
    dbcsr_cfg.multrec_limit = LIBXSMM_ACC_MULTREC_LIMIT;
  }
#endif

#if defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM)
# if defined(LIBXSMM_ACC_ACCDRV_POSTERIOR_STREAMS)
  dbcsr_cfg.accdrv_posterior_streams = LIBXSMM_ACC_ACCDRV_POSTERIOR_STREAMS;
# endif
# if defined(LIBXSMM_ACC_ACCDRV_POSTERIOR_BUFFERS)
  dbcsr_cfg.accdrv_posterior_buffers = LIBXSMM_ACC_ACCDRV_POSTERIOR_BUFFERS;
# endif
# if defined(LIBXSMM_ACC_ACCDRV_PRIORITY_STREAMS)
  dbcsr_cfg.accdrv_priority_streams = LIBXSMM_ACC_ACCDRV_PRIORITY_STREAMS;
# endif
# if defined(LIBXSMM_ACC_ACCDRV_PRIORITY_BUFFERS)
  dbcsr_cfg.accdrv_priority_buffers = LIBXSMM_ACC_ACCDRV_PRIORITY_BUFFERS;
# endif
# if defined(LIBXSMM_ACC_ACCDRV_MIN_NFLOPS_PERMM)
  dbcsr_cfg.accdrv_min_flop_process = LIBXSMM_ACC_ACCDRV_MIN_NFLOPS_PERMM;
# endif
#endif

  // check for multiple reconfigurations
  if (!once) once = true;
}


#if defined(__GNUC__)
LIBXSMM_ACC_EXTERN LIBXSMM_ACC_ATTRIBUTE(weak)
#else
LIBXSMM_ACC_EXTERN
#endif
void LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(LIBXSMM_ACC_CONCATENATE(__real_, CP2K_CONFIG_PREFIX), dbcsr_set_config))(LIBXSMM_ACC_CONFIG_SIGNATURE_SETTER);
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(LIBXSMM_ACC_CONCATENATE(__wrap_, CP2K_CONFIG_PREFIX), dbcsr_set_config))(LIBXSMM_ACC_CONFIG_SIGNATURE_SETTER)
{
  LIBXSMM_ACC_FSYMBOL(LIBXSMM_ACC_CONCATENATE(LIBXSMM_ACC_CONCATENATE(__real_, CP2K_CONFIG_PREFIX), dbcsr_set_config))(
    mm_driver, LIBXSMM_ACC_CONFIG_SIGNATURE_USE, mm_driver_strlen); // call original dbcsr_set_config

  if (libxsmm_acc_private::reconfigure) {
    libxsmm_acc_reconfigure(); // override settings
  }
}


LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_s)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const float* a, const float* b, float* c, LIBXSMM_ACC_FTYPE_LOGICAL* efficient);
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(xsmm_acc_process_mm_stack_s)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const float* a, const float* b, float* c, LIBXSMM_ACC_FTYPE_LOGICAL* efficient)
{
  if (libxsmm_acc_private::reconfigure) {
    libxsmm_acc_private::process_mm_stack(descriptor, params, myvalue, a, b, c, efficient);
  }
  else { /* CP2K/trunk/master code path */
    LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_s)(
      descriptor, params, myvalue, a, b, c, efficient);
  }
}


LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_d)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const double* a, const double* b, double* c, LIBXSMM_ACC_FTYPE_LOGICAL* efficient);
LIBXSMM_ACC_EXTERN void LIBXSMM_ACC_FSYMBOL(xsmm_acc_process_mm_stack_d)(
  const libxsmm_acc_stackdesc_type* descriptor, /*const*/ int* params, const int* myvalue,
  const double* a, const double* b, double* c, LIBXSMM_ACC_FTYPE_LOGICAL* efficient)
{
  if (libxsmm_acc_private::reconfigure) {
    libxsmm_acc_private::process_mm_stack(descriptor, params, myvalue, a, b, c, efficient);
  }
  else { /* CP2K/trunk/master code path */
    LIBXSMM_ACC_FSYMBOL(dbcsr_mm_hostdrv_mp_xsmm_process_mm_stack_d)(
      descriptor, params, myvalue, a, b, c, efficient);
  }
}

#endif // defined(CP2K_CONFIG_PREFIX)

#endif // defined(__LIBXSMM)
#endif // defined(__LIBXSMM) || (defined(__ACC) && defined(__ACC_MIC) && defined(__DBCSR_ACC) && defined(__LIBXSTREAM))
