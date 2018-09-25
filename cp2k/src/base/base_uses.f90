! Basic use statements and preprocessor macros
! should be included in the use statements

  USE base_hooks,                      ONLY: cp__a,&
                                             cp__b,&
                                             cp__w,&
                                             cp__l,&
                                             cp_abort,&
                                             cp_warn,&
                                             timeset,&
                                             timestop


! Dangerous: Full path can be arbitrarily long and might overflow Fortran line.
#if !defined(__SHORT_FILE__)
#define __SHORT_FILE__ __FILE__
#endif

#define __LOCATION__ cp__l(__SHORT_FILE__,__LINE__)
#define CPWARN(msg) CALL cp__w(__SHORT_FILE__,__LINE__,msg)
#define CPABORT(msg) CALL cp__b(__SHORT_FILE__,__LINE__,msg)

! CPASSERT appears to have side-effects, and it cannot be compiled out.
#if defined(NDEBUG) && 0
# define CPASSERT(cond)
#else
# define CPASSERT(cond) IF(.NOT.(cond))CALL cp__a(__SHORT_FILE__,__LINE__)
#endif
! CPXASSERT which is guaranteed to compile-out in release builds.
#if defined(NDEBUG)
# define CPXASSERT(cond)
#else
# define CPXASSERT(cond) CPASSERT(cond)
#endif

! The MARK_USED macro can be used to mark an argument/variable as used.
! It is intended to make it possible to switch on -Werror=unused-dummy-argument,
! but deal elegantly with e.g. library wrapper routines that take arguments only used if the library is linked in. 
! This code should be valid for any Fortran variable, is always standard conforming,
! and will be optimized away completely by the compiler
!
#define MARK_USED(foo) IF(.FALSE.)THEN; DO ; IF(SIZE(SHAPE(foo))==-1) EXIT ;  END DO ; ENDIF

! Calculate version number from 3-components.
#define CP_VERSION3(MAJOR, MINOR, UPDATE) ((MAJOR) * 10000 + (MINOR) * 100 + (UPDATE))
! Calculate version number from 4-components.
#define CP_VERSION4(MAJOR, MINOR, UPDATE, PATCH) ((MAJOR) * 100000000 + (MINOR) * 1000000 + (UPDATE) * 10000 + (PATCH))

! Evaluate the given symbol.
#define CP_EVAL(A) A
! Concatenate two symbols but expanding the arguments first.
#define CP_CONCATENATE(A, B) CP_EVAL(CP_EVAL(A) ## CP_EVAL(B))

! The CP_CONTIGUOUS macro may (or may not) expand to the CONTIGUOUS attribute
! depending on whether or not the compiler supports Fortran 2008. GNU Fortran
! needs -std=f2008 to make CONTIGUOUS available, hence the __F2008 symbol
! needs to be present (no way to detect standard by using the preprocessor).
!
#if !defined(CP_DISABLE_ATTRIBS) && defined(__F2008)
# if defined(__GFORTRAN__) && (CP_VERSION3(4, 6, 0) <= CP_VERSION3(__GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__))
#   define CP_CONTIGUOUS CONTIGUOUS
#   define CP_COMMA_CONTIGUOUS , CONTIGUOUS
# elif defined(__INTEL_COMPILER) && (1210 <= __INTEL_COMPILER)
#   define CP_CONTIGUOUS CONTIGUOUS
#   define CP_COMMA_CONTIGUOUS , CONTIGUOUS
# endif
#else
# define CP_CONTIGUOUS
# define CP_COMMA_CONTIGUOUS
#endif

! LIBXSMM added a FORTRAN-suitable header file with macro definitions.
! Allows macro-toggles in addition to parameters embedded into the module.
!
#if defined(__LIBXSMM)
# include <libxsmm_config.h>
#endif

#if defined(__MKL) || defined(__LIBXSMM)
# define CP_PURE
#else
# define CP_PURE PURE
#endif

