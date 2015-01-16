/******************************************************************************
** Copyright (c) 2014-2015, Intel Corporation                                **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#include <libxstream_test.hpp>
#include <libxstream.hpp>
#include <algorithm>
#include <limits>
#include <atomic>

#if defined(LIBXSTREAM_OFFLOAD)
# pragma offload_attribute(push,target(mic))
#endif
#include <cstring>
#if !defined(_WIN32)
# include <sys/mman.h>
# include <unistd.h>
#endif
#if defined(LIBXSTREAM_OFFLOAD)
# pragma offload_attribute(pop)
#endif

#if defined(__MKL)
# include <mkl.h>
#endif

#if defined(LIBXSTREAM_OFFLOAD)
# include <offload.h>
#endif

#if defined(_WIN32)
# include <windows.h>
#endif


namespace libxstream_internal {

unsigned int       abs(unsigned int        a)  { return a; }
unsigned long      abs(unsigned long       a)  { return a; }
unsigned long long abs(unsigned long long  a)  { return a; }

template<typename T>
inline T gcd(T a, T b)
{
  while (0 != b) {
    const T r = a % b;
    a = b;
    b = r;
  }
  return a;
}


template<typename T>
inline T lcm(T a, T b)
{
  //using libxstream_internal::abs;
  using std::abs;
  return abs(a * b) / gcd(a, b);
}


template<typename T>
inline T auto_alignment(T size)
{
  const T min_a = std::min(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
  const T max_a = lcm(LIBXSTREAM_MAX_SIMD, LIBXSTREAM_MAX_ALIGN);
  return max_a <= size ? max_a : (min_a < size ? min_a : sizeof(void*));
}


int allocate_real(unsigned char** memory, size_t size)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
#if defined(LIBXSTREAM_DEBUG)
      if (unsigned char *const buffer = new unsigned char[size]) {
        std::fill_n(buffer, size, 0);
        *memory = buffer;
      }
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
#elif defined(__MKL)
      unsigned char *const buffer = static_cast<unsigned char*>(mkl_malloc(size, static_cast<int>(libxstream_internal::auto_alignment(size))));
# if defined(LIBXSTREAM_CHECK)
      if (0 != buffer)
# endif
      {
        *memory = buffer;
      }
# if defined(LIBXSTREAM_CHECK)
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# endif
#elif defined(_WIN32)
      void *const buffer = _aligned_malloc(size, libxstream_internal::auto_alignment(size));
# if defined(LIBXSTREAM_CHECK)
      if (0 != buffer)
# endif
      {
        *memory = buffer;
      }
# if defined(LIBXSTREAM_CHECK)
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# endif
#elif defined(__GNUC__)
      unsigned char *const buffer = static_cast<unsigned char*>(_mm_malloc(size, static_cast<int>(libxstream_internal::auto_alignment(size))));
# if defined(LIBXSTREAM_CHECK)
      if (0 != buffer)
# endif
      {
        *memory = buffer;
      }
# if defined(LIBXSTREAM_CHECK)
      else {
        result = LIBXSTREAM_ERROR_RUNTIME;
      }
# endif
#else
# if defined(LIBXSTREAM_CHECK)
      result = (0 == posix_memalign(memory, libxstream_internal::auto_alignment(size), size) && 0 != *memory)
# else
      result = (0 == posix_memalign(memory, libxstream_internal::auto_alignment(size), size))
# endif
        ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#endif
    }
    else {
      *memory = 0;
    }
  }
#if defined(LIBXSTREAM_CHECK)
  else if (0 != size) {
    result = LIBXSTREAM_ERROR_CONDITION;
  }
#endif

  return result;
}


int deallocate_real(const unsigned char* memory)
{
  if (memory) {
#if defined(LIBXSTREAM_DEBUG)
    delete[] memory;
#elif defined(__MKL)
    mkl_free(const_cast<unsigned char*>(memory));
#elif defined(_WIN32)
    _aligned_free(const_cast<unsigned char*>(memory));
#elif defined(__GNUC__)
    _mm_free(const_cast<unsigned char*>(memory));
#else
    free(memory);
#endif
  }

  return LIBXSTREAM_ERROR_NONE;
}


unsigned char* vdata(unsigned char* memory)
{
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
  return memory;
#elif defined(_WIN32)
  return memory;
#else
  return memory + sizeof(size_t);
#endif
}


const unsigned char* vdata(const unsigned char* memory)
{
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
  return memory;
#elif defined(_WIN32)
  return memory;
#else
  return memory + sizeof(size_t);
#endif
}


int allocate_virtual(unsigned char** memory, size_t size, const void* data, size_t data_size = 0)
{
  LIBXSTREAM_CHECK_CONDITION(0 == data_size || (0 != data && data_size <= size));
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
    if (0 < size) {
      unsigned char* dst = 0;
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
      result = allocate_real(memory, size);
      LIBXSTREAM_CHECK_ERROR(result);
      dst = vdata(*memory);
#elif defined(_WIN32)
      void *const buffer = VirtualAlloc(0, size, MEM_RESERVE, PAGE_NOACCESS);
      LIBXSTREAM_CHECK_CONDITION(0 != buffer);
      dst = vdata(static_cast<unsigned char*>(VirtualAlloc(buffer, data_size, MEM_COMMIT, PAGE_READWRITE)));
      *memory = static_cast<unsigned char*>(buffer);
#else
      unsigned char *const buffer = static_cast<unsigned char*>(mmap(0, size, PROT_READ | PROT_WRITE/*PROT_NONE*/, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
      LIBXSTREAM_CHECK_CONDITION(MAP_FAILED != buffer);
      dst = vdata(buffer);
      LIBXSTREAM_ASSERT(sizeof(size) <= dst - buffer);
      *reinterpret_cast<size_t*>(buffer) = size;
      *memory = buffer;
#endif
      if (0 < data_size && 0 != data) {
        LIBXSTREAM_CHECK_CONDITION(0 != dst);
        const unsigned char *const src = static_cast<const unsigned char*>(data);
        for (size_t i = 0; i < data_size; ++i) dst[i] = src[i];
      }
    }
    else {
      *memory = 0;
    }
  }
#if defined(LIBXSTREAM_CHECK)
  else if (0 != size) {
    result = LIBXSTREAM_ERROR_CONDITION;
  }
#endif

  return result;
}


int deallocate_virtual(const unsigned char* memory)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
#if !defined(LIBXSTREAM_OFFLOAD) || defined(LIBXSTREAM_DEBUG)
    result = deallocate_real(memory);
#elif defined(_WIN32)
    result = FALSE != VirtualFree(const_cast<unsigned char*>(memory), 0, MEM_RELEASE) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#else
    const size_t size = *reinterpret_cast<const size_t*>(memory);
    result = 0 == munmap(const_cast<unsigned char*>(memory), size) ? LIBXSTREAM_ERROR_NONE : LIBXSTREAM_ERROR_RUNTIME;
#endif
  }

  return result;
}


class dev_info_type {
public:
  static std::atomic<int>& counter() {
    return m_counter;
  }

  static int& global_device() {
    return m_global_device;
  }

  // store the active device per host-thread
  static int& device() {
    static LIBXSTREAM_TLS int instance = -2;
    return instance;
  }

private:
  static std::atomic<int> m_counter;
  static int m_global_device;
} dev_info;

/*static*/std::atomic<int> dev_info_type::m_counter(0);
/*static*/int dev_info_type::m_global_device = -2;


LIBXSTREAM_EXPORT void mem_info(uint64_t& memory_physical, uint64_t& memory_not_allocated)
{
#if defined(_WIN32)
  MEMORYSTATUSEX status;
  status.dwLength = sizeof(status);
  const BOOL ok = GlobalMemoryStatusEx(&status);
  if (TRUE == ok) {
    memory_not_allocated = status.ullAvailPhys;
    memory_physical = status.ullTotalPhys;
  }
  else {
    memory_not_allocated = 0;
    memory_physical = 0;
  }
#else
  const long memory_pages_size = sysconf(_SC_PAGE_SIZE);
  const long memory_pages_phys = sysconf(_SC_PHYS_PAGES);
  const long memory_pages_avail = sysconf(_SC_AVPHYS_PAGES);
  memory_not_allocated = memory_pages_size * memory_pages_avail;
  memory_physical = memory_pages_size * memory_pages_phys;
#endif
}

} // namespace libxstream_internal


extern "C" int libxstream_get_ndevices(size_t* ndevices)
{
  LIBXSTREAM_CHECK_CONDITION(ndevices);

#if defined(LIBXSTREAM_OFFLOAD) && !defined(__MIC__)
  *ndevices = std::min(_Offload_number_of_devices(), LIBXSTREAM_MAX_DEVICES);
#else
  *ndevices = 1; // host
#endif

#if defined(LIBXSTREAM_DEBUG) && defined(LIBXSTREAM_MIC_STDTHREAD)
  static LIBXSTREAM_TLS bool print = true;
  if (print) {
    const std::thread::id id = std::this_thread::get_id();
    fprintf(stderr, "DBG libxstream_get_ndevices: ndevices=%lu thread=0x%lx\n",
      static_cast<unsigned long>(*ndevices),
      static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(&id)));
    print = false;
  }
#endif

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_get_active_device(int* device)
{
  LIBXSTREAM_CHECK_CONDITION(0 != device);
  int result = LIBXSTREAM_ERROR_NONE, active_device = libxstream_internal::dev_info_type::device();

  if (-1 > active_device) {
#if defined(LIBXSTREAM_DEBUG) && defined(LIBXSTREAM_MIC_STDTHREAD)
    const std::thread::id id = std::this_thread::get_id();
    fprintf(stderr, "DBG libxstream_get_active_device: device id not set for thread=0x%lx\n",
      static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(&id)));
#endif
#if defined(LIBXSTREAM_CHECK)
    result = LIBXSTREAM_ERROR_CONDITION;
#else
    active_device = libxstream_internal::dev_info_type::global_device();
    libxstream_internal::dev_info_type::device() = active_device;
#endif
  }

  *device = active_device;
  return result;
}


extern "C" int libxstream_set_active_device(int device)
{
  size_t ndevices = LIBXSTREAM_MAX_DEVICES;
  LIBXSTREAM_CHECK_CONDITION(-1 <= device && ndevices >= static_cast<size_t>(device + 1) && LIBXSTREAM_ERROR_NONE == libxstream_get_ndevices(&ndevices) && ndevices >= static_cast<size_t>(device + 1));

  if (-1 > libxstream_internal::dev_info_type::global_device() && 1 == ++libxstream_internal::dev_info_type::counter()) {
    libxstream_internal::dev_info_type::global_device() = device;
  }

  libxstream_internal::dev_info_type::device() = device;

#if defined(LIBXSTREAM_DEBUG) && defined(LIBXSTREAM_MIC_STDTHREAD)
  const std::thread::id id = std::this_thread::get_id();
  fprintf(stderr, "DBG libxstream_set_active_device: device=%i thread=0x%lx\n",
    device, static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(&id)));
#endif

#if defined(LIBXSTREAM_TEST) && (0 != (2*LIBXSTREAM_TEST+1)/2)
  const libxstream_test test;
#endif

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_priority_range(int* least, int* greatest)
{
  *least = -1;
  *greatest = -1;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_create(libxstream_stream** stream, int device, int priority, const char* name)
{
  LIBXSTREAM_CHECK_CONDITION(stream);
  libxstream_stream *const s = new libxstream_stream(device, priority, name);
  LIBXSTREAM_ASSERT(s);
  *stream = s;

#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_stream_create: stream=0x%lx name=\"%s\"\n",
    static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(stream)),
    name ? name : "");
#endif

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_stream_destroy(libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  if (0 != stream) {
    fprintf(stderr, "DBG libxstream_stream_destroy: stream=0x%lx name=\"%s\"\n",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)),
      stream->name());
  }
#endif

  const int result = stream ? stream->reset() : LIBXSTREAM_ERROR_NONE;
  delete stream;
  return result;
}


extern "C" int libxstream_stream_sync(libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  if (0 != stream) {
    fprintf(stderr, "DBG libxstream_stream_sync: stream=0x%lx name=\"%s\"\n",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)),
      stream->name());
  }
  else {
    fprintf(stderr, "DBG libxstream_stream_sync: synchronize all streams\n");
  }
#endif

  int result = LIBXSTREAM_ERROR_NONE;

  if (stream) {
    result = stream->reset();
    stream->wait(0);
  }
  else {
    libxstream_stream::sync();
  }

  return result;
}


extern "C" int libxstream_stream_wait_event(libxstream_stream* stream, libxstream_event* event)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_stream_wait_event: event=0x%lx stream=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));
#endif

  int result = LIBXSTREAM_ERROR_NONE;

  if (event) {
    if (stream) result = stream->reset();
    event->wait(stream);
  }
  else {
    if (stream) {
      result = stream->reset();
      stream->wait(0);
    }
    else {
      libxstream_stream::sync();
    }
  }

  return result;
}


extern "C" int libxstream_event_create(libxstream_event** event)
{
  LIBXSTREAM_CHECK_CONDITION(event);
  *event = new libxstream_event;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_event_destroy(libxstream_event* event)
{
  delete event;
  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_event_record(libxstream_event* event, libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_event_record: event=0x%lx stream=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));
#endif
  LIBXSTREAM_CHECK_CONDITION(event);
  int result = LIBXSTREAM_ERROR_NONE;

  if (stream) {
    result = stream->reset();
    event->enqueue(*stream, true);
  }
  else {
    libxstream_stream::enqueue(*event);
  }

  return result;
}


extern "C" int libxstream_event_query(const libxstream_event* event, int* has_occured)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_event_query: event=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)));
#endif
  LIBXSTREAM_CHECK_CONDITION(event && has_occured);

  bool occurred = true;
  event->query(occurred, 0);
  *has_occured = occurred ? 1 : 0;

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_event_synchronize(libxstream_event* event)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_event_synchronize: event=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(event)));
#endif
  if (event) {
    event->wait(0);
  }
  else {
    libxstream_stream::sync();
  }

  return LIBXSTREAM_ERROR_NONE;
}


extern "C" int libxstream_mem_info(int device, size_t* allocatable, size_t* physical)
{
  LIBXSTREAM_CHECK_CONDITION(allocatable || physical);
  uint64_t memory_physical = 0, memory_allocatable = 0;
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_OFFLOAD_BEGIN(0, device, &memory_physical, &memory_allocatable)
  {
    uint64_t& memory_physical = *ptr<uint64_t,1>();
    uint64_t& memory_allocatable = *ptr<uint64_t,2>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
#     pragma offload target(mic:LIBXSTREAM_OFFLOAD_DEVICE) //out(memory_physical, memory_allocatable)
      {
        libxstream_internal::mem_info(memory_physical, memory_allocatable);
      }
    }
    else
#endif
    {
      libxstream_internal::mem_info(memory_physical, memory_allocatable);
    }
  }
  LIBXSTREAM_OFFLOAD_END(true)

#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_mem_info: allocatable=%lu physical=%lu\n",
    static_cast<unsigned long>(memory_allocatable),
    static_cast<unsigned long>(memory_physical));
#endif
  LIBXSTREAM_CHECK_CONDITION(0 < memory_physical && 0 < memory_allocatable);

  if (allocatable) {
    *allocatable = memory_allocatable;
  }

  if (physical) {
    *physical = memory_physical;
  }

  return result;
}


extern "C" int libxstream_mem_allocate(int device, unsigned char** memory, size_t size, size_t /*TODO: alignment*/)
{
  int result = LIBXSTREAM_ERROR_NONE;

  LIBXSTREAM_OFFLOAD_BEGIN(0, device, memory, size, &result)
  {
    unsigned char*& memory = *ptr<unsigned char*,1>();
    const size_t size = val<const size_t,2>();
    int& result = *ptr<int,3>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      const int device = LIBXSTREAM_OFFLOAD_DEVICE;
      result = libxstream_internal::allocate_virtual(&memory, size, &device, sizeof(device));

      if (LIBXSTREAM_ERROR_NONE == result && 0 != memory) {
        unsigned char *const buffer = memory;
#       pragma offload_transfer target(mic:LIBXSTREAM_OFFLOAD_DEVICE) nocopy(buffer: length(size) alloc_if(true))
      }
    }
    else
#endif
    {
      result = libxstream_internal::allocate_real(&memory, size);
    }
  }
  LIBXSTREAM_OFFLOAD_END(true)

#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_mem_allocate: buffer=0x%lx size=%lu\n",
    memory ? static_cast<unsigned long>(*reinterpret_cast<const uintptr_t*>(memory)) : 0UL,
    static_cast<unsigned long>(size));
#endif

  return result;
}


extern "C" int libxstream_mem_deallocate(int device, const unsigned char* memory)
{
  int result = LIBXSTREAM_ERROR_NONE;

  if (memory) {
#if defined(LIBXSTREAM_DEBUG)
    fprintf(stderr, "DBG libxstream_mem_deallocate: buffer=0x%lx\n",
      static_cast<unsigned long>(reinterpret_cast<uintptr_t>(memory)));
#endif

    LIBXSTREAM_OFFLOAD_BEGIN(0, device, memory, &result)
    {
      const unsigned char *const memory = ptr<const unsigned char,1>();
      int& result = *ptr<int,2>();

#if defined(LIBXSTREAM_OFFLOAD)
      if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
# if defined(LIBXSTREAM_CHECK)
        const int memory_device = *reinterpret_cast<const int*>(libxstream_internal::vdata(memory));
        if (memory_device != LIBXSTREAM_OFFLOAD_DEVICE) {
#   if defined(LIBXSTREAM_DEBUG)
          fprintf(stderr, "\twarning: device %i does not match allocating device %i\n", LIBXSTREAM_OFFLOAD_DEVICE, memory_device);
#   endif
          LIBXSTREAM_OFFLOAD_DEVICE_UPDATE(memory_device);
        }
# endif
#       pragma offload_transfer target(mic:LIBXSTREAM_OFFLOAD_DEVICE) nocopy(memory: length(0) free_if(true))
        result = libxstream_internal::deallocate_virtual(memory);
      }
      else
#endif
      {
        result = libxstream_internal::deallocate_real(memory);
      }
    }
    LIBXSTREAM_OFFLOAD_END(true)
  }

  return result;
}


extern "C" int libxstream_memset_zero(unsigned char* memory, size_t size, libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_memset_zero: buffer=0x%lx size=%lu stream=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(memory)),
    static_cast<unsigned long>(size),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));
#endif
  LIBXSTREAM_CHECK_CONDITION(memory && stream);
  const int result = stream->reset();

  LIBXSTREAM_OFFLOAD_BEGIN(stream, memory, size)
  {
    unsigned char* dst = ptr<unsigned char,0>();
    const size_t size = val<const size_t,1>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(size) out(dst: length(0) alloc_if(false) free_if(false))
        {
          memset(dst, 0, size);
        }
      }
      else {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(size) out(dst: length(0) alloc_if(false) free_if(false))
        {
          memset(dst, 0, size);
        }
      }
    }
    else
#endif
    {
      memset(dst, 0, size);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false)

  return result;
}


extern "C" int libxstream_memcpy_h2d(const unsigned char* host_mem, unsigned char* dev_mem, size_t size, libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_memcpy_h2d: 0x%lx->0x%lx size=%lu stream=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(host_mem)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dev_mem)),
    static_cast<unsigned long>(size),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));
#endif
  LIBXSTREAM_CHECK_CONDITION(host_mem && dev_mem && stream);
  const int result = stream->reset();

  LIBXSTREAM_OFFLOAD_BEGIN(stream, host_mem, dev_mem, size)
  {
    const unsigned char *const src = ptr<const unsigned char,0>();
    unsigned char *const dst = ptr<unsigned char,1>();
    const size_t size = val<const size_t,2>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
      else {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
    }
    else
#endif
    {
      std::copy(src, src + size, dst);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false)

  return result;
}


extern "C" int libxstream_memcpy_d2h(const unsigned char* dev_mem, unsigned char* host_mem, size_t size, libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_memcpy_d2h: 0x%lx->0x%lx size=%lu stream=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dev_mem)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(host_mem)),
    static_cast<unsigned long>(size),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));
#endif
  LIBXSTREAM_CHECK_CONDITION(dev_mem && host_mem && stream);
  const int result = stream->reset();

  LIBXSTREAM_OFFLOAD_BEGIN(stream, dev_mem, host_mem, size)
  {
    const unsigned char* src = ptr<const unsigned char,0>();
    unsigned char *const dst = ptr<unsigned char,1>();
    const size_t size = val<const size_t,2>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          out(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
      else {
#       pragma offload_transfer LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          out(src: length(size) into(dst) alloc_if(false) free_if(false))
      }
    }
    else
#endif
    {
      std::copy(src, src + size, dst);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false)

  return result;
}


extern "C" int libxstream_memcpy_d2d(const unsigned char* src, unsigned char* dst, size_t size, libxstream_stream* stream)
{
#if defined(LIBXSTREAM_DEBUG)
  fprintf(stderr, "DBG libxstream_memcpy_d2d: 0x%lx->0x%lx size=%lu stream=0x%lx\n",
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(src)),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(dst)),
    static_cast<unsigned long>(size),
    static_cast<unsigned long>(reinterpret_cast<uintptr_t>(stream)));
#endif
  LIBXSTREAM_CHECK_CONDITION(src && dst && stream);
  const int result = stream->reset();

  LIBXSTREAM_OFFLOAD_BEGIN(stream, src, dst, size)
  {
    const uint64_t *const src = ptr<const uint64_t,0>();
    uint64_t* dst = ptr<uint64_t,1>();
    const size_t size = val<const size_t,2>();

#if defined(LIBXSTREAM_OFFLOAD)
    if (0 <= LIBXSTREAM_OFFLOAD_DEVICE) {
      // TODO: implement cross-device transfer

      if (LIBXSTREAM_OFFLOAD_READY) {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_SIGNAL \
          in(size) in(src: length(0) alloc_if(false) free_if(false)) out(dst: length(0) alloc_if(false) free_if(false))
        memcpy(dst, src, size);
      }
      else {
#       pragma offload LIBXSTREAM_OFFLOAD_TARGET_WAIT \
          in(size) in(src: length(0) alloc_if(false) free_if(false)) out(dst: length(0) alloc_if(false) free_if(false))
        memcpy(dst, src, size);
      }
    }
    else
#endif
    {
      memcpy(dst, src, size);
    }
  }
  LIBXSTREAM_OFFLOAD_END(false)

  return result;
}