/*!
 * Copyright 2016 Rory mitchell
 */
#pragma once
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/random.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#include <algorithm>
#include <chrono>
#include <ctime>
#include <sstream>
#include <string>
#include <vector>

#ifndef NCCL
#define NCCL 1
#endif

#if(NCCL)
#include "nccl.h"
#endif


// Uncomment to enable
//#define DEVICE_TIMER
#define TIMERS

namespace dh {

/*
 * Error handling  functions
 */

#define safe_cuda(ans) throw_on_cuda_error((ans), __FILE__, __LINE__)

inline cudaError_t throw_on_cuda_error(cudaError_t code, const char *file,
                                       int line) {
  if (code != cudaSuccess) {
    std::stringstream ss;
    ss << file << "(" << line << ")";
    std::string file_and_line;
    ss >> file_and_line;
    throw thrust::system_error(code, thrust::cuda_category(), file_and_line);
  }

  return code;
}

#define safe_nccl(ans) throw_on_nccl_error((ans), __FILE__, __LINE__)

#if(NCCL)
inline ncclResult_t throw_on_nccl_error(ncclResult_t code, const char *file,
                                       int line) {
  if (code != ncclSuccess) {
    std::stringstream ss;
    ss << "NCCL failure :" << ncclGetErrorString(code) << " ";
    ss << file << "(" << line << ")";
    throw std::runtime_error(ss.str());
  }

  return code;
}
#endif

#define gpuErrchk(ans)                          \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort) exit(code);
  }
}



inline int n_visible_devices() {
  int n_visgpus = 0;
 
  cudaGetDeviceCount(&n_visgpus);

  return n_visgpus;
}

inline int n_devices_all(int n_gpus) {
  if(NCCL==0 && n_gpus>1 || NCCL==0 && n_gpus!=0){
    if(n_gpus!=1 && n_gpus!=0){
      fprintf(stderr,"NCCL=0, so forcing n_gpus=1\n");
      fflush(stderr);
    }
    n_gpus=1;
  }
  int n_devices_visible = dh::n_visible_devices();
  int n_devices = n_gpus < 0 ? n_devices_visible : n_gpus;
  return(n_devices);
}
inline int n_devices(int n_gpus, int num_rows) {
  int n_devices = dh::n_devices_all(n_gpus);
  // fix-up device number to be limited by number of rows
  n_devices = n_devices > num_rows ? num_rows : n_devices;
  return(n_devices);
}

  // if n_devices=-1, then use all visible devices
inline void synchronize_n_devices(int n_devices, std::vector<int> dList) {
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaDeviceSynchronize());
  }
}
inline void synchronize_all() {
  for(int device_idx=0;device_idx<n_visible_devices();device_idx++){
    safe_cuda(cudaSetDevice(device_idx));
    safe_cuda(cudaDeviceSynchronize());
  }
}

inline std::string device_name(int device_idx) {
  cudaDeviceProp prop;
  dh::safe_cuda(cudaGetDeviceProperties(&prop, device_idx));
  return std::string(prop.name);
}
  

/*
 *  Timers
 */

#define MAX_WARPS 32  // Maximum number of warps to time
#define MAX_SLOTS 10
#define TIMER_BLOCKID 0  // Block to time
struct DeviceTimerGlobal {
#ifdef DEVICE_TIMER

  clock_t total_clocks[MAX_SLOTS][MAX_WARPS];
  int64_t count[MAX_SLOTS][MAX_WARPS];

#endif

  // Clear device memory. Call at start of kernel.
  __device__ void Init() {
#ifdef DEVICE_TIMER
    if (blockIdx.x == TIMER_BLOCKID && threadIdx.x < MAX_WARPS) {
      for (int SLOT = 0; SLOT < MAX_SLOTS; SLOT++) {
        total_clocks[SLOT][threadIdx.x] = 0;
        count[SLOT][threadIdx.x] = 0;
      }
    }
#endif
  }

  void HostPrint() {
#ifdef DEVICE_TIMER
    DeviceTimerGlobal h_timer;
    safe_cuda(
        cudaMemcpyFromSymbol(&h_timer, (*this), sizeof(DeviceTimerGlobal)));

    for (int SLOT = 0; SLOT < MAX_SLOTS; SLOT++) {
      if (h_timer.count[SLOT][0] == 0) {
        continue;
      }

      clock_t sum_clocks = 0;
      int64_t sum_count = 0;

      for (int WARP = 0; WARP < MAX_WARPS; WARP++) {
        if (h_timer.count[SLOT][WARP] == 0) {
          continue;
        }

        sum_clocks += h_timer.total_clocks[SLOT][WARP];
        sum_count += h_timer.count[SLOT][WARP];
      }

      printf("Slot %d: %d clocks per call, called %d times.\n", SLOT,
             sum_clocks / sum_count, h_timer.count[SLOT][0]);
    }
#endif
  }
};

struct DeviceTimer {
#ifdef DEVICE_TIMER
  clock_t start;
  int slot;
  DeviceTimerGlobal &GTimer;
#endif

#ifdef DEVICE_TIMER
  __device__ DeviceTimer(DeviceTimerGlobal &GTimer, int slot)  // NOLINT
      : GTimer(GTimer), start(clock()), slot(slot) {}
#else
  __device__ DeviceTimer(DeviceTimerGlobal &GTimer, int slot) {}  // NOLINT
#endif

  __device__ void End() {
#ifdef DEVICE_TIMER
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    if (blockIdx.x == TIMER_BLOCKID && lane_id == 0) {
      GTimer.count[slot][warp_id] += 1;
      GTimer.total_clocks[slot][warp_id] += clock() - start;
    }
#endif
  }
};

struct Timer {
  typedef std::chrono::high_resolution_clock ClockT;

  typedef std::chrono::high_resolution_clock::time_point TimePointT;
  TimePointT start;
  Timer() { reset(); }

  void reset() { start = ClockT::now(); }
  int64_t elapsed() const { return (ClockT::now() - start).count(); }
  void printElapsed(std::string label) {
    //    synchronize_n_devices(n_devices, dList);
    printf("%s:\t %lld\n", label.c_str(), elapsed());
    reset();
  }
};

/*
 * Range iterator
 */

class range {
 public:
  class iterator {
    friend class range;

   public:
    __host__ __device__ int64_t operator*() const { return i_; }
    __host__ __device__ const iterator &operator++() {
      i_ += step_;
      return *this;
    }
    __host__ __device__ iterator operator++(int) {
      iterator copy(*this);
      i_ += step_;
      return copy;
    }

    __host__ __device__ bool operator==(const iterator &other) const {
      return i_ >= other.i_;
    }
    __host__ __device__ bool operator!=(const iterator &other) const {
      return i_ < other.i_;
    }

    __host__ __device__ void step(int s) { step_ = s; }

   protected:
    __host__ __device__ explicit iterator(int64_t start) : i_(start) {}

   public:
    uint64_t i_;
    int step_ = 1;
  };

  __host__ __device__ iterator begin() const { return begin_; }
  __host__ __device__ iterator end() const { return end_; }
  __host__ __device__ range(int64_t begin, int64_t end)
      : begin_(begin), end_(end) {}
  __host__ __device__ void step(int s) { begin_.step(s); }

 private:
  iterator begin_;
  iterator end_;
};

template <typename T>
__device__ range grid_stride_range(T begin, T end) {
  begin += blockDim.x * blockIdx.x + threadIdx.x;
  range r(begin, end);
  r.step(gridDim.x * blockDim.x);
  return r;
}

template <typename T>
__device__ range block_stride_range(T begin, T end) {
  begin += threadIdx.x;
  range r(begin, end);
  r.step(blockDim.x);
  return r;
}

// Threadblock iterates over range, filling with value
template <typename IterT, typename ValueT>
__device__ void block_fill(IterT begin, size_t n, ValueT value) {
  for (auto i : block_stride_range(static_cast<size_t>(0), n)) {
    begin[i] = value;
  }
}

/*
 * Memory
 */

    enum memory_type {
    DEVICE,
    DEVICE_MANAGED
  };

  template <memory_type MemoryT>
class bulk_allocator;

  template <typename T>
class dvec {

 private:
  T *_ptr;
  size_t _size;
  int _device_idx;


 public:
  void external_allocate(int device_idx, void *ptr, size_t size) {
    if (!empty()) {
      throw std::runtime_error("Tried to allocate dvec but already allocated");
    }

    _ptr = static_cast<T *>(ptr);
    _size = size;
    _device_idx = device_idx;
  }

  dvec() : _ptr(NULL), _size(0), _device_idx(0) {}
  size_t size() const { return _size; }
  int device_idx() const { return _device_idx; }
  bool empty() const { return _ptr == NULL || _size == 0; }
  T *data() { return _ptr; }

  std::vector<T> as_vector() const {
    std::vector<T> h_vector(size());
    safe_cuda(cudaSetDevice(_device_idx));
    safe_cuda(cudaMemcpy(h_vector.data(), _ptr, size() * sizeof(T),
                         cudaMemcpyDeviceToHost));
    return h_vector;
  }

  void fill(T value) {
    safe_cuda(cudaSetDevice(_device_idx));
    thrust::fill_n(thrust::device_pointer_cast(_ptr), size(), value);
  }

  void print() {
    auto h_vector = this->as_vector();

    for (auto e : h_vector) {
      std::cout << e << " ";
    }

    std::cout << "\n";
  }

  thrust::device_ptr<T> tbegin() { return thrust::device_pointer_cast(_ptr); }

  thrust::device_ptr<T> tend() {
    return thrust::device_pointer_cast(_ptr + size());
  }

  template <typename T2>
  dvec &operator=(const std::vector<T2> &other) {
    this->copy(other.begin(), other.end());
    return *this;
  }

  dvec &operator=(dvec<T> &other) {
    if (other.size() != size()) {
      throw std::runtime_error(
          "Cannot copy assign dvec to dvec, sizes are different");
    }

    safe_cuda(cudaSetDevice(this->device_idx()));
    if(other.device_idx() == this->device_idx()){
      thrust::copy(other.tbegin(), other.tend(), this->tbegin());
    }
    else{
      throw std::runtime_error(
          "Cannot copy to/from different devices");
    }

    return *this;
  }

    template <typename IterT>
    void copy(IterT begin, IterT end){
      safe_cuda(cudaSetDevice(this->device_idx()));
      if (end-begin != size()) {
        throw std::runtime_error("Cannot copy assign vector to dvec, sizes are different");
      }
      thrust::copy(begin,end,this->tbegin());
    }
    
};


  template <memory_type MemoryT>
class bulk_allocator {
  std::vector<char*> d_ptr;
  std::vector<size_t> _size;
  std::vector<int> _device_idx;

  const size_t align = 256;


  template <typename SizeT>
  size_t align_round_up(SizeT n) {
    if (n % align == 0) {
      return n;
    } else {
      return n + align - (n % align);
    }
  }

  template <typename T, typename SizeT>
  size_t get_size_bytes(dvec<T> *first_vec, SizeT first_size) {
    return align_round_up(first_size * sizeof(T));
  }

  template <typename T, typename SizeT, typename... Args>
  size_t get_size_bytes(dvec<T> *first_vec, SizeT first_size, Args... args) {
    return align_round_up(first_size * sizeof(T)) + get_size_bytes(args...);
  }

  template <typename T, typename SizeT>
  void allocate_dvec(int device_idx, char *ptr, dvec<T> *first_vec, SizeT first_size) {
    first_vec->external_allocate(device_idx, static_cast<void *>(ptr), first_size);
  }

  template <typename T, typename SizeT, typename... Args>
  void allocate_dvec(int device_idx, char *ptr, dvec<T> *first_vec, SizeT first_size,
                     Args... args) {
    first_vec->external_allocate(device_idx, static_cast<void *>(ptr), first_size);
    ptr += align_round_up(first_size * sizeof(T));
    allocate_dvec(device_idx, ptr, args...);
  }

    //    template <memory_type MemoryT>
    char * allocate_device(int device_idx, size_t bytes, memory_type t){
      char * ptr;
      if(t==memory_type::DEVICE){
        safe_cuda(cudaSetDevice(device_idx));
        safe_cuda(cudaMalloc(&ptr, bytes));
      }
      else{
        safe_cuda(cudaMallocManaged(&ptr, bytes));
      }
      return ptr;
    }

 public:
  ~bulk_allocator() {
    for(int i=0;i<d_ptr.size();i++){
      if (!(d_ptr[i] == nullptr)) {
        safe_cuda(cudaSetDevice(_device_idx[i]));
        safe_cuda(cudaFree(d_ptr[i]));
      }
    }
  }

  // returns sum of bytes for all allocations
  size_t size() { return std::accumulate(_size.begin(),_size.end(),static_cast<size_t>(0)); }

  template <typename... Args>
  void allocate(int device_idx, Args... args) {

    size_t size= get_size_bytes(args...);

    char *ptr = allocate_device(device_idx, size, MemoryT);

    allocate_dvec(device_idx, ptr, args...);

    d_ptr.push_back(ptr);
    _size.push_back(size);
    _device_idx.push_back(device_idx);
  }
};

  

// Keep track of cub library device allocation
struct CubMemory {
  void *d_temp_storage;
  size_t temp_storage_bytes;

  CubMemory() : d_temp_storage(NULL), temp_storage_bytes(0) {}

  ~CubMemory() { Free(); }
  void Free() {
    if (d_temp_storage != NULL) {
      safe_cuda(cudaFree(d_temp_storage));
    }
  }

  void LazyAllocate(size_t n_bytes) {
    if (n_bytes > temp_storage_bytes) {
      Free();
      safe_cuda(cudaMalloc(&d_temp_storage, n_bytes));
      temp_storage_bytes = n_bytes;
    }
  }

  bool IsAllocated() { return d_temp_storage != NULL; }
};

inline size_t available_memory(int device_idx) {
  size_t device_free = 0;
  size_t device_total = 0;
  safe_cuda(cudaSetDevice(device_idx));
  dh::safe_cuda(cudaMemGetInfo(&device_free, &device_total));
  return device_free;
}


  

/*
 *  Utility functions
 */

template <typename T>
void print(const thrust::device_vector<T> &v, size_t max_items = 10) {
  thrust::host_vector<T> h = v;
  for (int i = 0; i < std::min(max_items, h.size()); i++) {
    std::cout << " " << h[i];
  }
  std::cout << "\n";
}

  template <typename T, memory_type MemoryT>
  void print(const dvec<T> &v, size_t max_items = 10) {
  std::vector<T> h = v.as_vector();
  for (int i = 0; i < std::min(max_items, h.size()); i++) {
    std::cout << " " << h[i];
  }
  std::cout << "\n";
}

template <typename T>
void print(char *label, const thrust::device_vector<T> &v,
           const char *format = "%d ", int max = 10) {
  thrust::host_vector<T> h_v = v;

  std::cout << label << ":\n";
  for (int i = 0; i < std::min(static_cast<int>(h_v.size()), max); i++) {
    printf(format, h_v[i]);
  }
  std::cout << "\n";
}

template <typename T1, typename T2>
T1 div_round_up(const T1 a, const T2 b) {
  return static_cast<T1>(ceil(static_cast<double>(a) / b));
}

template <typename T>
thrust::device_ptr<T> dptr(T *d_ptr) {
  return thrust::device_pointer_cast(d_ptr);
}

template <typename T>
T *raw(thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T>
const T *raw(const thrust::device_vector<T> &v) {  //  NOLINT
  return raw_pointer_cast(v.data());
}

template <typename T>
size_t size_bytes(const thrust::device_vector<T> &v) {
  return sizeof(T) * v.size();
}
/*
 * Kernel launcher
 */

template <typename L>
__global__ void launch_n_kernel(size_t begin, size_t end, L lambda) {
  for (auto i : grid_stride_range(begin, end)) {
    lambda(i);
  }
}
template <typename L>
__global__ void launch_n_kernel(int device_idx, size_t begin, size_t end, L lambda) {
  for (auto i : grid_stride_range(begin, end)) {
    lambda(i,device_idx);
  }
}

template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void launch_n(int device_idx, size_t n, L lambda) {
  safe_cuda(cudaSetDevice(device_idx));
  const int GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);
#if defined(__CUDACC__)
  launch_n_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(static_cast<size_t>(0),n, lambda);
#endif
}

  // if n_devices=-1, then use all visible devices
template <int ITEMS_PER_THREAD = 8, int BLOCK_THREADS = 256, typename L>
inline void multi_launch_n(size_t n, int n_devices, L lambda) {
  n_devices = n_devices<0 ? n_visible_devices() : n_devices;
  CHECK_LE(n_devices,n_visible_devices()) << "Number of devices requested needs to be less than equal to number of visible devices.";
  const int GRID_SIZE = div_round_up(n, ITEMS_PER_THREAD * BLOCK_THREADS);
#if defined(__CUDACC__)
  n_devices = n_devices > n ? n : n_devices;
  for(int device_idx=0;device_idx<n_devices;device_idx++){
    safe_cuda(cudaSetDevice(device_idx));
    size_t begin=(n/n_devices)*device_idx;
    size_t end=std::min((n/n_devices)*(device_idx+1),n);
    launch_n_kernel<<<GRID_SIZE, BLOCK_THREADS>>>(device_idx,begin, end, lambda);
  }
#endif
}

/*
 * Random
 */

struct BernoulliRng {
  float p;
  int seed;

  __host__ __device__ BernoulliRng(float p, int seed) : p(p), seed(seed) {}

  __host__ __device__ bool operator()(const int i) const {
    thrust::default_random_engine rng(seed);
    thrust::uniform_real_distribution<float> dist;
    rng.discard(i);

    return dist(rng) <= p;
  }
};

}  // namespace dh
