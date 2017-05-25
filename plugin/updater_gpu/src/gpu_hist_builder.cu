/*!
 * Copyright 2017 Rory mitchell
 */
#include <future>
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <cub/cub.cuh>
#include <algorithm>
#include <functional>
#include <numeric>
#include "common.cuh"
#include "device_helpers.cuh"
#include "gpu_hist_builder.cuh"

//#define _NCCL 1

namespace xgboost {
namespace tree {

  void DeviceGMat::Init(int device_idx, const common::GHistIndexMatrix& gmat, bst_uint begin, bst_uint end) {
    dh::safe_cuda(cudaSetDevice(device_idx));
  CHECK_EQ(gidx.size(), end-begin)
      << "gidx must be externally allocated";
  CHECK_EQ(ridx.size(), end-begin)
      << "ridx must be externally allocated";

  thrust::copy(&gmat.index[begin],&gmat.index[end],gidx.tbegin());
  thrust::device_vector<int> row_ptr = gmat.row_ptr;

  auto counting = thrust::make_counting_iterator(begin);
  thrust::upper_bound(row_ptr.begin(), row_ptr.end(), counting,
                      counting + gidx.size(), ridx.tbegin());
  thrust::transform(ridx.tbegin(), ridx.tend(), ridx.tbegin(),
                    [=] __device__(int val) { return val - 1; });
}

void DeviceHist::Init(int n_bins_in) {
  this->n_bins = n_bins_in;
  CHECK(!data.empty()) << "DeviceHist must be externally allocated";
}

void DeviceHist::Reset(int device_idx) {
  cudaSetDevice(device_idx);
  data.fill(gpu_gpair());
}

gpu_gpair* DeviceHist::GetLevelPtr(int depth) {
  return data.data() + n_nodes(depth - 1) * n_bins;
}

int DeviceHist::LevelSize(int depth) { return n_bins * n_nodes_level(depth); }

HistBuilder DeviceHist::GetBuilder() {
  return HistBuilder(data.data(), n_bins);
}

HistBuilder::HistBuilder(gpu_gpair* ptr, int n_bins)
    : d_hist(ptr), n_bins(n_bins) {}

__device__ void HistBuilder::Add(gpu_gpair gpair, int gidx, int nidx) const {
  int hist_idx = nidx * n_bins + gidx;
  atomicAdd(&(d_hist[hist_idx]._grad), gpair._grad);
  atomicAdd(&(d_hist[hist_idx]._hess), gpair._hess);
}

__device__ gpu_gpair HistBuilder::Get(int gidx, int nidx) const {
  return d_hist[nidx * n_bins + gidx];
}

GPUHistBuilder::GPUHistBuilder()
    : initialised(false),
      is_dense(false),
      p_last_fmat_(nullptr),
      prediction_cache_initialised(false) {}

GPUHistBuilder::~GPUHistBuilder() {
#ifdef _NCCL
  for(int d_idx=0; d_idx<n_devices; ++d_idx)
    {
      ncclCommDestroy(comms[d_idx]);
      dh::safe_cuda(cudaSetDevice(dList[d_idx]));
      dh::safe_cuda(cudaStreamDestroy(s[d_idx]));
    }
#endif
  
}

void GPUHistBuilder::Init(const TrainParam& param) {
  CHECK(param.max_depth < 16) << "Tree depth too large.";
  CHECK(param.grow_policy != TrainParam::kLossGuide)
      << "Loss guided growth policy not supported. Use CPU algorithm.";
  this->param = param;

  //  dh::safe_cuda(cudaSetDevice(param.gpu_id));
  CHECK(param.n_gpus!=0) << "Must have at least one device";
  int n_devices_all = dh::n_devices_all(param.n_gpus);
  for(int device_idx=0;device_idx<n_devices;device_idx++){
    if (!param.silent) {
      size_t free_memory = dh::available_memory(device_idx);
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Device: [" << device_idx << "] " << dh::device_name(device_idx) << " with " << free_memory / mb_size << " MB available device memory.";
    }
  }

  CHECK_LE(param.n_gpus,dh::n_visible_devices()) << "Specify number of GPUs to be less or equal to number of visible GPU devices.";

}
void GPUHistBuilder::InitData(const std::vector<bst_gpair>& gpair,
                              DMatrix& fmat,  // NOLINT
                              const RegTree& tree) {

  // set member num_rows and n_devices for rest of GPUHistBuilder members
  info = &fmat.info();
  num_rows = info->num_row;
  n_devices = dh::n_devices(param.n_gpus,num_rows);
  

  if (!initialised) {

    // set dList member
    dList.resize(n_devices);
    for (int i = 0; i < n_devices; ++i) dList[i] = i % n_devices;
    
#ifdef _NCCL
    // initialize nccl

    comms.resize(n_devices);
    streams.resize(n_devices);
    dh::safe_nccl(ncclCommInitAll(comms.data(), n_devices, dList.data())); // initialize communicator (One communicator per process)
    
    printf("# NCCL: Using devices\n");
    for (int d_idx = 0; d_idx < n_devices; ++d_idx) {

      dh::safe_cuda(cudaSetDevice(dList[d_idx]));
      dh::safe_cuda(cudaStreamCreate(streams[d_idx]));
      
      int cudaDev;
      int rank;
      cudaDeviceProp prop;
      dh::safe_nccl(ncclCommCuDevice(comms[d_idx], &cudaDev));
      dh::safe_nccl(ncclCommUserRank(comms[d_idx], &rank));
      dh::safe_cuda(cudaGetDeviceProperties(&prop, cudaDev));
      printf("#   Rank %2d uses device %2d [0x%02x] %s\n", rank, cudaDev,
             prop.pciBusID, prop.name); fflush(stdout);
    }
#endif    
    
    CHECK(fmat.SingleColBlock()) << "grow_gpu_hist: must have single column "
                                    "block. Try setting 'tree_method' "
                                    "parameter to 'exact'";
    is_dense = info->num_nonzero == info->num_col * info->num_row;
    hmat_.Init(&fmat, param.max_bin);
    gmat_.cut = &hmat_;
    gmat_.Init(&fmat);
    int n_bins = hmat_.row_ptr.back();
    int n_features = hmat_.row_ptr.size() - 1;

    // deliniate data onto multiple gpus
    
    device_row_segments.push_back(0);
    device_element_segments.push_back(0);
    bst_uint offset=0;
    size_t shard_size = std::ceil((double)(num_rows)/n_devices);
    for(int d_idx=0;d_idx<n_devices;d_idx++){
      int device_idx = dList[d_idx];
      offset+=shard_size;
      offset=std::min(offset,num_rows);
      device_row_segments.push_back(offset);
      device_element_segments.push_back(gmat_.row_ptr[offset]);
    }
    
    // Build feature segments
    std::vector<int> h_feature_segments;
    for (int node = 0; node < n_nodes_level(param.max_depth - 1); node++) {
      for (int fidx = 0; fidx < hmat_.row_ptr.size() - 1; fidx++) {
        h_feature_segments.push_back(hmat_.row_ptr[fidx] + node * n_bins);
      }
    }
    h_feature_segments.push_back(n_nodes_level(param.max_depth - 1) * n_bins);

    // Construct feature map
    std::vector<int> h_gidx_feature_map(n_bins);
    for (int row = 0; row < hmat_.row_ptr.size() - 1; row++) {
      for (int i = hmat_.row_ptr[row]; i < hmat_.row_ptr[row + 1]; i++) {
        h_gidx_feature_map[i] = row;
      }
    }
    
    int level_max_bins = n_nodes_level(param.max_depth - 1) * n_bins;

    // allocate unique common data
    int master_device=dList[0];
    ba.allocate(master_device,
                &hist_node_segments, n_nodes_level(param.max_depth - 1) + 1,
                &feature_segments,   h_feature_segments.size(),
                &gain, level_max_bins,
                &fidx_min_map, hmat_.min_val.size(),
                &argmax, n_nodes_level(param.max_depth - 1),
                &node_sums, n_nodes_level(param.max_depth - 1) * n_features,
                &hist_scan, level_max_bins,
                &feature_flags, n_features,
                &hist_temp.data,n_nodes(param.max_depth - 1) * n_bins
                );
    hist_temp.Init(n_bins);

    
    // allocate vectors across all devices
    hist_vec.resize(n_devices);
    left_child_smallest.resize(n_devices);
    nodes.resize(n_devices);
    prediction_cache.resize(n_devices);
    position.resize(n_devices);
    position_tmp.resize(n_devices);
    device_matrix.resize(n_devices);
    device_matrix.resize(n_devices);
    device_gpair.resize(n_devices);
    gidx_feature_map.resize(n_devices);
    gidx_fvalue_map.resize(n_devices);
    

    // shard rows onto gpus
    for(int d_idx=0;d_idx<n_devices;d_idx++){
      int device_idx = dList[d_idx];
      bst_uint num_rows_segment = device_row_segments[d_idx+1]-device_row_segments[d_idx];
      bst_uint num_elements_segment = device_element_segments[d_idx+1]-device_element_segments[d_idx];
      ba.allocate(device_idx,
                  &(hist_vec[d_idx].data),n_nodes(param.max_depth - 1) * n_bins,
                  &left_child_smallest[d_idx], n_nodes(param.max_depth),
                  &nodes[d_idx], n_nodes(param.max_depth),
                  &prediction_cache[d_idx], num_rows_segment,
                  &position[d_idx], num_rows_segment,
                  &position_tmp[d_idx], num_rows_segment,
                  &device_matrix[d_idx].gidx, num_elements_segment,
                  &device_matrix[d_idx].ridx, num_elements_segment,
                  &device_gpair[d_idx], num_rows_segment,
                  &gidx_feature_map[d_idx], n_bins,
                  &gidx_fvalue_map[d_idx], hmat_.cut.size()
                  );

      // Copy Host to Device
      
      // Construct device matrix
      device_matrix[d_idx].Init(device_idx,gmat_,device_element_segments[d_idx],device_element_segments[d_idx+1]);
      gidx_feature_map[d_idx] = h_gidx_feature_map;
      gidx_fvalue_map[d_idx] = hmat_.cut;

      // Initialize only, no copy
      hist_vec[d_idx].Init(n_bins);
      prediction_cache[d_idx].fill(0);

      // TODO: feature_flags need not be on multiple GPUs as not currently used (but could be used for more performance)
    }
  
        
    if (!param.silent) {
      const int mb_size = 1048576;
      LOG(CONSOLE) << "Allocated " << ba.size() / mb_size << " MB";
    }

    fidx_min_map = hmat_.min_val;

    thrust::sequence(hist_node_segments.tbegin(), hist_node_segments.tend(), 0,
                     n_bins);

    feature_flags.fill(1);

    feature_segments = h_feature_segments;



    initialised = true;
  }


  // shard rows onto gpus
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));

    nodes[d_idx].fill(Node());
    position[d_idx].fill(0);

    device_gpair[d_idx].copy(gpair.begin()+device_row_segments[d_idx],gpair.begin()+device_row_segments[d_idx+1]);
    
    subsample_gpair(&device_gpair[d_idx], param.subsample);

    hist_vec[d_idx].Reset(device_idx);
  }

  dh::synchronize_n_devices(n_devices, dList);


  p_last_fmat_ = &fmat;
}

void GPUHistBuilder::BuildHist(int depth) {

  dh::Timer time;
  
#if(1)
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    size_t begin = device_element_segments[d_idx];
    size_t end = device_element_segments[d_idx+1];
    size_t row_begin = device_row_segments[d_idx];
    
    dh::safe_cuda(cudaSetDevice(device_idx));

    auto d_ridx = device_matrix[d_idx].ridx.data();
    auto d_gidx = device_matrix[d_idx].gidx.data();
    auto d_position = position[d_idx].data();
    auto d_gpair = device_gpair[d_idx].data();
    auto d_left_child_smallest = left_child_smallest[d_idx].data();
    auto hist_builder = hist_vec[d_idx].GetBuilder();
    
    dh::launch_n(end-begin, [=] __device__(int local_idx) {
        
        int ridx = d_ridx[local_idx];
        int pos = d_position[ridx-row_begin];
        if (!is_active(pos, depth)) return;

        // Only increment smallest node
        bool is_smallest =
          (d_left_child_smallest[parent_nidx(pos)] && is_left_child(pos)) ||
          (!d_left_child_smallest[parent_nidx(pos)] && !is_left_child(pos));
        if (!is_smallest && depth > 0) return;
        
        int gidx = d_gidx[local_idx];
        gpu_gpair gpair = d_gpair[ridx-row_begin];
        
        hist_builder.Add(gpair, gidx, pos);
      });
  }
#else
  // TODO: doesn't work because need to pass all n_gpus of pointers of hist_builder
  dh::multi_launch_n(device_matrix.gidx.size(), param.n_gpus, [=] __device__(int idx, int device_idx) {
    int ridx = d_ridx[idx];
    int pos = d_position[ridx];
    if (!is_active(pos, depth)) return;

    // Only increment smallest node
    bool is_smallest =
        (d_left_child_smallest[parent_nidx(pos)] && is_left_child(pos)) ||
        (!d_left_child_smallest[parent_nidx(pos)] && !is_left_child(pos));
    if (!is_smallest && depth > 0) return;

    int gidx = d_gidx[idx];
    gpu_gpair gpair = d_gpair[ridx];

    hist_builder.Add(gpair, gidx, pos);
  });
 
#endif

  //  dh::safe_cuda(cudaDeviceSynchronize());
  dh::synchronize_n_devices(n_devices, dList);

  time.printElapsed("Add Time");


  // reduce each element of histogram across multiple gpus
  int master_device=dList[0];
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    auto master_hist_data = hist_vec[master_device].GetLevelPtr(depth);
    auto slave_hist_data = hist_vec[d_idx].GetLevelPtr(depth);
    size_t count_bytes = hist_temp.LevelSize(depth)*sizeof(gpu_gpair);
    fprintf(stderr,"device_idx=%d count_bytes=%zu counts=%d %d\n",device_idx,count_bytes,hist_vec[master_device].LevelSize(depth),hist_vec[d_idx].LevelSize(depth)); fflush(stderr);

    //#ifdef _NCCL
#if(0)
    dh::safe_cuda(cudaSetDevice(device_idx));
    ncclReduce((const void*)hist_vec[d_idx].GetLevelPtr(depth),(void*)hist_vec[d_idx].GetLevelPtr(depth),hist_vec[d_idx].LevelSize(depth), gpu_gpair, ncclSum, master_device, comms[d_idx]);
  
#else
    auto temp_hist_data = hist_temp.GetLevelPtr(depth);
    if(device_idx==master_device) continue;

    cudaMemcpyPeer(temp_hist_data,master_device,slave_hist_data,device_idx,count_bytes);
    
    dh::launch_n(hist_vec[master_device].LevelSize(depth), [=] __device__(int idx) {

        printf("idx=%d\n",idx);
        master_hist_data[idx] += temp_hist_data[idx];
        
      });
    dh::safe_cuda(cudaDeviceSynchronize());
#endif
  }
  fflush(stdout);
  time.printElapsed("Reduce-Add Time");

  dh::safe_cuda(cudaSetDevice(master_device)); // Apparently required, unsure why

#ifdef _NCCL
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    CUDACHECK(cudaSetDevice(device_idx));
    CUDACHECK(cudaStreamSynchronize(s[d_idx]));
  }
#endif
  
  
  // Subtraction trick
  auto hist_builder = hist_vec[master_device].GetBuilder();
  auto d_left_child_smallest = left_child_smallest[master_device].data();
  int n_sub_bins = (n_nodes_level(depth) / 2) * hist_builder.n_bins;
  if (depth > 0) {
    dh::launch_n(n_sub_bins, [=] __device__(int idx) {
      int nidx = n_nodes(depth - 1) + ((idx / hist_builder.n_bins) * 2);
      bool left_smallest = d_left_child_smallest[parent_nidx(nidx)];
      if (left_smallest) {
        nidx++;  // If left is smallest switch to right child
      }

      int gidx = idx % hist_builder.n_bins;
      gpu_gpair parent = hist_builder.Get(gidx, parent_nidx(nidx));
      int other_nidx = left_smallest ? nidx - 1 : nidx + 1;
      gpu_gpair other = hist_builder.Get(gidx, other_nidx);
      hist_builder.Add(parent - other, gidx, nidx);
    });
  }
  dh::safe_cuda(cudaDeviceSynchronize());
}


template <int BLOCK_THREADS>
__global__ void find_split_kernel(
    const gpu_gpair* d_level_hist, int* d_feature_segments, int depth,
    int n_features, int n_bins, Node* d_nodes, float* d_fidx_min_map,
    float* d_gidx_fvalue_map, GPUTrainingParam gpu_param,
    bool* d_left_child_smallest, bool colsample, int* d_feature_flags) {
  typedef cub::KeyValuePair<int, float> ArgMaxT;
  typedef cub::BlockScan<gpu_gpair, BLOCK_THREADS, cub::BLOCK_SCAN_WARP_SCANS>
      BlockScanT;
  typedef cub::BlockReduce<ArgMaxT, BLOCK_THREADS> MaxReduceT;
  typedef cub::BlockReduce<gpu_gpair, BLOCK_THREADS> SumReduceT;

  union TempStorage {
    typename BlockScanT::TempStorage scan;
    typename MaxReduceT::TempStorage max_reduce;
    typename SumReduceT::TempStorage sum_reduce;
  };

  struct UninitializedSplit : cub::Uninitialized<Split> {};
  struct UninitializedGpair : cub::Uninitialized<gpu_gpair> {};

  __shared__ UninitializedSplit uninitialized_split;
  Split& split = uninitialized_split.Alias();
  __shared__ UninitializedGpair uninitialized_sum;
  gpu_gpair& shared_sum = uninitialized_sum.Alias();
  __shared__ ArgMaxT block_max;
  __shared__ TempStorage temp_storage;

  if (threadIdx.x == 0) {
    split = Split();
  }

  __syncthreads();

  int node_idx = n_nodes(depth - 1) + blockIdx.x;

  for (int fidx = 0; fidx < n_features; fidx++) {
    if (colsample && d_feature_flags[fidx] == 0) continue;

    int begin = d_feature_segments[blockIdx.x * n_features + fidx];
    int end = d_feature_segments[blockIdx.x * n_features + fidx + 1];
    int gidx = (begin - (blockIdx.x * n_bins)) + threadIdx.x;
    bool thread_active = threadIdx.x < end - begin;

    gpu_gpair feature_sum = gpu_gpair();
    for (int reduce_begin = begin; reduce_begin < end;
         reduce_begin += BLOCK_THREADS) {
      // Scan histogram
      gpu_gpair bin = thread_active ? d_level_hist[reduce_begin + threadIdx.x]
                                    : gpu_gpair();

      feature_sum +=
          SumReduceT(temp_storage.sum_reduce).Reduce(bin, cub::Sum());
    }

    if (threadIdx.x == 0) {
      shared_sum = feature_sum;
    }
    //    __syncthreads(); // no need to synch because below there is a Scan

    GpairCallbackOp prefix_op = GpairCallbackOp();
    for (int scan_begin = begin; scan_begin < end;
         scan_begin += BLOCK_THREADS) {
      gpu_gpair bin =
          thread_active ? d_level_hist[scan_begin + threadIdx.x] : gpu_gpair();

      BlockScanT(temp_storage.scan)
          .ExclusiveScan(bin, bin, cub::Sum(), prefix_op);

      // Calculate gain
      gpu_gpair parent_sum = d_nodes[node_idx].sum_gradients;
      float parent_gain = d_nodes[node_idx].root_gain;

      gpu_gpair missing = parent_sum - shared_sum;

      bool missing_left;
      float gain = thread_active
                       ? loss_chg_missing(bin, missing, parent_sum, parent_gain,
                                          gpu_param, missing_left)
                       : -FLT_MAX;
      __syncthreads();

      // Find thread with best gain
      ArgMaxT tuple(threadIdx.x, gain);
      ArgMaxT best =
          MaxReduceT(temp_storage.max_reduce).Reduce(tuple, cub::ArgMax());

      if (threadIdx.x == 0) {
        block_max = best;
      }

      __syncthreads();

      // Best thread updates split
      if (threadIdx.x == block_max.key) {
        float fvalue;
        if (threadIdx.x == 0 &&
            begin == scan_begin) {  // check at start of first tile
          fvalue = d_fidx_min_map[fidx];
        } else {
          fvalue = d_gidx_fvalue_map[gidx - 1];
        }

        gpu_gpair left = missing_left ? bin + missing : bin;
        gpu_gpair right = parent_sum - left;

        split.Update(gain, missing_left, fvalue, fidx, left, right, gpu_param);
      }
      __syncthreads();
    }  // end scan
  }    // end over features

  // Create node
  if (threadIdx.x == 0) {
    d_nodes[node_idx].split = split;
    if (depth == 0) {
      // split.Print();
    }

    d_nodes[left_child_nidx(node_idx)] = Node(
        split.left_sum,
        CalcGain(gpu_param, split.left_sum.grad(), split.left_sum.hess()),
        CalcWeight(gpu_param, split.left_sum.grad(), split.left_sum.hess()));

    d_nodes[right_child_nidx(node_idx)] = Node(
        split.right_sum,
        CalcGain(gpu_param, split.right_sum.grad(), split.right_sum.hess()),
        CalcWeight(gpu_param, split.right_sum.grad(), split.right_sum.hess()));

    // Record smallest node
    if (split.left_sum.hess() <= split.right_sum.hess()) {
      d_left_child_smallest[node_idx] = true;
    } else {
      d_left_child_smallest[node_idx] = false;
    }
  }
}

#define MIN_BLOCK_THREADS 32
#define MAX_BLOCK_THREADS 1024  // hard-coded maximum block size

void GPUHistBuilder::FindSplit(int depth) {
  int master_device=dList[0];
  dh::safe_cuda(cudaSetDevice(master_device));
  // Specialised based on max_bins
  this->FindSplitSpecialize<MIN_BLOCK_THREADS>(depth);
}

template <>
void GPUHistBuilder::FindSplitSpecialize<MAX_BLOCK_THREADS>(int depth) {
  int master_device=dList[0];
  
  const int GRID_SIZE = n_nodes_level(depth);
  bool colsample =
      param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;

  find_split_kernel<
      MAX_BLOCK_THREADS><<<GRID_SIZE, MAX_BLOCK_THREADS>>>(
      hist_vec[master_device].GetLevelPtr(depth), feature_segments.data(), depth, info->num_col,
      hmat_.row_ptr.back(), nodes[master_device].data(), fidx_min_map.data(),
      gidx_fvalue_map[master_device].data(), gpu_param, left_child_smallest[master_device].data(), colsample,
      feature_flags.data());

  dh::safe_cuda(cudaDeviceSynchronize());
}
template <int BLOCK_THREADS>
void GPUHistBuilder::FindSplitSpecialize(int depth) {
  int master_device=dList[0];

  if (param.max_bin <= BLOCK_THREADS) {
    const int GRID_SIZE = n_nodes_level(depth);
    bool colsample =
        param.colsample_bylevel < 1.0 || param.colsample_bytree < 1.0;

    find_split_kernel<BLOCK_THREADS><<<GRID_SIZE, BLOCK_THREADS>>>(
        hist_vec[master_device].GetLevelPtr(depth), feature_segments.data(), depth, info->num_col,
        hmat_.row_ptr.back(), nodes[master_device].data(), fidx_min_map.data(),
        gidx_fvalue_map[master_device].data(), gpu_param, left_child_smallest[master_device].data(),
        colsample, feature_flags.data());
  } else {
    this->FindSplitSpecialize<BLOCK_THREADS + 32>(depth);
  }

  dh::safe_cuda(cudaDeviceSynchronize());
}

void GPUHistBuilder::SynchronizeTree(int depth) {
  int master_device=dList[0];
  dh::safe_cuda(cudaSetDevice(master_device));

 
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    if(device_idx==master_device) continue;

    
    auto master_node_data = nodes[master_device].data() + n_nodes(depth-1);
    auto slave_node_data = nodes[d_idx].data() + n_nodes(depth-1);
    int node_count_bytes = n_nodes_level(depth)*sizeof(Node);

    cudaMemcpyPeerAsync(slave_node_data,device_idx,master_node_data,master_device,node_count_bytes);

    auto master_left_child_smallest_data = left_child_smallest[master_device].data() + n_nodes(depth-1);
    auto slave_left_child_smallest_data = left_child_smallest[d_idx].data() + n_nodes(depth-1);
    int left_child_smallest_count_bytes = n_nodes_level(depth)*sizeof(bool);

    cudaMemcpyPeerAsync(slave_left_child_smallest_data,device_idx,master_left_child_smallest_data,master_device,left_child_smallest_count_bytes);
  }
  

}

void GPUHistBuilder::InitFirstNode(const std::vector<bst_gpair> &gpair) {

  // asynch reduce per device
  std::vector<std::future<gpu_gpair>> future_results(n_devices);
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    
    auto begin        = device_gpair[d_idx].tbegin();
    auto end          = device_gpair[d_idx].tend();
    gpu_gpair init = gpu_gpair();
    auto binary_op    = thrust::plus<gpu_gpair>();

    // std::async captures the algorithm parameters by value
    // use std::launch::async to ensure the creation of a new thread
    future_results[d_idx] = std::async(std::launch::async, [=]
                                                         {
                                                           dh::safe_cuda(cudaSetDevice(device_idx));
                                                           return thrust::reduce(begin, end, init, binary_op);
                                                         });
  }

  // sum over devices
  gpu_gpair sum=gpu_gpair();
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    sum+=future_results[d_idx].get();
  }

  int master_device=dList[0];
  auto d_nodes = nodes[master_device].data();
  auto gpu_param_alias = gpu_param;
  
  dh::safe_cuda(cudaSetDevice(master_device));
  
  dh::launch_n(1, [=] __device__(int idx) {
    gpu_gpair sum_gradients = sum;
    d_nodes[idx] = Node(
        sum_gradients,
        CalcGain(gpu_param_alias, sum_gradients.grad(), sum_gradients.hess()),
        CalcWeight(gpu_param_alias, sum_gradients.grad(),
                   sum_gradients.hess()));
  });
}

void GPUHistBuilder::UpdatePosition(int depth) {
  if (is_dense) {
    this->UpdatePositionDense(depth);
  } else {
    this->UpdatePositionSparse(depth);
  }
}

void GPUHistBuilder::UpdatePositionDense(int depth) {

  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));


    auto d_position = position[d_idx].data();
    Node* d_nodes = nodes[d_idx].data();
    auto d_gidx_fvalue_map = gidx_fvalue_map[d_idx].data();
    auto d_gidx = device_matrix[d_idx].gidx.data();
    int n_columns = info->num_col;
    size_t begin = device_row_segments[d_idx];
    size_t end = device_row_segments[d_idx+1];
    
    dh::launch_n(end-begin, [=] __device__(int local_idx) { // TODO: add device_idx and put set inside
	    NodeIdT pos = d_position[local_idx];
	    if (!is_active(pos, depth)) {
	      return;
	    }
	    Node node = d_nodes[pos];
	
	    if (node.IsLeaf()) {
	      return;
	    }
	
	    int gidx = d_gidx[local_idx * n_columns + node.split.findex];
	
	    float fvalue = d_gidx_fvalue_map[gidx];
	
	    if (fvalue <= node.split.fvalue) {
	      d_position[local_idx] = left_child_nidx(pos);
	    } else {
	      d_position[local_idx] = right_child_nidx(pos);
	    }
	  });
  }
  dh::safe_cuda(cudaDeviceSynchronize());
}

void GPUHistBuilder::UpdatePositionSparse(int depth) {

  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));
    
    auto d_position = position[d_idx].data();
    auto d_position_tmp = position_tmp[d_idx].data();
    Node* d_nodes = nodes[d_idx].data();
    auto d_gidx_feature_map = gidx_feature_map[d_idx].data();
    auto d_gidx_fvalue_map = gidx_fvalue_map[d_idx].data();
    auto d_gidx = device_matrix[d_idx].gidx.data();
    auto d_ridx = device_matrix[d_idx].ridx.data();

    size_t row_begin = device_row_segments[d_idx];
    size_t row_end = device_row_segments[d_idx+1];
    size_t element_begin = device_element_segments[d_idx];
    size_t element_end = device_element_segments[d_idx+1];

    // Update missing direction
    dh::launch_n(row_end - row_begin, [=] __device__(int local_idx) {
        NodeIdT pos = d_position[local_idx];
        if (!is_active(pos, depth)) {
          d_position_tmp[local_idx] = pos;
          return;
        }

        Node node = d_nodes[pos];

        if (node.IsLeaf()) {
          d_position_tmp[local_idx] = pos;
          return;
        } else if (node.split.missing_left) {
          d_position_tmp[local_idx] = pos * 2 + 1;
        } else {
          d_position_tmp[local_idx] = pos * 2 + 2;
        }
      });


    // Update node based on fvalue where exists
    dh::launch_n(element_end - element_begin, [=] __device__(int local_idx) {
        int ridx = d_ridx[local_idx];
        NodeIdT pos = d_position[ridx-row_begin];
        if (!is_active(pos, depth)) {
          return;
        }

        Node node = d_nodes[pos];

        if (node.IsLeaf()) {
          return;
        }

        int gidx = d_gidx[local_idx];
        int findex = d_gidx_feature_map[gidx];

        if (findex == node.split.findex) {
          float fvalue = d_gidx_fvalue_map[gidx];

          if (fvalue <= node.split.fvalue) {
            d_position_tmp[ridx-row_begin] = left_child_nidx(pos);
          } else {
            d_position_tmp[ridx-row_begin] = right_child_nidx(pos);
          }
        }
      });
    position[d_idx] = position_tmp[d_idx];
  }
  dh::synchronize_n_devices(n_devices, dList);

}

void GPUHistBuilder::ColSampleTree() {
  if (param.colsample_bylevel == 1.0 && param.colsample_bytree == 1.0) return;

  feature_set_tree.resize(info->num_col);
  std::iota(feature_set_tree.begin(), feature_set_tree.end(), 0);
  feature_set_tree = col_sample(feature_set_tree, param.colsample_bytree);
}

void GPUHistBuilder::ColSampleLevel() {
  if (param.colsample_bylevel == 1.0 && param.colsample_bytree == 1.0) return;

  feature_set_level.resize(feature_set_tree.size());
  feature_set_level = col_sample(feature_set_tree, param.colsample_bylevel);
  std::vector<int> h_feature_flags(info->num_col, 0);
  for (auto fidx : feature_set_level) {
    h_feature_flags[fidx] = 1;
  }
  feature_flags = h_feature_flags;
}


    
bool GPUHistBuilder::UpdatePredictionCache(
    const DMatrix* data, std::vector<bst_float>* p_out_preds) {
  std::vector<bst_float>& out_preds = *p_out_preds;

  /*
  if (nodes.empty() || !p_last_fmat_ || data != p_last_fmat_) {
    return false;
  }


  if (!prediction_cache_initialised) {
    for(int d_idx=0;d_idx<n_devices;d_idx++){
      int device_idx = dList[d_idx];
      size_t row_begin = device_row_segments[d_idx];
      size_t row_end = device_row_segments[d_idx+1];

      prediction_cache[d_idx].copy(out_preds.begin() + row_begin, out_preds.begin() + row_end);
    }
    prediction_cache_initialised = true;
  }
  dh::synchronize_n_devices(n_devices, dList);

  float eps = param.learning_rate;
  for(int d_idx=0;d_idx<n_devices;d_idx++){
    int device_idx = dList[d_idx];
    dh::safe_cuda(cudaSetDevice(device_idx));
    
    size_t row_begin = device_row_segments[d_idx];
    size_t row_end = device_row_segments[d_idx+1];

    auto d_nodes = nodes[d_idx].data();
    auto d_position = position[d_idx].data();
    auto d_prediction_cache = prediction_cache[d_idx].data();

    dh::launch_n(prediction_cache[d_idx].size(), [=] __device__(int local_idx) {
        int pos = d_position[local_idx];
        d_prediction_cache[local_idx] += d_nodes[pos].weight * eps;
      });

    thrust::copy(prediction_cache[d_idx].tbegin(), prediction_cache[d_idx].tend(),
                 &out_preds[row_begin]);

  }
  dh::synchronize_n_devices(n_devices, dList);

  return true;
  */
  return false;
}
  
void GPUHistBuilder::Update(const std::vector<bst_gpair>& gpair,
                            DMatrix* p_fmat, RegTree* p_tree) {
  this->InitData(gpair, *p_fmat, *p_tree);
  this->InitFirstNode(gpair); // only creates first node on master, which is used by FindSplit (currently only on master)
  this->ColSampleTree();
  long long int elapsed=0;
  for (int depth = 0; depth < param.max_depth; depth++) {
    this->ColSampleLevel();
    dh::Timer time;
    this->BuildHist(depth);
    elapsed+=time.elapsed();
    printf("depth=%d\n",depth);
    time.printElapsed("BH Time");
    dh::Timer timesplit;
    this->FindSplit(depth);
    timesplit.printElapsed("FS Time");
    dh::Timer timestree;
    this->SynchronizeTree(depth);
    timestree.printElapsed("ST Time");
    dh::Timer timeupdatepos;
    this->UpdatePosition(depth);
    timeupdatepos.printElapsed("UP Time");
  }
  printf("Total BuildHist Time=%lld\n",elapsed);
  int master_device=dList[0];
  dense2sparse_tree(p_tree, nodes[master_device].tbegin(), nodes[master_device].tend(), param);
}
}  // namespace tree
}  // namespace xgboost
