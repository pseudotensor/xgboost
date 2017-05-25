/*!
 * Copyright 2016 Rory mitchell
 */
#pragma once
#include <thrust/device_vector.h>
#include <xgboost/tree_updater.h>
#include <cub/util_type.cuh>  // Need key value pair definition
#include <vector>
#include "../../src/common/hist_util.h"
#include "../../src/tree/param.h"
#include "device_helpers.cuh"
#include "types.cuh"
#include "nccl.h"

namespace xgboost {

namespace tree {

  
struct DeviceGMat {
  dh::dvec<int> gidx;
  dh::dvec<int> ridx;
  void Init(int device_idx, const common::GHistIndexMatrix& gmat, bst_uint begin, bst_uint end);
};

struct HistBuilder {
  gpu_gpair *d_hist;
  int n_bins;
  __host__ __device__ HistBuilder(gpu_gpair *ptr, int n_bins);
  __device__ void Add(gpu_gpair gpair, int gidx, int nidx) const;
  __device__ gpu_gpair Get(int gidx, int nidx) const;
};

struct DeviceHist {
  int n_bins;
  dh::dvec<gpu_gpair> data;

  void Init(int max_depth);

  void Reset(int device_idx);

  HistBuilder GetBuilder();

  gpu_gpair *GetLevelPtr(int depth);

  int LevelSize(int depth);
};

class GPUHistBuilder {
 public:
  GPUHistBuilder();
  ~GPUHistBuilder();
  void Init(const TrainParam &param);

  void UpdateParam(const TrainParam &param) {
    this->param = param;
    this->gpu_param = GPUTrainingParam(param.min_child_weight, param.reg_lambda,
                                       param.reg_alpha, param.max_delta_step);
  }

  void InitData(const std::vector<bst_gpair> &gpair, DMatrix &fmat,  // NOLINT
                const RegTree &tree);
  void Update(const std::vector<bst_gpair> &gpair, DMatrix *p_fmat,
              RegTree *p_tree);
  void BuildHist(int depth);
  void FindSplit(int depth);
  void SynchronizeTree(int depth);
  template <int BLOCK_THREADS>
  void FindSplitSpecialize(int depth);
  void InitFirstNode(const std::vector<bst_gpair> &gpair);
  void UpdatePosition(int depth);
  void UpdatePositionDense(int depth);
  void UpdatePositionSparse(int depth);
  void ColSampleTree();
  void ColSampleLevel();
  bool UpdatePredictionCache(const DMatrix *data,
                             std::vector<bst_float> *p_out_preds);

  TrainParam param;
  GPUTrainingParam gpu_param;
  common::HistCutMatrix hmat_;
  common::GHistIndexMatrix gmat_;
  MetaInfo *info;
  bool initialised;
  bool is_dense;
  std::vector<DeviceGMat> device_matrix;
  const DMatrix *p_last_fmat_;

  // choose which memory type to use (DEVICE or DEVICE_MANAGED)
  dh::bulk_allocator<dh::memory_type::DEVICE> ba;
  //dh::bulk_allocator<dh::memory_type::DEVICE_MANAGED> ba;
  dh::CubMemory cub_mem;
  dh::dvec<int> hist_node_segments;
  dh::dvec<int> feature_segments;
  dh::dvec<float> gain;
  dh::dvec<float> fidx_min_map;
  DeviceHist hist_temp;
  dh::dvec<cub::KeyValuePair<int, float>> argmax;
  dh::dvec<gpu_gpair> node_sums;
  dh::dvec<gpu_gpair> hist_scan;
  dh::dvec<int> feature_flags;
  bool prediction_cache_initialised;

  bst_uint num_rows;
  int n_devices;
  std::vector<int> dList;

  std::vector<dh::dvec<bst_float>> prediction_cache;
  std::vector<dh::dvec<float>> gidx_fvalue_map;
  std::vector<dh::dvec<int>> gidx_feature_map;
  std::vector<dh::dvec<NodeIdT>> position;
  std::vector<dh::dvec<NodeIdT>> position_tmp;
  std::vector<dh::dvec<gpu_gpair>> device_gpair;
  std::vector<dh::dvec<Node>> nodes;
  std::vector<dh::dvec<bool>> left_child_smallest;
  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;
  std::vector<int> device_row_segments;
  std::vector<int> device_element_segments;
  std::vector<DeviceHist> hist_vec;
  std::vector<ncclComm_t> comms;
  std::vector<cudaStream_t> streams;
};
}  // namespace tree
}  // namespace xgboost
