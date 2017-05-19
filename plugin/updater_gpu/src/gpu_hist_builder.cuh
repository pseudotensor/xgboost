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

//#define WHICHDEVICE DEVICE // choose which memory type to use (DEVICE or DEVICE_MANAGE)D
#define WHICHDEVICE DEVICE_MANAGED // choose which memory type to use (DEVICE or DEVICE_MANAGED)

namespace xgboost {

namespace tree {

  
struct DeviceGMat {
  dh::dvec<int,dh::memory_type::WHICHDEVICE> gidx;
  dh::dvec<int,dh::memory_type::WHICHDEVICE> ridx;
  void Init(const common::GHistIndexMatrix &gmat);
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
  dh::dvec<gpu_gpair,dh::memory_type::WHICHDEVICE> hist;

  void Init(int max_depth);

  void Reset();

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
  template <int BLOCK_THREADS>
  void FindSplitSpecialize(int depth);
  void InitFirstNode();
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
  DeviceGMat device_matrix;
  const DMatrix *p_last_fmat_;

  dh::bulk_allocator<dh::memory_type::WHICHDEVICE> ba;
  dh::CubMemory cub_mem;
  dh::dvec<int,dh::memory_type::WHICHDEVICE> gidx_feature_map;
  dh::dvec<int,dh::memory_type::WHICHDEVICE> hist_node_segments;
  dh::dvec<int,dh::memory_type::WHICHDEVICE> feature_segments;
  dh::dvec<float,dh::memory_type::WHICHDEVICE> gain;
  dh::dvec<NodeIdT,dh::memory_type::WHICHDEVICE> position;
  dh::dvec<NodeIdT,dh::memory_type::WHICHDEVICE> position_tmp;
  dh::dvec<float,dh::memory_type::WHICHDEVICE> gidx_fvalue_map;
  dh::dvec<float,dh::memory_type::WHICHDEVICE> fidx_min_map;
  DeviceHist hist;
  dh::dvec<cub::KeyValuePair<int, float>,dh::memory_type::WHICHDEVICE> argmax;
  dh::dvec<gpu_gpair,dh::memory_type::WHICHDEVICE> node_sums;
  dh::dvec<gpu_gpair,dh::memory_type::WHICHDEVICE> hist_scan;
  dh::dvec<gpu_gpair,dh::memory_type::WHICHDEVICE> device_gpair;
  dh::dvec<Node,dh::memory_type::WHICHDEVICE> nodes;
  dh::dvec<int,dh::memory_type::WHICHDEVICE> feature_flags;
  dh::dvec<bool,dh::memory_type::WHICHDEVICE> left_child_smallest;
  dh::dvec<bst_float,dh::memory_type::WHICHDEVICE> prediction_cache;
  bool prediction_cache_initialised;

  std::vector<int> feature_set_tree;
  std::vector<int> feature_set_level;
};
}  // namespace tree
}  // namespace xgboost
