# CUDA Accelerated Tree Construction Algorithms
This plugin adds GPU accelerated tree construction algorithms to XGBoost.
## Usage
Specify the 'updater' parameter as one of the following algorithms. 

### Algorithms
| updater | Description |
| --- | --- |
grow_gpu | The standard XGBoost tree construction algorithm. Performs exact search for splits. Slower and uses considerably more memory than 'grow_gpu_hist' |
grow_gpu_hist | Equivalent to the XGBoost fast histogram algorithm. Faster and uses considerably less memory. Splits may be less accurate. |

### Supported parameters 
| parameter | grow_gpu | grow_gpu_hist |
| --- | --- | --- |
subsample | &#10004; | &#10004; |
colsample_bytree | &#10004; | &#10004;|
colsample_bylevel | &#10004; | &#10004; |
max_bin | &#10006; | &#10004; |
gpu_id | &#10004; | &#10004; | 
n_gpus | &#10006; | &#10004; | 

The device ordinal can be selected using the 'gpu_id' parameter, which defaults to 0.

Multiple GPUs can be used with the grow_gpu_hist parameter using the n_gpus parameter, which defaults to -1 (indicating use all visible GPUs).   NVIDIA NCCL is required to be installed.  If gpu_id is specified as non-zero, the gpu device order is mod(gpu_id + i) % n_visible_devices for i=0 to n_gpus-1.

This plugin currently works with the CLI version and python version.

Python example:
```python
param['gpu_id'] = 1
param['max_bin'] = 16
param['updater'] = 'grow_gpu_hist'
```
## Benchmarks
To run benchmarks on synthetic data for binary classification:
```bash
$ python benchmark/benchmark.py
```

Training time time on 1000000 rows x 50 columns with 500 boosting iterations on i7-6700K CPU @ 4.00GHz and Pascal Titan X.

| Updater | Time (s) |
| --- | --- |
| grow_gpu_hist | 11.09 |
| grow_fast_histmaker (histogram XGBoost - CPU) | 41.75 |
| grow_gpu | 193.90 |
| grow_colmaker (standard XGBoost - CPU) | 720.12 |


[See here](http://dmlc.ml/2016/12/14/GPU-accelerated-xgboost.html) for additional performance benchmarks of the 'grow_gpu' updater.

## Test
To run tests:
```bash
$ python -m nose test/python/
```
## Dependencies
A CUDA capable GPU with at least compute capability >= 3.5 (the algorithm depends on shuffle and vote instructions introduced in Kepler).

Building the plug-in requires CUDA Toolkit 7.5 or later (https://developer.nvidia.com/cuda-downloads)

The plugin also depends on CUB 1.6.4 - https://nvlabs.github.io/cub/ . CUB is a header only cuda library which provides sort/reduce/scan primitives.

```bash
cd <MY_CUB_DIRECTORY>
git clone https://github.com/NVlabs/cub.git
```
replacing <MY_CUB_DIRECTORY> with a directory of your choice.

On Linux, NVIDIA NCCL should be installed from: https://github.com/NVIDIA/nccl , installing as:

```bash
git clone https://github.com/NVIDIA/nccl.git
cd nccl
make
sudo ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib/
sudo make PREFIX=/usr/local/cuda/
sudo chmod a+r /usr/local/cuda/include/nccl*
sudo chmod a+r /usr/local/cuda/lib64/*nccl*
```
This ensures nccl is installed in same location as cuda libraries.

On Windows, NVIDIA NCCL is installed as:
```bash
# for windows support, which may eventually make it into the original repo.
git clone https://github.com/h2oai/nccl-windows
cd nccl
make
sudo ln -s /usr/local/cuda/lib64 /usr/local/cuda/lib/
sudo make PREFIX=/usr/local/cuda/
sudo chmod a+r /usr/local/cuda/include/nccl*
sudo chmod a+r /usr/local/cuda/lib64/*nccl*
```
This ensures nccl is installed in same location as cuda libraries.

On Windows, use the xgboost/windows/nccl.sln file in visual studio and build as release x64.  Copy windows\x64\Release/nccl.dll nccl.exp and nccl.lib to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\lib\x64 or wherever you installed CUDA Toolkit in windows.  Copy src/nccl.h to C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include .  You may have to copy these libraries to your path.

To include NCCL (and hence Multi-GPU support) ensure NCCL 1 in xgboost/plugin/updater_gpu/src/gpu_hist_builder.cuh and xgboost/plugin/updater_gpu/src/device_helpers.cuh and in xgboost/CMakelists.txt uncomment line with target_link_libraries(updater_gpu nccl) so the nccl library is loaded.  If you don't want to install NCCL and multi-GPU support, set NCCL 0 in those locations.

## Build

From the command line on Linux starting from the xgboost directory:

On Linux, from the xgboost directory:
```bash
$ mkdir build
$ cd build
$ cmake .. -DPLUGIN_UPDATER_GPU=ON
$ make
```
If 'make' fails try invoking make again. There can sometimes be problems with the order items are built.

On Windows using cmake, see what options for Generators you have for cmake, and choose one with [arch] replaced by Win64:
```bash
cmake -help
```
Then run cmake as:
```bash
$ cmake .. -G"Visual Studio 14 2015 Win64" -DPLUGIN_UPDATER_GPU=ON -DCUB_DIRECTORY=<MY_CUB_DIRECTORY>
```
where visual studio community 2015, supported by cuda toolkit (http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#axzz4isREr2nS), can be downloaded from: https://my.visualstudio.com/Downloads?q=Visual%20Studio%20Community%202015 .  You may also be able to use a later version of visual studio depending on whether the CUDA toolkit supports it.  Note that Mingw cannot be used with cuda.

### Using make
Now, it also supports the usual 'make' flow to build gpu-enabled tree construction plugins. It's currently only tested on Linux. From the xgboost directory
```bash
# make sure CUDA SDK bin directory is in the 'PATH' env variable
$ make PLUGIN_UPDATER_GPU=ON
```

### For Developers!

Now, some of the code-base inside gpu plugins have googletest unit-tests inside 'tests/'.
They can be enabled run along with other unit-tests inside '<xgboostRoot>/tests/cpp' using:
```bash
# make sure CUDA SDK bin directory is in the 'PATH' env variable
# below 2 commands need only be executed once
$ source ./dmlc-core/scripts/travis/travis_setup_env.sh
$ make -f dmlc-core/scripts/packages.mk gtest
$ make PLUGIN_UPDATER_GPU=ON GTEST_PATH=${CACHE_PREFIX} test
```

## Changelog
##### 2017/6/5

* Multi-GPU support for histogram method using NVIDIA NCCL.

## Changelog
##### 2017/5/31
* Faster version of the grow_gpu plugin
* Added support for building gpu plugin through 'make' flow too

## Changelog
##### 2017/5/19
* Further performance enhancements for histogram method.

##### 2017/5/5
* Histogram performance improvements
* Fix gcc build issues 

##### 2017/4/25
* Add fast histogram algorithm
* Fix Linux build
* Add 'gpu_id' parameter

## References
[Mitchell, Rory, and Eibe Frank. Accelerating the XGBoost algorithm using GPU computing. No. e2911v1. PeerJ Preprints, 2017.](https://peerj.com/preprints/2911/)

## Author
Rory Mitchell 

Please report bugs to the xgboost/issues page. You can tag me with @RAMitchell.

Otherwise I can be contacted at r.a.mitchell.nz at gmail.


