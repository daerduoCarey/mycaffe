#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/bn_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void MeanForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, Dtype* mean) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;
    atomicAdd(mean + c, bottom_data[index]);
  }
}

template <typename Dtype>
__global__ void DiffForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const Dtype* mean, Dtype* diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;
    diff[index] = bottom_data[index] - mean[c];
  }
}

template <typename Dtype>
__global__ void VarForward(const int nthreads,
    const int num, const int channels, const int height,
    const int width, const Dtype* diff, Dtype* var) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;

    atomicAdd(var + c, diff[index] * diff[index]);
  }
}

template <typename Dtype>
__global__ void NormalizeForward(const int nthreads, const Dtype* bottom_data,
    const int num, const int channels, const int height,
    const int width, const Dtype* means, const Dtype* vars,
    const Dtype* gammas, const Dtype* betas, Dtype* top_data) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;

    Dtype mean = means[c];
    Dtype gamma = gammas[c];
    Dtype beta = betas[c];
    Dtype stdev = sqrt(vars[c]);

    Dtype input = bottom_data[index];

    Dtype normed = (input - mean) / stdev;
    top_data[index] = gamma * normed + beta;
  }
}

template <typename Dtype>
__global__ void DlDsigmasqBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const Dtype* diffs, const Dtype* vars,
    const Dtype* gammas, Dtype* dl_dsigmasq) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;

    Dtype var = vars[c];
    Dtype gamma = gammas[c];
    Dtype var_sqrt = sqrtf(var);
    Dtype var_3_2 = var_sqrt * var_sqrt * var_sqrt;
    Dtype var_pow = 1.0 / var_3_2;

    atomicAdd(dl_dsigmasq + c, gamma * top_diff[index] * diffs[index] *
        -0.5 * var_pow);
  }
}

template <typename Dtype>
__global__ void DlDmuBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const Dtype* diffs, const Dtype* vars,
    const Dtype* gammas, const Dtype* dl_dsigmasq, Dtype* dl_dmu) {
  int count = num * height * width;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;

    Dtype var = vars[c];
    Dtype gamma = gammas[c];
    Dtype var_sqrt = sqrtf(var);
    Dtype inv_sqr_var = 1.0 / var_sqrt;
    Dtype dl_dsigmasq_val = dl_dsigmasq[c];

    atomicAdd(dl_dmu + c, gamma * top_diff[index] * -inv_sqr_var +
        dl_dsigmasq_val * -2.0 * diffs[index] / count);
  }
}

template <typename Dtype>
__global__ void DlDxBackward(const int nthreads, const Dtype* top_diff,
    const int num, const int channels, const int height,
    const int width, const Dtype* diffs, const Dtype* vars,
    const Dtype* gammas, const Dtype* dl_dsigmasq, const Dtype* dl_dmu,
    Dtype* bottom_diff) {
  int count = num * height * width;
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;

    Dtype var = vars[c];
    Dtype gamma = gammas[c];
    Dtype var_sqrt = sqrtf(var);
    Dtype inv_sqr_var = 1.0 / var_sqrt;
    Dtype dl_dsigmasq_val = dl_dsigmasq[c];
    Dtype dl_dmu_val = dl_dmu[c];

    bottom_diff[index] = gamma * top_diff[index] * inv_sqr_var +
        dl_dsigmasq_val * 2.0 * diffs[index] / count + dl_dmu_val / count;
  }
}

template <typename Dtype>
__global__ void ParamsBackward(const int nthreads, const Dtype* top_diff,
    const Dtype* top_data,
    const int num, const int channels, const int height,
    const int width, const Dtype* gammas, const Dtype* betas,
    Dtype* gamma_diff, Dtype* beta_diff) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int c = (index / width / height) % channels;

    Dtype gamma = gammas[c];
    Dtype beta = betas[c];

    atomicAdd(gamma_diff + c,
        top_diff[index] * (top_data[index] - beta) / gamma);
    atomicAdd(beta_diff + c, top_diff[index]);
  }
}

template <>
void BNLayer<double>::Forward_gpu(
    const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void BNLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom.size());
  CHECK_EQ(1, top.size());

  Blob<Dtype>* input = bottom.at(0);
  Blob<Dtype>* output = top.at(0);

  shared_ptr<Blob<Dtype> > gammas = this->blobs_.at(0);
  shared_ptr<Blob<Dtype> > betas = this->blobs_.at(1);

  Blob<Dtype> means;
  means.Reshape(1, input->channels(), 1, 1);
  caffe_gpu_set(means.count(), Dtype(0), means.mutable_gpu_data());

  vars_.Reshape(1, input->channels(), 1, 1);
  caffe_gpu_set(vars_.count(), Dtype(0), vars_.mutable_gpu_data());

  diffs_.ReshapeLike(*input);

  int num = input->num();
  int channels = input->channels();
  int height = input->height();
  int width = input->width();
  int nthreads = input->count();

  CHECK_EQ(num, output->num());
  CHECK_EQ(channels, output->channels());
  CHECK_EQ(height, output->height());
  CHECK_EQ(width, output->width());

  Timer timer;

  timer.Start();
  // NOLINT_NEXT_LINE(whitespace/operators)
  MeanForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>> (
      nthreads, input->gpu_data(), num, channels, height, width,
      means.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  timer.Stop();
  // DLOG(INFO) << "MeanForward: " << timer.MilliSeconds();

  Dtype count = num * height * width;

  caffe_gpu_scal(means.count(), Dtype(1.0 / count), means.mutable_gpu_data());

  timer.Start();
  // NOLINT_NEXT_LINE(whitespace/operators)
  DiffForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>> (
      nthreads, input->gpu_data(), num, channels, height, width,
      means.gpu_data(), diffs_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  timer.Stop();
  // DLOG(INFO) << "DiffForward: " << timer.MilliSeconds();

  timer.Start();
  // NOLINT_NEXT_LINE(whitespace/operators)
  VarForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads), CAFFE_CUDA_NUM_THREADS>>> (
      nthreads, num, channels, height, width, diffs_.gpu_data(),
      vars_.mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  timer.Stop();
  // DLOG(INFO) << "VarForward: " << timer.MilliSeconds();

  caffe_gpu_scal(vars_.count(), Dtype(1.0 / count),
      vars_.mutable_gpu_data());

  caffe_gpu_add_scalar(vars_.count(), Dtype(1e-6), vars_.mutable_gpu_data());

  timer.Start();
  // NOLINT_NEXT_LINE(whitespace/operators)
  NormalizeForward<Dtype><<<CAFFE_GET_BLOCKS(nthreads),
      CAFFE_CUDA_NUM_THREADS>>> (
      nthreads, input->gpu_data(), num, channels, height, width,
      means.gpu_data(), vars_.gpu_data(), gammas->gpu_data(), betas->gpu_data(),
      output->mutable_gpu_data());
  CUDA_POST_KERNEL_CHECK;
  timer.Stop();
  // DLOG(INFO) << "NormalizeForward: " << timer.MilliSeconds();
}

template <>
void BNLayer<double>::Backward_gpu(const vector<Blob<double>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<double>*>& bottom) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void BNLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype>* input = bottom.at(0);
  Blob<Dtype>* output = top.at(0);

  int num = input->num();
  int channels = input->channels();
  int height = input->height();
  int width = input->width();
  int count = input->count();

  shared_ptr<Blob<Dtype> > gammas = this->blobs_.at(0);
  shared_ptr<Blob<Dtype> > betas = this->blobs_.at(1);

  caffe_gpu_set(gammas->count(), Dtype(0), gammas->mutable_gpu_diff());
  caffe_gpu_set(betas->count(), Dtype(0), betas->mutable_gpu_diff());

  Timer timer;

  if (propagate_down.at(0)) {
    CHECK_EQ(num, output->num());
    CHECK_EQ(channels, output->channels());
    CHECK_EQ(height, output->height());
    CHECK_EQ(width, output->width());

    Blob<Dtype> dl_dsigmasq;
    dl_dsigmasq.Reshape(1, channels, 1, 1);
    caffe_gpu_set(dl_dsigmasq.count(), Dtype(0),
        dl_dsigmasq.mutable_gpu_data());

    timer.Start();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DlDsigmasqBackward<Dtype><<<CAFFE_GET_BLOCKS(count),
        CAFFE_CUDA_NUM_THREADS>>> (
        count, output->gpu_diff(), num, channels, height, width,
        diffs_.gpu_data(), vars_.gpu_data(), gammas->gpu_data(),
        dl_dsigmasq.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    timer.Stop();
    // DLOG(INFO) << "DlDsigmasqBackward: " << timer.MilliSeconds();

    Blob<Dtype> dl_dmu;
    dl_dmu.Reshape(1, channels, 1, 1);
    caffe_gpu_set(dl_dmu.count(), Dtype(0), dl_dmu.mutable_gpu_data());

    timer.Start();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DlDmuBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
        count, output->gpu_diff(), num, channels, height, width,
        diffs_.gpu_data(), vars_.gpu_data(), gammas->gpu_data(),
        dl_dsigmasq.gpu_data(), dl_dmu.mutable_gpu_data());
    CUDA_POST_KERNEL_CHECK;
    timer.Stop();
    // DLOG(INFO) << "DlDmuBackward: " << timer.MilliSeconds();

    caffe_gpu_set(count, Dtype(0), input->mutable_gpu_diff());

    timer.Start();
    // NOLINT_NEXT_LINE(whitespace/operators)
    DlDxBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
        count, output->gpu_diff(), num, channels, height, width,
        diffs_.gpu_data(), vars_.gpu_data(), gammas->gpu_data(),
        dl_dsigmasq.gpu_data(), dl_dmu.gpu_data(), input->mutable_gpu_diff());
    CUDA_POST_KERNEL_CHECK;
    timer.Stop();
    // DLOG(INFO) << "DlDxBackward: " << timer.MilliSeconds();
  }

  timer.Start();
  // NOLINT_NEXT_LINE(whitespace/operators)
  ParamsBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>> (
      count, output->gpu_diff(), output->gpu_data(), num, channels, height,
      width, gammas->gpu_data(), betas->gpu_data(), gammas->mutable_gpu_diff(),
      betas->mutable_gpu_diff());
  CUDA_POST_KERNEL_CHECK;
  timer.Stop();
  // DLOG(INFO) << "ParamsBackward: " << timer.MilliSeconds();
}

INSTANTIATE_LAYER_GPU_FUNCS(BNLayer);

}  // namespace caffe
