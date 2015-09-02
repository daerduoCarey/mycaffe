#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/bn_layer.hpp"

namespace caffe {

template <typename Dtype>
void BNLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom.size());

  int channels = bottom.at(0)->channels();

  this->blobs_.push_back(shared_ptr<Blob<Dtype> >(
      new Blob<Dtype>(1, channels, 1, 1)));
  this->blobs_.push_back(shared_ptr<Blob<Dtype> >(
      new Blob<Dtype>(1, channels, 1, 1)));

  caffe_set(this->blobs_.at(0)->count(), Dtype(1),
      this->blobs_.at(0)->mutable_cpu_data());
  caffe_set(this->blobs_.at(1)->count(), Dtype(0),
      this->blobs_.at(1)->mutable_cpu_data());
}

template <typename Dtype>
void BNLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom.size());
  CHECK_EQ(1, top.size());

  top.at(0)->ReshapeLike(*bottom.at(0));
}
template <typename Dtype>
void BNLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(1, bottom.size());
  CHECK_EQ(1, top.size());

  Blob<Dtype>* input = bottom.at(0);
  Blob<Dtype>* output = top.at(0);

  shared_ptr<Blob<Dtype> > gammas = this->blobs_.at(0);
  shared_ptr<Blob<Dtype> > betas = this->blobs_.at(1);

  Blob<Dtype> means;
  means.Reshape(1, input->channels(), 1, 1);
  caffe_set(means.count(), Dtype(0), means.mutable_cpu_data());

  vars_.Reshape(1, input->channels(), 1, 1);
  caffe_set(vars_.count(), Dtype(0), vars_.mutable_cpu_data());

  diffs_.ReshapeLike(*input);

  int num = input->num();
  int channels = input->channels();
  int height = input->height();
  int width = input->width();

  CHECK_EQ(num, output->num());
  CHECK_EQ(channels, output->channels());
  CHECK_EQ(height, output->height());
  CHECK_EQ(width, output->width());

  const Dtype* input_index = input->cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      Dtype* mean = means.mutable_cpu_data() + c;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          *mean += *input_index++;
        }
      }
    }
  }

  Dtype count = num * height * width;

  caffe_scal(means.count(), Dtype(1.0 / count), means.mutable_cpu_data());

  input_index = input->cpu_data();
  Dtype* diff_index = diffs_.mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      Dtype* var = vars_.mutable_cpu_data() + c;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          Dtype diff = *input_index++ - means.cpu_data()[c];
          *diff_index++ = diff;
          *var += diff * diff;
        }
      }
    }
  }

  caffe_scal(vars_.count(), Dtype(1.0 / count),
      vars_.mutable_cpu_data());

  caffe_add_scalar(vars_.count(), Dtype(1e-6), vars_.mutable_cpu_data());

  input_index = input->cpu_data();
  Dtype* output_index = output->mutable_cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      Dtype mean = means.cpu_data()[c];
      Dtype gamma = gammas->cpu_data()[c];
      Dtype beta = betas->cpu_data()[c];
      Dtype stdev = sqrt(vars_.cpu_data()[c]);
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          Dtype normed = (*input_index++ - mean) / stdev;
          *output_index++ = gamma * normed + beta;
        }
      }
    }
  }
}


template <typename Dtype>
void BNLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  Blob<Dtype>* input = bottom.at(0);
  Blob<Dtype>* output = top.at(0);

  int num = input->num();
  int channels = input->channels();
  int height = input->height();
  int width = input->width();

  shared_ptr<Blob<Dtype> > gammas = this->blobs_.at(0);
  shared_ptr<Blob<Dtype> > betas = this->blobs_.at(1);

  caffe_set(gammas->count(), Dtype(0), gammas->mutable_cpu_diff());
  caffe_set(betas->count(), Dtype(0), betas->mutable_cpu_diff());

  if (propagate_down.at(0)) {
    Dtype count = num * height * width;

    CHECK_EQ(num, output->num());
    CHECK_EQ(channels, output->channels());
    CHECK_EQ(height, output->height());
    CHECK_EQ(width, output->width());

    Blob<Dtype> dl_dsigmasq;
    dl_dsigmasq.Reshape(1, channels, 1, 1);
    caffe_set(dl_dsigmasq.count(), Dtype(0), dl_dsigmasq.mutable_cpu_data());

    const Dtype* diff = diffs_.cpu_data();
    const Dtype* output_diff = output->cpu_diff();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        Dtype var = vars_.cpu_data()[c];
        Dtype gamma = gammas->cpu_data()[c];
        Dtype* dl_dsigmasq_val = dl_dsigmasq.mutable_cpu_data() + c;
        Dtype var_pow = pow(var, -3.0/2.0);
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            *dl_dsigmasq_val += gamma * *output_diff++ * *diff++ * -0.5 *
                var_pow;
          }
        }
      }
    }

    Blob<Dtype> dl_dmu;
    dl_dmu.Reshape(1, channels, 1, 1);
    caffe_set(dl_dmu.count(), Dtype(0), dl_dmu.mutable_cpu_data());

    diff = diffs_.cpu_data();
    output_diff = output->cpu_diff();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        Dtype var = vars_.cpu_data()[c];
        Dtype dl_dsigmasq_val = dl_dsigmasq.cpu_data()[c];
        Dtype gamma = gammas->cpu_data()[c];
        Dtype* dl_dmu_val = dl_dmu.mutable_cpu_data() + c;
        Dtype inv_sqr_var = pow(var, -0.5);
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            *dl_dmu_val += gamma * *output_diff++ * -inv_sqr_var +
                dl_dsigmasq_val * -2.0 * *diff++ / count;
          }
        }
      }
    }

    caffe_set(input->count(), Dtype(0), input->mutable_cpu_diff());

    diff = diffs_.cpu_data();
    output_diff = output->cpu_diff();
    Dtype* input_diff = input->mutable_cpu_diff();
    for (int n = 0; n < num; ++n) {
      for (int c = 0; c < channels; ++c) {
        Dtype var = vars_.cpu_data()[c];
        Dtype dl_dsigmasq_val = dl_dsigmasq.cpu_data()[c];
        Dtype dl_dmu_val = dl_dmu.cpu_data()[c];
        Dtype gamma = gammas->cpu_data()[c];
        Dtype inv_sqr_var = pow(var, -0.5);
        for (int h = 0; h < height; ++h) {
          for (int w = 0; w < width; ++w) {
            *input_diff++ += gamma * *output_diff++ * inv_sqr_var +
                dl_dsigmasq_val * 2.0 * *diff++ / count + dl_dmu_val / count;
          }
        }
      }
    }
  }

  const Dtype* output_diff = output->cpu_diff();
  const Dtype* output_val = output->cpu_data();
  for (int n = 0; n < num; ++n) {
    for (int c = 0; c < channels; ++c) {
      Dtype gamma = gammas->cpu_data()[c];
      Dtype beta = betas->cpu_data()[c];
      Dtype* gamma_diff = gammas->mutable_cpu_diff() + c;
      Dtype* beta_diff = betas->mutable_cpu_diff() + c;
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          *gamma_diff += *output_diff * (*output_val++ - beta) / gamma;
          *beta_diff += *output_diff++;
        }
      }
    }
  }
}


#ifdef CPU_ONLY
STUB_GPU(BNLayer);
#endif

INSTANTIATE_CLASS(BNLayer);
REGISTER_LAYER_CLASS(BN);

}  // namespace caffe
