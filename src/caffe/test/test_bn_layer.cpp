#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/bn_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class BNLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  BNLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 6, 2, 2)),
        blob_top_data_(new Blob<Dtype>(10, 6, 2, 2)) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_std(10);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_data_);
  }
  virtual ~BNLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_data_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(BNLayerTest, TestGPUAndDouble);

TYPED_TEST(BNLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability0) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e0);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability1) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-1);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability2) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-2);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability3) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-3);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability4) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-4);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability5) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-5);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability6) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-6);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability7) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-7);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability8) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-8);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability9) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-9);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(BNLayerTest, TestGradientStability10) {
  typedef typename TypeParam::Dtype Dtype;
  FillerParameter filler_param;
  filler_param.set_std(1e-10);
  GaussianFiller<Dtype> filler(filler_param);
  filler.Fill(this->blob_bottom_data_);

  LayerParameter layer_param;
  BNLayer<Dtype> layer(layer_param);
  GradientChecker<Dtype> checker(1e-4, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
