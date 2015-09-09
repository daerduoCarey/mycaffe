#include <algorithm>
#include <vector>
#include <fstream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/power_file_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class PowerFileLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  PowerFileLayerTest()
      : blob_bottom_(new Blob<Dtype>(6, 3, 4, 2)),
        blob_top_(new Blob<Dtype>()),
        shift_(new Blob<Dtype>()) {

    Caffe::set_random_seed(1701);

    // reshape shift_ blob
    vector<int> shift_shape(1);
    shift_shape[0] = blob_bottom_->count(1);
    shift_->Reshape(shift_shape);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    filler.Fill(this->shift_);

    // output shift values to file
    const Dtype* shift = shift_->cpu_data();
    std::ofstream fout(".test_power_file_layer.txt");
    for(int i=0; i<shift_->count(); ++i) {
    	fout << shift[i] << "\t";
    }
    fout << std::endl;
    fout.close();

    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~PowerFileLayerTest() {
	  delete blob_bottom_;
	  delete blob_top_;
	  delete shift_;
  }

  void TestForward() {
    LayerParameter layer_param;
    PowerFileParameter *power_file_param = layer_param.mutable_power_file_param();
    power_file_param->set_shift_file(".test_power_file_layer.txt");
    PowerFileLayer<Dtype> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    // Now, check values
    const Dtype* shift = this->shift_->cpu_data();
    const int size = this->blob_bottom_->count(1);
    const Dtype* bottom_data = this->blob_bottom_->cpu_data();
    const Dtype* top_data = this->blob_top_->cpu_data();
    const Dtype min_precision = 1e-5;
    for (int i = 0; i < this->blob_bottom_->count(); ++i) {
      Dtype expected_value = shift[i % size] + bottom_data[i];

      Dtype precision = std::max(
    		  Dtype(std::abs(expected_value * Dtype(1e-4))), min_precision);
      EXPECT_NEAR(expected_value, top_data[i], precision);
    }
  }

  void TestBackward() {
	LayerParameter layer_param;
	PowerFileParameter *power_file_param = layer_param.mutable_power_file_param();
	power_file_param->set_shift_file(".test_power_file_layer.txt");
    PowerFileLayer<Dtype> layer(layer_param);
    GradientChecker<Dtype> checker(1e-3, 1e-2, 1701, 0., 0.01);
    checker.CheckGradientEltwise(&layer, this->blob_bottom_vec_,
        this->blob_top_vec_);
  }

  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  Blob<Dtype>* const shift_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(PowerFileLayerTest, TestGPUAndDouble);

TYPED_TEST(PowerFileLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestForward();
}

TYPED_TEST(PowerFileLayerTest, TestBackward) {
  typedef typename TypeParam::Dtype Dtype;
  this->TestBackward();
}

}  // namespace caffe
