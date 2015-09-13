#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <fstream>

#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/loc_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class LocLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  LocLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {

	// reshape theta blobs
	vector<int> locs_shape(2);
	locs_shape[0] = 10;
	locs_shape[1] = 2;
	blob_bottom_data_->Reshape(locs_shape);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~LocLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    LocLossParameter *loc_loss_param = layer_param.mutable_loc_loss_param();
    loc_loss_param->set_threshold(0.3);
    LocLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    LocLossLayer<Dtype> layer_weight_2(layer_param);
    layer_weight_2.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_2 =
        layer_weight_2.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype kErrorMargin = 1e-5;
    EXPECT_NEAR(loss_weight_1 * kLossWeight, loss_weight_2, kErrorMargin);
    // Make sure the loss is non-trivial.
    const Dtype kNonTrivialAbsThresh = 1e-1;
    EXPECT_GE(fabs(loss_weight_1), kNonTrivialAbsThresh);
  }

  void TestForwardWithPreDefinedTheta() {

	  std::ifstream fin("loc_loss_layer.testdata");

	  Dtype* locs = blob_bottom_data_->mutable_cpu_data();
	  for(int i=0; i<20; ++i) {
		  fin >> locs[i];
	  }

	  Dtype loss = (Dtype)0;
	  Dtype real_loss;
	  fin >> real_loss;

	  LayerParameter layer_param;
	  LocLossParameter *loc_loss_param = layer_param.mutable_loc_loss_param();
	  loc_loss_param->set_threshold(0.5);
	  LocLossLayer<Dtype> layer(layer_param);
	  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	  loss = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	  std::cout << "Computed loss = " << loss << ", real loss = " << real_loss << std::endl;

	  const Dtype kErrorMargin = 1e-5;
	  EXPECT_NEAR(loss, real_loss, kErrorMargin);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(LocLossLayerTest, TestGPUAndDouble);

TYPED_TEST(LocLossLayerTest, TestForward) {
	this->TestForward();
}


TYPED_TEST(LocLossLayerTest, TestForwardFabricatedData) {
	this->TestForwardWithPreDefinedTheta();
}

TYPED_TEST(LocLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  LocLossParameter *loc_loss_param = layer_param.mutable_loc_loss_param();
  loc_loss_param->set_threshold(0.3);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  LocLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
