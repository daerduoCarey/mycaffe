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
#include "caffe/st_loss_layer.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

namespace caffe {

template <typename TypeParam>
class STLossLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  STLossLayerTest()
      : blob_bottom_data_(new Blob<Dtype>()),
        blob_top_loss_(new Blob<Dtype>()) {

	// reshape theta blobs
	vector<int> theta_shape(3);
	theta_shape[0] = 10;
	theta_shape[1] = 2;
	theta_shape[2] = 3;
	blob_bottom_data_->Reshape(theta_shape);

    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_loss_);
  }
  virtual ~STLossLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_loss_;
  }

  void TestForward() {
    // Get the loss without a specified objective weight -- should be
    // equivalent to explicitly specifiying a weight of 1.
    LayerParameter layer_param;
    STLossParameter *st_loss_param = layer_param.mutable_st_loss_param();
    st_loss_param->set_output_h(10);
    st_loss_param->set_output_w(10);
    STLossLayer<Dtype> layer_weight_1(layer_param);
    layer_weight_1.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
    const Dtype loss_weight_1 =
        layer_weight_1.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

    // Get the loss again with a different objective weight; check that it is
    // scaled appropriately.
    const Dtype kLossWeight = 3.7;
    layer_param.add_loss_weight(kLossWeight);
    STLossLayer<Dtype> layer_weight_2(layer_param);
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

	  	std::ifstream fin("st_loss_layer.testdata");

	  	int n;
	  	fin >> n;

	  	std::cout << "There are " << n << " sets of theta's to test in parallel." << std::endl;

	  	// reshape theta blobs
	  	vector<int> theta_shape(3);
	  	theta_shape[0] = n;
	  	theta_shape[1] = 2;
	  	theta_shape[2] = 3;
	  	this->blob_bottom_data_->Reshape(theta_shape);

	  	Dtype* theta = this->blob_bottom_data_->mutable_cpu_data();

	      // fill the values
	  	for(int i=0; i<n; ++i) {
	  		for(int j=0; j<6; ++j) {
	  			fin >> theta[6 * i + j];
	  			std::cout << theta[6 * i + j] << "\t";
	  		}
	  		std::cout << "\n";
	  	}

	  	LayerParameter layer_param;
	  	STLossParameter *st_loss_param = layer_param.mutable_st_loss_param();
	  	st_loss_param->set_output_h(4);
	  	st_loss_param->set_output_w(4);

	  	STLossLayer<Dtype> layer(layer_param);
	  	layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
	  	const Dtype loss = layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);

	  	Dtype real_loss;
	  	fin >> real_loss;

	  	std::cout << "Computed loss is " << loss << ". Real loss is " << real_loss << std::endl;

	    const Dtype kErrorMargin = 1e-5;
	    EXPECT_NEAR(loss, real_loss, kErrorMargin);
  }

  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_top_loss_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(STLossLayerTest, TestGPUAndDouble);

TYPED_TEST(STLossLayerTest, TestForward) {
	this->TestForward();
}


TYPED_TEST(STLossLayerTest, TestForwardFabricatedData) {
	this->TestForwardWithPreDefinedTheta();
}

TYPED_TEST(STLossLayerTest, TestGradient) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  STLossParameter *st_loss_param = layer_param.mutable_st_loss_param();
  st_loss_param->set_output_h(10);
  st_loss_param->set_output_w(10);
  const Dtype kLossWeight = 3.7;
  layer_param.add_loss_weight(kLossWeight);
  STLossLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  GradientChecker<Dtype> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientExhaustive(&layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

}  // namespace caffe
