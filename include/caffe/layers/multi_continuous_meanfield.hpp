#ifndef CAFFE_MULTI_CONTINUOUS_MEANFIELD_HPP_
#define CAFFE_MULTI_CONTINUOUS_MEANFIELD_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/layers/continuous_meanfield_iteration.hpp"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class MultiContinuousMeanfieldLayer : public Layer<Dtype> {

 public:
  explicit MultiContinuousMeanfieldLayer(const LayerParameter& param) : Layer<Dtype>(param) {}

  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const {
    return "MultiContinuousMeanfield";
  }
  virtual inline int ExactNumBottomBlobs() const { return 3; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  virtual void compute_spatial_kernel(float* const output_kernel);
  virtual void compute_bilateral_kernel(const Blob<Dtype>* const rgb_blob, const int n, float* const output_kernel);

  int count_;
  int num_;
  int channels_;
  int height_;
  int width_;
  int num_pixels_;

  Dtype theta_alpha_;
  Dtype theta_beta_;
  Dtype theta_gamma_;

  Dtype spatial_weight_;
  Dtype bilateral_weight_;

  int num_iterations_;

  boost::shared_array<Dtype> norm_feed_;
  Blob<Dtype> spatial_norm_;
  Blob<Dtype> bilateral_norms_;

  Blob<Dtype> sum_multiplier_;

  vector<Blob<Dtype>*> split_layer_bottom_vec_;
  vector<Blob<Dtype>*> split_layer_top_vec_;

  vector<Blob<Dtype>*> split_layer_bottom_vec1_;
  vector<Blob<Dtype>*> split_layer_top_vec1_;

  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs_;
  vector<shared_ptr<Blob<Dtype> > > split_layer_out_blobs1_;
  vector<shared_ptr<Blob<Dtype> > > iteration_output_blobs_;
  vector<shared_ptr<ContinuousMeanfieldIteration<Dtype> > > meanfield_iterations_;

  shared_ptr<SplitLayer<Dtype> > split_layer_;
  shared_ptr<SplitLayer<Dtype> > split_layer1_;

  shared_ptr<ModifiedPermutohedral> spatial_lattice_;
  boost::shared_array<float> bilateral_kernel_buffer_;
  vector<shared_ptr<ModifiedPermutohedral> > bilateral_lattices_;
};

}  // namespace caffe

#endif  // CAFFE_MULTI_CONTINUOUS_MEANFIELD_HPP_
