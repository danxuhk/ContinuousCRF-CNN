/*!
 *  \brief     A CNN implementation for continuous mean-field updating with inspiration from the CRF-RNN implementation
 *  \authors   Dan Xu
 *  \version   1.0
 *  \date      2016
 *  \citation   If you use this code, please consider citing the paper:
 *      @inproceedings{xu2017multi,
 *         title={Multi-Scale Continuous CRFs as Sequential Deep Networks for Monocular Depth Estimation},
 *         author={Xu, Dan and Ricci, Elisa and Ouyang, Wanli and Wang, Xiaogang and Sebe, Nicu},
 *         journal={CVPR},
 *         year={2017}
 *      }
*/

#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

namespace caffe {

/**
 * To be invoked once only immediately after construction.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::OneTimeSetUp(
    Blob<Dtype>* const unary_terms,
    Blob<Dtype>* const score_map_,
    Blob<Dtype>* const output_blob_,
    const shared_ptr<ModifiedPermutohedral> spatial_lattice,
    const Blob<Dtype>* const spatial_norm,
    const Blob<Dtype>* const sum_multiplier_) {

  spatial_lattice_ = spatial_lattice;
  spatial_norm_ = spatial_norm;

  score_map = score_map_;
  output_blob = output_blob_;
  sum_multiplier = sum_multiplier_;

  count_ = unary_terms->count();
  num_ = unary_terms->num();
  channels_ = unary_terms->channels();
  height_ = unary_terms->height();
  width_ = unary_terms->width();
  num_pixels_ = height_ * width_;

  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Meanfield iteration skipping parameter initialization.";
  } else {
    blobs_.resize(2);
    blobs_[0].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // spatial kernel weight
    blobs_[1].reset(new Blob<Dtype>(1, 1, channels_, channels_)); // bilateral kernel weight
  }

  adding_output_blob_.Reshape(num_, channels_, height_, width_);
  spatial_out_blob_.Reshape(num_, channels_, height_, width_);
  bilateral_out_blob_.Reshape(num_, channels_, height_, width_);
  message_passing_.Reshape(num_, channels_, height_, width_);
  message_passing_norm_.Reshape(num_, channels_, height_, width_);
  spatial_out_blob_norm_.Reshape(num_, channels_, height_, width_);
  bilateral_out_blob_norm_.Reshape(num_, channels_, height_, width_);
  norm_out_.Reshape(num_, channels_, height_, width_);

  // Sum layer configuration
  sum_bottom_vec_.clear();
  sum_bottom_vec_.push_back(unary_terms);
//  sum_bottom_vec_.push_back(unary_terms1);
  sum_bottom_vec_.push_back(&message_passing_);

  sum_top_vec_.clear();
  sum_top_vec_.push_back(&adding_output_blob_);

  LayerParameter sum_param;
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
//  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));
  sum_param.mutable_eltwise_param()->add_coeff(Dtype(1.));

  sum_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_SUM);
  sum_layer_.reset(new EltwiseLayer<Dtype>(sum_param));
  sum_layer_->SetUp(sum_bottom_vec_, sum_top_vec_);

  //multiplication layer configuration
  mul_bottom_vec_.clear();
  mul_bottom_vec_.push_back(&adding_output_blob_);
  mul_bottom_vec_.push_back(&norm_out_);

  mul_top_vec_.clear();
  mul_top_vec_.push_back(output_blob);

  LayerParameter mul_param;

  mul_param.mutable_eltwise_param()->set_operation(EltwiseParameter_EltwiseOp_PROD);
  mul_layer_.reset(new EltwiseLayer<Dtype>(mul_param));
  mul_layer_->SetUp(mul_bottom_vec_, mul_top_vec_);

  //normalization layer configuration
  norm_bottom_vec_.clear();
  norm_bottom_vec_.push_back(&message_passing_norm_);
  
  norm_top_vec_.clear();
  norm_top_vec_.push_back(&norm_out_);

  LayerParameter norm_param;
  norm_param.mutable_power_param()->set_power(Dtype(-1.));
  norm_param.mutable_power_param()->set_scale(Dtype(1.));
  norm_param.mutable_power_param()->set_shift(Dtype(0.));

  norm_layer_.reset(new PowerLayer<Dtype>(norm_param));
  norm_layer_->SetUp(norm_bottom_vec_, norm_top_vec_);

}

/**
 * To be invoked before every call to the Forward_cpu() method.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::PrePass(
    const vector<shared_ptr<Blob<Dtype> > >& parameters_to_copy_from,
    const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
    const Blob<Dtype>* const bilateral_norms) {

  bilateral_lattices_ = bilateral_lattices;
  bilateral_norms_ = bilateral_norms;

  // Get copies of the up-to-date parameters.
  for (int i = 0; i < parameters_to_copy_from.size(); ++i) {
    blobs_[i]->CopyFrom(*(parameters_to_copy_from[i].get()));
  }
}

/**
 * Forward pass during the inference.
 */
template <typename Dtype>
void MeanfieldIteration<Dtype>::Forward_cpu() {

  //-----------------------------------Message passing-----------------------
  for (int n = 0; n < num_; ++n) {

    Dtype* spatial_out_data = spatial_out_blob_.mutable_cpu_data() + spatial_out_blob_.offset(n);
    const Dtype* score_map_input_data = score_map->cpu_data() + score_map->offset(n);

    spatial_lattice_->compute(spatial_out_data, score_map_input_data, channels_, false);

    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
          spatial_out_data + channel_id * num_pixels_,
          spatial_out_data + channel_id * num_pixels_);
    }

    Dtype* bilateral_out_data = bilateral_out_blob_.mutable_cpu_data() + bilateral_out_blob_.offset(n);

    (*bilateral_lattices_)[n]->compute(bilateral_out_data, score_map_input_data, channels_, false);
    // Pixel-wise normalization.
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
          bilateral_out_data + channel_id * num_pixels_,
          bilateral_out_data + channel_id * num_pixels_);
    }
  }

  caffe_set(count_, Dtype(0.), message_passing_.mutable_cpu_data());

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, Dtype(2.),
        this->blobs_[0]->cpu_data(), spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), Dtype(0.),
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, Dtype(2.),
        this->blobs_[1]->cpu_data(), bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), Dtype(1.),
        message_passing_.mutable_cpu_data() + message_passing_.offset(n));
  }

  //------------------------- Adding unaries --------------
  sum_layer_->Forward(sum_bottom_vec_, sum_top_vec_);

  //------------------------ Message Passing and weighting for normalizing
  for (int n = 0; n < num_; ++n) {

      Dtype* spatial_out_data_norm = spatial_out_blob_norm_.mutable_cpu_data() + spatial_out_blob_norm_.offset(n);
      const Dtype* score_map_input_data_norm = sum_multiplier->cpu_data();

      spatial_lattice_->compute(spatial_out_data_norm, score_map_input_data_norm, channels_, false);

      // Pixel-wise normalization.
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
          caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
                  spatial_out_data_norm + channel_id * num_pixels_,
                  spatial_out_data_norm + channel_id * num_pixels_);
      }

      Dtype* bilateral_out_data_norm = bilateral_out_blob_norm_.mutable_cpu_data() + bilateral_out_blob_norm_.offset(n);

      (*bilateral_lattices_)[n]->compute(bilateral_out_data_norm, score_map_input_data_norm, channels_, false);
      // Pixel-wise normalization.
      for (int channel_id = 0; channel_id < channels_; ++channel_id) {
          caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
                  bilateral_out_data_norm + channel_id * num_pixels_,
                  bilateral_out_data_norm + channel_id * num_pixels_);
      }
  }

  caffe_set(count_, Dtype(0.), message_passing_norm_.mutable_cpu_data());

  for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, Dtype(2.),
              this->blobs_[0]->cpu_data(), spatial_out_blob_norm_.cpu_data() + spatial_out_blob_norm_.offset(n), Dtype(0.),
              message_passing_norm_.mutable_cpu_data() + message_passing_norm_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, channels_, num_pixels_, channels_, Dtype(2.),
              this->blobs_[1]->cpu_data(), bilateral_out_blob_norm_.cpu_data() + bilateral_out_blob_norm_.offset(n), Dtype(1.),
              message_passing_norm_.mutable_cpu_data() + message_passing_norm_.offset(n));
  }
  //add the sum of weights
  caffe_add_scalar(count_, Dtype(1.), message_passing_norm_.mutable_cpu_data()); 

  //------------------------- Normalizing ----------------
  norm_layer_->Forward(norm_bottom_vec_, norm_top_vec_);
  mul_layer_->Forward(mul_bottom_vec_, mul_top_vec_);
}

template<typename Dtype>
void MeanfieldIteration<Dtype>::Backward_cpu() {
  
  //initialization of the diff of the parameters of spatial and bilateral weights
  caffe_set(this->blobs_[0]->count(), Dtype(0.), this->blobs_[0]->mutable_cpu_diff());
  caffe_set(this->blobs_[1]->count(), Dtype(0.), this->blobs_[1]->mutable_cpu_diff());

  //---------------------------- Gradient for normalizing ------------------
  vector<bool> eltwise_propagate_down(2, true);
  mul_layer_->Backward(mul_top_vec_, eltwise_propagate_down, mul_bottom_vec_);

  vector<bool> power_propagate_down(1, true);
  norm_layer_->Backward(norm_top_vec_, power_propagate_down, norm_bottom_vec_);

  //gradient w.r.t. the kernel weights in normalization path 
  for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
              num_pixels_, Dtype(2.), message_passing_norm_.cpu_diff() + message_passing_norm_.offset(n),
              spatial_out_blob_norm_.cpu_data() + spatial_out_blob_norm_.offset(n), Dtype(1.),
              this->blobs_[0]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
      caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
              num_pixels_, Dtype(2.), message_passing_norm_.cpu_diff() + message_passing_norm_.offset(n),
              bilateral_out_blob_norm_.cpu_data() + bilateral_out_blob_norm_.offset(n), Dtype(1.),
              this->blobs_[1]->mutable_cpu_diff());
  } 
  //---------------------------- Add unary gradient --------------------------
  vector<bool> eltwise_propagate_down1(2, true);
  sum_layer_->Backward(sum_top_vec_, eltwise_propagate_down1, sum_bottom_vec_);

  // ------------------------- Gradient w.r.t. kernels weights in the message passing path ------------
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, Dtype(2.), message_passing_.cpu_diff() + message_passing_.offset(n),
                          spatial_out_blob_.cpu_data() + spatial_out_blob_.offset(n), Dtype(1.),
                          this->blobs_[0]->mutable_cpu_diff());
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, channels_, channels_,
                          num_pixels_, Dtype(2.), message_passing_.cpu_diff() + message_passing_.offset(n),
                          bilateral_out_blob_.cpu_data() + bilateral_out_blob_.offset(n), Dtype(1.),
                          this->blobs_[1]->mutable_cpu_diff());
  }

  // gradient for weightsing (messing_passing_-->spatial_out_blog_/bilateral_out_blob_)
  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, Dtype(2.),
                          this->blobs_[0]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          Dtype(0.),
                          spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n));
  }

  for (int n = 0; n < num_; ++n) {
    caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, channels_, num_pixels_, channels_, Dtype(2.),
                          this->blobs_[1]->cpu_data(), message_passing_.cpu_diff() + message_passing_.offset(n),
                          Dtype(0.),
                          bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n));
  }

  //---------------------------- BP thru normalization --------------------------
  for (int n = 0; n < num_; ++n) {

    Dtype *spatial_out_diff = spatial_out_blob_.mutable_cpu_diff() + spatial_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, spatial_norm_->cpu_data(),
                spatial_out_diff + channel_id * num_pixels_,
                spatial_out_diff + channel_id * num_pixels_);
    }

    Dtype *bilateral_out_diff = bilateral_out_blob_.mutable_cpu_diff() + bilateral_out_blob_.offset(n);
    for (int channel_id = 0; channel_id < channels_; ++channel_id) {
      caffe_mul(num_pixels_, bilateral_norms_->cpu_data() + bilateral_norms_->offset(n),
                bilateral_out_diff + channel_id * num_pixels_,
                bilateral_out_diff + channel_id * num_pixels_);
    }
  }
  //--------------------------- Gradient for message passing ---------------
  for (int n = 0; n < num_; ++n) {

    spatial_lattice_->compute(score_map->mutable_cpu_diff() + score_map->offset(n),
                              spatial_out_blob_.cpu_diff() + spatial_out_blob_.offset(n), channels_,
                              true, false);

    (*bilateral_lattices_)[n]->compute(score_map->mutable_cpu_diff() + score_map->offset(n),
                                       bilateral_out_blob_.cpu_diff() + bilateral_out_blob_.offset(n),
                                       channels_, true, true);
  }
}

INSTANTIATE_CLASS(MeanfieldIteration);
}  // namespace caffe
