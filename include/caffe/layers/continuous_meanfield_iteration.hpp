#ifndef CAFFE_CONTINUOUS_MEANFIELD_Iteration_HPP_
#define CAFFE_CONTINUOUS_MEANFIELD_Iteration_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/eltwise_layer.hpp"
#include "caffe/layers/power_layer.hpp"
#include "caffe/layers/split_layer.hpp"

#include "caffe/util/modified_permutohedral.hpp"
#include "caffe/layers/loss_layer.hpp"
#include <boost/shared_array.hpp>

namespace caffe {

 template <typename Dtype>
    class ContinuousMeanfieldIteration {

     public:
      /**
       * Must be invoked only once after the construction of the layer.
       */
      void OneTimeSetUp(
          Blob<Dtype>* const unary_terms,
          Blob<Dtype>* const score_map_,
          Blob<Dtype>* const output_blob_,
          const shared_ptr<ModifiedPermutohedral> spatial_lattice,
          const Blob<Dtype>* const spatial_norm,
          const Blob<Dtype>* const sum_multiplier_);

      /**
       * Must be invoked before invoking {@link Forward_cpu()}
       */
      void PrePass(
      //virtual void PrePass(
          const vector<shared_ptr<Blob<Dtype> > >&  parameters_to_copy_from,
          const vector<shared_ptr<ModifiedPermutohedral> >* const bilateral_lattices,
          const Blob<Dtype>* const bilateral_norms);

      /**
       * Forward pass - to be called during inference.
       */
      //virtual void Forward_cpu();
      void Forward_cpu();
 


      /**
       * Backward pass - to be called during training.
       */
      //virtual void Backward_cpu();
      void Backward_cpu();

      // A quick hack. This should be properly encapsulated.
      vector<shared_ptr<Blob<Dtype> > >& blobs() {
        return blobs_;
      }

     protected:
      vector<shared_ptr<Blob<Dtype> > > blobs_;

      int count_;
      int num_;
      int channels_;
      int height_;
      int width_;
      int num_pixels_;

      Blob<Dtype>* score_map;
      Blob<Dtype>* output_blob;

      Blob<Dtype> spatial_out_blob_;
      Blob<Dtype> bilateral_out_blob_;
      Blob<Dtype> adding_output_blob_;
      Blob<Dtype> message_passing_;

      Blob<Dtype> spatial_out_blob_norm_;
      Blob<Dtype> bilateral_out_blob_norm_;
      Blob<Dtype> message_passing_norm_;
      Blob<Dtype> norm_out_;

      vector<Blob<Dtype>*> sum_top_vec_;
      vector<Blob<Dtype>*> sum_bottom_vec_;
      shared_ptr<EltwiseLayer<Dtype> > sum_layer_;

      vector<Blob<Dtype>*> mul_top_vec_;
      vector<Blob<Dtype>*> mul_bottom_vec_;
      shared_ptr<EltwiseLayer<Dtype> > mul_layer_;

      vector<Blob<Dtype>*> norm_top_vec_;
      vector<Blob<Dtype>*> norm_bottom_vec_;
      shared_ptr<PowerLayer<Dtype> > norm_layer_;

      shared_ptr<ModifiedPermutohedral> spatial_lattice_;
      const vector<shared_ptr<ModifiedPermutohedral> >* bilateral_lattices_;

      const Blob<Dtype>* spatial_norm_;
      const Blob<Dtype>* bilateral_norms_;
      const Blob<Dtype>* sum_multiplier;

 };

}  // namespace caffe

#endif  // CAFFE_CONTINUOUS_MEANFIELD_ITERATION_HPP_
