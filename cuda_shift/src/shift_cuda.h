void shift_featuremap_cuda_forward(THCudaTensor *data,
                                   THCudaIntTensor *shift, THCudaTensor *out);

void shift_featuremap_cuda_backward(THCudaTensor *grad_output,
                                   THCudaIntTensor *shift, THCudaTensor *grad_input);
