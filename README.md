# FNet Implementation with TensorFlow & PyTorch.
TensorFlow & PyTorch implementation of the paper "FNet: Mixing Tokens with Fourier Transforms".

## Overview
The FNet model was proposed in "FNet: Mixing Tokens with Fourier Transforms" by James Lee-Thorp, Joshua Ainslie, Ilya Eckstein, Santiago Ontanon. The model replaces the self-attention layer in a BERT model with a Fourier transform which returns only the real parts of the transform. The model is significantly faster than the BERT model since it has fewer parameters and is more memory efficient. The model achieves about 92%~97% accuracy of BERT counterparts on the GLUE benchmark and trains much faster than the BERT model. Find the abstract from the paper below.

## Abstract
We show that Transformer encoder architectures can be sped up, with limited accuracy costs, by replacing the self-attention sublayers with simple linear transformations that "mix" input tokens. These linear mixers, along with standard nonlinearities in feed-forward layers, prove competent at modeling semantic relationships in several text classification tasks. Most surprisingly, we find that replacing the self-attention sublayer in a Transformer encoder with a standard, unparameterized Fourier Transform achieves 92-97% of the accuracy of BERT counterparts on the GLUE benchmark, but trains 80% faster on GPUs and 70% faster on TPUs at standard 512 input lengths. At longer input lengths, our FNet model is significantly faster: when compared to the "efficient" Transformers on the Long Range Arena benchmark, FNet matches the accuracy of the most accurate models, while outpacing the fastest models across all sequence lengths on GPUs (and across relatively shorter lengths on TPUs). Finally, FNet has a light memory footprint and is particularly efficient at smaller model sizes; for a fixed speed and accuracy budget, small FNet models outperform Transformer counterparts.

<p align="center">
  <img src="https://raw.githubusercontent.com/abdelghanibelgaid/FNet-TensorFlow-PyTorch/main/fnet_architecture.png" width="350" title="FNet encoder architecture with N encoder. (Lee-Thorp et al., 2021)">
</p>

## Additional Links

- [Text Generation using FNet - Keras](https://keras.io/examples/nlp/text_generation_fnet)
- [FNet PyTorch - GitHub Repository](https://github.com/erksch/fnet-pytorch)
- [FNet - Hugging Face](https://huggingface.co/docs/transformers/model_doc/fnet)
- [FNet - Papers With Code](https://paperswithcode.com/paper/fnet-mixing-tokens-with-fourier-transforms)
- [FNet - Paper Explained - YouTube](https://youtu.be/j7pWPdGEfMA)

## Reference

Lee-Thorp, J., Ainslie, J., Eckstein, I., & Ontanon S. (2021). *FNet: Mixing Tokens with Fourier Transforms.* arXiv preprint arXiv:2105.03824)

## Contributing
Github issues and pull requests are welcome. Your feedback is much appreciated!

February 2022, Abdelghani Belgaid
