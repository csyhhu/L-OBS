# Layer-wise Optimal Brain Surgeon
This repo is for Layer-wise Optimal Brain Surgeon (L-OBS), which will appear in NIPS 2017. Codes are based on [Tensorflow](https://www.tensorflow.org/) r1.0+

Paper link: [Learning to Prune Deep Neural Networks via Layer-wise Optimal Brain Surgeon](https://arxiv.org/abs/1705.07565)

## This repo contains:

1. Experiment codes for MNIST using lenet300-100: (folder: lenet300-100)

    This is a toy code for L-OBS, you can know how L-OBS is implemented using lenet300-100 as an examples.

    **How to use it**:

    Run lenet300-100/LOBS.py to prune lenet300-100

2. Experiment codes for Imagenet using ResNet-50: (folder: ResNet-50)

    **Explaination**:

    This folder is for conducting L-OBS on ResNet-50

    To facilitate the process of getting Imagenet data, we use [this](https://github.com/ethereon/caffe-tensorflow)
    to generate image batches and build network models. But we also make some modifications.

    To run this code, you need to first download the whole [ILSVRC2012](http://www.image-net.org/challenges/LSVRC/2012/)
    dataset. Then specify the dataset root in the codes.

    **How to use it**:

    run the following .py file:

    - calculate_hessian_inverse.py:

        This code calculates hessian inverse for every layer in ResNet-50.
        It will generate 54 hessian_inverse.npy files in hessian_inverse/ folder

    - prune_weights.py:

        This code prunes weights and biases for ResNet-50.

    - validate.py:

        This code test the pruned weights efficiency.

    **Notice**:

    - It may takes some time to calculate the hessian inverse. In our experiment server:
    64-CPUs Intel(R) Xeon(R) CPU E5-2697A v4 @ 2.60GHz, 512Gb memory (without GPUs, haha, maybe you can
    speed up with GPUs), it takes about 33 hours to calculate all the hessian inverse.

    - This folder has already contains all the layer that current popular models use: *fully-connected*, *convolution*, *res*.
    And it implements APIs for calculating hessian inverse and performing pruning. The utility folder is still under maintain,
    we are making our best to provide you user-friendly interface for conducting experiments on L-OBS. But currently,
    if you want to conduct experiments on other models, please first get familiar with the APIs provided in this folder.
    Then you can easily deploy L-OBS on other models.


3. A utility code folder, which provides APIs for any network model and dataset. It is under maintain.
4. To deploy L-OBS on large-scale dataset efficiently, multiprocessor-version codes will be released later.


## Support
Please use github issues for any problem related to the code. Send email to the authors for general questions related to the paper.
