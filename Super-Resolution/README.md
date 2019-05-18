# Super-Resolution
A collection of state-of-the-art video or single-image super-resolution architectures.

## Network list and reference (Updating)
The hyperlink directs to paper site, follows the official codes if the authors open sources.

|Model |Published |Code* |Keywords|
|:-----|:---------|:-----|:-------|
|Image Super-Resolution Using Deep Convolutional Networks (SRCNN)|[ECCV14](https://arxiv.org/abs/1501.00092)|[Keras](https://github.com/qobilidop/srcnn)| 1st DL SR |
|RAISR: Rapid and Accurate Image Super Resolution (RAISR)|[arXiv](https://arxiv.org/abs/1606.01299)|-|| Google, Pixel 3 |
|Single Image Super-resolution from Transformed Self-Exemplars (SelfExSR)|[CVPR15](http://vision.ai.illinois.edu/publications/huangcvpr2015.pdf)|[Matlab](https://github.com/jiaming-wang/Personal-Summarize/tree/master/Matlab-code/SelfExSR)| Without training data |
|Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network (ESPCN)|[CVPR16](https://arxiv.org/abs/1609.05158)|[Keras](https://github.com/qobilidop/srcnn)| Real time |
|Accurate Image Super-Resolution Using Very Deep Convolutional Networks (VDSR)|[CVPR16](https://arxiv.org/abs/1511.04587)|[Caffe](https://github.com/huangzehao/caffe-vdsr)| Deep, Residual |
|Deeply-Recursive Convolutional Network for Image Super-Resolution (DRCN)|[CVPR16](https://arxiv.org/abs/1511.04491)|-| Recurrent |
|Perceptual Losses for Real-Time Style Transfer and Super-Resolution (PLSR)|[ECCV16](https://cs.stanford.edu/people/jcjohns/papers/eccv16/JohnsonECCV16.pdf)|[Torch](https://github.com/jcjohnson/fast-neural-style)| 1st Perceptual Loss |
|Image Super-Resolution via Deep Recursive Residual Network (DRRN)|[CVPR17](http://cvlab.cse.msu.edu/pdfs/Tai_Yang_Liu_CVPR2017.pdf)|[Caffe](https://github.com/tyshiwo/DRRN_CVPR17), [PyTorch](https://github.com/jt827859032/DRRN-pytorch)| Recurrent |
|Fast and Accurate Image Super-Resolution with Deep Laplacian Pyramid Networks (LapSRN)|[CVPR17](http://vllab.ucmerced.edu/wlai24/LapSRN/)|[Matlab](https://github.com/phoenix104104/LapSRN)| Huber loss |
|Enhanced Deep Residual Networks for Single Image Super-Resolution (EDSR)|[CVPR17](https://arxiv.org/abs/1707.02921)|[PyTorch](https://github.com/thstkdgus35/EDSR-PyTorch)| NTIRE17 Champion |
|Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (SRGAN)|[CVPR17](https://arxiv.org/abs/1609.04802)|[Tensorflow](https://github.com/jiaming-wang/SRGAN-tensorflow)| 1st proposed GAN |
|Real-Time Video Super-Resolution with Spatio-Temporal Networks and Motion Compensation (VESPCN)|[CVPR17](https://arxiv.org/abs/1611.05250)|-| VideoSR |
|MemNet: A Persistent Memory Network for Image Restoration (MemNet)|[ICCV17](https://arxiv.org/abs/1708.02209)|[Caffe](https://github.com/tyshiwo/MemNet)||
|Image Super-Resolution Using Dense Skip Connections (SRDenseNet)|[ICCV17](http://openaccess.thecvf.com/content_ICCV_2017/papers/Tong_Image_Super-Resolution_Using_ICCV_2017_paper.pdf)|[PyTorch](https://github.com/wxywhu/SRDenseNet-pytorch)| Dense |
|Detail-revealing Deep Video Super-resolution (SPMC)|[ICCV17](https://arxiv.org/abs/1704.02738)|[Tensorflow](https://github.com/jiangsutx/SPMC_VideoSR)| VideoSR |
|Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (DnCNN)|[TIP17](http://ieeexplore.ieee.org/document/7839189/)|[Matlab](https://github.com/cszn/DnCNN)| Denoise |
|Fast and Accurate Image Super Resolution by Deep CNN with Skip Connection and Network in Network (DCSCN)|[arXiv](https://arxiv.org/abs/1707.05425)|[Tensorflow](https://github.com/jiny2001/dcscn-super-resolution)||
|Fast and Accurate Single Image Super-Resolution via Information Distillation Network (IDN)|[CVPR18](https://arxiv.org/abs/1803.09454)|[Caffe](https://github.com/Zheng222/IDN-Caffe)| Fast |
|Residual Dense Network for Image Super-Resolution (RDN)|[CVPR18](https://arxiv.org/abs/1802.08797)|[Torch](https://github.com/yulunzhang/RDN)| Deep, BI-BD-DN |
|Learning a Single Convolutional Super-Resolution Network for Multiple Degradations (SRMD)|[CVPR18](https://arxiv.org/abs/1712.06116)|[Matlab](https://github.com/cszn/SRMD)| Denoise/Deblur/SR |
|Deep Back-Projection Networks For Super-Resolution (DBPN)|[CVPR18](https://arxiv.org/abs/1803.02735)|[PyTorch](https://github.com/alterzero/DBPN-Pytorch)| NTIRE18 Champion |
|"Zero Shot" Super-Resolution using Deep Internal Learning (ZSSR)|[CVPR18](http://www.wisdom.weizmann.ac.il/~vision/zssr/)|[Tensorflow](https://github.com/assafshocher/ZSSR)| Zero-shot |
|Frame-Recurrent Video Super-Resolution (FRVSR)|[CVPR18](https://arxiv.org/abs/1801.04590)|[PDF](https://github.com/msmsajjadi/FRVSR)| VideoSR |
|Deep Video Super-Resolution Network Using Dynamic Upsampling Filters Without Explicit Motion Compensation (DUF)|[CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.pdf)|[Tensorflow](https://github.com/yhjo09/VSR-DUF)| VideoSR |
|Scale-recurrent Network for Deep Image Deblurring (SRN)|[CVPR18](http://openaccess.thecvf.com/content_cvpr_2018/papers/Tao_Scale-Recurrent_Network_for_CVPR_2018_paper.pdf)|[Tensorflow](https://github.com/cgtuebingen/learning-blind-motion-deblurring)| Deblur |
|Deep Image Prior(DIP)|[CVPR18](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)|[PyTorch](https://github.com/DmitryUlyanov/deep-image-prior)| Learning-free |
|Fast, Accurate, and Lightweight Super-Resolution with Cascading Residual Network (CARN)|[ECCV18](https://arxiv.org/abs/1803.08664)|[PyTorch](https://github.com/nmhkahn/CARN-pytorch)| Fast |
|Image Super-Resolution Using Very Deep Residual Channel Attention Networks (RCAN)|[ECCV18](https://arxiv.org/abs/1807.02758)|[PyTorch](https://github.com/yulunzhang/RCAN)| Deep, BI-BD-DN |
|Multi-scale Residual Network for Image Super-Resolution (MSRN)|[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Juncheng_Li_Multi-scale_Residual_Network_ECCV_2018_paper.pdf)|[PyTorch](https://github.com/MIVRC/MSRN-PyTorch)| |
|SRFeat: Single Image Super-Resolution with Feature Discrimination (SRFeat)|[ECCV18](http://openaccess.thecvf.com/content_ECCV_2018/papers/Seong-Jin_Park_SRFeat_Single_Image_ECCV_2018_paper.pdf)|[Tensorflow](https://github.com/HyeongseokSon1/SRFeat)| GAN |
|Non-Local Recurrent Network for Image Restoration (NLRN)|[NIPS18](https://papers.nips.cc/paper/7439-non-local-recurrent-network-for-image-restoration.pdf)|[Tensorflow](https://github.com/Ding-Liu/NLRN)| Non-local, Recurrent |
|Joint Sub-bands Learning with Clique Structures for Wavelet Domain Super-Resolution (SRCliqueNet)|[NIPS18](https://arxiv.org/abs/1809.04508)|-| Wavelet |
|FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising (FFDNet)|[TIP18](https://ieeexplore.ieee.org/document/8365806/)|[Matlab](https://github.com/cszn/FFDNet)| Conditional denoise|
|Multi-Memory Convolutional Neural Network for Video Super-Resolution (MMCNN)|[TIP18](https://ieeexplore.ieee.org/document/8579237/)|[Tensorflow](https://github.com/psychopa4/MMCNN)| Multi-Memory CNN|
|Toward Convolutional Blind Denoising of Real Photographs (CBDNet)|[arXiv](https://arxiv.org/abs/1807.04686)|[Matlab](https://github.com/GuoShi28/CBDNet)| Blind-denoise |
|Learning for Video Super-Resolution through HR Optical Flow Estimation (SOFVSR)|[ACCV18](http://arxiv.org/abs/1809.08573)|[PyTorch](https://github.com/LongguangWang/SOF-VSR)| VideoSR |
|ESRGAN: Enhanced Super-Resolution Generative Adversarial Networks (ESRGAN)|[ECCVW18](http://arxiv.org/abs/1809.00219)|[PyTorch](https://github.com/xinntao/ESRGAN)|1st place PIRM 2018|
|Temporally Coherent GANs for Video Super-Resolution (TecoGAN)|[arXiv](http://arxiv.org/abs/1811.09393)|[Tensorflow](https://github.com/thunil/TecoGAN)| VideoSR GAN|
|Recurrent Back-Projection Network for Video Super-Resolution (RBPN)|[CVPR19](https://arxiv.org/abs/1903.10128)|[PyTorch](https://github.com/alterzero/RBPN-PyTorch)| VideoSR |
|Fast Spatio-Temporal Residual Network for Video Super-Resolution (FSTRN)|[CVPR19](https://arxiv.org/pdf/1904.02870.pdf)|-| Fast VideoSR |
|Meta-SR: A Magnification-Arbitrary Network for Super-Resolution (Meta-SR)|[CVPR19](https://arxiv.org/pdf/1903.00875.pdf)|-| Meta SR |
|Feedback Network for Image Super-Resolution (SRFBN)|[CVPR19](https://arxiv.org/pdf/1903.09814.pdf)|[PyTorch](https://github.com/Paper99/SRFBN_CVPR19)| Feedback |
|Learning Parallax Attention for Stereo Image Super-Resolution (PASSRnet)|[CVPR19](https://arxiv.org/pdf/1903.05784.pdf)|[PyTorch](https://github.com/LongguangWang/PASSRnet)| Parallax-attention |

\*The 1st repo is by paper author.

## Link of datasets
*(please contact me if any of links offend you or any one disabled)*

|Name|Usage|#|Site|Comments|
|:---|:----|:----|:---|:-----|
|SET5|Test|5|[download](https://uofi.box.com/shared/static/kfahv87nfe8ax910l85dksyl2q212voc.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|SET14|Test|14|[download](https://uofi.box.com/shared/static/igsnfieh4lz68l926l8xbklwsnnk8we9.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|SunHay80|Test|80|[download](https://uofi.box.com/shared/static/rirohj4773jl7ef752r330rtqw23djt8.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|Urban100|Test|100|[download](https://uofi.box.com/shared/static/65upg43jjd0a4cwsiqgl6o6ixube6klm.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|VID4|Test|4|[download](https://people.csail.mit.edu/celiu/CVPR2011/videoSR.zip)|4 videos|
|BSD100|Train|300|[download](https://uofi.box.com/shared/static/qgctsplb8txrksm9to9x01zfa4m61ngq.zip)|[jbhuang0604](https://github.com/jbhuang0604/SelfExSR)|
|BSD300|Train/Val|300|[download](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/segbench/BSDS300-images.tgz)|-|
|BSD500|Train/Val|500|[download](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz)|-|
|91-Image|Train|91|[download](http://www.ifp.illinois.edu/~jyang29/codes/ScSR.rar)|Yang|
|DIV2K|Train/Val|900|[website](https://data.vision.ee.ethz.ch/cvl/DIV2K/)|NTIRE17|
|Waterloo|Train|4741|[website](https://ece.uwaterloo.ca/~k29ma/exploration/)|-|
|MCL-V|Train|12|[website](http://mcl.usc.edu/mcl-v-database/)|12 videos|
|GOPRO|Train/Val|33|[website](https://github.com/SeungjunNah/DeepDeblur_release)|33 videos, deblur|
|CelebA|Train|202599|[website](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)|Human faces|
|Sintel|Train/Val|35|[website](http://sintel.is.tue.mpg.de/downloads)|Optical flow|
|FlyingChairs|Train|22872|[website](https://lmb.informatik.uni-freiburg.de/resources/datasets/FlyingChairs.en.html#flyingchairs)|Optical flow|
|DND|Test|50|[website](https://noise.visinf.tu-darmstadt.de/)|Real noisy photos|
|RENOIR|Train|120|[website](http://ani.stat.fsu.edu/~abarbu/Renoir.html)|Real noisy photos|
|NC|Test|60|[website](http://demo.ipol.im/demo/125/)|Noisy photos|
|SIDD(M)|Train/Val|200|[website](https://www.eecs.yorku.ca/~kamel/sidd/)|NTIRE 2019 Real Denoise|
|RSR|Train/Val|80|[download]()|NTIRE 2019 Real SR|
|Vimeo-90k|Train/Test|89800|[website](http://toflow.csail.mit.edu/)|90k HQ videos|
|General-100|Test|100|[website](http://mmlab.ie.cuhk.edu.hk/projects/FSRCNN.html/)|Chao Dong|

Other open datasets:
[Kaggle](https://www.kaggle.com/datasets), [ImageNet](http://www.image-net.org/), [COCO](http://cocodataset.org/)
