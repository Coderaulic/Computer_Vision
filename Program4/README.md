# Neural Network Trainer: 
==============================================================================<br>
Title:&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;
Neural Network Trainer<br>
Date:&emsp;&emsp;&emsp;&nbsp;&nbsp;
12/06/2016<br>
Authors:&emsp;&emsp;&nbsp;&nbsp;Ryu Muthui and Robert Griswold<br>
Description:&emsp;Optimizing Neural Networks using deformable parts model
==============================================================================<br>

## <a href="https://github.com/Coderaulic/Computer_Vision/blob/master/Program4/neural_network_trainer.cpp">Neural Network Trainer</a>:

Final project utilizing techniques learned to explore our own Computer Vision application with Machine Learning aspects:

A neural network trainer that can read in images that are labeled. Different keypoint detectors can be used,
confidence levels are provided for most likely and next most likely, and training data can be loaded from file.

Credits:<br>
- Implementation is based largely on Abner Matheus' <a href="http://picoledelimao.github.io/blog/2016/01/31/is-it-a-cat-or-dog-a-neural-network-application-in-opencv/">Dog or Cat Neural Network Application</a>
in Open CV Jan 31st, 2016.
- Deformable Part Models are optionally used to optimize the neural network training and testing. DPM implementation is based largely on Jiaolong Xu's OpenCV-contribution <a href="https://github.com/opencv/opencv_contrib/tree/master/modules/dpm">here</a>.
- <a href="https://github.com/andrewssobral/vehicle_detection_haarcascades">Vehicle detection</a> by Haar Cascades with OpenCV by Andrews Sobral.
- OpenCV repository, <a href="https://github.com/opencv/opencv_extra/tree/master/testdata/cv/dpm/VOC2007_Cascade">data files</a> used in our code for DPM.

See docs for more information.
Images from project:
![1](https://cloud.githubusercontent.com/assets/10789046/24436169/dd5cf49a-13ee-11e7-9813-1d19847e3364.jpg)
![2](https://cloud.githubusercontent.com/assets/10789046/24436170/dd77a7fe-13ee-11e7-9531-cc32cf3fb166.jpg)
![3](https://cloud.githubusercontent.com/assets/10789046/24436168/dd5bc732-13ee-11e7-977d-fc189e4eef85.jpg)
![4](https://cloud.githubusercontent.com/assets/10789046/24436173/dfb904d6-13ee-11e7-8dae-ec28d753330f.jpg)

