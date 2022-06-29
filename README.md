# BeamsNet

This git repo contains the dataset, code and weights of the 
deep learning architecture, **_BeamsNet_**, which was introduced
in the paper **_BeamsNet: A data-driven Approach Enhancing Doppler Velocity Log Measurements for Autonomous Underwater Vehicle Navigation_**.

## Introduction

Autonomous underwater vehicles (AUV) perform various applications such as
seafloor mapping and underwater structure health monitoring. Commonly, an
inertial navigation system aided by a Doppler velocity log (DVL) is used to
provide the vehicle’s navigation solution. In such fusion, the DVL provides the
velocity vector of the AUV, which determines the navigation solution’s accu-
racy and helps estimate the navigation states. In our paper we proposed BeamsNet,
an end-to-end deep learning framework to regress the estimated DVL velocity
vector that improves the accuracy of the velocity vector estimate, and could
replace the model-based approach. Two versions of BeamsNet, differing in their
input to the network, are suggested. The first uses the current DVL beam
measurements and inertial sensors data, while the other utilizes only DVL data,
taking the current and past DVL measurements for the regression process. Both
simulation and sea experiments were made to validate the proposed learning ap-
proach relative to the model-based approach. Sea experiments were made with
the Snapir AUV in the Mediterranean Sea, collecting approximately four hours
of DVL and inertial sensor data. Our results show that the proposed approach
achieved an improvement of more than 60% in estimating the DVL velocity
vector

## Dataset

The dataset was collected using the "Snapir" AUV in Mediterranean Sea. The
Snapir is an A18-D, ECA GROUP mid-size AUV for deep water applications.
Capable of rapidly and accurately mapping large areas of the sea floor, Snapir
has a length of 5.5[m], a diameter of 0.5[m], 24 hours’ endurance, and a depth
rating of 3000[m]. Snapir carries several sensors as its payload, including an
interferometric authorized synthetic aperture sonar (SAS) and Teledyne RD
Instruments, Navigator DVL.

![Alt text](/Figures/snapir.jpg "Snapir AUV")

**Additional information regarding the dataset is located in the _dataset_ folder.**


## Architectures

#### BeamsNetV1 architecture
![Alt text](/Figures/BeamsNetV1.png "BeamsNetV1 architecture")

To cope with the different input sizes, BeamsNetV1 architecture
contains three heads. The first is for the 100 samples of the three-axes accelerometer, and the second is for the 100 samples of the three-axes gyroscope,
operating simultaneously. The last head takes the DVL beam measurements.
The raw accelerometer and gyroscopes measurements pass through a one dimensional convolutional (1DCNN) layer consisting of six filters of size 2 × 1
that extract features from the data. Next, the features extracted from the accelerometers and gyroscopes are flattened, combined, and then passed through
a dropout layer with p = 0.2. After a sequence of fully connected layers, the current DVL measurement is combined and goes through the last fully connected
layer that produces the 3×1 vector, which is the estimated DVL velocity vector.
The architecture and the activation functions after each layer are presented in
Figure above.

#### BeamsNetV2 architecture
![Alt text](/Figures/BeamsNetV2.png "BeamsNetV2 architecture")

The network’s input is n past samples of the DVL beam measurements. Same as for
the BeamsNetV1 architecture, the input goes through a one-dimensional convolutional layer with the same specifications. The output from the convolutional
layer is flattened and passes through two fully connected layers. After that,
the current DVL measurement is combined with the last fully connected layer
output and goes into the last fully connected layer that generates the output

**A code for networks is provided and can be seen in the _code_ folder with additional information.**