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

## Architectures

![Alt text](/Figures/BeamsNetV1.png "Optional title")

