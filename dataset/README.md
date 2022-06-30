- **TrainAndValidation** folder contains approximately 4 hours data from Snapir's IMU and DVL.
The 4 hours dataset is built from 9 different missions with different parameters regarding the length of the mission,
the depth, the speed and more.

### How the dataset was divided to train and validation

![Alt text](/Figures/datasetfig.jpg "Dataset")

M.1 = Mission 1, and so on.

- **Test** folder contains approximately 30 minutes data from Snapir's IMU and DVL.
The dataset is built from a mission that was executed in another date.

___

- Inside the **TrainAndValidation** and **Test** folders there are _IMU_in.npy_, _V.npy_ and _IMU_in_test.npy_, _V_test.npy_ respectively.

- _IMU_in.npy_ and _IMU_in_test.npy_ are 3 dimensional matrices with a size of (a,N,s)

  - 'a' is the sensor's axis. a = 0 is the x axis,a = 1 is the y axis and a = 2 is the z axis
  - 'N' is the number of samples
  - 's' is the type of sensor. s = 0 is the accelerometer and s = 1 is the gyroscope

- _V.npy_ and _V_test.npy_ are 2 dimensional matrices with a size of (a,N)

  - 'a' is the sensor's axis. a = 0 is the x axis,a = 1 is the y axis and a = 2 is the z axis
  - 'N' is the number of samples