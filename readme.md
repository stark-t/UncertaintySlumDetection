# UncertaintySlumDetection

Welcome to the official repository for the paper "Quantifying Uncertainty in Slum Detection: Advancing Transfer-Learning with Limited Data in Noisy Urban Environments".

<small><i>Stark, T., Wurm, M., Zhu, X. X., & Taubenböck, H. (2024). Quantifying Uncertainty in Slum Detection: Advancing Transfer-Learning with Limited Data in Noisy Urban Environments. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, submitted</i></small>

![UncertaintySlumDetection](https://github.com/stark-t/UncertaintySlumDetection/blob/master/images_repo/results.png)


## Abstract Summary:

In urban slum mapping, the importance of efficient techniques is often underestimated, limiting comprehensive research and solutions for the complex challenges faced by these settlements. We prioritize efficient methods to detect urban slum morphologies, utilizing transfer-learning with minimal samples. By estimating prediction probabilities, employing Monte Carlo Dropout, and addressing uncertainties using our custom CNN STnet. This approach enhances AI model training in noisy datasets, providing insights into slum dynamics and intra-urban variabilities.


## Project Guide:

### **Install Repository Dependencies**

1. install requirements.txt
2. change config_example.py to config.py and adjust paths accordingly

### **Get Data**

**Note:** The data provided here is only for example purposes and differs from the original paper. In the original paper, we used resampled 3-meter PlanetScope data. However, due to copyright reasons, we are using 3-meter resampled RGB Sentinel-2 imagery from the cities of Caracas and Mumbai as an example.

To use the example data, follow these steps:

1. Download the data from Figshare https://figshare.com/articles/dataset/Dataset/24988959.
2. Extract the data into the `/data/` directory.


### **Run Code**

1. **run_split_raster.py**
    - This script splits the remote sensing data into small tiles and creates the labels.
    - The labels are created for 3 classes: 0 and 1 for background and urban areas, and class 2 for slum polygons.
    - All data should be of the same extent and resolution.
    - Labels are added to the image tile file name and saved in the `data/datasets` directory, along with image statistics used for normalization.

2. **run_train_pretraining.py**
    - This script pretrains the STnet using the example data.
    - The pretraining is performed on the Caracas dataset.

3. **run_train_transferlearning.py**
    - This script finetunes the STnet using the example data.
    - The pretrained STnet is transfer-learned on the Mumbai dataset.

4. **run_inference.py**
    - This script creates results and maps using the Mumbai dataset.
    - Note: Since the example dataset is small, the same data is used for both transfer-learning and testing.
    - In the original paper, a 2-fold split is used for transfer-learning and testing, and the results are merged afterwards.

## Results

### Example results

**Note:** Please note that the results shown here are based on a limited amount of data and may not be ideal. They should be used as a reference for structuring other data to achieve similar results as described in the original paper. If you require additional data or access to the original model weights, feel free to contact the authors. They will be happy to assist you.

| Sentinel2 RGB | Slum Propbability |
|---------|---------|
| ![alt text](https://github.com/stark-t/UncertaintySlumDetection/blob/master/images_repo/mumbai.png) | ![alt text](https://github.com/stark-t/UncertaintySlumDetection/blob/master/images_repo/mumbai_results.png) |
