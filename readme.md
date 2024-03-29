# UncertaintySlumDetection

Welcome to the official repository for the paper "Quantifying Uncertainty in Slum Detection: Advancing Transfer-Learning with Limited Data in Noisy Urban Environments".

<small><i>T. Stark, M. Wurm, X. X. Zhu and H. Taubenbock, "Quantifying Uncertainty in Slum Detection: Advancing Transfer-Learning with Limited Data in Noisy Urban Environments," in IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, doi: 10.1109/JSTARS.2024.3359636</i></small>

https://ieeexplore.ieee.org/document/10416343

![UncertaintySlumDetection](https://github.com/stark-t/UncertaintySlumDetection/blob/master/images_repo/Results.png)


## Abstract Summary:

In urban slum mapping, the importance of efficient techniques is often underestimated, limiting comprehensive research and solutions for the complex challenges faced by these settlements. We prioritize efficient methods to detect urban slum morphologies, utilizing transfer-learning with minimal samples. By estimating prediction probabilities, employing Monte Carlo Dropout, and addressing uncertainties using our custom CNN STnet. This approach enhances AI model training in noisy datasets, providing insights into slum dynamics and intra-urban variabilities.


## Project Guide:

### **Install Repository Dependencies**

1. install requirements.txt
2. change config_example.py to config.py and adjust paths accordingly

### **Get Data**

**Note:** The data provided here is solely intended for illustrative purposes and deviates from the original paper. In the original study, we utilized resampled 3-meter PlanetScope data. However, due to copyright restrictions, we have substituted it with 3-meter resampled RGB Sentinel-2 imagery from the cities of Caracas and Mumbai as a demonstrative example.

To use the example data, follow these steps:

1. Download the data from Figshare https://figshare.com/articles/dataset/Dataset/24988959.
2. Extract the data into the `/data/` directory.

To employ your custom dataset, it is crucial to adhere to a specific data structure. For each area of interest (AOI) within the data directory, three requisite files are essential, each sharing identical resolution and extent:

1. **Remote Sensing Imagery:** This should be in RGB format and resampled to a 3-meter resolution. It must be named AOI_3m.tif (e.g., Mumbai_3m.tif).

2. **Urban-Background Mask:** Utilize values of 0 for background and 1 for urban areas. In our case, we employed Local Climate Zones as delineated by Zhu et al., 2019. The data must be named AOI_urban.tif (e.g. Mumbai_urban.tif).

3. **Slum Reference Mask:** Employ values of 1 to represent slum areas. The data must be named AOI_slum_reference.tif (e.g. Mumbai_slum_reference.tif).

Ensuring uniform resolution and extent across all three files is imperative for seamless integration into our processing pipeline.

<small><i>Zhu, X. X., Hu, J., Qiu, C., Shi, Y., Kang, J., Mou, L., ... & Wang, Y. (2019). So2Sat LCZ42: A benchmark dataset for global local climate zones classification. arXiv preprint arXiv:1912.12171.</i></small>


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


## Citation

@ARTICLE{10416343,
  author={Stark, Thomas and Wurm, Michael and Zhu, Xiao Xiang and Taubenbock, Hannes},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing}, 
  title={Quantifying Uncertainty in Slum Detection: Advancing Transfer-Learning with Limited Data in Noisy Urban Environments}, 
  year={2024},
  volume={},
  number={},
  pages={1-15},
  keywords={Artificial intelligence;Remote sensing;Urban areas;Training;Noise measurement;Uncertainty;Task analysis;Transfer learning;learning from few samples;uncertainty estimation;noisy dataset;imbalanced dataset;slum mapping},
  doi={10.1109/JSTARS.2024.3359636}}

## Further Reading

This study is rooted in our preceding research focused on the identification of slum settlements. For a more comprehensive understanding of our work, we invite you to consult the following publications, which offer additional insights into our research efforts.

<small><i>
Wurm, M., Stark, T., Zhu, X. X., Weigand, M., & Taubenböck, H. (2019). Semantic segmentation of slums in satellite images using transfer learning on fully convolutional neural networks. ISPRS journal of photogrammetry and remote sensing, 150, 59-69.

Stark, T., Wurm, M., Zhu, X. X., & Taubenböck, H. (2020). Satellite-based mapping of urban poverty with transfer-learned slum morphologies. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 13, 5251-5263.

Stark, T., Wurm, M., Taubenböck, H., & Zhu, X. X. (2019, May). Slum mapping in imbalanced remote sensing datasets using transfer learned deep features. In 2019 Joint Urban Remote Sensing Event (JURSE) (pp. 1-4). IEEE.</i></small>