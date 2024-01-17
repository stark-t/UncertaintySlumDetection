# UncertaintySlumDetection

Welcome to the official repository for the paper "Quantifying Uncertainty in Slum Detection: Advancing Transfer-Learning with Limited Data in Noisy Urban Environments".

## Abstract Summary:

In urban slum mapping, the importance of efficient techniques is often underestimated, limiting comprehensive research and solutions for the complex challenges faced by these settlements. We prioritize efficient methods to detect urban slum morphologies, utilizing transfer-learning with minimal samples. By estimating prediction probabilities, employing Monte Carlo Dropout, and addressing uncertainties using our custom CNN STnet. This approach enhances AI model training in noisy datasets, providing insights into slum dynamics and intra-urban variabilities.



## Project Guide:

* Install Repository Dependencies

    * requirements
    * change config.py

* Get Data
This is only example data and is very different to the original paper. In the paper we use resampled 3 meter planetscope data, due to copyright reasons and for an example purpose we use 3m resampled RGB snetinel 2 imagery from the city of Caracas and Mumbai. IN the original paper we used >XXXXX for pre trainined and 1 to 100

    * download data from figshare
    * extract data into /data/

* Run Code

    1. run_split_raster.py
        This will split the remote sensing data into smal tiles and create the labels. The labels are created into 3 classes. 0 and 1 for background and urban areas, this is achieved by using reclassified Local Climate Zones data from Zhu et al. And class 2 form slum polygons. All data should be of the same extent and resolution. Labels are created using the the distribution of the three classes and added to the image tile in its file name which is saved in data/datasets including its image satatistics used for normalizition the image dataset.
    2. run_train_pretraining.py
        This will pretrain the STnet. Using the example data we will pretrain the STnet on the Caracas dataset.
    3. run_train_transferlearning.py
        This will finetune the STnet. Using the example data the pretrained STnet will be transfer-learned of the Mumbai dataset.
    4. run_inference.py
        Using the Mumbai dataset the results and maps will be created. Note since the example dataset is quite small the same data is used for transfer-learning and testing. In the original paper a 2 fold split is used for transfer-learning and testing and the results are megred afterwards.