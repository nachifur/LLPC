<p align="center">
<a href="https://github.com/nachifur/automatic-label-correction-CCEDD" target="_blank">
<img align="center" alt="CCEDD" src="https://github.com/nachifur/automatic-label-correction-CCEDD/blob/main/CCEDD.jpg" />
</a>
</p>

# 1. Resources
* Dataset: [CCEDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/nachifur_mail_ustc_edu_cn/Es3O42XCo6dDtPuLXh-e_y8BOao96q0GWVyfBuKmr51M4A?e=O0RPgO)
* Results on CCEDD: [ENDE_BCELoss + corrected label](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EYqcLCKYIyBCkuV9clTGN1ABHi72SUV6SL_dzdnLookx2A?e=eFxUHt)
* [paper](https://www.frontiersin.org/articles/10.3389/fninf.2022.895290/full)

# 2. Environments
ubuntu18.04+cuda10.2+pytorch1.1.0

create environments
```
conda env create -f install.yaml
```
activate environments
```
conda activate automatic_label_correction_based_CCEDD
```
# 3. Datset - CCEDD
**Although we provide uncorrected labels, we recommend that you use corrected labels to train the model.**
## 3.1 Download Datset
[Download CCEDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/nachifur_mail_ustc_edu_cn/Es3O42XCo6dDtPuLXh-e_y8BOao96q0GWVyfBuKmr51M4A?e=O0RPgO)

The data folders should be:
```
automatic_label_correction_based_CCEDD
    * cell_data
        - label_correct
            - edge
            - png
        - label_no_correct
            - edge
            - png
```
unzip: 
* CCEDD/edge_correct.zip -> automatic_label_correction_based_CCEDD/cell_data/label_correct
* CCEDD/png.zip -> automatic_label_correction_based_CCEDD/cell_data/label_correct
* CCEDD/edge_no_correct.zip -> automatic_label_correction_based_CCEDD/cell_data/label_no_correct
* CCEDD/png.zip -> automatic_label_correction_based_CCEDD/cell_data/label_no_correct

## 3.2 Generate Dataset
```
cd data_processing
python data_processing/datasets_generate.py 0
```

# 4. Training
```
cp data_processing/label_correct_config.yml ENDE_BCEloss/config.yml.example
conda activate automatic_label_correction_based_CCEDD
cd ENDE_BCEloss
python run.py
```
# 5. Test / Result
`python run.py` can complete training, testing, and evaluation. 

You can use the following command to view the evaluation result.
```
cd ENDE_BCEloss
python ENDE_BCEloss/show_eval_result.py
```
ENDE_BCEloss Result:
```
average precision mean:    0.676
average recall mean:    0.449
ODS:    F(0.641,0.542) = 0.588    [th=0.310]
OIS:    F(0.687,0.522) = 0.593
AP:    AP = 0.614
```
# 6. Automatic Label Correction algorithm
Our correction algorithm requires the original annotation file. We currently do not publish annotated files, but provide [5 json files](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EcMzbZ5P6d5LhsczwZLqsNABKy-5zNsaERh6hA3XbatEDA?e=ydMerP) for testing. 
## 1. Unzip Annotated Files
unzip: 
* json.zip -> automatic_label_correction_based_CCEDD/cell_data/label_correct
* json.zip -> automatic_label_correction_based_CCEDD/cell_data/label_no_correct
## 2. Debug for Automatic Label Correction algorithm
Edit `data_processing/data_processing.py`, uncomment the code below:
```
    generate edge from points
    time_start=time.time()
    print(time_start)
    if label_correct:
        gen_edge_from_point_base_gradient(DATA_PATH, debug)
    else:
        gen_edge_from_point(DATA_PATH, debug)
    time_end=time.time()
    print(time_end)
    print('generate edge from points time cost',time_end-time_start,'s')
```
Debug for automatic label correction algorithm.
```
cd data_processing
python data_processing/datasets_generate.py 1
```
# 7. Citation
If you find our work useful in your research, please consider citing:
```
@ARTICLE{10.3389/fninf.2022.895290,
AUTHOR={Liu, Jiawei and Fan, Huijie and Wang, Qiang and Li, Wentao and Tang, Yandong and Wang, Danbo and Zhou, Mingyi and Chen, Li},   
TITLE={Local Label Point Correction for Edge Detection of Overlapping Cervical Cells},      
JOURNAL={Frontiers in Neuroinformatics},      
VOLUME={16},      
YEAR={2022},      
URL={https://www.frontiersin.org/article/10.3389/fninf.2022.895290},       
DOI={10.3389/fninf.2022.895290},      
ISSN={1662-5196},   
ABSTRACT={Accurate labeling is essential for supervised deep learning methods. However, it is almost impossible to accurately and manually annotate thousands of images, which results in many labeling errors for most datasets. We proposes a local label point correction (LLPC) method to improve annotation quality for edge detection and image segmentation tasks. Our algorithm contains three steps: gradient-guided point correction, point interpolation, and local point smoothing. We correct the labels of object contours by moving the annotated points to the pixel gradient peaks. This can improve the edge localization accuracy, but it also causes unsmooth contours due to the interference of image noise. Therefore, we design a point smoothing method based on local linear fitting to smooth the corrected edge. To verify the effectiveness of our LLPC, we construct a largest overlapping cervical cell edge detection dataset (CCEDD) with higher precision label corrected by our label correction method. Our LLPC only needs to set three parameters, but yields 30–40% average precision improvement on multiple networks. The qualitative and quantitative experimental results show that our LLPC can improve the quality of manual labels and the accuracy of overlapping cell edge detection. We hope that our study will give a strong boost to the development of the label correction for edge detection and image segmentation. We will release the dataset and code at: <ext-link ext-link-type="uri" xlink:href="https://github.com/nachifur/LLPC" xmlns:xlink="http://www.w3.org/1999/xlink">https://github.com/nachifur/LLPC</ext-link>.}
}
```
# 8. Terms of use 
Terms of use: by downloading the CCEDD you agree to the following terms:

- You will use the data only for non-commercial research and educational purposes.
- You will NOT distribute the above images.
- Shenyang Institute of Automation, Chinese Academy of Sciences makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
- You accept full responsibility for your use of the data and shall defend and indemnify Shenyang Institute of Automation, Chinese Academy of Sciences, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

# 9. Contact
Please contact Jiawei Liu if there is any question (liujiawei18@mails.ucas.ac.cn).
