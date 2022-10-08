# Local Label Point Correction for Edge Detection of Overlapping Cervical Cells
Our unique contributions are summarized as follows:
* We are the **first to propose a label correction method based on
annotation points for edge detection and image segmentation**.
By correcting the position of these label points, our label
correction method can generate higher-quality label, which
contributes 30–40 AP improvement on multiple baseline
models.
* We construct a **largest publicly cervical cell edge detection
dataset** based on our LLPC. Our dataset is ten times larger than
the previous datasets, which greatly facilitates the development
of overlapping cell edge detection.
* We present the **first publicly available label correction
benchmark for improving contour annotation**. Our study
serves as a potential catalyst to promote label correction
research and further paves the way to construct accurately
annotated datasets for edge detection and image segmentation.


<p align="center">
<a href="https://github.com/nachifur/LLPC" target="_blank">
<img align="center" alt="Visual comparison of the original label and our corrected label" src="https://github.com/nachifur/LLPC/blob/main/img/fig1.jpg" />
</a>
</p>

<p align="center">
<a href="https://github.com/nachifur/LLPC" target="_blank">
<img align="center" alt="Label correction for edge detection and semantic segmentation" src="https://github.com/nachifur/LLPC/blob/main/img/fig8.jpg" />
</a>
</p>

# 1. Resources
* Dataset: [CCEDD](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EX0v39dd8kRDhavIoFqxHyoBuqXql9sdPXoyaWptsUvfKw?e=8lMSga)
* Results on CCEDD: [unet_plus_plus_BCEloss + corrected label](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/Ed4SzbyvuXdCuuaJ-twoEjgB_DBYhc4bwzen4qOE32ZevQ?e=QVyuCK)
* [paper](https://www.frontiersin.org/articles/10.3389/fninf.2022.895290/full)
* [Early manuscripts in arXiv (Mon, 5 Oct 2020 11:01:45 UTC)](https://arxiv.org/abs/2010.01919v1)

# 2. Environments
ubuntu18.04+cuda10.2+pytorch1.1.0

create environments
```
conda env create -f install.yaml
```
activate environments
```
conda activate LLPC
```
# 3. Datset - CCEDD
<p align="center">
<a href="https://github.com/nachifur/LLPC" target="_blank">
<img align="center" alt="CCEDD" width = "300" height = "120" src="https://github.com/nachifur/LLPC/blob/main/img/CCEDD.jpg" />
</a>
</p>

**Although we provide original labels (uncorrected), we recommend that you use corrected labels to train deep networks.**
## 3.1 Datset Structure
The data folders should be:
```
LLPC
    * data_processing
    * unet_plus_plus_BCEloss
    * CCEDD
        - label_correct
            - edge
            - png
            - json
        - label_no_correct
            - edge
            - png
            - json
```
1. download CCEDD

unzip: 
* CCEDD/png.zip -> LLPC/CCEDD/label_correct
* CCEDD/png.zip -> LLPC/CCEDD/label_no_correct
2. Generate Dataset
```
cd data_processing
python data_processing/datasets_generate.py 0
```
## 3.2 Local Label Point Correction Algorithm
Our LLPC requires a original annotation file (`CCEDD/label_correct/json`) and a original image (`CCEDD/label_correct/png`) to generate a corrected edge label (`CCEDD/label_correct/edge`). In fact, we have already provided the edge label (`CCEDD/label_correct/edge`). Therefore, if you are not interested in point correction techniques and focus on the design of edge detection networks, you can skip this section.
### 3.2.1. Unzip Annotated Files
unzip: 
* CCEDD/json.zip -> LLPC/CCEDD/label_correct
* CCEDD/json.zip -> LLPC/CCEDD/label_no_correct
### 3.2.2. Debug for LLPC
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
Debug for LLPC:
```
cd data_processing
python data_processing/datasets_generate.py 1
```
### 3.2.3. Generating Edges from Points by Our LLPC
```
cd data_processing
python data_processing/datasets_generate.py 0
```
# 4. Training
```
cp data_processing/label_correct_config.yml unet_plus_plus_BCEloss/config.yml.example
conda activate LLPC
cd unet_plus_plus_BCEloss
python run.py
```
# 5. Test / Result
`python run.py` can complete training, testing, and evaluation. 

You can use the following command to view the evaluation result.
```
cd unet_plus_plus_BCEloss
python unet_plus_plus_BCEloss/show_eval_result.py
```
unet_plus_plus_BCEloss Result:
```
average precision mean:    0.763
average recall mean:    0.613
ODS:    F(0.709,0.674) = 0.691    [th=0.350]
OIS:    F(0.739,0.667) = 0.701
AP:    AP = 0.755
```

# 6. Citation
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


# 7. Terms of Use 
Terms of use: by downloading the CCEDD you agree to the following terms:

- You will use the data only for non-commercial research and educational purposes.
- You will NOT distribute the above images.
- Shenyang Institute of Automation, Chinese Academy of Sciences makes no representations or warranties regarding the data, including but not limited to warranties of non-infringement or fitness for a particular purpose.
- You accept full responsibility for your use of the data and shall defend and indemnify Shenyang Institute of Automation, Chinese Academy of Sciences, including its employees, officers and agents, against any and all claims arising from your use of the data, including but not limited to your use of any copies of copyrighted images that you may create from the data.

# 8. Contact
Please contact Jiawei Liu if there is any question (liujiawei18@mails.ucas.ac.cn).
