# automatic-label-correction-CCEDD
**This is an informal version, please do not cite this article for now.
You can use CCEDD, but you need to cite the official version of the article.**
# 1. Resources
* Dataset: [CCEDD](https://mailustceducn-my.sharepoint.com/:f:/g/personal/nachifur_mail_ustc_edu_cn/Es3O42XCo6dDtPuLXh-e_y8BOao96q0GWVyfBuKmr51M4A?e=O0RPgO)
* Results on CCEDD: [ENDE_BCELoss + corrected label](https://mailustceducn-my.sharepoint.com/:u:/g/personal/nachifur_mail_ustc_edu_cn/EYqcLCKYIyBCkuV9clTGN1ABHi72SUV6SL_dzdnLookx2A?e=eFxUHt)
* paper: [Automatic label correction for accurate edge detection of
overlapping cervical cell](https://arxiv.org/abs/2010.01919)

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
waiting...
<!-- If you find our work useful in your research, please consider citing:
```
@article{liu2020automatic,
  title={Automatic label correction based on CCESD},
  author={Liu, Jiawei and Wang, Qiang and Fan, Huijie and Tang, Yandong},
  journal={arXiv preprint arXiv:2010.01919},
  year={2020}
}
```--> 
# 8. Contact
Please contact me if there is any question (Jiawei Liu liujiawei18@mails.ucas.ac.cn)

