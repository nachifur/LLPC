from data_processing import data_processing
import sys
# run
# python datasets_generate.py 0
# debug
# python datasets_generate.py 1

# inputs = sys.argv[1]
# if inputs=="1":
#     debug=1
# else:
#     debug=0

debug=0

# label_no_correct
DATA_PATH = "/home/liu/disk12t/liu_data/cell_edge_detection/paper_test/data_processing_improve/new_ccedd/label_no_correct"
train_ratio = 6
val_ratio = 1
test_ratio = 3
ratio_list = [train_ratio, val_ratio, test_ratio]
data_processing(DATA_PATH, ratio_list, debug, label_correct=False)

# label_correct
DATA_PATH = "/home/liu/disk12t/liu_data/cell_edge_detection/paper_test/data_processing_improve/new_ccedd/label_correct"
train_ratio = 6
val_ratio = 1
test_ratio = 3
ratio_list = [train_ratio, val_ratio, test_ratio]
data_processing(DATA_PATH, ratio_list, debug, label_correct=True)



