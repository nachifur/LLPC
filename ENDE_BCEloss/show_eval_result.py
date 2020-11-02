import numpy as np
log_eval_PR = np.genfromtxt(
    'checkpoints/cell/log_eval_PR.txt', dtype=np.str, encoding='utf-8').astype(np.float)
eval_bdry = np.genfromtxt('checkpoints/cell/eval_bdry.txt',
                          dtype=np.str, encoding='utf-8').astype(np.float)
eval_bdry_img = np.genfromtxt(
    'checkpoints/cell/eval_bdry_img.txt', dtype=np.str, encoding='utf-8').astype(np.float)
eval_bdry_thr = np.genfromtxt(
    'checkpoints/cell/eval_bdry_thr.txt', dtype=np.str, encoding='utf-8').astype(np.float)
print('\naverage precision mean:    {:.3f}'.format(log_eval_PR[0][0]))
print('average recall mean:    {:.3f}'.format(log_eval_PR[1][0]))
print('ODS:    F({:.3f},{:.3f}) = {:.3f}    [th={:.3f}]'.format(
    eval_bdry[1], eval_bdry[2], eval_bdry[3], eval_bdry[0]))
print('OIS:    F({:.3f},{:.3f}) = {:.3f}'.format(
    eval_bdry[4], eval_bdry[5], eval_bdry[6]))
print('AP:    AP = {:.3f}'.format(eval_bdry[7]))
