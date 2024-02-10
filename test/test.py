import torch
import os
import numpy as np
import cv2
import glob



if __name__ == '__main__':
    gt_root = ''
    pred_root = ""

    resize = (512,512)

    sooth = 0.00001
    all_IOU = []
    all_DICE = []
    all_MSE = []
    for dataset in ['Kvasir-SEG','ETIS-LaribPolypDB','CVC-ColonDB','CVC-ClinicDB']:
        gts = glob.glob(gt_root.format(dataset))
        preds = glob.glob(pred_root.format(dataset))

        gts.sort()
        preds.sort()
        assert len(gts)==len(preds)

        IOU = []
        DICE = []
        MSE = []


        for gt, pred in zip(gts,preds):
            gt_im = cv2.imread(gt,cv2.IMREAD_GRAYSCALE)
            pd_im = cv2.imread(pred,cv2.IMREAD_GRAYSCALE)

            gt_im = cv2.resize(gt_im,resize)
            pd_im = cv2.resize(pd_im,resize)

            gt_im = np.where(gt_im>=128,1,0)
            pd_im = np.where(pd_im >= 128, 1, 0)

            # IOU calculation
            union = np.where((gt_im+pd_im)>0,1,0)
            intersection = np.where((gt_im*pd_im)>0,1,0)

            iou = (intersection.sum()+sooth)/(union.sum()+sooth)
            IOU.append(iou)
            all_IOU.append(iou)

            # Dice calculation
            dice = (2.*intersection.sum()+sooth)/(gt_im.sum()+pd_im.sum()+sooth)
            DICE.append(dice)
            all_DICE.append(dice)

            # MSE calculation
            err = np.sum((gt_im - pd_im) ** 2)
            err /= float(gt_im.shape[0] * gt_im.shape[1])
            MSE.append(err)
            all_MSE.append(err)

            # target = torch.from_numpy(gt_im)
            # predic = torch.from_numpy(pd_im)
        mIoU = sum(IOU)/len(IOU)
        mDice = sum(DICE)/len(DICE)
        mMSE = sum(MSE) / len(MSE)
        print("{} | mIOU: {:.4f} | mDice: {:.4f} | mMSE: {:.4f}.".format(dataset,mIoU,mDice,mMSE))

    aIoU = sum(all_IOU)/len(all_IOU)
    aDice = sum(all_DICE)/len(all_DICE)
    aMSE = sum(all_MSE) / len(all_MSE)
    print("{} | mIOU: {:.4f} | mDice: {:.4f} | mMSE: {:.4f}.".format('all',aIoU,aDice,aMSE))