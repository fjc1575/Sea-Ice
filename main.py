import time
import os
import torch
import wjc_core
import argparse
import get_feturse
from tensorboardX import SummaryWriter
from attention_unet import AttU_Net
from segnet import SegNet
from unet import Unet
from Unet_plus_plus import Unet_plus_plus
from TSDUC_Net import TSDCU_net
from Deeplab_v3_plus import DeepLabv3_plus
from DAUnet import DAUnet

if __name__ == '__main__':

    print("train_GLCM")
    # train_GLCM
    train_inPath = './data/train/image/original'
    train_outPath = './data/train/image/glcm'
    train_image = len(os.listdir(train_inPath))
    for i in range(train_image):
        traininFile=os.path.join(train_inPath, os.listdir(train_inPath)[i])
        trainoutFile = os.path.join(train_outPath, os.listdir(train_inPath)[i])
        get_feturse.GLCM_Features(traininFile, trainoutFile)
    print("val_GLCM")
    # val_GLCM
    val_inPath = './data/val/image/original'
    val_outPath = './data/val/image/glcm'
    val_image = len(os.listdir(val_inPath))
    for i in range(val_image):
        valinFile = os.path.join(val_inPath, os.listdir(val_inPath)[i])
        valoutFile = os.path.join(val_outPath, os.listdir(val_inPath)[i])
        get_feturse.GLCM_Features(valinFile, valoutFile)
    print("test_GLCM")
    # test_GLCM
    test_inPath = './data/test/image/original'
    test_outPath = './data/test/image/glcm'
    test_image = len(os.listdir(test_inPath))
    for i in range(test_image):
        testinFile = os.path.join(test_inPath, os.listdir(test_inPath)[i])
        testoutFile = os.path.join(test_outPath, os.listdir(test_inPath)[i])
        get_feturse.GLCM_Features(testinFile, testoutFile)


    model, name = TSDCU_net(3, 1), 'data_TSDCU_net_5epoch'
    parse = argparse.ArgumentParser()
    parse.add_argument("--model_name", type=str, default=name)
    parse.add_argument("--batch_size", type=int, default=1)
    parse.add_argument("--epoch", type=int, default=5)
    parse.add_argument("--data_file", type=str, default="data")
    parse.add_argument("--ckpt", type=str, help="the path of model weight file", default="./" + name + "/weights")
    args = parse.parse_args()
    # Prepare a space for saving trained model and predicted results.
    wjc_core.init_work_space(args)
    # Train a model.
    start_time = time.time()
    writer = SummaryWriter('./' + args.model_name + '/runs')
    wjc_core.train(args, writer, model)
    writer.close()
    end_time = time.time()
    print("Training cost ", end_time - start_time, " seconds")
    # Test a model.
    start_time = time.time()
    # test the model trained
    wjc_core.test(args)
    # or test a certain model
    # wjc_core.test(args, save_gray=True, manual=True, weight_path='./weights/data_TSDCU_net_150epoch.pth')
    end_time = time.time()
    print("Testing cost ", end_time - start_time, " seconds")
    # Print the validation accuracy of the MODAU-net model. *You can change the pth file.
    print(wjc_core.validation(args, torch.load('./data_TSDCU_net_5epoch/weights/data_TSDCU_net_5epoch.pth', map_location='cuda')))



    # Print parameter number of each model.
    wjc_core.model_print(model)
