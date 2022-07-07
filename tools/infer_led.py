# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
from distutils.command import check
import os
import csv

import mmcv
from os import path as op

from mmcls.apis import inference_model, init_model, show_result_pyplot


def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', help='Image file dictionary')
    parser.add_argument('--config', help='Config file(str or list)')
    parser.add_argument('--checkpoint', help='Checkpoint file(str or list)')
    parser.add_argument(
        '--result_csv_file', default='result.csv', help='the result csv file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--pu_label',  action='store_true', default='false', 
                    help='Whether or not use pseudo technique')
    args = parser.parse_args()
    result = args.result_csv_file

    # 注意，这里是模型集成的部分代码，假如你要使用模型集成，这里并未完全实现，期待您的实现
    if isinstance(args.config, list) and isinstance(args.checkpoint, list):  # 这一行代码有误 args.config， args.checkpoint 需要你自行修改
        for config, checkpoint in zip(args.config, args.checkpoint):
            models = init_model(config, checkpoint, device=args.device) 
    else:
        # build the model from a config file and a checkpoint file
        models = init_model(args.config, args.checkpoint, device=args.device)    


    # write to the csv file
    headers = ['image', 'label']
    with open(result, 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)

        # test  images
        for img in mmcv.track_iter_progress(os.listdir(args.img_dir)):
            _img = op.join(args.img_dir, img)
            results = []
            # 当存在多个模型时
            if isinstance(models, list):
                for model in models:
                    result = inference_model(model, _img)
                    results.append(result)
            else:
                # 当仅为单模型时
                result = inference_model(models, _img)
                results = result
            # 模型集成，伪标签技术（是比赛中常用的很有效的涨点方式），这里我们用pred_score分数去做(PS: 未完整实现，大家如果感兴趣，可以去实现它们)
            pred_label, pred_score, pred_class = results['pred_label'], results['pred_score'], results['pred_class']  
            # write  rows
            
            # 本次比赛为分类问题：这里我们设定一个足够高的阈值, 选择预测概率最有把握的样本的标签作为真实的标签（例如概率为0.99或者概率未0.01的预测标签
            # 将预测然后将得到的有标注的数据加入原始数据继续进行训练，再预测，一直到达停止条件（例如大部分甚至全部unlabeled的样本都被打上了标签），
            # 此时，我们就把未标注的样本通过这种方式标注出来了
            # 这里阈值可以根据实际情况自行设定
            if args.pu_label == True:
                # import pdb;pdb.set_trace()
                if pred_score > 0.9:
                    row = [img, pred_label]
                    f_csv.writerow(row)
            else:
                row = [img, pred_label]
                f_csv.writerow(row)




if __name__ == '__main__':
    main()
