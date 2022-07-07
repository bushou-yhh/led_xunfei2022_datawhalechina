IMAGE_DIR=data/led/test  
CONFIG_FILE="configs/resnet50_1xb32_led.py"  
WEIGHT="training/resnet50_1xb32_led/epoch_50.pth" 
RESULT=resnet50_1xb32_led.csv



python tools/infer_led.py ${IMAGE_DIR}   --config ${CONFIG_FILE[*]}  --checkpoint  ${WEIGHT[*]}   --result_csv_file ${RESULT}\
                # --pu_label  #去掉注释进行伪标签标注，不去则生成结果csv文件