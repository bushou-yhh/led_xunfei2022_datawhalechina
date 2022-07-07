# 1. base training
CONFIG_FILE=configs/xunfei2022led/resnet34_1xb32_led.py
RESULT_DIR=training/resnet34_1xb32_led
python tools/train.py ${CONFIG_FILE}  --work-dir ${RESULT_DIR}  --gpu-id  0


# 2. pseudo traning
#...
#...