# 1. base training
CONFIG_FILE=configs/resnet50_1xb32_led.py
RESULT_DIR=training/resnet50_1xb32_led
python tools/train.py ${CONFIG_FILE}  --work-dir ${RESULT_DIR}  --gpu-id  0

# 2. pseudo traning
# CONFIG_FILE=configs/led/swin-tiny_1xb64_led_pseudo.py
# RESULT_DIR=training/swin-tiny_1xb64_led_pseudo
# python tools/train.py ${CONFIG_FILE}  --work-dir ${RESULT_DIR}  --gpu-id  0