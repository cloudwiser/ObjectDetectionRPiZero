<img align="left" width="120" height="120" src="https://avatars0.githubusercontent.com/u/15658638?s=200&v=4">

### TensorFlow Lite : label_image object detection on Raspberry Pi Zero W

This is a follow-on to compiling TensorFlow Lite for the Raspberry Pi Zero W as detailed at https://github.com/cloudwiser/TensorFlowLiteRPIZero

Having got the C++ `minimal` example code compiled, this extends the `label_image` example to demonstrate object detection and bounding box extraction using the SSD Mobilenet model trained against the COCO traning set

Copy the directories and files in this repo to the equivalent paths in your local `./tensorflow/tensorflow/lite/...` directory structure

Please note that `label_image` has not been tested with other quantized or non-quantized models and it expects 1 input tensor and 4 output tensors...so running it with other models may either fail this test and/or cause the dreaded `Segmentation error` after loading and running the model

If this is the case, it should be a case of adjusting the input and output tensor size checks and ensuring the correct handling of TfLite float vs u_int8 output tensors

Once you have installed (and edited) the code, build it via...

```sh
$ cd ~/tensorflow/tensorflow/lite/tools/make/

$ ./build_rpi_armv6_label_image.sh
```

and then run `label_image` with...

```sh
$ cd ~/tensorflow/tensorflow/lite/tools/make/gen/rpi_armv6/bin

$ ./label_image -i fruit.bmp -m ./coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/detect.tflite -l ./coco_ssd_mobilenet_v1_1.0_quant_2018_06_29/labelmap.txt -v 1
```

Run `label_image` with `-h` and/or read the source code for the complete set of command line options

Note these and the output has been refactored from the current `label_image` code in master branch in the TensorFlow repo 
