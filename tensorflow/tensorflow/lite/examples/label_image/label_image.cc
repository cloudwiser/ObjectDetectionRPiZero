/* Copyright 2017 The TensorFlow Authors.
   Portions copyright 2019 clouwdwise consulting. 
   All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <fcntl.h>      // NOLINT(build/include_order)
#include <getopt.h>     // NOLINT(build/include_order)
#include <sys/time.h>   // NOLINT(build/include_order)
#include <sys/types.h>  // NOLINT(build/include_order)
#include <sys/uio.h>    // NOLINT(build/include_order)
#include <unistd.h>     // NOLINT(build/include_order)

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/string_util.h"

#define LOG(x) std::cerr
#define EXPECTED_NUM_INPUTS 1
#define EXPECTED_NUM_OUTPUTS 4
#define IMAGE_WIDTH 300
#define IMAGE_HEIGHT 300
#define IMAGE_CHANNELS 3
#define THRESHOLD 0.6f

#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"
#include "tensorflow/lite/examples/label_image/get_top_n.h"


namespace tflite {
namespace label_image {

std::vector<std::array<unsigned char, 4>> ColorPalette {
    {255, 0, 0, 255},   // red
    {0, 255, 0, 255},   // lime
    {0, 0, 255, 255},   // blue
    {255, 128, 0, 255}, // orange
    {255, 0, 255, 255}, // magenta
    {0, 255, 255, 255}, // cyan
    {255, 255, 0, 255}, // yellow
    {255, 0, 127, 255}, // pink
    {128, 255, 0, 255}, // light green
    {0, 191, 255, 255}  // deep sky blue
};
// auto color = ColorPalette[<detection_number>];
// auto R = color[0], G = color[1], B = color[2], A = color[3];

template<typename T>
T* TensorData(TfLiteTensor* tensor);

template<>
float* TensorData(TfLiteTensor* tensor) {
    switch (tensor->type) {
        case kTfLiteFloat32:
            return tensor->data.f;
        default:
            LOG(FATAL) << "Unknown or mismatched tensor type\n";
    }
    return nullptr;
}

template<>
uint8_t* TensorData(TfLiteTensor* tensor) {
    switch (tensor->type) {
        case kTfLiteUInt8:
            return tensor->data.uint8;
        default:
            LOG(FATAL) << "Unknown or mistmatched tensor type\n";
    }
    return nullptr;
}

double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

// Takes a file name, and loads a list of labels from it, one per line, and
// returns a vector of the strings. It pads with empty strings so the length
// of the result is a multiple of 16, because our model expects that.
TfLiteStatus ReadLabelsFile(const string& file_name,
                            std::vector<string>* result,
                            size_t* found_label_count) {
  std::ifstream file(file_name);
  if (!file) {
    LOG(FATAL) << "Labels file: " << file_name << " not found\n";
    return kTfLiteError;
  }
  result->clear();
  string line;
  while (std::getline(file, line)) {
    result->push_back(line);
  }
  *found_label_count = result->size();
  const int padding = 16;
  while (result->size() % padding) {
    result->emplace_back();
  }
  return kTfLiteOk;
}

void PrintProfilingInfo(const profiling::ProfileEvent* e, uint32_t op_index,
                        TfLiteRegistration registration) {
  // output something like
  // time (ms) , Node xxx, OpCode xxx, symblic name
  //      5.352, Node   5, OpCode   4, DEPTHWISE_CONV_2D
  LOG(INFO) << std::fixed << std::setw(10) << std::setprecision(3)
            << (e->end_timestamp_us - e->begin_timestamp_us) / 1000.0
            << ", Node " << std::setw(3) << std::setprecision(3) << op_index
            << ", OpCode " << std::setw(3) << std::setprecision(3)
            << registration.builtin_code << ", "
            << EnumNameBuiltinOperator(
                   static_cast<BuiltinOperator>(registration.builtin_code))
            << "\n";
}

void RunInference(Settings* s) {
  if (!s->model_name.c_str()) {
    LOG(ERROR) << "No model file specified\n";
    exit(-1);
  }

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(s->model_name.c_str());
  if (!model) {
    LOG(FATAL) << "Failed to mmap model: " << s->model_name << "\n";
    exit(-1);
  }
  LOG(INFO) << "Loaded model: " << s->model_name << "\n";
  model->error_reporter();
  tflite::ops::builtin::BuiltinOpResolver resolver;

  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }

  interpreter->UseNNAPI(s->accel);
  interpreter->SetAllowFp16PrecisionForFp32(s->allow_fp16);

  if (s->verbose) {
    LOG(INFO) << "tensors size : " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size   : " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs       : " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  if (s->number_of_threads != -1) {
    interpreter->SetNumThreads(s->number_of_threads);
  }

  int image_width = IMAGE_WIDTH;
  int image_height = IMAGE_HEIGHT;
  int image_channels = IMAGE_CHANNELS;
  
  std::ifstream file(s->input_bmp_name, std::ios::in | std::ios::binary);
  if (!file) {
    LOG(FATAL) << "input file " << s->input_bmp_name << " not found\n";
    exit(-1);
  }
  BMP bmp(s->input_bmp_name.c_str());
  std::vector<uint8_t> in = parse_bmp(&bmp, &image_width, &image_height, &image_channels, s);

  int input = interpreter->inputs()[0];
  if (s->verbose) 
    LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  if (s->verbose) {
    LOG(INFO) << "number of inputs : " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }
  // Check the input and output geometry is correct for what we are expecting
  if (inputs.size() != EXPECTED_NUM_INPUTS) {
    LOG(FATAL) << "Expecting " << EXPECTED_NUM_INPUTS << " inputs\n";
    exit(-1);
  }
  if (outputs.size() != EXPECTED_NUM_OUTPUTS) {
    LOG(FATAL) << "Expecting " << EXPECTED_NUM_OUTPUTS << " outputs\n";
    exit(-1);
  }
  
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!\n";
  }

  if (s->verbose)
    PrintInterpreterState(interpreter.get());

  // get input dimension from the input tensor metadata
  // assuming one input only
  TfLiteIntArray* dims = interpreter->tensor(input)->dims;
  int wanted_height = dims->data[1];
  int wanted_width = dims->data[2];
  int wanted_channels = dims->data[3];
  if (s->verbose) 
    LOG(INFO) << "wanted height: " << wanted_height << " | "
              << "wanted width: "  << wanted_width <<  " | "
              << "wanted channels: " << wanted_channels << "\n";

  switch (interpreter->tensor(input)->type) {
    case kTfLiteFloat32:
      s->input_floating = true;
      resize<float>(interpreter->typed_tensor<float>(input), in.data(),
                    image_height, image_width, image_channels, wanted_height,
                    wanted_width, wanted_channels, s);
      break;
    case kTfLiteUInt8:
      resize<uint8_t>(interpreter->typed_tensor<uint8_t>(input), in.data(),
                      image_height, image_width, image_channels, wanted_height,
                      wanted_width, wanted_channels, s);
      break;
    default:
      LOG(FATAL) << "Tensor input type: "
                 << interpreter->tensor(input)->type << " not supported\n";
      exit(-1);
  }

  profiling::Profiler* profiler = new profiling::Profiler();
  interpreter->SetProfiler(profiler);

  if (s->profiling) profiler->StartProfiling();

  struct timeval start_time, stop_time;
  gettimeofday(&start_time, nullptr);
  LOG(INFO) << "Invoke started...\n";
  for (int i = 0; i < s->loop_count; i++) {
    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }
  }
  gettimeofday(&stop_time, nullptr);
  LOG(INFO) << "Invoke finished\n";
  LOG(INFO) << "(average time: "
            << (get_us(stop_time) - get_us(start_time)) / (s->loop_count * 1000)
            << " ms)\n";

  if (s->profiling) {
    profiler->StopProfiling();
    auto profile_events = profiler->GetProfileEvents();
    for (int i = 0; i < profile_events.size(); i++) {
      auto op_index = profile_events[i]->event_metadata;
      const auto node_and_registration =
          interpreter->node_and_registration(op_index);
      const TfLiteRegistration registration = node_and_registration->second;
      PrintProfilingInfo(profile_events[i], op_index, registration);
    }
  }

  const float threshold = THRESHOLD;
  std::vector<std::pair<float, int>> top_results;
  int output = interpreter->outputs()[0];

  std::vector<string> labels;
  size_t label_count;

  if (ReadLabelsFile(s->labels_file_name, &labels, &label_count) != kTfLiteOk) {
    LOG(FATAL) << "Failed to load labels file: " << s->labels_file_name << " \n";
    exit(-1);
  }

  // Derived from Yijin Liu's repo at https://github.com/YijinLiu/tf-cpu
  TfLiteTensor* output_locations_ = interpreter->tensor(interpreter->outputs()[0]);
  TfLiteTensor* output_classes_ = interpreter->tensor(interpreter->outputs()[1]);
  TfLiteTensor* output_scores_ = interpreter->tensor(interpreter->outputs()[2]);
  TfLiteTensor* num_detections_ = interpreter->tensor(interpreter->outputs()[3]);
  if (s->verbose) 
    LOG(INFO) << "graph output size: " << interpreter->outputs().size() << "\n";

  const float* detection_locations = TensorData<float>(output_locations_);
  const float* detection_classes = TensorData<float>(output_classes_);
  const float* detection_scores = TensorData<float>(output_scores_);
  const int num_detections = *TensorData<float>(num_detections_);
  if (s->verbose) {
    LOG(INFO) << "number of detections: " << num_detections << "\n";
    // Display all the detection details...
    for (int d = 0; d < num_detections; d++) {
      const std::string cls = labels[detection_classes[d] + 1]; // + 1 given tflite conversion removes the initial background class 
      const float score = detection_scores[d];
      const float ymin = detection_locations[(sizeof(float) * d) + 0] * image_height;
      const float xmin = detection_locations[(sizeof(float) * d) + 1] * image_width;
      const float ymax = detection_locations[(sizeof(float) * d) + 2] * image_height;
      const float xmax = detection_locations[(sizeof(float) * d) + 3] * image_width;
      LOG(INFO) << "------ detection: " << d << " ------\n";
      LOG(INFO) << " score = " << score << "\n";
      LOG(INFO) << " class = " << cls << "\n";
      LOG(INFO) << " ymin  = " << static_cast<unsigned int>(ymin) << "\n";
      LOG(INFO) << " xmin  = " << static_cast<unsigned int>(xmin) << "\n";
      LOG(INFO) << " ymax  = " << static_cast<unsigned int>(ymax) << "\n";
      LOG(INFO) << " xmax  = " << static_cast<unsigned int>(xmax) << "\n";
    }
  }

  switch (interpreter->tensor(output)->type) {
    case kTfLiteFloat32:
      get_top_n<float>(detection_scores, detection_classes, 
                      num_detections, s->number_of_results, 
                      threshold, &top_results, true);
      break;
    case kTfLiteUInt8:
      get_top_n<uint8_t>(detection_scores, detection_classes,
                        num_detections, s->number_of_results, 
                        threshold, &top_results, false);
      break;
    default:
      LOG(FATAL) << "Tensor output type: "
                 << interpreter->tensor(output)->type << " not supported\n";
      exit(-1);
  }

  // Display dectection summaries for those above the threshold
  LOG(INFO) << "------ detections > " << threshold << " ------\n";
  for (const auto& result : top_results) {
    const float confidence = result.first;
    const int index = result.second;
    // + 1 on the class index given tflite conversion removes the initial background class
    LOG(INFO) << "object = " << labels[index + 1] << " @ " << confidence << "\n";
  }
}

void display_usage() {
  LOG(INFO)
      << "label_image\n"
      << "--accelerated, -a: [0|1], use Android NNAPI or not\n"
      << "--count, -c: loop interpreter->Invoke() for certain times\n"
      << "--allow_fp16, -f: [0|1], allow running fp32 models with fp16 not\n"
      << "--image, -i: image_name.bmp\n"
      << "--labels, -l: labels for the model\n"
      << "--tflite_model, -m: model_name.tflite\n"
      << "--profiling, -p: [0|1], profiling or not\n"
      << "--num_results, -r: number of results to show\n"
      << "--threads, -t: number of threads\n"
      << "--verbose, -v: [0|1] print more information\n"
      << "\n";
}

int Main(int argc, char** argv) {
  Settings s;

  int c;
  while (1) {
    static struct option long_options[] = {
        {"accelerated", required_argument, nullptr, 'a'},
        {"count", required_argument, nullptr, 'c'},
        {"allow_fp16", required_argument, nullptr, 'f'},
        {"image", required_argument, nullptr, 'i'},
        {"labels", required_argument, nullptr, 'l'},
        {"tflite_model", required_argument, nullptr, 'm'},
        {"profiling", required_argument, nullptr, 'p'},
        {"num_results", required_argument, nullptr, 'r'},
        {"threads", required_argument, nullptr, 't'},
        {"verbose", required_argument, nullptr, 'v'},
        {nullptr, 0, nullptr, 0}};

    /* getopt_long stores the option index here. */
    int option_index = 0;

    c = getopt_long(argc, argv, "a:c:f:i:l:m:p:r:t:v:", long_options,
                    &option_index);

    /* Detect the end of the options. */
    if (c == -1) break;

    switch (c) {
      case 'a':
        s.accel = strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'c':
        s.loop_count =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'f':
        s.allow_fp16 =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'i':
        s.input_bmp_name = optarg;
        break;
      case 'l':
        s.labels_file_name = optarg;
        break;
      case 'm':
        s.model_name = optarg;
        break;
      case 'p':
        s.profiling =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'r':
        s.number_of_results =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 't':
        s.number_of_threads = strtol(  // NOLINT(runtime/deprecated_fn)
            optarg, nullptr, 10);
        break;
      case 'v':
        s.verbose =
            strtol(optarg, nullptr, 10);  // NOLINT(runtime/deprecated_fn)
        break;
      case 'h':
      case '?':
        /* getopt_long already printed an error message. */
        display_usage();
        exit(-1);
      default:
        exit(-1);
    }
  }
  RunInference(&s);
  return 0;
}

}  // namespace label_image
}  // namespace tflite

int main(int argc, char** argv) {
  return tflite::label_image::Main(argc, argv);
}
