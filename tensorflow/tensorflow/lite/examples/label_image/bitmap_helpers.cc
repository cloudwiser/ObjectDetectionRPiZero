/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>

#include <unistd.h>  // NOLINT(build/include_order)

#include "tensorflow/lite/examples/label_image/bitmap_helpers.h"

#define LOG(x) std::cerr

namespace tflite {
namespace label_image {

std::vector<uint8_t> decode_bmp(const uint8_t* input, int row_size, int width,
                                int height, int channels, bool top_down) {
  std::vector<uint8_t> output(height * width * channels);
  for (int i = 0; i < height; i++) {
    int src_pos;
    int dst_pos;

    for (int j = 0; j < width; j++) {
      if (!top_down) {
        src_pos = ((height - 1 - i) * row_size) + j * channels;
      } else {
        src_pos = i * row_size + j * channels;
      }

      dst_pos = (i * width + j) * channels;

      switch (channels) {
        case 1:
          output[dst_pos] = input[src_pos];
          break;
        case 3:
          // BGR -> RGB
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          break;
        case 4:
          // BGRA -> RGBA
          output[dst_pos] = input[src_pos + 2];
          output[dst_pos + 1] = input[src_pos + 1];
          output[dst_pos + 2] = input[src_pos];
          output[dst_pos + 3] = input[src_pos + 3];
          break;
        default:
          LOG(FATAL) << "Unexpected number of channels: " << channels;
          break;
      }
    }
  }
  return output;
}

std::vector<uint8_t> parse_bmp(BMP* bmp, int* width, int* height, int* channels, Settings* s) {

  *width = bmp->bmp_info_header.width;
  *height = bmp->bmp_info_header.height;
  *channels = bmp->bmp_info_header.bit_count / 8;

  if (s->verbose)
    LOG(INFO) << "width, height, channels: " << *width << ", " << *height << ", " << *channels << "\n";

  // there may be padding bytes when the width is not a multiple of 4 bytes
  // 8 * channels == bits per pixel
  const int row_size = (8 * *channels * *width + 31) / 32 * 4;

  // if height is negative, data layout is top down
  // otherwise, it's bottom up
  bool top_down = (*height < 0);

  if (s->verbose)
    LOG(INFO) << "row_size: " << row_size << ", top_down: " << top_down << "\n";

  // Decode image, allocating tensor once the image size is known
  auto* bmp_pixels = &(bmp->data[0]);
  return decode_bmp(bmp_pixels, row_size, *width, abs(*height), *channels, top_down);
}

void write_bmp(BMP* bmp, Settings* s) {

  std::string output_bmp_name = "out_" + s->input_bmp_name;  
  bmp->write(output_bmp_name.c_str()); 
}

}  // namespace label_image
}  // namespace tflite
