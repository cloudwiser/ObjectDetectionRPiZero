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

#ifndef TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_GET_TOP_N_IMPL_H_
#define TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_GET_TOP_N_IMPL_H_

#include <algorithm>
#include <queue>

namespace tflite {
namespace label_image {

extern bool input_floating;

// Returns the top N confidence values over threshold in the provided vector,
// sorted by confidence in descending order.
template <class T>
void get_top_n(const float* scores, const float* classes,
                int num_detections, size_t num_results,
                float threshold, std::vector<std::tuple<float, int, int>>* top_results,
                bool input_floating) {
  // Will contain top N results in ascending order
  std::priority_queue<std::tuple<float, int, int>, std::vector<std::tuple<float, int, int>>,
                      std::greater<std::tuple<float, int, int>>>
      top_result_pq;
  
  for (int i = 0; i < num_detections; ++i) {
    float detection_score;
    int detection_class;
    // Get the confidence for this detection
    if (input_floating)
      detection_score = scores[i];
    else
      detection_score = scores[i] / 255.0;
    // Get the class label for this detection
    detection_class = classes[i];
    // Only add this detection to our top_results if it beats the threshold
    if (detection_score < threshold) {
      continue;
    }
    // Add the detection in terms of score and class
    top_result_pq.push(std::tuple<float, int, int>(detection_score, detection_class, i));

    // If at capacity, kick the smallest value out
    if (top_result_pq.size() > num_results) {
      top_result_pq.pop();
    }
  }
  
  // Copy to output vector and reverse into descending order
  while (!top_result_pq.empty()) {
    top_results->push_back(top_result_pq.top());
    top_result_pq.pop();
  }
  std::reverse(top_results->begin(), top_results->end());
}

}  // namespace label_image
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXAMPLES_LABEL_IMAGE_GET_TOP_N_IMPL_H_
