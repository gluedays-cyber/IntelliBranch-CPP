#pragma once

#include "intellibranch/common.hpp"
#include "intellibranch/runtime.hpp"
#include <iostream>
#include <memory>
#include <string>

namespace intellibranch {

void serialize_model(std::ostream& os, const InferenceModel& model);
std::shared_ptr<InferenceModel> deserialize_model(std::istream& is);

void save_binary_model(const std::string& file_path, const InferenceModel& model);
std::shared_ptr<InferenceModel> load_binary_model(const std::string& file_path);

} // namespace intellibranch
