#include "generative-models/quantize-profile.h"

namespace vpipe::genai::quant {

std::string
family_for_class(std::string_view class_name)
{
  if (class_name.empty()) { return {}; }
  for (const std::string& fam : profile::Registry::get().families(kDomain)) {
    const FlexData* p = find(fam);
    if (text(p, kDitClassName, "") == class_name) { return fam; }
  }
  return {};
}

}  // namespace vpipe::genai::quant
