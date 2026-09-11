#include "generative-models/detect-profile.h"

namespace vpipe::genai::detect {

std::string
model_type_for_class(std::string_view class_name, bool edit)
{
  if (class_name.empty()) { return {}; }
  for (const std::string& fam : profile::Registry::get().families(kDomain)) {
    const FlexData* p = find(fam);
    bool owned = false;
    for (const std::string& c : strings(p, kClassNames)) {
      if (c == class_name) { owned = true; break; }
    }
    if (!owned) { continue; }
    if (edit) {
      const std::string e = text(p, kModelTypeEdit, "");
      if (!e.empty()) { return e; }
    }
    const std::string t = text(p, kModelType, "");
    // A profile that names classes and no type has said nothing usable;
    // fall back to the family TAG, which is what every registry already
    // calls it.
    return t.empty() ? fam : t;
  }
  return {};
}

const FlexData*
for_model_type(std::string_view model_type)
{
  if (model_type.empty()) { return nullptr; }
  for (const std::string& fam : profile::Registry::get().families(kDomain)) {
    const FlexData* p = find(fam);
    if (text(p, kModelType, "") == model_type ||
        text(p, kModelTypeEdit, "") == model_type || fam == model_type) {
      return p;
    }
  }
  return nullptr;
}

}  // namespace vpipe::genai::detect
