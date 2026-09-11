#include "generative-models/family-profile.h"

#include <utility>

namespace vpipe::genai::profile {

Registry&
Registry::get() noexcept
{
  static Registry r;
  return r;
}

bool
Registry::add(std::string domain, std::string family, FlexData profile)
{
  if (domain.empty() || family.empty()) { return false; }
  std::lock_guard<std::mutex> lk(_mu);
  return _by_key
      .emplace(std::make_pair(std::move(domain), std::move(family)),
               std::move(profile))
      .second;
}

const FlexData*
Registry::find(std::string_view domain,
               std::string_view family) const noexcept
{
  std::lock_guard<std::mutex> lk(_mu);
  const auto it =
      _by_key.find(std::make_pair(std::string(domain), std::string(family)));
  return it == _by_key.end() ? nullptr : &it->second;
}

std::vector<std::string>
Registry::families(std::string_view domain) const
{
  std::vector<std::string> out;
  std::lock_guard<std::mutex> lk(_mu);
  for (const auto& kv : _by_key) {
    if (kv.first.first == domain) { out.push_back(kv.first.second); }
  }
  return out;
}

}  // namespace vpipe::genai::profile
