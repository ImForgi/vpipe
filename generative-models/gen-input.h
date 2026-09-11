#ifndef VPIPE_GENERATIVE_MODELS_GEN_INPUT_H
#define VPIPE_GENERATIVE_MODELS_GEN_INPUT_H

// AN OPTIONAL NAMED INPUT to a generation, looked up by name instead of
// carried as a struct field.
//
// This exists because the request structs were growing, and growing in
// a way that costs everyone. VideoGenRequest wears the history plainly:
// `cond`, then `neg`, then `audio_cond`, then `ref`, then `ref_last`,
// then `ref_video_rows` and `ref_audio_rows`. Every one of those arrived
// with a capability, changed the struct's layout, and invalidated every
// family binary -- including the families that ignore it. It is the
// same shape as the acceleration settings before they became a bag, and
// it has the same answer.
//
// So: the typed fields that exist STAY, because they work and they are
// documented. What changes is where the NEXT input goes. A mask, a
// control image, a depth map, a third reference, a per-region prompt --
// each of those is a name and a tensor, and none of them needs a field.
//
// WHY A CALLBACK AND NOT A FLEXDATA. A FlexData cannot hold a
// `const float*` to a tensor the stage already owns without either
// copying it or smuggling a pointer through an integer. The lookup is
// one member whose layout never moves, and it borrows exactly the way
// the typed pointers beside it do.
//
// Scalars are a different question and have a different answer: the
// `extras` FlexData beside this one. Use that for a number or a flag,
// and this for anything with a shape.

#include "common/flex-data.h"

#include <cstddef>
#include <functional>
#include <string_view>
#include <vector>

namespace vpipe::genai {

// A tensor the stage is lending for the duration of the call. BORROWED,
// like every other pointer on a request: valid until the generate call
// returns, and owned by a beat the stage is holding.
struct NamedTensor {
  const void*      data = nullptr;
  // Row-major over these extents. Empty means the input exists and the
  // stage could not describe its shape, which a family should treat as
  // a refusal rather than guess at.
  std::vector<int> shape;
  // 2 for f16 and bf16, 4 for f32. `is_bf16` separates the two 2-byte
  // cases, which are not interchangeable: reading bf16 as f16 gives
  // values of roughly the right magnitude and entirely the wrong
  // content.
  int  elem_size = 4;
  bool is_bf16   = false;

  std::size_t elems() const noexcept
  {
    if (data == nullptr || shape.empty()) { return 0; }
    std::size_t n = 1;
    for (const int d : shape) {
      if (d <= 0) { return 0; }
      n *= (std::size_t)d;
    }
    return n;
  }
};

// Look one up. False means the graph wired nothing under that name,
// which is the answer for every name on a graph that predates it -- so
// a family MUST work when this returns false for everything.
//
// NAMES ARE THE HOST'S and are lower_snake. They are added, never
// renamed or repurposed: a name is a promise to every binary already
// asking for it. The set is empty today, which is the honest state of
// it -- this is the seam, not a backlog.
using NamedInputFn =
    std::function<bool(std::string_view name, NamedTensor* out)>;

}  // namespace vpipe::genai

#endif
