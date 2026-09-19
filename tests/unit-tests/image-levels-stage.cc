// `image-levels`: the tone curve, and the order the knobs apply in.
//
// The mapping is per-sample and position-independent, so it is pinned
// here through map_unit() rather than by pushing frames through a
// runtime -- what matters is the arithmetic and that the defaults are
// the identity.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/typed-stage.h"
#include "stages/audio-video/image-levels-stage.h"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

std::unique_ptr<ImageLevelsStage>
make_(Session& s, const char* id, FlexData cfg)
{
  return std::make_unique<ImageLevelsStage>(&s, id, std::vector<InEdge>{},
                                            std::move(cfg));
}

FlexData
cfg_(std::initializer_list<std::pair<const char*, double>> kv)
{
  auto o = FlexData::make_object();
  for (auto& p : kv) {
    o.as_object().insert_or_assign(p.first, FlexData::make_real(p.second));
  }
  return o;
}

bool near_(double a, double b, double tol = 1e-9)
{
  return std::fabs(a - b) <= tol;
}

}  // namespace

TEST(image_levels, defaults_are_the_identity)
{
  Session s;
  auto st = make_(s, "lv", FlexData::make_object());
  EXPECT_TRUE(st->config_error().empty());
  for (double x : {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}) {
    EXPECT_TRUE(near_(st->map_unit(x), x));
  }
}

TEST(image_levels, brightness_is_an_offset_and_clamps)
{
  Session s;
  auto st = make_(s, "lv", cfg_({{"brightness", 0.02}}));
  EXPECT_TRUE(near_(st->map_unit(0.5), 0.52));
  EXPECT_TRUE(near_(st->map_unit(0.0), 0.02));
  // and it saturates rather than wrapping
  EXPECT_TRUE(near_(st->map_unit(1.0), 1.0));

  auto dn = make_(s, "dn", cfg_({{"brightness", -0.02}}));
  EXPECT_TRUE(near_(dn->map_unit(0.0), 0.0));   // clamped, not negative
}

TEST(image_levels, contrast_holds_the_pivot_fixed)
{
  Session s;
  auto st = make_(s, "lv", cfg_({{"contrast", 2.0}, {"pivot", 0.5}}));
  EXPECT_TRUE(near_(st->map_unit(0.5), 0.5));          // the pivot
  EXPECT_TRUE(near_(st->map_unit(0.6), 0.7));
  EXPECT_TRUE(near_(st->map_unit(0.4), 0.3));

  // A pivot elsewhere holds THAT level instead -- the knob a dark clip
  // wants, so its shadows are not the first thing to move.
  auto lo = make_(s, "lo", cfg_({{"contrast", 2.0}, {"pivot", 0.25}}));
  EXPECT_TRUE(near_(lo->map_unit(0.25), 0.25));
  EXPECT_TRUE(near_(lo->map_unit(0.35), 0.45));
}

TEST(image_levels, gamma_moves_midtones_and_pins_the_ends)
{
  Session s;
  auto st = make_(s, "lv", cfg_({{"gamma", 2.0}}));
  EXPECT_TRUE(near_(st->map_unit(0.0), 0.0));
  EXPECT_TRUE(near_(st->map_unit(1.0), 1.0));
  EXPECT_TRUE(near_(st->map_unit(0.5), 0.25));     // darkened midtone
  std::printf("[image_levels] gamma 2.0: 0.5 -> %.4f\n", st->map_unit(0.5));
}

TEST(image_levels, the_knobs_apply_gamma_contrast_brightness_in_that_order)
{
  Session s;
  auto st = make_(s, "lv",
                  cfg_({{"gamma", 2.0}, {"contrast", 2.0},
                        {"pivot", 0.5}, {"brightness", 0.1}}));
  // 0.5 -> gamma 0.25 -> contrast about 0.5 gives 0.0 -> +0.1
  EXPECT_TRUE(near_(st->map_unit(0.5), 0.1));
  // Brightness applied FIRST would give ((0.5^2)+0.1-0.5)*2+0.5 = 0.2,
  // so this value is what distinguishes the two orders.
  EXPECT_TRUE(!near_(st->map_unit(0.5), 0.2));
}

TEST(image_levels, a_bad_gamma_or_pivot_is_refused_at_config)
{
  Session s;
  EXPECT_FALSE(make_(s, "g0", cfg_({{"gamma", 0.0}}))->config_error().empty());
  EXPECT_FALSE(make_(s, "gn", cfg_({{"gamma", -1.0}}))->config_error().empty());
  EXPECT_FALSE(make_(s, "p2", cfg_({{"pivot", 1.5}}))->config_error().empty());
  EXPECT_TRUE(make_(s, "ok", cfg_({{"gamma", 2.2}}))->config_error().empty());
}

// The measured MiniMax-H3 seam: a continuation lands about 6 levels
// darker than the guide it continued from, near-uniformly. This pins the
// compensation a graph applies to the guide.
TEST(image_levels, compensates_the_h3_continuation_step)
{
  Session s;
  auto st = make_(s, "lv", cfg_({{"brightness", 6.0 / 255.0}}));
  for (double v : {37.0, 60.0, 136.0, 200.0}) {
    const double out = st->map_unit(v / 255.0) * 255.0;
    EXPECT_TRUE(std::fabs(out - (v + 6.0)) < 0.01);
  }
}
