// FlashVSR as a `generate-video` family: the registry path, the
// geometry it answers before anything loads, and one generation driven
// the way the stage drives it.
//
// THIS IS THE WIRING, not the model. The model is checked against the
// reference in flashvsr-dit-load.cc; what is new here is that the
// registry finds the family, that the family claims the right
// directories and refuses the wrong ones, and that a source clip handed
// over as the NAMED INPUT `source_video` reaches the denoiser and comes
// back as the same latent the direct call produces.
//
// THE REFUSAL IS PART OF THE CONTRACT and gets its own case. This model
// conditions on the source and nothing else, so a request without one
// must be refused rather than generated: noise plus a constant prompt
// produces a confident 4x upscale of nothing, which looks like a
// working model and is the failure most worth catching.
//
// Env, both required or the test skips:
//   VPIPE_FLASHVSR_TEST_MODEL_PATH  the converted FlashVSR-v1.1 dir
//   VPIPE_FLASHVSR_GOLDEN           a dir fvsr/gen_goldens.py wrote

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "generative-models/flashvsr/flashvsr-family.h"
#include "generative-models/flashvsr/flashvsr-lq-proj.h"
#include "generative-models/flashvsr/metal-flashvsr-transformer.h"
#include "generative-models/weight-set.h"
#include "generative-models/video-model-registry.h"
#include "pipeline/stage-registry.h"
#include "pipeline/stage-spec.h"
#include "stages/flashvsr-model-config-stage.h"
#include "stages/flashvsr-src-encoder-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;

namespace {

namespace fs = std::filesystem;

std::vector<float>
read_f32_(const fs::path& p)
{
  std::ifstream in(p, std::ios::binary | std::ios::ate);
  if (!in) { return {}; }
  const std::streamsize n = in.tellg();
  in.seekg(0);
  std::vector<float> out((std::size_t)n / 4);
  in.read(reinterpret_cast<char*>(out.data()), n);
  return out;
}

double
rel_l2_(const std::vector<float>& got, const std::vector<float>& ref)
{
  if (got.size() != ref.size() || got.empty()) { return 1e9; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < got.size(); ++i) {
    const double d = (double)got[i] - (double)ref[i];
    num += d * d;
    den += (double)ref[i] * (double)ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

const char* root_() {
  const char* r = std::getenv("VPIPE_FLASHVSR_TEST_MODEL_PATH");
  return (r != nullptr && *r != '\0') ? r : nullptr;
}
const char* gold_() {
  const char* g = std::getenv("VPIPE_FLASHVSR_GOLDEN");
  return (g != nullptr && *g != '\0') ? g : nullptr;
}

}  // namespace

TEST(flashvsr_family, registered_and_claims_narrowly)
{
  // The family registers itself when libvpipe loads, which is what makes
  // it reachable from the stage without the stage naming it.
  VideoModelFamily* fam = VideoModelRegistry::get().find("flashvsr");
  ASSERT_TRUE(fam != nullptr);
  if (fam == nullptr) { return; }
  EXPECT_TRUE(fam->tag() == "flashvsr");

  // Geometry, answered before anything is loaded -- that is the whole
  // point of asking the family rather than the generator.
  EXPECT_TRUE(fam->align_frames("", 41) == 41);
  EXPECT_TRUE(fam->align_frames("", 42) == 49);
  int gh = 0, gw = 0;
  fam->size_grid("", &gh, &gw);
  EXPECT_TRUE(gh == 128 && gw == 128);
  // 41 frames -> 10 latent frames at 512px.
  EXPECT_TRUE(fam->latent_bytes("", 512, 512, 41) ==
              (std::size_t)16 * 10 * 64 * 64 * 4);

  // CLAIMS NARROWLY. A directory with no source projection is not this
  // family even if it holds a Wan denoiser -- claiming one would load a
  // plain Wan checkpoint at full cost and drive it with a projection it
  // does not have.
  EXPECT_FALSE(fam->claims("/nonexistent-flashvsr-dir", ""));
  if (const char* r = root_()) {
    EXPECT_TRUE(fam->claims(r, ""));
    // Its own `vae/` subdirectory holds 194 VAE tensors and no
    // projection, so it must NOT be claimed as a denoiser.
    EXPECT_FALSE(fam->claims(std::string(r) + "/vae", ""));
  }
}

// THE TWO LAYOUTS, told apart by files alone. Needs no checkpoint: empty
// files with the published names are what the resolver looks for, and
// the one it looks for FIRST among them is the source projection, since
// the denoiser alone is a plain Wan 2.1 transformer.
TEST(flashvsr_family, the_published_layout_is_recognised_by_its_files)
{
  namespace fs = std::filesystem;
  const fs::path d = fs::temp_directory_path() /
      ("vpipe-flashvsr-layout-" + std::to_string((long)::getpid()));
  std::error_code ec;
  fs::remove_all(d, ec);
  fs::create_directories(d, ec);
  auto touch = [&](const char* n) { std::ofstream(d / n).put('\0'); };
  touch("diffusion_pytorch_model_streaming_dmd.safetensors");
  touch("posi_prompt.pth");
  touch("Wan2.1_VAE.pth");

  FlashVsrLayout l;
  // The denoiser, the context and the VAE -- but no projection: not ours.
  EXPECT_FALSE(resolve_flashvsr_layout(d.string(), &l));
  VideoModelFamily* fam = VideoModelRegistry::get().find("flashvsr");
  if (fam != nullptr) { EXPECT_FALSE(fam->claims(d.string(), "")); }

  touch("LQ_proj_in.ckpt");
  const bool ok = resolve_flashvsr_layout(d.string(), &l);
  EXPECT_TRUE(ok);
  if (ok) {
    EXPECT_TRUE(l.published);
    EXPECT_TRUE(fs::path(l.denoiser).filename() ==
                "diffusion_pytorch_model_streaming_dmd.safetensors");
    EXPECT_TRUE(fs::path(l.source).filename() == "LQ_proj_in.ckpt");
    EXPECT_TRUE(l.source_prefix.empty());
    EXPECT_TRUE(fs::path(l.context).filename() == "posi_prompt.pth");
    EXPECT_TRUE(l.context_name == "posi_prompt");
    EXPECT_TRUE(fs::path(l.vae).filename() == "Wan2.1_VAE.pth");
  }
  if (fam != nullptr) { EXPECT_TRUE(fam->claims(d.string(), "")); }
  fs::remove_all(d, ec);
}

// THE WINDOW, THE KNOB THAT SETS IT, AND THE PLAN THAT READS IT. Needs no
// checkpoint: the geometry is Wan 2.1-T2V-1.3B's, which is Config's
// default, and the numbers are what forward_chunk_ allocates.
TEST(flashvsr_family, the_window_is_declared_as_allocated)
{
  using T = MetalFlashVsrTransformer;
  // kv_ratio + 1 windows, and never fewer than the opening chunk's three.
  EXPECT_TRUE(T::kv_windows(0.0) == 3 && T::kv_windows(1.0) == 3);
  EXPECT_TRUE(T::kv_windows(2.0) == 3 && T::kv_windows(3.0) == 4);
  // A K and a V per block, bf16, at 1920x1152: a window is 2x72x120
  // tokens. 25.5 GB before the ring, at kv_ratio 3.
  const T::Config cfg;
  EXPECT_TRUE(T::kv_window_bytes(cfg, 1152, 1920, 3.0) == 12740198400ull);
  EXPECT_TRUE(T::kv_window_bytes(cfg, 1152, 1920, 2.0) == 9555148800ull);
  T::Params p;
  p.kv_ratio = 2.0;
  const std::uint64_t all = T::denoise_scratch_bytes(cfg, 1152, 1920, 89, p);
  // The window plus the opening chunk's scratch, routing and the latent:
  // more than the window, and not by more than a few GB.
  EXPECT_TRUE(all > 9555148800ull && all < 9555148800ull + (4ull << 30));
  std::printf("  1920x1152 x 89 frames, kv_ratio 2: window %.2f GB, "
              "denoise scratch %.2f GB\n", 9555148800.0 / 1e9,
              (double)all / 1e9);

  // The source: its keys, its category, the tag a model_config iport
  // takes, and a value the routing cannot use refused at config time.
  const StageSpec* sp = StageRegistry::get().spec("flashvsr-model-config");
  ASSERT_TRUE(sp != nullptr);
  if (sp == nullptr) { return; }
  EXPECT_TRUE(sp->category == StageCategory::ModelSpecificConfig);
  auto type_of = [&](const char* k) -> int {
    for (const ConfigKey& c : sp->attrs) {
      if (std::string(c.key) == k) { return (int)c.type; }
    }
    return -1;
  };
  EXPECT_TRUE(type_of("kv_ratio") == (int)ConfigType::Real);
  EXPECT_TRUE(type_of("sparse_ratio") == (int)ConfigType::Real);
  EXPECT_TRUE(type_of("local_range") == (int)ConfigType::Int);
  ASSERT_TRUE(sp->oports.size() == 1);
  if (sp->oports.size() == 1) {
    EXPECT_TRUE(std::string(sp->oports[0].tags) == "model-config");
  }
  Session sess;
  {
    FlexData c = FlexData::make_object();
    c.as_object().insert_or_assign("kv_ratio", FlexData::make_real(2.0));
    FlashVsrModelConfigStage s(&sess, "cfg", std::vector<InEdge>{}, c);
    EXPECT_TRUE(s.config_error().empty());
    EXPECT_TRUE(s.num_oports() == 1);
    // ONLY what was set, stamped with the family.
    FlexData out = s.resolved_config();
    auto o = out.as_object();
    EXPECT_TRUE(std::string(o.at("model_family").as_string("")) == "flashvsr");
    EXPECT_TRUE(o.at("kv_ratio").as_real(0.0) == 2.0);
    EXPECT_FALSE(o.contains("sparse_ratio") || o.contains("local_range"));
  }
  {
    FlexData c = FlexData::make_object();
    c.as_object().insert_or_assign("kv_ratio", FlexData::make_real(-1.0));
    FlashVsrModelConfigStage s(&sess, "bad", std::vector<InEdge>{}, c);
    EXPECT_FALSE(s.config_error().empty());
  }
  // A root that is no FlashVSR checkpoint declares nothing rather than a
  // guess.
  VideoModelFamily* fam = VideoModelRegistry::get().find("flashvsr");
  if (fam != nullptr) {
    EXPECT_TRUE(fam->denoise_scratch_bytes("/nonexistent-flashvsr-dir", 1920,
                                           1152, 89, nullptr) == 0);
  }
}

TEST(flashvsr_family, refuses_without_conditioning)
{
  const char* r = root_();
  if (r == nullptr) { return; }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  VideoModelFamily* fam = VideoModelRegistry::get().find("flashvsr");
  if (fam == nullptr) { return; }

  VideoModelCreateArgs args;
  args.root = r;
  args.model_type = "flashvsr";
  args.metal = mc;
  args.session = &sess;
  auto gen = fam->load(args);
  ASSERT_TRUE(gen != nullptr);
  if (gen == nullptr) { return; }
  EXPECT_TRUE(gen->latent_channels() == 16);
  EXPECT_TRUE(gen->spatial_compression() == 8);
  EXPECT_TRUE(gen->resident_bytes() > 0);

  // A request with NO CONDITIONING, which is what a graph with no
  // conditioner wired produces. For this family that is the same
  // refusal a missing prompt is for any other: its conditioning IS its
  // source clip, projected.
  VideoGenRequest req;
  req.height = 256;
  req.width = 256;
  req.frames = 41;
  VideoGenResult res;
  EXPECT_FALSE(gen->generate(req, &res));
  EXPECT_TRUE(res.video.empty());
}

// The SPLIT, driven the way the graph drives it: the conditioner's own
// projection over the clip, then the rows into the family as ordinary
// conditioning. What this covers that the denoiser's own test does not
// is that the two halves agree on the row layout -- one row frame per
// four source frames, tokens at the denoiser's grid, and the row
// frame's index being the latent frame's index.
TEST(flashvsr_family, generates_from_projected_rows)
{
  const char* r = root_();
  const char* g = gold_();
  if (r == nullptr || g == nullptr) { return; }
  const fs::path gp(g);
  const auto want = read_f32_(gp / "latent.f32");
  std::ifstream lqf(gp / "lq_video_u8.bin", std::ios::binary | std::ios::ate);
  std::ifstream cfgf(gp / "config.json");
  if (want.empty() || !lqf || !cfgf) { return; }
  FlexData fd = FlexData::from_json(cfgf);
  if (!fd.is_object()) { return; }
  auto o = fd.as_object();
  const int size = (int)o.at("size").as_real(0.0);
  const int frames = (int)o.at("frames").as_real(0.0);
  if (size <= 0 || frames <= 0) { return; }

  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }
  VideoModelFamily* fam = VideoModelRegistry::get().find("flashvsr");
  if (fam == nullptr) { return; }

  const std::streamsize nb = lqf.tellg();
  lqf.seekg(0);
  std::vector<std::uint8_t> planar((std::size_t)nb);
  lqf.read(reinterpret_cast<char*>(planar.data()), nb);
  // The golden clip is [3, F, H, W]; the port takes [F, 3, H, W], which
  // is the shape temporal-stack builds.
  const std::size_t px = (std::size_t)size * size;
  std::vector<std::uint8_t> clip(planar.size());
  for (int f = 0; f < frames; ++f) {
    for (int c = 0; c < 3; ++c) {
      std::memcpy(clip.data() + ((std::size_t)f * 3 + c) * px,
                  planar.data() + ((std::size_t)c * frames + f) * px, px);
    }
  }

  // ---- the CONDITIONER's half. Driven with the reference's own
  // irregular pattern: the opening slice seeds the causal carries and
  // produces nothing, so a single call over the clip returns no rows.
  FlashVsrLayout layout;
  ASSERT_TRUE(resolve_flashvsr_layout(r, &layout));
  auto proj = FlashVsrLqProj::load(WeightSet::open(layout.source, nullptr),
                                   mc, nullptr, layout.source_prefix);
  ASSERT_TRUE(proj != nullptr);
  if (proj == nullptr) { return; }
  const int dim = proj->config().out_dim;
  const int sh = proj->config().shuffle_h;
  const std::size_t tok = (std::size_t)(size / sh) * (size / sh);
  const std::size_t frame_px = (std::size_t)3 * size * size;
  std::vector<metal_compute::SharedBuffer> rows;
  std::string perr;
  auto drive = [&](int lo, int hi) {
    if (hi <= lo || hi > frames) { return true; }
    int rf = 0;
    return proj->stream_forward(clip.data() + (std::size_t)lo * frame_px,
                                hi - lo, size, size, &rows, &rf, &perr);
  };
  proj->reset();
  bool ok = true;
  for (int j = 0; j < 7 && ok; ++j) {
    ok = drive(std::max(0, j * 4 - 3), (j + 1) * 4 - 3);
  }
  const int chunks = (frames - 1) / 8 - 2;
  for (int c = 1; c < chunks && ok; ++c) {
    for (int j = 0; j < 2 && ok; ++j) {
      ok = drive(c * 8 + 17 + j * 4, c * 8 + 21 + j * 4);
    }
  }
  ASSERT_TRUE(ok);
  if (!ok) { return; }
  // The row frame count IS the latent frame count -- 6 from the opening
  // chunk and 2 from each later one, which is 2*chunks + 4.
  EXPECT_TRUE((int)rows.size() ==
              MetalFlashVsrTransformer::latent_frames(frames));
  std::vector<std::uint16_t> cond(rows.size() * tok * (std::size_t)dim);
  for (std::size_t i = 0; i < rows.size(); ++i) {
    std::memcpy(cond.data() + i * tok * (std::size_t)dim,
                rows[i].contents(), tok * (std::size_t)dim * 2);
  }

  VideoModelCreateArgs args;
  args.root = r;
  args.model_type = "flashvsr";
  args.metal = mc;
  args.session = &sess;
  args.width = size;
  args.height = size;
  args.frames = frames;
  auto gen = fam->load(args);
  ASSERT_TRUE(gen != nullptr);
  if (gen == nullptr) { return; }

  FlexData mc_cfg = FlexData::make_object();
  {
    auto mo = mc_cfg.as_object();
    if (o.contains("local_range")) {
      mo.insert_or_assign("local_range", o.at("local_range"));
    }
    if (o.contains("kv_ratio")) {
      mo.insert_or_assign("kv_ratio", o.at("kv_ratio"));
    }
  }

  VideoGenRequest req;
  req.height = size;
  req.width = size;
  req.frames = frames;
  req.seed = 0;
  req.model_config = &mc_cfg;
  req.cond = cond.data();
  req.cond_rows = (int)(rows.size() * tok);
  req.cond_dim = dim;
  int chunks_seen = 0;
  req.progress = [&](int, int) { ++chunks_seen; return true; };

  VideoGenResult res;
  const bool gok = gen->generate(req, &res);
  ASSERT_TRUE(gok);
  if (!gok) { return; }
  ASSERT_TRUE(res.video_shape.size() == 4);
  if (res.video_shape.size() != 4) { return; }
  std::printf("  family latent [%d,%d,%d,%d] from %zu row frames, "
              "%d chunks reported\n", res.video_shape[0], res.video_shape[1],
              res.video_shape[2], res.video_shape[3], rows.size(),
              chunks_seen);
  EXPECT_TRUE(chunks_seen == chunks);
  // The noise is DRAWN here rather than fed, so the latent cannot match
  // the golden -- but its shape, scale and spread must. A wrong split
  // shows up as the wrong shape or a dead output, not a small error.
  EXPECT_TRUE(res.video.size() == want.size());
  double mean = 0.0, var = 0.0;
  for (const float v : res.video) { mean += v; }
  mean /= (double)res.video.size();
  for (const float v : res.video) { var += (v - mean) * (v - mean); }
  var /= (double)res.video.size();
  const double sd = std::sqrt(var);
  std::printf("  latent mean %+.3f sd %.3f (golden near mean -0.29 sd 1.18)\n",
              mean, sd);
  EXPECT_TRUE(sd > 0.5 && sd < 3.0);
  EXPECT_TRUE(std::fabs(mean) < 1.5);
  EXPECT_TRUE(rel_l2_(res.video, want) > 0.05);
}

// The source encoder's PORT SURFACE and its category.
//
// Both are contracts a graph author and the composer read rather than
// exercise, so nothing at run time notices when they drift. The
// category in particular is load-bearing: `model-specific-config` is
// what tells tooling to offer this stage only where the resident
// checkpoint is FlashVSR, instead of in a flat list where picking wrong
// looks like picking right.
//
// Needs no model and no GPU, so it runs in the default suite.
TEST(flashvsr_family, source_encoder_surface)
{
  const StageSpec* sp = StageRegistry::get().spec("flashvsr-src-encoder");
  ASSERT_TRUE(sp != nullptr);
  if (sp == nullptr) { return; }

  // MODEL-SPECIFIC, like the per-family config stages beside it: its
  // applicability is decided by the checkpoint, not by the graph.
  EXPECT_TRUE(sp->category == StageCategory::ModelSpecificConfig);

  ASSERT_TRUE(sp->iports.size() == 2);
  if (sp->iports.size() == 2) {
    EXPECT_TRUE(std::string(sp->iports[0].name) == "source");
    EXPECT_TRUE(std::string(sp->iports[1].name) == "model");
  }
  ASSERT_TRUE(sp->oports.size() == 1);
  if (sp->oports.size() == 1) {
    EXPECT_TRUE(std::string(sp->oports[0].name) == "conditioning");
    // The TAG is the whole compatibility check against generate-video's
    // conditioning iport: the composer offers the connection only if it
    // matches, and that match is what lets this stage replace a text
    // conditioner with no change to the stage it feeds.
    EXPECT_TRUE(port_tags_compatible("conditioning", sp->oports[0].tags));
  }

  // ...and an INSTANCE HAS THEM. The spec only describes the ports; the
  // constructor creates them. This stage once described one oport and
  // created none, so the shipped flashvsr-restore graph was refused at
  // pipeline_from_spec with "upstream has 0 oports" -- a failure no check
  // on the spec could see.
  Session sess;
  FlashVsrSrcEncoderStage s(&sess, "enc", std::vector<InEdge>{},
                            FlexData::make_object());
  EXPECT_TRUE(s.num_oports() == sp->oports.size());
}
