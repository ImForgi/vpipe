// minimax-h3-context-export / minimax-h3-context-import /
// minimax-h3-context-trim, driven through real pipelines with synthetic
// beats -- no model, no GPU.
//
//   * export -> file -> import: the tail that comes out is the tail of
//     what went in, byte for byte, with the audio end-aligned and the
//     trim the importer reports;
//   * import from the graph (iports 0/1) makes the same context;
//   * trim drops the overlap from the frames AND the PCM, renumbers the
//     frames and conforms the PCM to the kept picture.

#include "minitest.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/session.h"
#include "pipeline/pipeline-runtime.h"
#include "pipeline/pipeline.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/minimax-h3-context-export-stage.h"
#include "stages/minimax-h3-context-import-stage.h"
#include "stages/minimax-h3-chain-normalize-stage.h"
#include "stages/minimax-h3-context-trim-stage.h"
#include "stages/minimax-h3-context.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

// A 124-frame clip on a tiny 4x6 latent canvas: 37 video latents, 207
// audio latents -- H3's own grid, so the +1/3 overhang is real.
constexpr int kZ = 24, kT = 37, kH = 4, kW = 6, kA = 207;

float
video_val_(int c, int t, int i)
{
  return (float)(c * 1000 + t * 10) + (float)i * 0.25f;
}

float
audio_val_(int s, int ch, int i)
{
  return -(float)(s * 100000 + ch * 1000 + i);
}

TensorBeat
video_latent_()
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {kZ, kT, kH, kW};
  tb.resize_contiguous((std::size_t)kZ * kT * kH * kW);
  float* d = tb.as_f32();
  for (int c = 0; c < kZ; ++c) {
    for (int t = 0; t < kT; ++t) {
      for (int i = 0; i < kH * kW; ++i) {
        d[((std::size_t)c * kT + t) * kH * kW + i] = video_val_(c, t, i);
      }
    }
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
  sb.as_object().insert_or_assign("frames", FlexData::make_int(124));
  tb.sideband = std::move(sb);
  return tb;
}

TensorBeat
audio_latent_()
{
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::F32;
  tb.shape = {2, 32, kA};
  tb.resize_contiguous((std::size_t)2 * 32 * kA);
  float* d = tb.as_f32();
  for (int s = 0; s < 2; ++s) {
    for (int ch = 0; ch < 32; ++ch) {
      for (int i = 0; i < kA; ++i) {
        d[((std::size_t)s * 32 + ch) * kA + i] = audio_val_(s, ch, i);
      }
    }
  }
  return tb;
}

// Emits one beat per oport, in oport order, then finishes.
class MultiSource : public TypedStage<MultiSource> {
public:
  static constexpr const char* kTypeName = "ut-h3ctx-source";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> beats;   // index = oport
  bool done = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (done) { ctx.signal_done(); co_return; }
    done = true;
    for (std::size_t p = 0; p < beats.size(); ++p) {
      if (beats[p]) { co_await ctx.write((unsigned)p, std::move(beats[p])); }
    }
  }
};

// Emits a list of beats on oport 0, REBUILT for every launch -- what a
// real source does, and what the relaunch tests below need.
class RefillSource : public TypedStage<RefillSource> {
public:
  static constexpr const char* kTypeName = "ut-h3ctx-refill";
  using TypedStage::TypedStage;
  std::function<std::vector<std::unique_ptr<BeatPayloadIntf>>()> make;
  std::vector<std::unique_ptr<BeatPayloadIntf>> beats;
  std::size_t next = 0;
  bool        filled = false;
  Job
  process(RuntimeContext& ctx) override
  {
    if (!filled) {
      beats  = make();
      next   = 0;
      filled = true;
    }
    if (next >= beats.size()) {
      ctx.signal_done();
      co_return;
    }
    co_await ctx.write(0, std::move(beats[next++]));
  }
  void
  reset_run_state() override
  {
    beats.clear();
    next   = 0;
    filled = false;
  }
};

// Emits a list of beats on oport 0.
class ListSource : public TypedStage<ListSource> {
public:
  static constexpr const char* kTypeName = "ut-h3ctx-list";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> beats;
  std::size_t next = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (next >= beats.size()) { ctx.signal_done(); co_return; }
    co_await ctx.write(0, std::move(beats[next++]));
  }
};

// Emits one beat on oport 0, but only after `quiet` turns of saying
// nothing -- so whatever it carries is guaranteed to arrive after beats
// from sources that speak immediately, whichever clock domain the runtime
// happens to schedule first.
class LateSource : public TypedStage<LateSource> {
public:
  static constexpr const char* kTypeName = "ut-h3ctx-late";
  using TypedStage::TypedStage;
  std::unique_ptr<BeatPayloadIntf> beat;
  int quiet = 3;
  int turn  = 0;
  Job
  process(RuntimeContext& ctx) override
  {
    if (turn++ < quiet) { co_return; }
    if (!beat) { ctx.signal_done(); co_return; }
    co_await ctx.write(0, std::move(beat));
  }
  void
  reset_run_state() override
  {
    turn = 0;
  }
};

class Collect : public TypedStage<Collect> {
public:
  static constexpr const char* kTypeName = "ut-h3ctx-collect";
  using TypedStage::TypedStage;
  std::vector<std::unique_ptr<BeatPayloadIntf>> got;
  Job
  process(RuntimeContext& ctx) override
  {
    auto b = co_await ctx.read(0);
    if (!b) { ctx.signal_done(); co_return; }
    got.push_back(std::move(b));
  }
};

std::unique_ptr<BeatPayloadIntf>
tensor_(TensorBeat tb)
{
  return make_payload<TensorBeatPayload>(std::move(tb));
}

std::string
tmp_(const char* name)
{
  return (std::filesystem::temp_directory_path() / name).string();
}

FlexData
cfg_(std::initializer_list<std::pair<const char*, FlexData>> kv)
{
  FlexData o = FlexData::make_object();
  for (auto& p : kv) { o.as_object().insert_or_assign(p.first, p.second); }
  return o;
}

std::int64_t
sb_int_(const FlexData& sb, const char* k, std::int64_t def = -1)
{
  if (!sb.is_object()) { return def; }
  FlexData s = sb;
  const auto o = s.as_object();
  return o.contains(k) ? o.at(k).as_int(def) : def;
}

double
sb_real_(const FlexData& sb, const char* k, double def = 1e9)
{
  if (!sb.is_object()) { return def; }
  FlexData s = sb;
  const auto o = s.as_object();
  return o.contains(k) ? o.at(k).as_real(def) : def;
}

// The context an importer emitted, checked against the synthetic clip.
// Returns the first mismatch (empty = all good): a helper outside a TEST
// body cannot use EXPECT_TRUE.
std::string
context_mismatch_(const Collect& video, const Collect& audio,
                  const Collect& info)
{
  if (video.got.size() != 1 || audio.got.size() != 1 || info.got.size() != 1) {
    return "expected one beat per oport, got " +
           std::to_string(video.got.size()) + "/" +
           std::to_string(audio.got.size()) + "/" +
           std::to_string(info.got.size());
  }
  const auto* v = dynamic_cast<const TensorBeatPayload*>(video.got[0].get());
  const auto* a = dynamic_cast<const TensorBeatPayload*>(audio.got[0].get());
  const auto* f = dynamic_cast<const FlexDataPayload*>(info.got[0].get());
  if (v == nullptr || a == nullptr || f == nullptr) { return "payload types"; }

  if (v->shape != std::vector<std::int64_t>{kZ, 7, kH, kW}) {
    return "video tail shape";
  }
  const float* vd = v->as_f32();
  for (int c = 0; c < kZ; ++c) {
    for (int t = 0; t < 7; ++t) {
      for (int i = 0; i < kH * kW; ++i) {
        if (vd[((std::size_t)c * 7 + t) * kH * kW + i] !=
            video_val_(c, kT - 7 + t, i)) {
          return "video tail values";
        }
      }
    }
  }
  if (sb_int_(v->sideband, "context_frames") != 22 ||
      sb_int_(v->sideband, "start_frame") != 0) {
    return "video sideband";
  }
  if (a->shape != std::vector<std::int64_t>{2, 32, 37}) {
    return "audio tail shape";
  }
  const float* ad = a->as_f32();
  if (ad[0] != audio_val_(0, 0, kA - 37) ||
      ad[((std::size_t)1 * 32 + 31) * 37 + 36] != audio_val_(1, 31, kA - 1)) {
    return "audio tail values";
  }
  // 5/3 * 22 + (207 - 5/3 * 124) - 37 = 0.
  if (!(std::fabs(sb_real_(a->sideband, "audio_offset")) < 1e-9)) {
    return "audio_offset";
  }
  if (sb_int_(f->data, "trim_frames") != 22 ||
      sb_int_(f->data, "source_frames") != 124) {
    return "info";
  }
  return {};
}

}  // namespace

TEST(minimax_h3_context_stages, types_are_registered)
{
  Session sess;
  Pipeline pl("p", &sess);
  EXPECT_TRUE(pl.insert_stage("minimax-h3-context-export", "e", {},
                              cfg_({{"output_url",
                                     FlexData::make_string(tmp_("x.h3ctx"))}})) !=
              nullptr);
  EXPECT_TRUE(pl.insert_stage("minimax-h3-context-import", "i", {},
                              FlexData::make_object()) != nullptr);
  EXPECT_TRUE(pl.insert_stage("minimax-h3-context-trim", "t", {},
                              FlexData::make_object()) != nullptr);
}

TEST(minimax_h3_context_stages, bad_config_is_deferred)
{
  Session sess;
  MiniMaxH3ContextImportStage bad_frames(
      &sess, "i", {}, cfg_({{"context_frames", FlexData::make_int(20)}}));
  EXPECT_FALSE(bad_frames.config_error().empty());
  MiniMaxH3ContextImportStage bad_mode(
      &sess, "i", {},
      cfg_({{"duration_mode", FlexData::make_string("seconds")}}));
  EXPECT_FALSE(bad_mode.config_error().empty());
  MiniMaxH3ContextExportStage no_url(&sess, "e", {}, FlexData::make_object());
  EXPECT_FALSE(no_url.config_error().empty());
  EXPECT_TRUE(MiniMaxH3ContextExportStage::path_for_clip(
                  "/a/clip.h3ctx.safetensors", 0) ==
              "/a/clip.h3ctx.safetensors");
  EXPECT_TRUE(MiniMaxH3ContextExportStage::path_for_clip(
                  "/a/clip.safetensors", 2) == "/a/clip-000002.safetensors");
}

TEST(minimax_h3_context_stages, export_then_import_carries_the_tail)
{
  const std::string path = tmp_("vpipe-h3ctx-stages.safetensors");
  std::error_code ec;
  std::filesystem::remove(path, ec);

  // ---- export -------------------------------------------------------
  {
    Session sess;
    Pipeline pl("export", &sess);
    auto src_u = std::make_unique<MultiSource>(&sess, "src",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
    src_u->beats.push_back(tensor_(video_latent_()));
    src_u->beats.push_back(tensor_(audio_latent_()));
    src_u->allocate_oports(2);
    auto* src = static_cast<MultiSource*>(pl.insert_stage(std::move(src_u)));
    pl.insert_stage(std::make_unique<MiniMaxH3ContextExportStage>(
        &sess, "export", std::vector<InEdge>{{src, 0}, {src, 1}},
        cfg_({{"output_url", FlexData::make_string(path)}})));
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
  }
  h3ctx::ContextFile file;
  std::string err;
  ASSERT_TRUE(h3ctx::read_context_file(path, &file, &err));
  EXPECT_TRUE(file.video_shape == (std::vector<std::int64_t>{kZ, kT, kH, kW}));
  EXPECT_TRUE(file.audio_shape == (std::vector<std::int64_t>{2, 32, kA}));
  EXPECT_TRUE(file.metadata["format"] == h3ctx::kFormat &&
              file.metadata["frames"] == "124");

  // ---- import -------------------------------------------------------
  {
    Session sess;
    Pipeline pl("import", &sess);
    auto* imp = pl.insert_stage(std::make_unique<MiniMaxH3ContextImportStage>(
        &sess, "import", std::vector<InEdge>{},
        cfg_({{"input_url", FlexData::make_string(path)}})));
    auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
        &sess, "cv", std::vector<InEdge>{{imp, 0}}, FlexData::make_object())));
    auto* ca = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
        &sess, "ca", std::vector<InEdge>{{imp, 1}}, FlexData::make_object())));
    auto* ci = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
        &sess, "ci", std::vector<InEdge>{{imp, 2}}, FlexData::make_object())));
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    const std::string why = context_mismatch_(*cv, *ca, *ci);
    if (!why.empty()) {
      std::printf("[minimax_h3_context_stages] %s\n", why.c_str());
    }
    EXPECT_TRUE(why.empty());
  }
  std::filesystem::remove(path, ec);
}

TEST(minimax_h3_context_stages, import_from_the_graph)
{
  Session sess;
  Pipeline pl("graph", &sess);
  auto src_u = std::make_unique<MultiSource>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  src_u->beats.push_back(tensor_(video_latent_()));
  src_u->beats.push_back(tensor_(audio_latent_()));
  src_u->allocate_oports(2);
  auto* src = static_cast<MultiSource*>(pl.insert_stage(std::move(src_u)));
  auto* imp = pl.insert_stage(std::make_unique<MiniMaxH3ContextImportStage>(
      &sess, "import", std::vector<InEdge>{{src, 0}, {src, 1}},
      FlexData::make_object()));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{imp, 0}}, FlexData::make_object())));
  auto* ca = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "ca", std::vector<InEdge>{{imp, 1}}, FlexData::make_object())));
  auto* ci = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "ci", std::vector<InEdge>{{imp, 2}}, FlexData::make_object())));
  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  const std::string why = context_mismatch_(*cv, *ca, *ci);
  if (!why.empty()) {
    std::printf("[minimax_h3_context_stages] %s\n", why.c_str());
  }
  EXPECT_TRUE(why.empty());
}

TEST(minimax_h3_context_stages, trim_drops_the_overlap_from_frames_and_pcm)
{
  constexpr int kFrames = 124, kSr = 32000;
  constexpr std::int64_t kSamples = 207 * 800;   // 8.3 ms longer than 124 frames

  Session sess;
  Pipeline pl("trim", &sess);

  auto frames_u = std::make_unique<ListSource>(&sess, "frames",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  for (int f = 0; f < kFrames; ++f) {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = {3, 2, 2};
    tb.resize_contiguous(12);
    tb.data[0] = (std::uint8_t)f;          // which frame this was
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("frame", FlexData::make_int(f));
    sb.as_object().insert_or_assign("frames", FlexData::make_int(kFrames));
    sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
    tb.sideband = std::move(sb);
    frames_u->beats.push_back(tensor_(std::move(tb)));
  }
  frames_u->allocate_oports(1);
  auto* frames = pl.insert_stage(std::move(frames_u));

  auto pcm_u = std::make_unique<ListSource>(&sess, "pcm",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {2, kSamples};
    tb.resize_contiguous((std::size_t)2 * kSamples);
    float* d = tb.as_f32();
    for (int c = 0; c < 2; ++c) {
      for (std::int64_t i = 0; i < kSamples; ++i) {
        d[(std::size_t)c * kSamples + i] = (float)(c * 1000000 + i);
      }
    }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("sample_rate", FlexData::make_int(kSr));
    sb.as_object().insert_or_assign("channels", FlexData::make_int(2));
    tb.sideband = std::move(sb);
    pcm_u->beats.push_back(tensor_(std::move(tb)));
  }
  pcm_u->allocate_oports(1);
  auto* pcm = pl.insert_stage(std::move(pcm_u));

  auto info_u = std::make_unique<ListSource>(&sess, "info",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  info_u->beats.push_back(make_payload<FlexDataPayload>(
      cfg_({{"trim_frames", FlexData::make_int(22)}})));
  info_u->allocate_oports(1);
  auto* info = pl.insert_stage(std::move(info_u));

  auto* trim = pl.insert_stage(std::make_unique<MiniMaxH3ContextTrimStage>(
      &sess, "trim", std::vector<InEdge>{{frames, 0}, {pcm, 0}, {info, 0}},
      // info wins over trim_frames. This one follows the bytes through, so
      // the two corrections that rewrite them are off: they have their own
      // tests below.
      cfg_({{"trim_frames", FlexData::make_int(5)},
            {"luma_match", FlexData::make_bool(false)},
            {"declick_ms", FlexData::make_real(0.0)}})));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{trim, 0}}, FlexData::make_object())));
  auto* ca = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "ca", std::vector<InEdge>{{trim, 1}}, FlexData::make_object())));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  EXPECT_TRUE(cv->got.size() == (std::size_t)(kFrames - 22));
  if (!cv->got.empty()) {
    const auto* first = dynamic_cast<const TensorBeatPayload*>(cv->got[0].get());
    const auto* last =
        dynamic_cast<const TensorBeatPayload*>(cv->got.back().get());
    EXPECT_TRUE(first != nullptr && first->data[0] == 22 &&
                sb_int_(first->sideband, "frame") == 0 &&
                sb_int_(first->sideband, "frames") == kFrames - 22);
    EXPECT_TRUE(last != nullptr && last->data[0] == kFrames - 1 &&
                sb_int_(last->sideband, "frame") == kFrames - 23);
  }

  ASSERT_TRUE(ca->got.size() == 1);
  if (ca->got.size() == 1) {
    const auto* a = dynamic_cast<const TensorBeatPayload*>(ca->got[0].get());
    ASSERT_TRUE(a != nullptr);
    if (a != nullptr) {
      // 102 frames at 24 fps and 32 kHz, exactly.
      EXPECT_TRUE(a->shape == (std::vector<std::int64_t>{2, 136000}));
      const float* d = a->as_f32();
      // 22 frames = 29333 samples dropped from each channel's head.
      EXPECT_TRUE(d[0] == 29333.0f && d[136000] == 1029333.0f);
      EXPECT_TRUE(sb_int_(a->sideband, "samples") == 136000 &&
                  sb_int_(a->sideband, "sample_rate") == kSr);
    }
  }
}

// A STOPPED pipeline keeps its stages, so everything a stage learned
// during a run has to be put back -- the bug class stage-relaunch-sweep
// exists for, which cannot see these two: the exporter is a sink and the
// trim has a required iport.
TEST(minimax_h3_context_stages, export_rewrites_its_file_on_a_relaunch)
{
  const std::string path   = tmp_("vpipe-h3ctx-relaunch.safetensors");
  const std::string second =
      MiniMaxH3ContextExportStage::path_for_clip(path, 1);
  std::error_code ec;
  std::filesystem::remove(path, ec);
  std::filesystem::remove(second, ec);

  Session sess;
  Pipeline pl("export-twice", &sess);
  auto src_u = std::make_unique<MultiSource>(&sess, "src",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  src_u->allocate_oports(1);
  auto* src = static_cast<MultiSource*>(pl.insert_stage(std::move(src_u)));
  pl.insert_stage(std::make_unique<MiniMaxH3ContextExportStage>(
      &sess, "export", std::vector<InEdge>{{src, 0}},
      cfg_({{"output_url", FlexData::make_string(path)}})));

  for (int run = 0; run < 2; ++run) {
    src->beats.clear();
    src->beats.push_back(tensor_(video_latent_()));
    src->done = false;
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
  }
  // The second launch must REWRITE the configured file: the next
  // pipeline in a chain reads that name, and a file beside it is one
  // nobody imports.
  EXPECT_TRUE(std::filesystem::exists(path, ec));
  EXPECT_FALSE(std::filesystem::exists(second, ec));
  std::filesystem::remove(path, ec);
  std::filesystem::remove(second, ec);
}

TEST(minimax_h3_context_stages, trim_starts_each_launch_from_frame_zero)
{
  constexpr int kFrames = 10, kTrim = 5;
  Session sess;
  Pipeline pl("trim-twice", &sess);

  // Frames with NO sideband index, so the stage falls back to its own
  // counter -- which a launch that did not reset it would carry past the
  // new clip's first frames and drop every one of them.
  auto frames_u = std::make_unique<RefillSource>(&sess, "frames",
                                                 std::vector<InEdge>{},
                                                 FlexData::make_object());
  frames_u->make = []() {
    std::vector<std::unique_ptr<BeatPayloadIntf>> out;
    for (int f = 0; f < kFrames; ++f) {
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::U8;
      tb.shape = {3, 2, 2};
      tb.resize_contiguous(12);
      tb.data[0] = (std::uint8_t)f;
      out.push_back(tensor_(std::move(tb)));
    }
    return out;
  };
  frames_u->allocate_oports(1);
  auto* frames = pl.insert_stage(std::move(frames_u));

  auto* trim = pl.insert_stage(std::make_unique<MiniMaxH3ContextTrimStage>(
      &sess, "trim", std::vector<InEdge>{{frames, 0}},
      cfg_({{"trim_frames", FlexData::make_int(kTrim)}})));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{trim, 0}}, FlexData::make_object())));

  std::size_t per_run[2] = {0, 0};
  for (int run = 0; run < 2; ++run) {
    const std::size_t before = cv->got.size();
    PipelineRuntime rt(&pl, &sess);
    ASSERT_TRUE(rt.launch());
    rt.wait_idle();
    rt.stop();
    per_run[run] = cv->got.size() - before;
  }
  EXPECT_TRUE(per_run[0] == (std::size_t)(kFrames - kTrim));
  EXPECT_TRUE(per_run[1] == per_run[0]);
}

// The join's own two seams: the picture coming off the pinned context at
// the context's level, and the PCM starting mid-waveform.
TEST(minimax_h3_context_stages, luma_match_levels_the_frames_after_the_cut)
{
  constexpr int kFrames = 40, kTrim = 22, kBody = 120;

  // `hot` is the artifact: the first frames after the cut come out above
  // the level the clip then settles at. Returns the level of each
  // delivered frame, read inside the scope that owns the pipeline.
  auto run = [](bool match, const std::vector<int>& hot) {
    Session sess;
    Pipeline pl("luma", &sess);
    auto frames_u = std::make_unique<ListSource>(&sess, "frames",
                                                 std::vector<InEdge>{},
                                                 FlexData::make_object());
    for (int f = 0; f < kFrames; ++f) {
      const int kept = f - kTrim;
      int v = kBody;
      if (kept >= 0 && kept < (int)hot.size()) { v = hot[(std::size_t)kept]; }
      TensorBeat tb;
      tb.dtype = TensorBeat::DType::U8;
      tb.shape = {3, 2, 2};
      tb.resize_contiguous(12);
      for (std::size_t i = 0; i < 12; ++i) { tb.data[i] = (std::uint8_t)v; }
      FlexData sb = FlexData::make_object();
      sb.as_object().insert_or_assign("frame", FlexData::make_int(f));
      sb.as_object().insert_or_assign("frames", FlexData::make_int(kFrames));
      tb.sideband = std::move(sb);
      frames_u->beats.push_back(tensor_(std::move(tb)));
    }
    frames_u->allocate_oports(1);
    auto* frames = pl.insert_stage(std::move(frames_u));
    auto* trim = pl.insert_stage(std::make_unique<MiniMaxH3ContextTrimStage>(
        &sess, "trim", std::vector<InEdge>{{frames, 0}},
        cfg_({{"trim_frames", FlexData::make_int(kTrim)},
              {"luma_match", FlexData::make_bool(match)}})));
    auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
        &sess, "cv", std::vector<InEdge>{{trim, 0}}, FlexData::make_object())));
    PipelineRuntime rt(&pl, &sess);
    std::vector<int> out;
    if (!rt.launch()) { return out; }
    rt.wait_idle();
    rt.stop();
    for (auto& b : cv->got) {
      const auto* t = dynamic_cast<const TensorBeatPayload*>(b.get());
      out.push_back(t == nullptr ? -1 : (int)t->data[0]);
    }
    return out;
  };

  const std::vector<int> hot = {138, 126, 120};   // +15%, +5%, settled
  const std::vector<int> off = run(false, hot);
  const std::vector<int> on  = run(true, hot);
  ASSERT_TRUE(off.size() == (std::size_t)(kFrames - kTrim));
  ASSERT_TRUE(on.size() == off.size());
  if (on.size() != off.size() || on.size() < 4) { return; }

  // Untouched, the step into the clip is the whole 15%.
  EXPECT_TRUE(off[0] == 138 && off[1] == 126);
  // Corrected, the first frames arrive on the body's level...
  EXPECT_TRUE(std::abs(on[0] - kBody) <= 1);
  EXPECT_TRUE(std::abs(on[1] - kBody) < std::abs(off[1] - kBody));
  // ...and every frame that was already there is delivered untouched --
  // the window bounds how far the correction reaches, it does not grade
  // the frames it reaches over.
  for (std::size_t k = 2; k < on.size(); ++k) {
    EXPECT_TRUE(on[k] == kBody);
  }

  // A clean join is left exactly as it is: nothing to put back.
  const std::vector<int> flat = run(true, {kBody, kBody, kBody});
  ASSERT_TRUE(flat.size() == off.size());
  if (flat.size() == off.size()) {
    for (int v : flat) { EXPECT_TRUE(v == kBody); }
  }
}

TEST(minimax_h3_context_stages, declick_fades_the_pcm_cut)
{
  constexpr int kFrames = 124, kSr = 32000, kTrim = 22;
  constexpr std::int64_t kSamples = 207 * 800;
  constexpr double kMs = 12.0;

  Session sess;
  Pipeline pl("declick", &sess);

  // The picture the PCM is conformed to; no frame survives the trim's
  // interest here, only the fps and the frame count do.
  auto frames_u = std::make_unique<ListSource>(&sess, "frames",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  for (int f = 0; f < kFrames; ++f) {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = {3, 2, 2};
    tb.resize_contiguous(12);
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("frame", FlexData::make_int(f));
    sb.as_object().insert_or_assign("frames", FlexData::make_int(kFrames));
    sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
    tb.sideband = std::move(sb);
    frames_u->beats.push_back(tensor_(std::move(tb)));
  }
  frames_u->allocate_oports(1);
  auto* frames = pl.insert_stage(std::move(frames_u));

  auto pcm_u = std::make_unique<ListSource>(&sess, "pcm",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {1, kSamples};
    tb.resize_contiguous((std::size_t)kSamples);
    float* d = tb.as_f32();
    for (std::int64_t i = 0; i < kSamples; ++i) { d[i] = 1.0f; }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("sample_rate", FlexData::make_int(kSr));
    tb.sideband = std::move(sb);
    pcm_u->beats.push_back(tensor_(std::move(tb)));
  }
  pcm_u->allocate_oports(1);
  auto* pcm = pl.insert_stage(std::move(pcm_u));

  auto* trim = pl.insert_stage(std::make_unique<MiniMaxH3ContextTrimStage>(
      &sess, "trim", std::vector<InEdge>{{frames, 0}, {pcm, 0}},
      cfg_({{"trim_frames", FlexData::make_int(kTrim)},
            {"declick_ms", FlexData::make_real(kMs)}})));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{trim, 0}}, FlexData::make_object())));
  auto* ca = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "ca", std::vector<InEdge>{{trim, 1}}, FlexData::make_object())));
  (void)cv;

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  ASSERT_TRUE(ca->got.size() == 1);
  if (ca->got.size() != 1) { return; }
  const auto* a = dynamic_cast<const TensorBeatPayload*>(ca->got[0].get());
  ASSERT_TRUE(a != nullptr);
  if (a == nullptr) { return; }
  const std::int64_t fade = (std::int64_t)(kMs * 0.001 * kSr);   // 384
  ASSERT_TRUE(a->shape.back() > fade);
  if (a->shape.back() <= fade) { return; }
  const float* d = a->as_f32();
  EXPECT_TRUE(d[0] == 0.0f);                       // in from silence
  EXPECT_TRUE(d[fade / 2] > 0.4f && d[fade / 2] < 0.6f);   // raised cosine
  EXPECT_TRUE(d[fade - 1] > 0.999f);               // and back to the signal
  EXPECT_TRUE(d[fade] == 1.0f && d[fade + 1000] == 1.0f);  // nothing beyond
}

// The luma match holds the clip's first frames back until it has seen the
// body it levels them against. Two ways that hold can swallow them: a clip
// with fewer kept frames than the hold, and the run ending between the last
// frame and the close. drain() is what has to let them out.
TEST(minimax_h3_context_stages, a_clip_shorter_than_the_luma_hold_still_arrives)
{
  constexpr int kFrames = 26, kTrim = 22;   // 4 kept, against a hold of 10

  Session sess;
  Pipeline pl("short", &sess);
  auto frames_u = std::make_unique<ListSource>(&sess, "frames",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  for (int f = 0; f < kFrames; ++f) {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = {3, 2, 2};
    tb.resize_contiguous(12);
    for (std::size_t i = 0; i < 12; ++i) { tb.data[i] = (std::uint8_t)(100 + f); }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("frame", FlexData::make_int(f));
    sb.as_object().insert_or_assign("frames", FlexData::make_int(kFrames));
    tb.sideband = std::move(sb);
    frames_u->beats.push_back(tensor_(std::move(tb)));
  }
  frames_u->allocate_oports(1);
  auto* frames = pl.insert_stage(std::move(frames_u));
  auto* trim = pl.insert_stage(std::make_unique<MiniMaxH3ContextTrimStage>(
      &sess, "trim", std::vector<InEdge>{{frames, 0}},
      cfg_({{"trim_frames", FlexData::make_int(kTrim)}})));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{trim, 0}}, FlexData::make_object())));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // Every kept frame comes out, once, in order -- none of them stranded in
  // the hold, which only ever fills at 10.
  EXPECT_TRUE(cv->got.size() == (std::size_t)(kFrames - kTrim));
  if (cv->got.size() != (std::size_t)(kFrames - kTrim)) { return; }
  const auto* first = dynamic_cast<const TensorBeatPayload*>(cv->got[0].get());
  const auto* last =
      dynamic_cast<const TensorBeatPayload*>(cv->got.back().get());
  EXPECT_TRUE(first != nullptr && sb_int_(first->sideband, "frame") == 0);
  EXPECT_TRUE(last != nullptr && sb_int_(last->sideband, "frame") ==
                                     (std::int64_t)(kFrames - kTrim - 1));
}

// The info beat decides how much to cut, and the PCM is one beat that is
// cut once -- so a PCM beat that arrives BEFORE the info beat must still
// wait for it. Getting this wrong trimmed the whole soundtrack by the
// stage's own default, visibly only as a flaky test, because which of the
// two landed first was down to clock-domain scheduling order.
TEST(minimax_h3_context_stages, the_pcm_waits_for_the_info_beat)
{
  constexpr int kFrames = 124, kSr = 32000, kInfoTrim = 22, kCfgTrim = 5;
  constexpr std::int64_t kSamples = 207 * 800;

  Session sess;
  Pipeline pl("late-info", &sess);

  auto frames_u = std::make_unique<ListSource>(&sess, "frames",
                                               std::vector<InEdge>{},
                                               FlexData::make_object());
  for (int f = 0; f < kFrames; ++f) {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::U8;
    tb.shape = {3, 2, 2};
    tb.resize_contiguous(12);
    tb.data[0] = (std::uint8_t)f;
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("frame", FlexData::make_int(f));
    sb.as_object().insert_or_assign("frames", FlexData::make_int(kFrames));
    sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
    tb.sideband = std::move(sb);
    frames_u->beats.push_back(tensor_(std::move(tb)));
  }
  frames_u->allocate_oports(1);
  auto* frames = pl.insert_stage(std::move(frames_u));

  auto pcm_u = std::make_unique<ListSource>(&sess, "pcm",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  {
    TensorBeat tb;
    tb.dtype = TensorBeat::DType::F32;
    tb.shape = {1, kSamples};
    tb.resize_contiguous((std::size_t)kSamples);
    float* d = tb.as_f32();
    for (std::int64_t i = 0; i < kSamples; ++i) { d[i] = (float)i; }
    FlexData sb = FlexData::make_object();
    sb.as_object().insert_or_assign("sample_rate", FlexData::make_int(kSr));
    tb.sideband = std::move(sb);
    pcm_u->beats.push_back(tensor_(std::move(tb)));
  }
  pcm_u->allocate_oports(1);
  auto* pcm = pl.insert_stage(std::move(pcm_u));

  auto info_u = std::make_unique<LateSource>(&sess, "info",
                                             std::vector<InEdge>{},
                                             FlexData::make_object());
  info_u->beat = make_payload<FlexDataPayload>(
      cfg_({{"trim_frames", FlexData::make_int(kInfoTrim)}}));
  info_u->allocate_oports(1);
  auto* info = pl.insert_stage(std::move(info_u));

  auto* trim = pl.insert_stage(std::make_unique<MiniMaxH3ContextTrimStage>(
      &sess, "trim", std::vector<InEdge>{{frames, 0}, {pcm, 0}, {info, 0}},
      cfg_({{"trim_frames", FlexData::make_int(kCfgTrim)},
            {"luma_match", FlexData::make_bool(false)},
            {"declick_ms", FlexData::make_real(0.0)}})));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{trim, 0}}, FlexData::make_object())));
  auto* ca = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "ca", std::vector<InEdge>{{trim, 1}}, FlexData::make_object())));

  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();

  // The importer's 22 wins over the configured 5, in BOTH media.
  EXPECT_TRUE(cv->got.size() == (std::size_t)(kFrames - kInfoTrim));
  ASSERT_TRUE(ca->got.size() == 1);
  if (ca->got.size() != 1) { return; }
  const auto* a = dynamic_cast<const TensorBeatPayload*>(ca->got[0].get());
  ASSERT_TRUE(a != nullptr);
  if (a == nullptr) { return; }
  // 102 frames at 24 fps and 32 kHz; 22 frames = 29333 samples dropped.
  EXPECT_TRUE(a->shape == (std::vector<std::int64_t>{1, 136000}));
  if (a->shape != (std::vector<std::int64_t>{1, 136000})) { return; }
  EXPECT_TRUE(a->as_f32()[0] == 29333.0f);
}

// ---- the chain normaliser ------------------------------------------
namespace {

// Broadband value noise, the way footage is. `fine` scales the top two
// octaves, `mid` the two the 5-to-17 px band sits in, `bias` the level.
// Both gains are needed: the correction under test removes the MIDDLE
// band, so a drift built only from fine octaves is invisible to it --
// measured, a clip with 2.2x the fine detail moves the band by -2 %.
std::unique_ptr<BeatPayloadIntf>
noise_beat_(int h, int w, unsigned seed, double fine, double mid, int bias)
{
  std::vector<double> img((std::size_t)h * w, 128.0 + bias);
  std::uint64_t s = (std::uint64_t)seed * 2654435761u + 1;
  auto rnd = [&]() {
    s ^= s << 13; s ^= s >> 7; s ^= s << 17;
    return (double)((s >> 11) & 0xffff) / 65535.0 - 0.5;
  };
  for (int oct = 0; oct < 6; ++oct) {
    const int step = 1 << (5 - oct);
    double gain = 1.0;
    if (oct >= 4)                  { gain = fine; }
    else if (oct == 1 || oct == 2) { gain = mid; }
    const double amp = 18.0 * gain / (1.0 + oct * 0.5);
    const int gh = h / step + 2, gw = w / step + 2;
    std::vector<double> g((std::size_t)gh * gw);
    for (auto& v : g) { v = rnd() * amp; }
    for (int y = 0; y < h; ++y) {
      for (int x = 0; x < w; ++x) {
        const int gy = y / step, gx = x / step;
        const double fy = (double)(y % step) / step;
        const double fx = (double)(x % step) / step;
        const double a = g[(std::size_t)gy * gw + gx];
        const double b = g[(std::size_t)gy * gw + gx + 1];
        const double c = g[(std::size_t)(gy + 1) * gw + gx];
        const double d = g[(std::size_t)(gy + 1) * gw + gx + 1];
        img[(std::size_t)y * w + x] +=
            a + (b - a) * fx + ((c - a) + ((d - c) - (b - a)) * fx) * fy;
      }
    }
  }
  TensorBeat tb;
  tb.dtype = TensorBeat::DType::U8;
  tb.shape = {3, h, w};
  tb.resize_contiguous((std::size_t)3 * h * w);
  const std::size_t plane = (std::size_t)h * w;
  for (int c = 0; c < 3; ++c) {
    for (std::size_t i = 0; i < plane; ++i) {
      tb.data[(std::size_t)c * plane + i] =
          (std::uint8_t)std::clamp((int)std::llround(img[i]), 0, 255);
    }
  }
  FlexData sb = FlexData::make_object();
  sb.as_object().insert_or_assign("fps", FlexData::make_real(24.0));
  tb.sideband = std::move(sb);
  return tensor_(std::move(tb));
}

}  // namespace

TEST(minimax_h3_context_stages, chain_normalize_levels_a_drifting_take)
{
  constexpr int kH = 72, kW = 96;
  constexpr int kSkip = 2, kBase = 14, kTail = 24;

  auto run = [&](double strength, bool colour) {
    Session sess;
    Pipeline pl("normalize", &sess);
    auto src_u = std::make_unique<ListSource>(&sess, "frames",
                                              std::vector<InEdge>{},
                                              FlexData::make_object());
    for (int i = 0; i < kSkip + kBase; ++i) {
      src_u->beats.push_back(noise_beat_(kH, kW, 100 + i, 1.0, 1.0, 0));
    }
    // The continuation: finer and brighter, which is what a clip that
    // continues another comes out as.
    for (int i = 0; i < kTail; ++i) {
      src_u->beats.push_back(noise_beat_(kH, kW, 300 + i, 2.2, 2.0, 20));
    }
    src_u->allocate_oports(1);
    auto* src = pl.insert_stage(std::move(src_u));
    auto* norm = pl.insert_stage(
        std::make_unique<MiniMaxH3ChainNormalizeStage>(
            &sess, "norm", std::vector<InEdge>{{src, 0}},
            cfg_({{"skip_seconds", FlexData::make_real(kSkip / 24.0)},
                  {"baseline_seconds", FlexData::make_real(kBase / 24.0)},
                  {"fps", FlexData::make_real(24.0)},
                  {"ema", FlexData::make_real(1.0)},
                  {"strength", FlexData::make_real(strength)},
                  {"colour_match", FlexData::make_bool(colour)}})));
    auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
        &sess, "cv", std::vector<InEdge>{{norm, 0}}, FlexData::make_object())));
    PipelineRuntime rt(&pl, &sess);
    // band energy, level -- the band is what the correction removes, so
    // it is what says whether it worked.
    std::vector<std::pair<double, double>> out;
    if (!rt.launch()) { return out; }
    rt.wait_idle();
    rt.stop();
    for (auto& b : cv->got) {
      const auto* t = dynamic_cast<const TensorBeatPayload*>(b.get());
      if (t == nullptr) { out.push_back({-1.0, -1.0}); continue; }
      out.push_back({h3ctx::structure_band_energy(t->data.data(), 3, kH, kW),
                     h3ctx::frame_luma(t->data.data(),
                                       (std::int64_t)t->data.size())});
    }
    return out;
  };

  const auto off  = run(0.0, false);   // measures, touches nothing
  const auto both = run(1.0, true);
  ASSERT_TRUE(off.size() == (std::size_t)(kSkip + kBase + kTail));
  ASSERT_TRUE(both.size() == off.size());
  if (both.size() != off.size() ||
      off.size() < (std::size_t)(kSkip + kBase + 4)) {
    return;
  }
  const std::size_t head = (std::size_t)(kSkip + kBase);
  auto mean = [](const std::vector<std::pair<double, double>>& v,
                 std::size_t a, std::size_t b, bool level) {
    double s = 0.0;
    for (std::size_t i = a; i < b; ++i) { s += level ? v[i].second : v[i].first; }
    return s / (double)(b - a);
  };

  // Uncorrected, the take drifts on both axes.
  EXPECT_TRUE(mean(off, head, off.size(), false) >
              mean(off, kSkip, head, false) * 1.15);
  EXPECT_TRUE(mean(off, head, off.size(), true) >
              mean(off, kSkip, head, true) + 10.0);

  // Corrected, the grade comes back onto the first clip's...
  EXPECT_TRUE(std::abs(mean(both, head, both.size(), true) -
                       mean(both, kSkip, head, true)) < 3.0);
  // ...and the band the correction removes comes down. The reference
  // does not claim to remove the drift entirely -- it measured 1.49x to
  // 1.22x -- so this asks for a clear reduction, not for none.
  EXPECT_TRUE(mean(both, head, both.size(), false) <
              mean(off, head, off.size(), false));

  // WHAT IT DOES NOT DO, recorded because it is surprising and it is the
  // reference's own behaviour, not this port's: the contrast-normalised
  // Laplacian the correction is DRIVEN by does not come down with it.
  // The band removed is mid-scale and the measure divides by whole-frame
  // variance, which that band carries most of, so taking it out raises
  // the number -- by about a quarter at full correction on this content.
  // There is no feedback: each frame's target is computed from the frame
  // as it arrives, never from a corrected one, so nothing runs away.
  // Assert it, so a future change to either half is noticed here.
  {
    auto frame = noise_beat_(kH, kW, 300, 2.2, 2.0, 20);
    auto* t = dynamic_cast<TensorBeatPayload*>(frame.get());
    ASSERT_TRUE(t != nullptr);
    if (t != nullptr) {
      const double lap0 = h3ctx::norm_laplacian(t->data.data(), 3, kH, kW);
      const double band0 =
          h3ctx::structure_band_energy(t->data.data(), 3, kH, kW);
      std::vector<float> scratch;
      h3ctx::subtract_structure_band(t->data.data(), 3, kH, kW, 0.85,
                                     &scratch);
      EXPECT_TRUE(h3ctx::structure_band_energy(t->data.data(), 3, kH, kW) <
                  band0);                       // the band: down
      EXPECT_TRUE(h3ctx::norm_laplacian(t->data.data(), 3, kH, kW) > lap0);
    }                                           // the measure: up
  }
}

TEST(minimax_h3_context_stages, chain_normalize_passes_a_short_take_through)
{
  constexpr int kH = 40, kW = 56;
  Session sess;
  Pipeline pl("short-take", &sess);
  auto src_u = std::make_unique<ListSource>(&sess, "frames",
                                            std::vector<InEdge>{},
                                            FlexData::make_object());
  // Under the reference's floor of 8 frames: nothing is levelled, and
  // every frame comes out as it went in. The window below is set so the
  // colour reference IS picked (frame 4 of a window of 8) before the
  // take runs out -- untouched has to mean the grade too.
  std::vector<std::vector<std::uint8_t>> want;   // plain copies
  for (int i = 0; i < 6; ++i) {
    auto b = noise_beat_(kH, kW, 11 + i, 1.0, 1.0, 0);
    const auto& d = dynamic_cast<const TensorBeatPayload*>(b.get())->data;
    want.emplace_back(d.begin(), d.end());
    src_u->beats.push_back(std::move(b));
  }
  src_u->allocate_oports(1);
  auto* src = pl.insert_stage(std::move(src_u));
  auto* norm = pl.insert_stage(std::make_unique<MiniMaxH3ChainNormalizeStage>(
      &sess, "norm", std::vector<InEdge>{{src, 0}},
      cfg_({{"skip_seconds", FlexData::make_real(0.0)},
            {"baseline_seconds", FlexData::make_real(0.1)},
            {"fps", FlexData::make_real(24.0)}})));
  auto* cv = static_cast<Collect*>(pl.insert_stage(std::make_unique<Collect>(
      &sess, "cv", std::vector<InEdge>{{norm, 0}}, FlexData::make_object())));
  PipelineRuntime rt(&pl, &sess);
  ASSERT_TRUE(rt.launch());
  rt.wait_idle();
  rt.stop();
  EXPECT_TRUE(cv->got.size() == want.size());
  if (cv->got.size() != want.size()) { return; }
  for (std::size_t k = 0; k < want.size(); ++k) {
    const auto* t = dynamic_cast<const TensorBeatPayload*>(cv->got[k].get());
    EXPECT_TRUE(t != nullptr &&
                std::equal(t->data.begin(), t->data.end(),
                           want[k].begin(), want[k].end()));
  }
}
