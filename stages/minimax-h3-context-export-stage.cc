#include "stages/minimax-h3-context-export-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "stages/minimax-h3-context.h"
#include "stages/model-provenance.h"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>

namespace vpipe {

namespace {

constexpr ConfigKey kAttrs[] = {
  {.key = "output_url", .type = ConfigType::String, .required = true,
   .doc = "context file to write (.safetensors). A launch that emits several "
          "clips writes the second and later ones with a -000001, -000002 "
          "suffix before the extension, as save-image does",
   .is_path = true, .path_write = true},
  {.key = "overwrite_existing", .type = ConfigType::Bool,
   .doc = "replace an existing file (default); false keeps it and skips the "
          "clip",
   .def_bool = true},
};
const PortSpec kIports[] = {
  {.name = "latent",
   .doc = "the SAMPLED MiniMax-H3 video latent, f32 [24, T, H/16, W/16] -- "
          "generate-video oport0, fanned out beside vae-decode",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
  {.name = "audio_latent",
   .doc = "OPTIONAL: the sampled audio latent, f32 [2, 32, A] -- "
          "generate-video oport1. Unwired, the context carries no sound",
   .type = &typeid(TensorBeatPayload), .tags = "latent", .clock_group = 0},
};
const StageSpec kSpec = {
  .type_name = "minimax-h3-context-export",
  .doc       = "Sink: saves a MiniMax-H3 clip's sampled video (and audio) "
               "latents to a safetensors context file, whole and without a "
               "VAE round trip, so minimax-h3-context-import can continue the "
               "clip in a later launch.",
  .display_name = "MiniMax-H3 Context Export",
  .category  = StageCategory::Generative,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};

std::string
shape_str_(const std::vector<std::int64_t>& s)
{
  std::string out = "[";
  for (std::size_t i = 0; i < s.size(); ++i) {
    if (i) { out += ", "; }
    out += std::to_string(s[i]);
  }
  return out + "]";
}

}  // namespace

MiniMaxH3ContextExportStage::MiniMaxH3ContextExportStage(
    const SessionContextIntf* s, std::string id,
    std::vector<InEdge> iports, FlexData config)
  : TypedStage<MiniMaxH3ContextExportStage>(s, std::move(id), std::move(iports),
                                            std::move(config))
{
  _output_url = attr_path("output_url", /*for_write=*/true);
  _overwrite  = attr_bool("overwrite_existing");
  if (_output_url.empty()) {
    fail_config(fmt("MiniMaxH3ContextExportStage('{}'): config.output_url is "
                    "required", this->id()));
  }
}

const StageSpec&
MiniMaxH3ContextExportStage::spec() const noexcept
{
  return kSpec;
}

std::string
MiniMaxH3ContextExportStage::path_for_clip(const std::string& base,
                                           std::uint64_t index)
{
  // save-image's rule, so a chain names its files the way the rest of
  // the tree does: clip 0 writes `base` verbatim, later clips get a
  // zero-padded suffix before the extension.
  if (index == 0) { return base; }
  char suf[16];
  std::snprintf(suf, sizeof suf, "-%06u", (unsigned)index);
  const std::filesystem::path p(base);
  std::filesystem::path out =
      p.parent_path() / (p.stem().string() + suf + p.extension().string());
  return out.string();
}

void
MiniMaxH3ContextExportStage::reset_run_state()
{
  // Per-launch reset. `_clips` is the filename index: clip 0 writes
  // `output_url` verbatim and later clips get a suffix, so that a run
  // emitting several clips does not clobber itself. Across runs that is
  // wrong -- the stage survives a stop/relaunch -- and the next
  // pipeline in a chain reads the configured name, so without this a
  // second launch would leave it holding the FIRST launch's clip while
  // writing `name-000001.safetensors` nobody reads.
  _clips = 0;
}

Job
MiniMaxH3ContextExportStage::process(RuntimeContext& ctx)
{
  auto vb = co_await ctx.read(0);
  if (!vb) {
    ctx.signal_done();
    co_return;
  }
  // The audio half of the SAME generation: generate-video writes oport0
  // then oport1 per clip, so a read here pairs them.
  std::unique_ptr<BeatPayloadIntf> ab;
  if (ctx.num_iports() > 1 && ctx.iport_connected(1)) {
    ab = co_await ctx.read(1);
  }
  if (_output_url.empty()) { co_return; }   // config already failed

  const auto* v = dynamic_cast<const TensorBeatPayload*>(vb.get());
  if (v == nullptr || v->dtype != TensorBeat::DType::F32 ||
      v->shape.size() != 4 || v->shape[0] != h3ctx::kVideoChannels ||
      !v->is_contiguous()) {
    session()->warn(fmt(
        "MiniMaxH3ContextExportStage('{}'): expected a MiniMax-H3 video "
        "latent, f32 [{}, T, h, w], got {}; not saved", this->id(),
        h3ctx::kVideoChannels, vb->describe()));
    co_return;
  }
  const int frames = h3ctx::frames_for_latents((int)v->shape[1]);
  if (frames <= 0) {
    session()->warn(fmt(
        "MiniMaxH3ContextExportStage('{}'): a {}-frame latent is not an H3 "
        "clip (5n + 2 latent frames); not saved", this->id(), v->shape[1]));
    co_return;
  }

  // The beats are READ where they are: a 10 s 720p clip is ~100 MB per
  // latent, and this stage runs while the VAE decode of the same clip is
  // asking for its own working set.
  h3ctx::ContextFileView file;
  file.video       = v->as_f32();
  file.video_shape = v->shape;

  const auto* a =
      ab ? dynamic_cast<const TensorBeatPayload*>(ab.get()) : nullptr;
  if (a != nullptr) {
    if (a->dtype == TensorBeat::DType::F32 && a->shape.size() == 3 &&
        a->shape[0] == h3ctx::kStereo &&
        a->shape[1] == h3ctx::kAudioLatentChannels && a->shape[2] > 0 &&
        a->is_contiguous()) {
      file.audio       = a->as_f32();
      file.audio_shape = a->shape;
    } else {
      session()->warn(fmt(
          "MiniMaxH3ContextExportStage('{}'): the audio latent is {}, not f32 "
          "[{}, {}, A]; saving the video only", this->id(), ab->describe(),
          h3ctx::kStereo, h3ctx::kAudioLatentChannels));
    }
  }

  double fps = h3ctx::kFps;
  if (v->sideband.is_object()) {
    FlexData sb = v->sideband;              // as_object() is a view: keep it
    const auto o = sb.as_object();
    if (o.contains("fps") && o.at("fps").as_real(0.0) > 0.0) {
      fps = o.at("fps").as_real(fps);
    }
    if (o.contains("context_frames")) {
      file.metadata["continued_with_context_frames"] =
          std::to_string(o.at("context_frames").as_int(0));
    }
  }
  file.metadata["format"]         = h3ctx::kFormat;
  file.metadata["frames"]         = std::to_string(frames);
  file.metadata["fps"]            = fmt("{}", fps)();
  file.metadata["width"]          = std::to_string(v->shape[3] * 16);
  file.metadata["height"]         = std::to_string(v->shape[2] * 16);
  file.metadata["audio_latents"]  =
      std::to_string(file.audio == nullptr ? 0 : file.audio_shape[2]);
  const std::string model = provenance::model_name(v->sideband);
  if (!model.empty()) { file.metadata["model"] = model; }

  const std::string path = path_for_clip(_output_url, _clips++);
  std::error_code ec;
  if (!_overwrite && std::filesystem::exists(path, ec)) {
    session()->warn(fmt(
        "MiniMaxH3ContextExportStage('{}'): '{}' exists and overwrite is "
        "false; clip not saved", this->id(), path));
    co_return;
  }
  std::string err;
  if (!h3ctx::write_context_file(path, file, &err)) {
    session()->warn(fmt("MiniMaxH3ContextExportStage('{}'): {}; clip not saved",
                        this->id(), err));
    co_return;
  }
  session()->info(fmt(
      "MiniMaxH3ContextExportStage('{}'): saved {} frames -- video {}{} -- to "
      "'{}'",
      this->id(), frames, shape_str_(file.video_shape),
      file.audio == nullptr ? std::string(", no audio")
                            : " + audio " + shape_str_(file.audio_shape),
      path));
}

VPIPE_REGISTER_STAGE(MiniMaxH3ContextExportStage)
VPIPE_REGISTER_SPEC(MiniMaxH3ContextExportStage, kSpec)

}  // namespace vpipe
