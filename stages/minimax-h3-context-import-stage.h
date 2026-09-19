#ifndef VPIPE_STAGES_MINIMAX_H3_CONTEXT_IMPORT_STAGE_H
#define VPIPE_STAGES_MINIMAX_H3_CONTEXT_IMPORT_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"
#include "stages/minimax-h3-context.h"

#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Source: turns a previous MiniMax-H3 clip into a CONTINUATION CONTEXT for
// `generate-video` (iports 11/12), so the next clip continues its motion
// and its sound instead of starting over.
//
// It carries the last `context_frames` frames of the clip's SAMPLED video
// latent and the matching tail of its audio latent. `generate-video` pins
// them as clean rows on the new clip's own timeline (over its first
// frames), which the model reads as "the clip so far". The new clip
// regenerates those overlap frames; `minimax-h3-context-trim` drops them after
// decoding (wire this stage's `info` oport to it and it knows how many).
//
// Two sources:
//   * a context FILE written by `minimax-h3-context-export` (config
//     input_url) -- one context, for chaining across launches;
//   * the latents of a previous `generate-video` IN THE SAME GRAPH, on
//     iport0/1 -- one context per clip, no file.
//
//   iport0  latent        OPTIONAL: a previous clip's video latent
//   iport1  audio_latent  OPTIONAL: its audio latent
//   oport0  h3_context        f32 [24, n, H/16, W/16] -> generate-video iport11
//   oport1  h3_context_audio  f32 [2, 32, a]          -> generate-video iport12
//   oport2  info              FlexData {trim_frames, context_frames, ...}
//                             -> minimax-h3-context-trim iport2
//
// Config:
//   input_url            (path)        the context file; required unless
//                                      iport0 is wired
//   context_frames       (int, 22)     video frames carried: 17k + 5
//                                      (5, 22, 39, 56, ...)
//   audio_context_frames (int, 0)      audio frames carried; 0 = same as
//                                      video, -1 = no audio
//   start_frame          (int, 0)      where the context sits on the new
//                                      clip; 0 = continuation (anything
//                                      else is an interior guide, no trim)
//   duration_mode        (string)      clip | new_footage | new_footage_min
//                                      -- whether generate-video's `frames`
//                                      is the clip length or the footage
//                                      kept after the trim
//   audio_align          (string)      exact | round -- carried audio ends
//                                      exactly where the picture ends, or
//                                      snapped to the generated audio grid
class MiniMaxH3ContextImportStage final
  : public TypedStage<MiniMaxH3ContextImportStage> {
public:
  static constexpr const char* kTypeName = "minimax-h3-context-import";

  MiniMaxH3ContextImportStage(const SessionContextIntf* session, std::string id,
                              std::vector<InEdge> iports, FlexData config);

  // A stopped pipeline keeps its stages: re-arm the file import so the
  // next launch emits the context again.
  void reset_run_state() override;

  Job process(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // The three beats of one context.
  struct Beats {
    std::unique_ptr<BeatPayloadIntf> video;
    std::unique_ptr<BeatPayloadIntf> audio;
    std::unique_ptr<BeatPayloadIntf> info;
  };
  // What to carry out of a clip of these shapes (an empty `audio_shape`
  // = no soundtrack). False after warning; `source` names the clip in
  // messages.
  bool plan_(const std::vector<std::int64_t>& video_shape,
             const std::vector<std::int64_t>& audio_shape,
             const std::string& source, h3ctx::TailPlan* plan) const;
  // The three beats of a context whose tails are already cut.
  bool emit_(const h3ctx::TailPlan& plan, const float* video_tail,
             const std::vector<std::int64_t>& video_shape,
             const float* audio_tail,
             const std::vector<std::int64_t>& audio_shape,
             const std::string& source, Beats* out) const;
  // plan_ + emit_ over a WHOLE clip held in memory, which is what the
  // in-graph mode has (`audio` may be null). The file mode reads only
  // the tail instead and calls emit_ directly.
  bool build_(const float* video, const std::vector<std::int64_t>& video_shape,
              const float* audio, const std::vector<std::int64_t>& audio_shape,
              const std::string& source, Beats* out) const;

  std::string          _input_url;
  int                  _context_frames       = 0;
  int                  _audio_context_frames = 0;
  int                  _start_frame          = 0;
  h3ctx::DurationMode  _duration             = h3ctx::DurationMode::kClip;
  h3ctx::AudioAlign    _audio_align          = h3ctx::AudioAlign::kExact;
  bool                 _file_done            = false;
};

}  // namespace vpipe

#endif
