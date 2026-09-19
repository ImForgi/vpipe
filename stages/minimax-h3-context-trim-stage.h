#ifndef VPIPE_STAGES_MINIMAX_H3_CONTEXT_TRIM_STAGE_H
#define VPIPE_STAGES_MINIMAX_H3_CONTEXT_TRIM_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Drops the regenerated OVERLAP from the head of a decoded MiniMax-H3 clip
// that was generated with a continuation context -- picture and sound by
// the same duration, so the clip can be appended to the previous one.
//
// A clip continued from a 22-frame context re-renders those 22 frames at
// its start (they are the model's reading of the context, not new
// footage). This stage removes them from the per-frame RGB beats of
// `vae-decode` and the matching samples from `audio-vae-decode`'s PCM.
// Trimming only the picture would put the soundtrack trim/fps seconds
// ahead of it.
//
// With `conform_audio` it also cuts or pads the PCM to exactly
// (frames - trim) / fps seconds. H3 rounds its 40 Hz audio grid to the
// nearest latent, so a decoded soundtrack is up to ~8 ms longer or shorter
// than its picture; left alone that error accumulates along a chain.
//
//   iport0  image   per-frame planar u8 RGB from vae-decode oport0
//   iport1  audio   OPTIONAL: PCM [channels, samples] from audio-vae-decode
//   iport2  info    OPTIONAL: minimax-h3-context-import oport2 -- its
//                   trim_frames replaces the config value, per clip
//   oport0  image   the frames after the overlap, renumbered from 0
//   oport1  audio   the PCM after the overlap
//
// It also owns the two seams the cut leaves behind: `luma_match` puts the
// clip's first frames back on the level it settles at (they come off the
// pinned context carrying ITS exposure, which reads as a flash), and
// `declick_ms` fades the PCM in over a few ms instead of starting it
// mid-waveform.
//
// Config:
//   trim_frames       (int, 22)    frames dropped when `info` is not wired
//   conform_audio     (bool, true) fit the PCM to the trimmed picture exactly
//   fps               (real, 24)   frame rate when a beat does not carry one
//   luma_match        (bool, true) level the first frames after the cut
//   luma_match_frames (int, 6)     how many of them (0 = off)
//   declick_ms        (real, 12)   fade at the PCM cut (0 = off)
class MiniMaxH3ContextTrimStage final
  : public TypedStage<MiniMaxH3ContextTrimStage> {
public:
  static constexpr const char* kTypeName = "minimax-h3-context-trim";

  MiniMaxH3ContextTrimStage(const SessionContextIntf* session, std::string id,
                            std::vector<InEdge> iports, FlexData config);

  // Per-launch reset -- see the comment on the definition.
  void reset_run_state() override;

  Job process(RuntimeContext& ctx) override;

  // Frames still held when the run ends -- a clip shorter than the hold,
  // or a stop -- leave here. The runtime calls this after the process
  // loop however it exited, which the tail of process() cannot see: the
  // call that observes EOS on every iport returns early, before it.
  Job drain(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  void apply_info_(const BeatPayloadIntf* beat);
  // Level the held frames and hand them over, in arrival order, leaving
  // nothing behind. Every path that lets frames out goes through it.
  std::vector<std::unique_ptr<BeatPayloadIntf>> release_held_();
  // Level the held frames onto the trend of the ones behind them.
  void luma_match_();

  // How many frames are held back so `luma_match_` can read the level
  // this clip settles at before it corrects its first frames -- the head
  // cannot be judged until the body it belongs to has been seen. 0 when
  // there is nothing to correct.
  int luma_hold_() const;

  // Config.
  int         _trim_cfg      = 0;
  bool        _conform       = true;
  double      _fps_cfg       = 24.0;
  bool        _luma_match    = true;
  int         _luma_frames   = 0;
  double      _declick_ms    = 0.0;
  // Per run: reset_run_state() puts the two back to their config value.
  int          _trim          = 0;
  double       _fps           = 24.0;
  bool         _info_latched  = false;
  bool         _short_said    = false;
  bool         _fps_said      = false;
  bool         _luma_said     = false;
  bool         _luma_done     = false;  // this clip's hold has been released
  std::int64_t _frame_counter = 0;   // for frames without a sideband index
  int          _clip_frames   = 0;   // the picture the PCM is conformed to
  std::vector<std::unique_ptr<BeatPayloadIntf>> _held;   // see luma_hold_()
};

}  // namespace vpipe

#endif
