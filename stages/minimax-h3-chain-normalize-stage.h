#ifndef VPIPE_STAGES_MINIMAX_H3_CHAIN_NORMALIZE_STAGE_H
#define VPIPE_STAGES_MINIMAX_H3_CHAIN_NORMALIZE_STAGE_H

#include "common/beat-payload-intf.h"
#include "common/job.h"
#include "pipeline/runtime-context.h"
#include "pipeline/typed-stage.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vpipe {

// Levels a JOINED MiniMax-H3 chain: the texture that ratchets up clip by
// clip, and the grade that wanders with it.
//
// Every clip after the first is generated from the previous clip's
// output, and that feedback loop accretes. ComfyUI-H3-Multishot measures
// it at "about +13 % fine texture per join" at 736x1280, calls it the
// TEXTURE RATCHET, and documents that under about four windows it is
// slight while at seven it is visible sharpening. Its own in-loop
// controls "damp it but never fully remove it", so it ships a node that
// levels the finished take instead: H3ChainNormalize.
//
// THIS IS THAT NODE, ported as it stands. The measure (a contrast-
// normalised Laplacian), the band it corrects (5 to 17 px, the
// difference of two box blurs), the deadband, the smoothing and the
// constants are its choices, arrived at against far more footage than we
// have here, and they are kept rather than second-guessed. Its own notes
// on why each is what it is:
//
//   * the correction goes to the STRUCTURE band only, so grain and
//     sensor noise pass through -- a whole-frame blur levels the drift
//     just as well but costs about 20 % of the fine band, the
//     "camcorder texture" worth keeping;
//   * the baseline is the MEDIAN of the first clip, taken after its
//     opening exposure fade settles, with a deadband, so the first clip
//     is never softened against its own average;
//   * every frame is histogram-matched to one reference frame from the
//     first clip, which is what pulls colour and exposure drift back --
//     "the half of the problem no sharpening control ever touched";
//   * the correction is smoothed so it eases in rather than stepping at
//     a window boundary.
//
// Measured there on an 84 s lamp-lit gauge: drift 1.49x down to 1.22x
// with the fine band's grain at or above the original.
//
// WHY NOT IN THE TRIM. `minimax-h3-context-trim` corrects the luma step
// AT a join, from inside one clip, while it is generated. Both
// corrections here are properties of the TAKE: they are measured against
// what the FIRST clip looked like, which a stage that sees one clip at a
// time cannot know.
//
//   iport0  image   per-frame planar u8 RGB [3, H, W], the joined chain
//   oport0  image   the same frames, levelled
//
// The reference sees the whole take at once and corrects every frame
// from the first, including the ones it measured the baseline on. To
// deliver the same frames, this holds the picture until the baseline
// window closes -- `skip_seconds + baseline_seconds` of it -- then emits
// those in order and streams the rest. So the memory is the baseline
// window, not the take -- at the defaults, 12 s at 24 fps, which is 288
// frames and about 780 MB at 1280x704. Shorten the window on a machine
// that cannot spare it; the correction only needs enough of clip 1 to
// take a stable median.
//
// Config (the reference's own inputs, same names and defaults):
//   skip_seconds      (real, 2)     opening fade, excluded from the baseline
//   baseline_seconds  (real, 10)    how much of clip 1 sets the house level
//   fps               (real, 24)    when a beat's sideband carries none
//   strength          (real, 1)     how hard excess structure is pulled back
//   deadband          (real, 1.06)  within this ratio of the baseline, leave alone
//   ema               (real, 0.10)  smoothing, so nothing steps at a join
//   colour_match      (bool, true)  histogram-match every frame to clip 1
class MiniMaxH3ChainNormalizeStage final
  : public TypedStage<MiniMaxH3ChainNormalizeStage> {
public:
  static constexpr const char* kTypeName = "minimax-h3-chain-normalize";

  MiniMaxH3ChainNormalizeStage(const SessionContextIntf* session,
                               std::string id,
                               std::vector<InEdge> iports, FlexData config);

  // Per-launch reset -- see the comment on the definition.
  void reset_run_state() override;

  Job process(RuntimeContext& ctx) override;

  // Whatever is still held when the run ends -- a take shorter than the
  // baseline window, or a stop part-way through one.
  Job drain(RuntimeContext& ctx) override;

  const StageSpec& spec() const noexcept override;

private:
  // Close the baseline window: pick the median and the colour reference.
  void settle_();
  // The reference's per-frame body: match, measure, smooth, subtract.
  void level_(std::uint8_t* px, int channels, int h, int w);
  // Level everything held and hand it over, in arrival order.
  std::vector<std::unique_ptr<BeatPayloadIntf>> release_held_();

  // Config.
  double _skip_s   = 2.0;
  double _base_s   = 10.0;
  double _fps_cfg  = 24.0;
  double _strength = 1.0;
  double _deadband = 1.06;
  double _ema      = 0.10;
  bool   _colour   = true;

  // Per run.
  double       _fps      = 24.0;
  std::int64_t _seen     = 0;      // frames in, this run
  bool         _settled  = false;  // the baseline window has closed
  double       _baseline = 0.0;
  double       _sigma    = 0.0;    // the smoothed correction
  double       _peak     = 0.0;
  bool         _said     = false;
  bool         _odd_said = false;
  bool         _have_ref = false;
  double       _ref_cdf[3][256] = {};
  int          _ref_channels    = 0;
  std::vector<double> _baseline_samples;
  std::vector<float>  _scratch;
  std::vector<std::unique_ptr<BeatPayloadIntf>> _held;
};

}  // namespace vpipe

#endif
