#ifndef GENERATIVE_MODELS_FLASHVSR_FLASHVSR_FAMILY_H
#define GENERATIVE_MODELS_FLASHVSR_FLASHVSR_FAMILY_H

#include "generative-models/video-model-registry.h"

#include <string>

namespace vpipe {
namespace genai {

// FlashVSR as a `generate-video` family, registered rather than added as
// a fourth branch of that stage's built-in dispatch.
//
// WHY REGISTERED, when it is in-tree and the built-in families are not.
// The registry is the newer path and the stage consults it first, so
// this costs the built-ins nothing. What decided it is the SOURCE CLIP:
// a family reached through the registry takes its extra input through
// the request's named-input lookup, which is the seam that exists for
// exactly this, where a built-in branch would invite one more typed
// field on a struct whose comment already lists six such fields as the
// reason not to add a seventh.
//
// THE NAME IT ASKS FOR is `source_video`, and it is the first name in
// the set -- see gen-input.h, which says the set is empty and that
// names are added, never renamed. It resolves to the bicubic-upscaled
// low-quality clip as planar u8 RGB [frames, 3, H, W], which is the
// shape temporal-stack builds and the reference ports already take. A
// request that does not carry it is REFUSED rather than run: this model
// conditions on nothing else, so without it there is noise and a
// constant prompt, and what comes out is a confident 4x upscale of
// nothing.
//
// WHAT IT IGNORES, and why none of it is an error. The conditioning
// port carries a text encoding this family never reads -- its own
// context is a constant that ships with the checkpoint -- so a graph
// still has to wire something there because the stage's first port
// drives the beat, and the cheapest honest thing to wire is a
// text-prompt source with an empty string. The negative port, the
// sampler and the scheduler are all inert: one step at a fixed
// timestep, guidance distilled away, no schedule to select.
//
// GEOMETRY IS NOT NEGOTIABLE HERE the way it is elsewhere. Output width
// and height must be multiples of 128 because the window partition
// needs the token grid to divide by 8, and frames must be 8k+1 because
// the denoise consumes 8 source frames per chunk. Both are reported and
// rounded up, per the contract's rule that a family adjusts rather than
// refuses so a graph can change families without being re-authored.
//
// THE SOURCE SETS THE SIZE. The output is 4x the source clip, so a
// graph that resamples its input has already chosen the output
// resolution; the stage's own width and height are read as a cap when
// they are set and ignored when they are not.
// WHERE FLASHVSR'S PARTS LIVE under one checkpoint root, in either of the
// two layouts it arrives in. Every consumer -- the denoiser's load, the
// source encoder stage, claims(), the memory declarations -- goes through
// this, so the two layouts are told apart in exactly one place.
//
//   CONVERTED (tools/flashvsr_prepare_checkpoint.py): one index names the
//     denoiser, `lq_proj.*` and `posi_context`; the VAE is `vae/`.
//   PUBLISHED (the model page, as the catalogue fetches it): the denoiser
//     safetensors, `LQ_proj_in.ckpt` and `posi_prompt.pth` as torch
//     files, and `Wan2.1_VAE.pth` bare. Read in place -- torch files map
//     as shards (shared/torch-zip.h) -- so nothing is converted or copied.
//
// Converted wins when both are present, which is what a directory the
// converter ran over looks like: its index then names the same bytes.
struct FlashVsrLayout {
  std::string denoiser;        // what the denoiser's WeightSet opens
  std::string source;          // ...the source projection's
  std::string source_prefix;   // tensor-name prefix inside `source`
  std::string context;         // ...the fixed context's
  std::string context_name;    // its tensor name
  std::string vae;             // the VAE, or empty when absent
  bool        published = false;
};

bool resolve_flashvsr_layout(const std::string& root, FlashVsrLayout* out);

class FlashVsrVideoFamily final : public VideoModelFamily {
 public:
  std::string_view tag() const noexcept override { return "flashvsr"; }

  // Claims a directory holding the four files the checkpoint ships. The
  // check has to be narrow: the denoiser alone is Wan 2.1-T2V-1.3B and
  // would claim a plain Wan checkpoint, which would then be driven with
  // a source projection it does not have. So the projection is what is
  // looked for, not the denoiser.
  bool claims(const std::string& root,
              const std::string& model_type) const override;

  int  align_frames(const std::string& root, int frames) const override;
  void size_grid(const std::string& root, int* gh, int* gw) const override;

  std::vector<ResourceClaim>
  declare_resources(const std::string& root) const override;
  std::vector<StageHolding>
  declare_holdings(const std::string& root) const override;
  std::size_t latent_bytes(const std::string& root, int width, int height,
                           int frames) const override;
  // The key/value window and the activation scratch, which at full HD
  // are several times the checkpoint. The window follows `kv_ratio` from
  // the flashvsr-model-config beat when one is wired.
  std::size_t denoise_scratch_bytes(const std::string& root, int width,
                                    int height, int frames,
                                    const FlexData* model_config)
      const override;

  std::unique_ptr<VideoGenerator>
  load(const VideoModelCreateArgs& args) override;
};

// Register the family with the process-wide VideoModelRegistry. Called
// once from the stage library's registration point, the way the other
// in-tree registries are seeded.
void register_flashvsr_video_family();

}  // namespace genai
}  // namespace vpipe

#endif
