#ifndef VPIPE_STAGES_MINIMAX_H3_CONTEXT_H
#define VPIPE_STAGES_MINIMAX_H3_CONTEXT_H

// MiniMax-H3 CONTINUATION CONTEXT: the arithmetic and the file format
// shared by `minimax-h3-context-export`, `minimax-h3-context-import`,
// `minimax-h3-context-trim` and the context ports of `generate-video`.
//
// The idea (the one ComfyUI's H3 Motion Context / Extender nodes use):
// carry the LAST few frames of a generated clip -- its sampled latents,
// never a decode/re-encode -- into the next clip as clean conditioning
// rows pinned on the new clip's OWN timeline, over its first frames. The
// model then reads them as "the clip so far" and continues their motion
// and their sound, where a reference would only be imitated. The new
// clip regenerates those overlap frames, which `minimax-h3-context-trim` drops.
//
// Everything here is plain integer / double arithmetic and byte I/O --
// no GPU, no weights -- so it builds on every platform and is tested on
// its own. The rotary placement itself lives in the H3 layout
// (genai::minimax_h3::ContextGuide); this file only decides WHAT to
// carry and WHERE it ends.
//
// Three facts about H3 every function below leans on:
//
//   * A clip is 17n + 5 pixel frames and packs into 5n + 2 latents; the
//     latents cover 1, 4, 4, 4, 4 frames per group of five. A TAIL of
//     17k + 5 frames (5, 22, 39, ...) is therefore 5k + 2 latents and
//     always starts at cycle phase 0, which is what makes it a valid
//     clip on its own and lets its coordinates coincide with a target's
//     first latents.
//   * Audio is 40 latents/s against 24 fps, so one video frame is 5/3 of
//     an audio latent -- and a clip's audio latent count is ROUNDED to
//     the nearest integer (124 frames want 206.67 and get 207). The
//     signed remainder, the "overhang", is what end-aligns carried audio
//     with carried picture.
//   * The trained clip length tops out at 362 frames (15.08 s).

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace vpipe {
namespace h3ctx {

constexpr int    kFramesPerChunk        = 17;
constexpr int    kLatentsPerChunk       = 5;
constexpr int    kMinClipFrames         = 5;
constexpr int    kMaxTrainedFrames      = 362;
constexpr double kFps                   = 24.0;
constexpr int    kAudioLatentsPerSecond = 40;
constexpr double kFrameRescale          = 5.0 / 3.0;  // audio latents per frame
constexpr int    kVideoChannels         = 24;         // H3 video latent z
constexpr int    kAudioLatentChannels   = 32;         // H3 audio latent width
constexpr int    kStereo                = 2;
constexpr const char* kFormat           = "minimax-h3-context/1";

// ---- the frame grid ----------------------------------------------------

// True for a length H3 can generate or carry: 17n + 5, n >= 0.
bool is_clip_frames(int frames);
// Latents of a 17n + 5 clip (5n + 2); 0 when `frames` is not one.
int latents_for_frames(int frames);
// Pixel frames covered by `latents` counted from cycle phase 0; 0 unless
// `latents` is 5n + 2 (the only counts a clip or a phase-0 tail has).
int frames_for_latents(int latents);
// The nearest 17n + 5 at or above / to `frames` (ties go up).
int align_frames_up(int frames);
int align_frames_nearest(int frames);
// Audio latents H3 gives a clip of `frames` frames (rounded, like the
// generator).
int audio_latents_for_frames(int frames, double fps = kFps);

// ---- how long to generate ---------------------------------------------

enum class DurationMode {
  kClip,           // `frames` is the generated clip; the overlap eats into it
  kNewFootage,     // `frames` is the footage KEPT; overlap added, nearest grid
  kNewFootageMin,  // same, rounded up so the kept footage is never shorter
};
bool parse_duration_mode(const std::string& s, DurationMode* out);
const char* duration_mode_name(DurationMode m);

struct DurationPlan {
  int  generated_frames = 0;  // 17n + 5, what the DiT is asked for
  int  overlap_frames   = 0;  // regenerated context frames the trim drops
  int  new_frames       = 0;  // generated_frames - overlap_frames
  bool capped           = false;  // clamped to kMaxTrainedFrames
};
// Resolve the generated length for a request of `frames` with a context
// of `context_frames` placed at `start_frame`. The overlap is the context
// only when it sits at the head (start_frame 0) -- an interior guide
// replaces nothing the trim should drop. False (with `err`) when the
// context does not leave at least one generated frame after it.
bool resolve_duration(int frames, int context_frames, int start_frame,
                      DurationMode mode, DurationPlan* out, std::string* err);

// ---- what to carry -----------------------------------------------------

enum class AudioAlign {
  kExact,  // end exactly where the carried picture ends (fractional offset)
  kRound,  // snap the offset to the generated audio's integer latent grid
};
bool parse_audio_align(const std::string& s, AudioAlign* out);

struct TailPlan {
  int    source_frames     = 0;  // pixel frames of the source clip
  int    context_frames    = 0;  // carried video frames (17k + 5)
  int    video_latents     = 0;  // carried video latents (5k + 2)
  int    audio_latents     = 0;  // carried audio latents per channel (0 = none)
  double audio_overhang    = 0.0;// source audio latents - 5/3 * source frames
  double audio_offset      = 0.0;// rotary time of carried audio latent 0, from origin
  int    trim_frames       = 0;  // what minimax-h3-context-trim should drop
  bool   overhang_ignored  = false;  // the source grid was not H3's; offset assumes 0
};
// Plan a tail of `context_frames` video frames and `audio_context_frames`
// audio frames (0 = same as video, negative = no audio) out of a source
// clip of `source_latents` video latents and `source_audio_latents` audio
// latents per channel (0 = the source has no soundtrack), placed at
// `start_frame` of the new clip.
bool plan_tail(int source_latents, int source_audio_latents,
               int context_frames, int audio_context_frames, int start_frame,
               AudioAlign align, TailPlan* out, std::string* err);

// Copy the last `latents` frames of a [z, T, h, w] latent (row-major f32)
// into `out` as [z, latents, h, w].
bool slice_video_tail(const float* src, const std::vector<std::int64_t>& shape,
                      int latents, std::vector<float>* out,
                      std::vector<std::int64_t>* out_shape);
// Copy the last `latents` time steps of a [stereo, channels, A] latent.
bool slice_audio_tail(const float* src, const std::vector<std::int64_t>& shape,
                      int latents, std::vector<float>* out,
                      std::vector<std::int64_t>* out_shape);

// ---- trimming the decoded clip ----------------------------------------

struct PcmTrim {
  std::int64_t drop = 0;  // leading samples removed
  std::int64_t keep = 0;  // samples in the result (may exceed what remains:
                          // the tail is then padded with silence)
};
// The PCM counterpart of dropping `trim_frames` video frames. `video_frames`
// is the decoded clip's frame count when known (0 = derive it from the
// sample count). With `conform`, the result is exactly
// (video_frames - trim_frames) / fps seconds, absorbing H3's ±1/3-latent
// audio rounding so it cannot accumulate along a chain.
PcmTrim plan_pcm_trim(std::int64_t samples, int sample_rate, double fps,
                      int trim_frames, int video_frames, bool conform);

// ---- the luma step at the join -----------------------------------------
//
// The first frames delivered after a pinned context carry the CONTEXT's
// exposure rather than the one the model settles on, which on a hard cut
// reads as a flash. The correction puts them back on the level THIS clip
// settles at -- never on the previous clip's, which would be a second
// correction of the same error and would show as a ramp over the frames
// after it.
//
// The level is read over the frames just past the unstable head and next
// to the cut, which is where continuity lives; reading the middle of the
// clip instead measures the shot's content as if it were an error.
//
// ComfyUI's H3Studio hit the same artifact in its own chains and its
// "luma-match join" is the same correction, in its reel export; the window
// and the target here follow what it arrived at.

// The body window. It starts past the worst of the head and stops well
// short of the middle of the clip: reading the middle measures the shot's
// own content as if it were an error. It OVERLAPS the frames being
// corrected, which is deliberate -- the levels are all measured before
// any gain is applied, and a frame that is already on the trend
// contributes the trend. Push `luma_match_frames` past kLumaBodyEnd and
// the target is fitted entirely on frames the correction then rewrites,
// so the default keeps the window the shorter of the two.
constexpr int kLumaBodyBegin = 2;
constexpr int kLumaBodyEnd   = 10;

// The mean sample value of one decoded frame, 0..255.
double frame_luma(const std::uint8_t* rgb, std::int64_t samples);

// A least-squares line through the levels of frames [a0, a1), in frame
// units. A line and not an average: a shot whose exposure genuinely moves
// -- going into shade, say -- would be pulled off its own trend by a flat
// target, turning a smooth decline into a dip and a bump, which is itself
// what an eye reads as a flash. On a steady shot the fit IS the average.
struct LumaTrend {
  double slope     = 0.0;
  double intercept = 0.0;
  bool   ok        = false;
  double at(int frame) const { return slope * (double)frame + intercept; }
};
LumaTrend luma_trend(const double* levels, int count, int a0, int a1);

// The gain putting frame `k` back on `body`, released linearly over
// `frames`. A frame already on the settled level gets exactly 1.0 whatever
// the window, so the window only bounds how far in the correction reaches
// and can never drag a frame that was right.
double luma_gain(const LumaTrend& body, double level, int k, int frames);

// Multiply one frame by `gain`, saturating at 0 and 255.
void apply_luma_gain(std::uint8_t* rgb, std::int64_t samples, double gain);

// ---- the context file --------------------------------------------------
//
// One safetensors file:
//   "video"  F32 [24, T, H/16, W/16]  the sampled (whitened) video latent
//   "audio"  F32 [2, 32, A]           the sampled audio latent (optional)
//   __metadata__                      string -> string, `format` =
//                                     minimax-h3-context/1
// The WHOLE clip is stored, not a tail, so the context length is chosen
// when importing and the clip can still be re-decoded later.

struct ContextFile {
  std::vector<float>                 video;
  std::vector<std::int64_t>          video_shape;
  std::vector<float>                 audio;        // empty = no soundtrack
  std::vector<std::int64_t>          audio_shape;
  std::map<std::string, std::string> metadata;
};

// The same clip WITHOUT owning it: a whole H3 latent is ~100 MB at 720p
// and the exporter already holds one in the beat it was handed, so the
// writer reads it where it is rather than taking a second copy at the
// exact moment the VAE decode needs the room.
struct ContextFileView {
  const float*                       video = nullptr;
  std::vector<std::int64_t>          video_shape;
  const float*                       audio = nullptr;   // null = no sound
  std::vector<std::int64_t>          audio_shape;
  std::map<std::string, std::string> metadata;
};

// What the header says, without reading a byte of the tensors: enough to
// plan a tail and then read only that tail.
struct ContextHeader {
  std::vector<std::int64_t>          video_shape;
  std::vector<std::int64_t>          audio_shape;   // empty = no soundtrack
  std::map<std::string, std::string> metadata;
};

bool write_context_file(const std::string& path, const ContextFileView& file,
                        std::string* err);
bool write_context_file(const std::string& path, const ContextFile& file,
                        std::string* err);
bool read_context_file(const std::string& path, ContextFile* file,
                       std::string* err);
// The header alone.
bool read_context_header(const std::string& path, ContextHeader* out,
                         std::string* err);
// Only the LAST `video_latents` video latents and `audio_latents` audio
// latents (0 = none of that modality), as `slice_*_tail` would have cut
// them from the whole clip. A chain imports a fraction of a second out of
// a file that holds minutes, so this reads that fraction: the tail of each
// channel is contiguous, one seek per channel.
bool read_context_tail(const std::string& path, int video_latents,
                       int audio_latents, ContextFile* out, std::string* err);

}  // namespace h3ctx
}  // namespace vpipe

#endif
