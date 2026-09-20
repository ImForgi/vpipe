#include "stages/minimax-h3-context.h"

#include "apple-silicon/tensor-beat.h"
#include "common/beat-payload-intf.h"
#include "common/flex-data.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace vpipe {
namespace h3ctx {

namespace {

constexpr int kFramesPerLatent[5] = {1, 4, 4, 4, 4};

void
set_err_(std::string* err, std::string msg)
{
  if (err != nullptr) { *err = std::move(msg); }
}

// -1 on anything that is not a usable element count. Every dimension is
// a COUNT, so zero or negative is malformed, and the running product is
// checked before it is taken: a header is file content, and a shape like
// [3, 6148914691236517206, 1, 1] would otherwise overflow to a small
// positive number that matches the byte range while the shape itself
// does not describe the buffer -- which the tail slicing then walks off.
std::int64_t
product_(const std::vector<std::int64_t>& shape)
{
  std::int64_t n = 1;
  for (std::int64_t d : shape) {
    if (d <= 0) { return -1; }
    if (n > std::numeric_limits<std::int64_t>::max() / d) { return -1; }
    n *= d;
  }
  return n;
}

}  // namespace

// ---- the frame grid ----------------------------------------------------

bool
is_clip_frames(int frames)
{
  return frames >= kMinClipFrames &&
         frames % kFramesPerChunk == kMinClipFrames;
}

int
latents_for_frames(int frames)
{
  if (!is_clip_frames(frames)) { return 0; }
  return (frames - kMinClipFrames) / kFramesPerChunk * kLatentsPerChunk + 2;
}

int
frames_for_latents(int latents)
{
  if (latents < 2 || latents % kLatentsPerChunk != 2) { return 0; }
  int frames = 0;
  for (int i = 0; i < latents; ++i) { frames += kFramesPerLatent[i % 5]; }
  return frames;
}

int
align_frames_up(int frames)
{
  if (frames <= kMinClipFrames) { return kMinClipFrames; }
  const int r = frames % kFramesPerChunk;
  const int add = (kMinClipFrames - r + kFramesPerChunk) % kFramesPerChunk;
  return frames + add;
}

int
align_frames_nearest(int frames)
{
  const int up = align_frames_up(frames);
  const int down = up - kFramesPerChunk;
  if (down >= kMinClipFrames && (frames - down) < (up - frames)) {
    return down;
  }
  return up;
}

int
audio_latents_for_frames(int frames, double fps)
{
  if (frames < 0 || !(fps > 0.0)) { return 0; }
  return (int)std::llround((double)frames / fps *
                           (double)kAudioLatentsPerSecond);
}

// ---- how long to generate ---------------------------------------------

bool
parse_duration_mode(const std::string& s, DurationMode* out)
{
  if (out == nullptr) { return false; }
  if (s.empty() || s == "clip") { *out = DurationMode::kClip; return true; }
  if (s == "new_footage") { *out = DurationMode::kNewFootage; return true; }
  if (s == "new_footage_min") {
    *out = DurationMode::kNewFootageMin;
    return true;
  }
  return false;
}

const char*
duration_mode_name(DurationMode m)
{
  switch (m) {
    case DurationMode::kClip:          return "clip";
    case DurationMode::kNewFootage:    return "new_footage";
    case DurationMode::kNewFootageMin: return "new_footage_min";
  }
  return "clip";
}

bool
resolve_duration(int frames, int context_frames, int start_frame,
                 DurationMode mode, DurationPlan* out, std::string* err)
{
  if (out == nullptr) { return false; }
  if (frames <= 0 || context_frames < 0 || start_frame < 0) {
    set_err_(err, "frames must be positive and the context non-negative");
    return false;
  }
  DurationPlan p;
  // Only a HEAD context is regenerated footage the trim drops; a guide
  // placed inside the clip replaces nothing.
  p.overlap_frames = start_frame == 0 ? context_frames : 0;
  switch (mode) {
    case DurationMode::kClip:
      p.generated_frames = align_frames_up(frames);
      break;
    case DurationMode::kNewFootage:
      p.generated_frames = align_frames_nearest(frames + p.overlap_frames);
      break;
    case DurationMode::kNewFootageMin:
      p.generated_frames = align_frames_up(frames + p.overlap_frames);
      break;
  }
  // Nearest can round a short request down onto the overlap itself; one
  // chunk up is the least that still generates something. Not in `clip`
  // mode, where the configured length is the contract and a clip too
  // short for its context is an error below.
  while (mode != DurationMode::kClip &&
         p.generated_frames <= p.overlap_frames) {
    p.generated_frames += kFramesPerChunk;
  }
  // The footage modes grow the clip on the caller's behalf, so they are
  // also the ones that must not push it past what the model was trained
  // on. `clip` generates exactly what was configured, as without a context.
  if (mode != DurationMode::kClip && p.generated_frames > kMaxTrainedFrames) {
    p.generated_frames = kMaxTrainedFrames;
    p.capped = true;
  }
  if (p.generated_frames <= start_frame + context_frames) {
    set_err_(err, "a " + std::to_string(context_frames) +
                      "-frame context at frame " +
                      std::to_string(start_frame) + " leaves no room in a " +
                      std::to_string(p.generated_frames) + "-frame clip");
    return false;
  }
  p.new_frames = p.generated_frames - p.overlap_frames;
  *out = p;
  return true;
}

// ---- what to carry -----------------------------------------------------

bool
parse_audio_align(const std::string& s, AudioAlign* out)
{
  if (out == nullptr) { return false; }
  if (s.empty() || s == "exact") { *out = AudioAlign::kExact; return true; }
  if (s == "round") { *out = AudioAlign::kRound; return true; }
  return false;
}

bool
plan_tail(int source_latents, int source_audio_latents, int context_frames,
          int audio_context_frames, int start_frame, AudioAlign align,
          TailPlan* out, std::string* err)
{
  if (out == nullptr) { return false; }
  TailPlan p;
  p.source_frames = frames_for_latents(source_latents);
  if (p.source_frames <= 0) {
    set_err_(err, "the source latent has " + std::to_string(source_latents) +
                      " frames, which is not an H3 clip (5n + 2)");
    return false;
  }
  if (!is_clip_frames(context_frames)) {
    set_err_(err, "context_frames " + std::to_string(context_frames) +
                      " is not 17k + 5 (5, 22, 39, 56, ...) -- only those "
                      "tails start on a VAE cycle boundary");
    return false;
  }
  if (context_frames > p.source_frames) {
    set_err_(err, "context_frames " + std::to_string(context_frames) +
                      " is longer than the " +
                      std::to_string(p.source_frames) + "-frame source clip");
    return false;
  }
  if (start_frame < 0) {
    set_err_(err, "start_frame must be non-negative");
    return false;
  }
  p.context_frames = context_frames;
  p.video_latents  = latents_for_frames(context_frames);
  p.trim_frames    = start_frame == 0 ? context_frames : 0;

  if (source_audio_latents > 0 && audio_context_frames >= 0) {
    const int a_frames =
        audio_context_frames == 0 ? context_frames : audio_context_frames;
    int rt = (int)std::llround((double)a_frames / kFps *
                               (double)kAudioLatentsPerSecond);
    rt = std::min(rt, source_audio_latents);
    if (rt < 1) {
      set_err_(err, "the audio context window is empty");
      return false;
    }
    p.audio_latents = rt;
    // H3 rounds the audio grid to the NEAREST latent, so the source's
    // soundtrack ends up to 1/2 latent past or short of its last frame.
    // Carry that phase: the audio tail must end where the picture tail
    // ends, or every join drifts by a fraction of a latent.
    p.audio_overhang = (double)source_audio_latents -
                       kFrameRescale * (double)p.source_frames;
    if (!(std::fabs(p.audio_overhang) < 0.5)) {
      p.audio_overhang   = 0.0;
      p.overhang_ignored = true;
    }
    const double end = kFrameRescale * (double)(start_frame + context_frames) +
                       p.audio_overhang;
    p.audio_offset = end - (double)rt;
    if (align == AudioAlign::kRound) {
      p.audio_offset = std::nearbyint(p.audio_offset);
    }
  }
  *out = p;
  return true;
}

bool
slice_video_tail(const float* src, const std::vector<std::int64_t>& shape,
                 int latents, std::vector<float>* out,
                 std::vector<std::int64_t>* out_shape)
{
  if (src == nullptr || out == nullptr || out_shape == nullptr ||
      shape.size() != 4 || latents <= 0 || shape[1] < latents) {
    return false;
  }
  const std::int64_t z = shape[0], t = shape[1], h = shape[2], w = shape[3];
  const std::int64_t plane = h * w;
  out->assign((std::size_t)(z * latents * plane), 0.0f);
  for (std::int64_t c = 0; c < z; ++c) {
    const float* s = src + (c * t + (t - latents)) * plane;
    float* d = out->data() + c * latents * plane;
    std::memcpy(d, s, (std::size_t)(latents * plane) * sizeof(float));
  }
  *out_shape = {z, (std::int64_t)latents, h, w};
  return true;
}

bool
slice_audio_tail(const float* src, const std::vector<std::int64_t>& shape,
                 int latents, std::vector<float>* out,
                 std::vector<std::int64_t>* out_shape)
{
  if (src == nullptr || out == nullptr || out_shape == nullptr ||
      shape.size() != 3 || latents <= 0 || shape[2] < latents) {
    return false;
  }
  const std::int64_t s0 = shape[0], ch = shape[1], a = shape[2];
  out->assign((std::size_t)(s0 * ch * latents), 0.0f);
  for (std::int64_t i = 0; i < s0 * ch; ++i) {
    std::memcpy(out->data() + i * latents, src + i * a + (a - latents),
                (std::size_t)latents * sizeof(float));
  }
  *out_shape = {s0, ch, (std::int64_t)latents};
  return true;
}

// ---- decoded frames ----------------------------------------------------

namespace {

// A planar frame this tree decodes is RGB; the caps keep the `int`
// arithmetic every measure does on a frame inside its range.
constexpr int kMaxFrameChannels = 4;
constexpr int kMaxFrameElements = 1 << 30;

}  // namespace

std::uint8_t*
frame_pixels(BeatPayloadIntf* beat, int* channels, int* h, int* w,
             std::int64_t* samples)
{
  auto* tb = dynamic_cast<TensorBeatPayload*>(beat);
  if (tb == nullptr || tb->dtype != TensorBeat::DType::U8 ||
      tb->shape.size() != 3 || !tb->is_contiguous()) {
    return nullptr;
  }
  const std::int64_t c = tb->shape[0], hh = tb->shape[1], ww = tb->shape[2];
  if (c <= 0 || c > kMaxFrameChannels || hh <= 0 || ww <= 0) {
    return nullptr;
  }
  // One bound, on the element count: a measure indexes with `int`, so a
  // frame whose product does not fit that arithmetic is refused rather
  // than measured wrongly. The frame's DIMENSIONS are not bounded here
  // -- each measure states its own minimum.
  if (hh > (std::int64_t)kMaxFrameElements / ww ||
      hh * ww > (std::int64_t)kMaxFrameElements / c) {
    return nullptr;
  }
  if (channels != nullptr) { *channels = (int)c; }
  if (h != nullptr)        { *h = (int)hh; }
  if (w != nullptr)        { *w = (int)ww; }
  if (samples != nullptr)  { *samples = c * hh * ww; }
  return tb->as_u8();
}

// ---- trimming the decoded clip ----------------------------------------

PcmTrim
plan_pcm_trim(std::int64_t samples, int sample_rate, double fps,
              int trim_frames, int video_frames, bool conform)
{
  PcmTrim t;
  if (samples <= 0 || sample_rate <= 0 || !(fps > 0.0)) { return t; }
  const double spf = (double)sample_rate / fps;   // samples per frame
  const int frames =
      video_frames > 0 ? video_frames
                       : (int)std::llround((double)samples / spf);
  const int trim = std::max(0, trim_frames);
  t.drop = std::min<std::int64_t>(samples,
                                  (std::int64_t)std::llround(trim * spf));
  if (conform) {
    t.keep = std::max<std::int64_t>(
        0, (std::int64_t)std::llround((double)(frames - trim) * spf));
  } else {
    t.keep = samples - t.drop;
  }
  return t;
}

// ---- the luma step at the join -----------------------------------------

namespace {

// Below this the frame is black enough that a ratio of two levels says
// nothing, so nothing is done to it.
constexpr double kLumaLevelFloor = 0.25;
// The correction is a nudge back onto the clip's own trend, not a grade:
// a join needing more than this is a cut, and pushing it further would
// cost more in banding and clipping than the step it removes.
constexpr double kLumaGainMin = 0.82;
constexpr double kLumaGainMax = 1.22;

}  // namespace

double
frame_luma(const std::uint8_t* rgb, std::int64_t samples)
{
  if (rgb == nullptr || samples <= 0) { return 0.0; }
  std::uint64_t sum = 0;
  for (std::int64_t i = 0; i < samples; ++i) { sum += rgb[i]; }
  return (double)sum / (double)samples;
}

LumaTrend
luma_trend(const double* levels, int count, int a0, int a1)
{
  LumaTrend tr;
  if (levels == nullptr || count <= 0) { return tr; }
  a0 = std::max(0, std::min(a0, count - 1));
  a1 = std::min(a1, count);
  const int n = a1 - a0;
  if (n < 2) { return tr; }
  double mx = 0.0, my = 0.0;
  for (int k = a0; k < a1; ++k) {
    if (levels[k] <= kLumaLevelFloor) { return tr; }
    mx += (double)k;
    my += levels[k];
  }
  mx /= (double)n;
  my /= (double)n;
  double num = 0.0, den = 0.0;
  for (int k = a0; k < a1; ++k) {
    const double dx = (double)k - mx;
    num += dx * (levels[k] - my);
    den += dx * dx;
  }
  if (!(den > 0.0)) { return tr; }
  tr.slope     = num / den;
  tr.intercept = my - tr.slope * mx;
  tr.ok        = true;
  return tr;
}

double
luma_gain(const LumaTrend& body, double level, int k, int frames)
{
  if (!body.ok || frames <= 0 || k < 0 || k >= frames) { return 1.0; }
  if (level <= kLumaLevelFloor) { return 1.0; }
  const double want = body.at(k);
  if (want <= kLumaLevelFloor) { return 1.0; }
  const double g = std::clamp(want / level, kLumaGainMin, kLumaGainMax);
  const double w = 1.0 - (double)k / (double)frames;
  return 1.0 + (g - 1.0) * w;
}

void
apply_luma_gain(std::uint8_t* rgb, std::int64_t samples, double gain)
{
  if (rgb == nullptr || samples <= 0) { return; }
  for (std::int64_t i = 0; i < samples; ++i) {
    const long v = std::lround((double)rgb[i] * gain);
    rgb[i] = (std::uint8_t)std::clamp(v, 0L, 255L);
  }
}

// ---- levelling a whole chain -------------------------------------------

namespace {

// The structure band: a box blur of radius 2 against one of radius 8,
// about 5 px against about 17 px.
constexpr int kBandInner = 2;
constexpr int kBandOuter = 8;

// One separable box blur with replicated edges, in place, using `line`
// as the row/column buffer.
void
box_blur_(float* img, int h, int w, int r, std::vector<float>* line)
{
  if (r <= 0) { return; }
  line->resize((std::size_t)std::max(h, w));
  const double inv = 1.0 / (double)(2 * r + 1);
  for (int y = 0; y < h; ++y) {
    float* p = img + (std::size_t)y * w;
    double sum = 0.0;
    for (int x = -r; x <= r; ++x) { sum += p[std::clamp(x, 0, w - 1)]; }
    for (int x = 0; x < w; ++x) {
      (*line)[(std::size_t)x] = (float)(sum * inv);
      sum += p[std::clamp(x + r + 1, 0, w - 1)];
      sum -= p[std::clamp(x - r, 0, w - 1)];
    }
    std::memcpy(p, line->data(), (std::size_t)w * sizeof(float));
  }
  for (int x = 0; x < w; ++x) {
    double sum = 0.0;
    for (int y = -r; y <= r; ++y) {
      sum += img[(std::size_t)std::clamp(y, 0, h - 1) * w + x];
    }
    for (int y = 0; y < h; ++y) {
      (*line)[(std::size_t)y] = (float)(sum * inv);
      sum += img[(std::size_t)std::clamp(y + r + 1, 0, h - 1) * w + x];
      sum -= img[(std::size_t)std::clamp(y - r, 0, h - 1) * w + x];
    }
    for (int y = 0; y < h; ++y) {
      img[(std::size_t)y * w + x] = (*line)[(std::size_t)y];
    }
  }
}

// The grey both measures are taken on -- the mean over channels -- into
// `g`, returning the population variance of the WHOLE frame, floored, as
// the reference does. Both measures divide by it, which is what makes
// them contrast-normalised: a brighter or flatter take does not read as
// a different amount of detail.
double
grey_and_variance_(const std::uint8_t* rgb, int channels, int h, int w,
                   float* g)
{
  const std::size_t plane = (std::size_t)h * w;
  double sum = 0.0, sum2 = 0.0;
  for (std::size_t i = 0; i < plane; ++i) {
    double v = 0.0;
    for (int c = 0; c < channels; ++c) { v += rgb[(std::size_t)c * plane + i]; }
    v /= (double)channels;
    g[i] = (float)v;
    sum += v;
    sum2 += v * v;
  }
  const double mean = sum / (double)plane;
  return std::max(sum2 / (double)plane - mean * mean, 1e-9);
}

}  // namespace

double
norm_laplacian(const std::uint8_t* rgb, int channels, int h, int w)
{
  if (rgb == nullptr || channels <= 0 || h < 3 || w < 3) { return 0.0; }
  const std::size_t plane = (std::size_t)h * w;
  std::vector<float> g(plane);
  // The variance is the whole frame's; the Laplacian below is taken over
  // the interior only, as the reference does.
  const double var = grey_and_variance_(rgb, channels, h, w, g.data());
  double energy = 0.0;
  for (int y = 1; y < h - 1; ++y) {
    for (int x = 1; x < w - 1; ++x) {
      const std::size_t i = (std::size_t)y * w + x;
      const double k =
          4.0 * g[i] - g[i - w] - g[i + w] - g[i - 1] - g[i + 1];
      energy += k * k;
    }
  }
  energy /= (double)((h - 2) * (w - 2));
  return energy / var;
}

void
subtract_structure_band(std::uint8_t* rgb, int channels, int h, int w,
                        double amount, std::vector<float>* scratch)
{
  if (rgb == nullptr || channels <= 0 || h < 3 || w < 3) { return; }
  if (!(amount > 0.0) || scratch == nullptr) { return; }
  const std::size_t plane = (std::size_t)h * w;
  scratch->resize(plane * 2);
  float* inner = scratch->data();
  float* outer = inner + plane;
  std::vector<float> line;
  for (int c = 0; c < channels; ++c) {
    std::uint8_t* p = rgb + (std::size_t)c * plane;
    for (std::size_t i = 0; i < plane; ++i) { inner[i] = (float)p[i]; }
    std::memcpy(outer, inner, plane * sizeof(float));
    box_blur_(inner, h, w, kBandInner, &line);
    box_blur_(outer, h, w, kBandOuter, &line);
    for (std::size_t i = 0; i < plane; ++i) {
      const double band = (double)inner[i] - (double)outer[i];
      const long   v    = std::lround((double)p[i] - amount * band);
      p[i] = (std::uint8_t)std::clamp(v, 0L, 255L);
    }
  }
}

double
structure_band_energy(const std::uint8_t* rgb, int channels, int h, int w)
{
  if (rgb == nullptr || channels <= 0 || h < 3 || w < 3) { return 0.0; }
  const std::size_t plane = (std::size_t)h * w;
  std::vector<float> inner(plane), outer(plane);
  const double var = grey_and_variance_(rgb, channels, h, w, inner.data());
  outer = inner;
  std::vector<float> line;
  box_blur_(inner.data(), h, w, kBandInner, &line);
  box_blur_(outer.data(), h, w, kBandOuter, &line);
  double energy = 0.0;
  for (std::size_t i = 0; i < plane; ++i) {
    const double d = (double)inner[i] - (double)outer[i];
    energy += d * d;
  }
  return (energy / (double)plane) / var;
}

void
channel_cdf(const std::uint8_t* rgb, int channels, int h, int w, int channel,
            double* cdf256)
{
  if (cdf256 == nullptr) { return; }
  for (int i = 0; i < 256; ++i) { cdf256[i] = 0.0; }
  if (rgb == nullptr || channel < 0 || channel >= channels || h <= 0 ||
      w <= 0) {
    return;
  }
  const std::size_t plane = (std::size_t)h * w;
  const std::uint8_t* p = rgb + (std::size_t)channel * plane;
  double hist[256] = {};
  for (std::size_t i = 0; i < plane; ++i) { hist[p[i]] += 1.0; }
  double run = 0.0;
  for (int i = 0; i < 256; ++i) {
    run += hist[i];
    cdf256[i] = run / (double)plane;
  }
}

void
match_lut(const double* src_cdf256, const double* ref_cdf256,
          std::uint8_t* lut256)
{
  if (lut256 == nullptr) { return; }
  for (int i = 0; i < 256; ++i) { lut256[i] = (std::uint8_t)i; }
  if (src_cdf256 == nullptr || ref_cdf256 == nullptr) { return; }
  // searchsorted(ref, src) with the reference's default side: the first
  // reference level whose CDF is not below this source level's. Both are
  // non-decreasing, so one walk covers all 256.
  int j = 0;
  for (int i = 0; i < 256; ++i) {
    while (j < 255 && ref_cdf256[j] < src_cdf256[i]) { ++j; }
    lut256[i] = (std::uint8_t)j;
  }
}

void
apply_lut(std::uint8_t* rgb, int channels, int h, int w, int channel,
          const std::uint8_t* lut256)
{
  if (rgb == nullptr || lut256 == nullptr || channel < 0 ||
      channel >= channels || h <= 0 || w <= 0) {
    return;
  }
  const std::size_t plane = (std::size_t)h * w;
  std::uint8_t* p = rgb + (std::size_t)channel * plane;
  for (std::size_t i = 0; i < plane; ++i) { p[i] = lut256[p[i]]; }
}

// ---- the context file --------------------------------------------------

namespace {

constexpr std::uint64_t kMaxHeaderBytes = 100ull << 20;

void
put_u64_le_(std::uint64_t v, char* out)
{
  for (int i = 0; i < 8; ++i) { out[i] = (char)((v >> (8 * i)) & 0xff); }
}

std::uint64_t
get_u64_le_(const unsigned char* in)
{
  std::uint64_t v = 0;
  for (int i = 0; i < 8; ++i) { v |= (std::uint64_t)in[i] << (8 * i); }
  return v;
}

FlexData
tensor_entry_(const std::vector<std::int64_t>& shape, std::uint64_t begin,
              std::uint64_t end)
{
  FlexData e = FlexData::make_object();
  auto o = e.as_object();
  o.insert_or_assign("dtype", FlexData::make_string("F32"));
  FlexData sh = FlexData::make_array();
  for (std::int64_t d : shape) { sh.as_array().push_back(FlexData::make_int(d)); }
  o.insert_or_assign("shape", std::move(sh));
  FlexData off = FlexData::make_array();
  off.as_array().push_back(FlexData::make_uint(begin));
  off.as_array().push_back(FlexData::make_uint(end));
  o.insert_or_assign("data_offsets", std::move(off));
  return e;
}

}  // namespace

bool
write_context_file(const std::string& path, const ContextFileView& file,
                   std::string* err)
{
  if (std::endian::native != std::endian::little) {
    set_err_(err, "the context file format is little-endian only");
    return false;
  }
  const std::int64_t nv = product_(file.video_shape);
  if (file.video == nullptr || file.video_shape.size() != 4 || nv <= 0) {
    set_err_(err, "the video latent is not a non-empty [z, T, h, w] tensor");
    return false;
  }
  const bool has_audio = file.audio != nullptr && !file.audio_shape.empty();
  const std::int64_t na = has_audio ? product_(file.audio_shape) : 0;
  if (has_audio && (file.audio_shape.size() != 3 || na <= 0)) {
    set_err_(err, "the audio latent is not a [stereo, channels, A] tensor");
    return false;
  }

  const std::uint64_t vbytes = (std::uint64_t)nv * sizeof(float);
  const std::uint64_t abytes = (std::uint64_t)na * sizeof(float);
  FlexData header = FlexData::make_object();
  {
    auto o = header.as_object();
    FlexData meta = FlexData::make_object();
    for (const auto& kv : file.metadata) {
      meta.as_object().insert_or_assign(kv.first,
                                        FlexData::make_string(kv.second));
    }
    o.insert_or_assign("__metadata__", std::move(meta));
    o.insert_or_assign("video", tensor_entry_(file.video_shape, 0, vbytes));
    if (has_audio) {
      o.insert_or_assign("audio", tensor_entry_(file.audio_shape, vbytes,
                                                vbytes + abytes));
    }
  }
  std::string json = header.to_json(false);
  // Pad with spaces to an 8-byte boundary, as the format allows, so the
  // data starts aligned for anyone who maps the file.
  while ((json.size() + 8) % 8 != 0) { json.push_back(' '); }

  const std::string tmp = path + ".tmp";
  {
    std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
    if (!f) {
      set_err_(err, "cannot open '" + tmp + "' for writing");
      return false;
    }
    char len[8];
    put_u64_le_((std::uint64_t)json.size(), len);
    f.write(len, 8);
    f.write(json.data(), (std::streamsize)json.size());
    f.write((const char*)file.video, (std::streamsize)vbytes);
    if (has_audio) {
      f.write((const char*)file.audio, (std::streamsize)abytes);
    }
    f.flush();
    if (!f) {
      set_err_(err, "short write to '" + tmp + "'");
      std::remove(tmp.c_str());
      return false;
    }
  }
  // Written aside and renamed, so a reader never sees half a file -- a
  // chain that imports while another launch exports is the case.
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    set_err_(err, "cannot rename '" + tmp + "' to '" + path + "'");
    std::remove(tmp.c_str());
    return false;
  }
  return true;
}

bool
write_context_file(const std::string& path, const ContextFile& file,
                   std::string* err)
{
  const std::int64_t nv = product_(file.video_shape);
  if (nv <= 0 || (std::size_t)nv != file.video.size()) {
    set_err_(err, "the video latent is not a non-empty [z, T, h, w] tensor");
    return false;
  }
  const bool has_audio = !file.audio.empty();
  if (has_audio) {
    const std::int64_t na = product_(file.audio_shape);
    if (na <= 0 || (std::size_t)na != file.audio.size()) {
      set_err_(err, "the audio latent is not a [stereo, channels, A] tensor");
      return false;
    }
  }
  ContextFileView v;
  v.video       = file.video.data();
  v.video_shape = file.video_shape;
  v.audio       = has_audio ? file.audio.data() : nullptr;
  if (has_audio) { v.audio_shape = file.audio_shape; }
  v.metadata    = file.metadata;
  return write_context_file(path, v, err);
}

namespace {

// One tensor as the header describes it: shape plus its byte range
// within the data block.
struct Entry {
  std::vector<std::int64_t> shape;
  std::uint64_t             begin = 0;
  std::uint64_t             end   = 0;
  bool                      found = false;
};

// Open `path`, validate the safetensors header, and describe the two
// tensors this format carries. `f` is left positioned anywhere; every
// read below seeks. Errors are the caller's message, already prefixed
// with the path.
bool
open_header_(const std::string& path, std::ifstream* f,
             std::uint64_t* data_start, std::uint64_t* data_size,
             std::map<std::string, std::string>* metadata, Entry* video,
             Entry* audio, std::string* err)
{
  if (std::endian::native != std::endian::little) {
    set_err_(err, "the context file format is little-endian only");
    return false;
  }
  f->open(path, std::ios::binary);
  if (!*f) {
    set_err_(err, "cannot open '" + path + "'");
    return false;
  }
  f->seekg(0, std::ios::end);
  const std::int64_t size = (std::int64_t)f->tellg();
  f->seekg(0, std::ios::beg);
  unsigned char len[8];
  if (size < 8 || !f->read((char*)len, 8)) {
    set_err_(err, "'" + path + "' is too short to be a safetensors file");
    return false;
  }
  const std::uint64_t hlen = get_u64_le_(len);
  if (hlen == 0 || hlen > kMaxHeaderBytes || (std::int64_t)(8 + hlen) > size) {
    set_err_(err, "'" + path + "' has an invalid safetensors header length");
    return false;
  }
  std::string json((std::size_t)hlen, '\0');
  if (!f->read(json.data(), (std::streamsize)hlen)) {
    set_err_(err, "cannot read the header of '" + path + "'");
    return false;
  }
  FlexData header;
  try {
    header = FlexData::from_json(json);
  } catch (const std::exception& e) {
    set_err_(err, "'" + path + "' header is not valid JSON: " + e.what());
    return false;
  }
  if (!header.is_object()) {
    set_err_(err, "'" + path + "' header is not a JSON object");
    return false;
  }
  *data_start = 8 + hlen;
  *data_size  = (std::uint64_t)size - *data_start;

  const auto ho = header.as_object();
  if (ho.contains("__metadata__")) {
    const FlexData meta = ho.at("__metadata__");
    if (meta.is_object()) {
      for (const auto& kv : meta.as_object()) {
        if (kv.second.is_string()) {
          (*metadata)[std::string(kv.first)] = std::string(kv.second.as_string());
        }
      }
    }
  }

  // -1 malformed, 0 absent, 1 described. Shapes are file content: every
  // dimension must be a positive count and the element count must match
  // the byte range exactly, so nothing downstream indexes past the data.
  auto entry = [&](const char* name, std::size_t rank, Entry* out) -> int {
    if (!ho.contains(name)) { return 0; }
    const FlexData e = ho.at(name);
    if (!e.is_object()) { return -1; }
    const auto eo = e.as_object();
    if (!eo.contains("dtype") || eo.at("dtype").as_string() != "F32" ||
        !eo.contains("shape") || !eo.contains("data_offsets")) {
      return -1;
    }
    const FlexData sh = eo.at("shape");
    const FlexData off = eo.at("data_offsets");
    if (!sh.is_array() || !off.is_array() || off.as_array().size() != 2 ||
        sh.as_array().size() != rank) {
      return -1;
    }
    out->shape.clear();
    for (const FlexData& d : sh.as_array()) { out->shape.push_back(d.as_int(-1)); }
    const std::int64_t n = product_(out->shape);
    out->begin = off.as_array()[0].as_uint();
    out->end   = off.as_array()[1].as_uint();
    if (n <= 0 || out->end < out->begin || out->end > *data_size ||
        out->end - out->begin != (std::uint64_t)n * sizeof(float)) {
      return -1;
    }
    out->found = true;
    return 1;
  };

  if (entry("video", 4, video) <= 0) {
    set_err_(err, "'" + path + "' has no valid F32 [z, T, h, w] \"video\" "
                  "tensor");
    return false;
  }
  if (entry("audio", 3, audio) < 0) {
    set_err_(err, "'" + path + "' has an invalid \"audio\" tensor");
    return false;
  }
  return true;
}

// Read `count` floats starting `offset` floats into `e`'s byte range.
bool
read_floats_(std::ifstream* f, std::uint64_t data_start, const Entry& e,
             std::int64_t offset, std::int64_t count, float* out)
{
  if (count <= 0) { return true; }
  f->clear();
  f->seekg((std::streamoff)(data_start + e.begin +
                            (std::uint64_t)offset * sizeof(float)),
           std::ios::beg);
  return (bool)f->read((char*)out, (std::streamsize)count * sizeof(float));
}

}  // namespace

bool
read_context_file(const std::string& path, ContextFile* file,
                  std::string* err)
{
  if (file == nullptr) { return false; }
  std::ifstream f;
  std::uint64_t data_start = 0, data_size = 0;
  ContextFile out;
  Entry video, audio;
  if (!open_header_(path, &f, &data_start, &data_size, &out.metadata, &video,
                    &audio, err)) {
    return false;
  }
  out.video_shape = video.shape;
  out.video.assign((std::size_t)product_(video.shape), 0.0f);
  if (!read_floats_(&f, data_start, video, 0, (std::int64_t)out.video.size(),
                    out.video.data())) {
    set_err_(err, "cannot read the \"video\" tensor of '" + path + "'");
    return false;
  }
  if (audio.found) {
    out.audio_shape = audio.shape;
    out.audio.assign((std::size_t)product_(audio.shape), 0.0f);
    if (!read_floats_(&f, data_start, audio, 0, (std::int64_t)out.audio.size(),
                      out.audio.data())) {
      set_err_(err, "cannot read the \"audio\" tensor of '" + path + "'");
      return false;
    }
  }
  *file = std::move(out);
  return true;
}

bool
read_context_header(const std::string& path, ContextHeader* out,
                    std::string* err)
{
  if (out == nullptr) { return false; }
  std::ifstream f;
  std::uint64_t data_start = 0, data_size = 0;
  ContextHeader h;
  Entry video, audio;
  if (!open_header_(path, &f, &data_start, &data_size, &h.metadata, &video,
                    &audio, err)) {
    return false;
  }
  h.video_shape = video.shape;
  if (audio.found) { h.audio_shape = audio.shape; }
  *out = std::move(h);
  return true;
}

bool
read_context_tail(const std::string& path, int video_latents,
                  int audio_latents, ContextFile* out, std::string* err)
{
  if (out == nullptr) { return false; }
  std::ifstream f;
  std::uint64_t data_start = 0, data_size = 0;
  ContextFile tail;
  Entry video, audio;
  if (!open_header_(path, &f, &data_start, &data_size, &tail.metadata, &video,
                    &audio, err)) {
    return false;
  }
  if (video_latents > 0) {
    const std::int64_t z = video.shape[0], t = video.shape[1];
    const std::int64_t plane = video.shape[2] * video.shape[3];
    if (t < video_latents) {
      set_err_(err, "'" + path + "' holds " + std::to_string(t) +
                        " video latents, fewer than the " +
                        std::to_string(video_latents) + " asked for");
      return false;
    }
    tail.video_shape = {z, (std::int64_t)video_latents, video.shape[2],
                        video.shape[3]};
    tail.video.assign((std::size_t)(z * video_latents * plane), 0.0f);
    // One channel's tail is contiguous, so this is `z` reads of the
    // carried frames rather than the whole clip.
    for (std::int64_t c = 0; c < z; ++c) {
      if (!read_floats_(&f, data_start, video, (c * t + (t - video_latents)) * plane,
                        video_latents * plane,
                        tail.video.data() + c * video_latents * plane)) {
        set_err_(err, "cannot read the \"video\" tail of '" + path + "'");
        return false;
      }
    }
  }
  if (audio_latents > 0 && audio.found) {
    const std::int64_t s0 = audio.shape[0], ch = audio.shape[1];
    const std::int64_t a = audio.shape[2];
    if (a < audio_latents) {
      set_err_(err, "'" + path + "' holds " + std::to_string(a) +
                        " audio latents, fewer than the " +
                        std::to_string(audio_latents) + " asked for");
      return false;
    }
    tail.audio_shape = {s0, ch, (std::int64_t)audio_latents};
    tail.audio.assign((std::size_t)(s0 * ch * audio_latents), 0.0f);
    for (std::int64_t i = 0; i < s0 * ch; ++i) {
      if (!read_floats_(&f, data_start, audio, i * a + (a - audio_latents),
                        audio_latents,
                        tail.audio.data() + i * audio_latents)) {
        set_err_(err, "cannot read the \"audio\" tail of '" + path + "'");
        return false;
      }
    }
  }
  *out = std::move(tail);
  return true;
}

}  // namespace h3ctx
}  // namespace vpipe
