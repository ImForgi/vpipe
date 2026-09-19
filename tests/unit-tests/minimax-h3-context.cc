// The MiniMax-H3 continuation-context helpers
// (stages/minimax-h3-context.h): the 17n + 5 frame grid, the duration
// modes, the tail plan the importer builds, the PCM trim, and the context
// file round trip. Pure arithmetic and byte I/O -- no model, no GPU -- so
// nothing here is gated.

#include "minitest.h"

#include "stages/minimax-h3-context.h"

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;

TEST(minimax_h3_context, frame_grid)
{
  EXPECT_TRUE(h3ctx::is_clip_frames(5));
  EXPECT_TRUE(h3ctx::is_clip_frames(22));
  EXPECT_TRUE(h3ctx::is_clip_frames(124));
  EXPECT_TRUE(h3ctx::is_clip_frames(362));
  EXPECT_FALSE(h3ctx::is_clip_frames(1));
  EXPECT_FALSE(h3ctx::is_clip_frames(120));
  EXPECT_TRUE(h3ctx::latents_for_frames(5) == 2);
  EXPECT_TRUE(h3ctx::latents_for_frames(22) == 7);
  EXPECT_TRUE(h3ctx::latents_for_frames(124) == 37);
  EXPECT_TRUE(h3ctx::latents_for_frames(120) == 0);
  EXPECT_TRUE(h3ctx::frames_for_latents(37) == 124);
  EXPECT_TRUE(h3ctx::frames_for_latents(7) == 22);
  EXPECT_TRUE(h3ctx::frames_for_latents(36) == 0);   // not 5n + 2
  EXPECT_TRUE(h3ctx::align_frames_up(120) == 124);
  EXPECT_TRUE(h3ctx::align_frames_up(124) == 124);
  EXPECT_TRUE(h3ctx::align_frames_up(1) == 5);
  EXPECT_TRUE(h3ctx::align_frames_nearest(142) == 141);
  EXPECT_TRUE(h3ctx::align_frames_nearest(150) == 158);   // 9 vs 8: up
  EXPECT_TRUE(h3ctx::audio_latents_for_frames(124) == 207);
  EXPECT_TRUE(h3ctx::audio_latents_for_frames(260) == 433);
}

TEST(minimax_h3_context, duration_modes)
{
  h3ctx::DurationPlan p;
  std::string err;
  // clip: what was configured; the overlap comes out of it.
  ASSERT_TRUE(h3ctx::resolve_duration(120, 22, 0, h3ctx::DurationMode::kClip,
                                      &p, &err));
  EXPECT_TRUE(p.generated_frames == 124 && p.overlap_frames == 22 &&
              p.new_frames == 102 && !p.capped);
  // new_footage: 120 kept + 22 overlap = 142 -> nearest 141 -> 119 new.
  ASSERT_TRUE(h3ctx::resolve_duration(120, 22, 0,
                                      h3ctx::DurationMode::kNewFootage, &p,
                                      &err));
  EXPECT_TRUE(p.generated_frames == 141 && p.new_frames == 119);
  // new_footage_min: 142 -> up 158 -> 136 new.
  ASSERT_TRUE(h3ctx::resolve_duration(120, 22, 0,
                                      h3ctx::DurationMode::kNewFootageMin, &p,
                                      &err));
  EXPECT_TRUE(p.generated_frames == 158 && p.new_frames == 136);
  // 15 s asked as new footage would exceed the trained 362: capped.
  ASSERT_TRUE(h3ctx::resolve_duration(360, 22, 0,
                                      h3ctx::DurationMode::kNewFootage, &p,
                                      &err));
  EXPECT_TRUE(p.generated_frames == 362 && p.capped && p.new_frames == 340);
  // clip mode never caps -- it is exactly what the graph configured.
  ASSERT_TRUE(h3ctx::resolve_duration(379, 22, 0, h3ctx::DurationMode::kClip,
                                      &p, &err));
  EXPECT_TRUE(p.generated_frames == 379 && !p.capped);
  // An interior guide replaces nothing: no overlap.
  ASSERT_TRUE(h3ctx::resolve_duration(124, 22, 34,
                                      h3ctx::DurationMode::kNewFootage, &p,
                                      &err));
  EXPECT_TRUE(p.overlap_frames == 0 && p.generated_frames == 124);
  // A context that fills the clip leaves nothing to generate.
  EXPECT_FALSE(h3ctx::resolve_duration(22, 22, 0, h3ctx::DurationMode::kClip,
                                       &p, &err));
  // Parsing.
  h3ctx::DurationMode m;
  EXPECT_TRUE(h3ctx::parse_duration_mode("", &m) &&
              m == h3ctx::DurationMode::kClip);
  EXPECT_TRUE(h3ctx::parse_duration_mode("new_footage_min", &m) &&
              m == h3ctx::DurationMode::kNewFootageMin);
  EXPECT_FALSE(h3ctx::parse_duration_mode("seconds", &m));
}

TEST(minimax_h3_context, tail_plan)
{
  h3ctx::TailPlan t;
  std::string err;
  // 124-frame source: 37 video latents, 207 audio latents (+1/3 overhang).
  ASSERT_TRUE(h3ctx::plan_tail(37, 207, 22, 0, 0, h3ctx::AudioAlign::kExact,
                               &t, &err));
  EXPECT_TRUE(t.source_frames == 124 && t.context_frames == 22 &&
              t.video_latents == 7 && t.audio_latents == 37 &&
              t.trim_frames == 22 && !t.overhang_ignored);
  EXPECT_TRUE(std::fabs(t.audio_overhang - 1.0 / 3.0) < 1e-12);
  // 5/3 * 22 + 1/3 - 37 = 0: the carried audio lands on the target grid.
  EXPECT_TRUE(std::fabs(t.audio_offset) < 1e-12);

  // 260-frame source rounds DOWN (433 for 433.33): overhang -1/3.
  ASSERT_TRUE(h3ctx::plan_tail(77, 433, 22, 0, 0, h3ctx::AudioAlign::kExact,
                               &t, &err));
  EXPECT_TRUE(t.source_frames == 260);
  EXPECT_TRUE(std::fabs(t.audio_overhang + 1.0 / 3.0) < 1e-9);
  EXPECT_TRUE(std::fabs(t.audio_offset - (5.0 / 3.0 * 22 - 1.0 / 3.0 - 37)) <
              1e-9);
  // round snaps that -2/3 to the integer grid.
  ASSERT_TRUE(h3ctx::plan_tail(77, 433, 22, 0, 0, h3ctx::AudioAlign::kRound,
                               &t, &err));
  EXPECT_TRUE(t.audio_offset == -1.0);

  // A longer audio window reaches back before the picture window.
  ASSERT_TRUE(h3ctx::plan_tail(37, 207, 22, 48, 0, h3ctx::AudioAlign::kExact,
                               &t, &err));
  EXPECT_TRUE(t.audio_latents == 80 && t.audio_offset < 0.0);
  // No audio: negative window, or a source without a soundtrack.
  ASSERT_TRUE(h3ctx::plan_tail(37, 207, 22, -1, 0, h3ctx::AudioAlign::kExact,
                               &t, &err));
  EXPECT_TRUE(t.audio_latents == 0);
  ASSERT_TRUE(h3ctx::plan_tail(37, 0, 22, 0, 0, h3ctx::AudioAlign::kExact,
                               &t, &err));
  EXPECT_TRUE(t.audio_latents == 0);
  // Interior placement: no trim, audio still ends with the picture.
  ASSERT_TRUE(h3ctx::plan_tail(37, 207, 22, 0, 34, h3ctx::AudioAlign::kExact,
                               &t, &err));
  EXPECT_TRUE(t.trim_frames == 0 &&
              std::fabs(t.audio_offset - (5.0 / 3.0 * 56 + 1.0 / 3.0 - 37)) <
                  1e-9);

  EXPECT_FALSE(h3ctx::plan_tail(37, 207, 20, 0, 0, h3ctx::AudioAlign::kExact,
                                &t, &err));   // not 17k + 5
  EXPECT_FALSE(h3ctx::plan_tail(37, 207, 141, 0, 0, h3ctx::AudioAlign::kExact,
                                &t, &err));   // longer than the source
  EXPECT_FALSE(h3ctx::plan_tail(36, 207, 22, 0, 0, h3ctx::AudioAlign::kExact,
                                &t, &err));   // source is not 5n + 2
}

TEST(minimax_h3_context, tail_slices)
{
  // [z=2, T=7, h=2, w=3]: value encodes (c, t, cell).
  const std::vector<std::int64_t> vs = {2, 7, 2, 3};
  std::vector<float> v(2 * 7 * 6);
  for (int c = 0; c < 2; ++c) {
    for (int t = 0; t < 7; ++t) {
      for (int i = 0; i < 6; ++i) {
        v[(std::size_t)((c * 7 + t) * 6 + i)] = (float)(c * 100 + t * 10 + i);
      }
    }
  }
  std::vector<float> out;
  std::vector<std::int64_t> os;
  ASSERT_TRUE(h3ctx::slice_video_tail(v.data(), vs, 2, &out, &os));
  EXPECT_TRUE(os == (std::vector<std::int64_t>{2, 2, 2, 3}));
  EXPECT_TRUE(out[0] == 50.0f && out[6] == 60.0f && out[12] == 150.0f &&
              out[23] == 165.0f);
  EXPECT_FALSE(h3ctx::slice_video_tail(v.data(), vs, 8, &out, &os));

  const std::vector<std::int64_t> as = {2, 3, 5};
  std::vector<float> a(30);
  for (int i = 0; i < 30; ++i) { a[(std::size_t)i] = (float)i; }
  ASSERT_TRUE(h3ctx::slice_audio_tail(a.data(), as, 2, &out, &os));
  EXPECT_TRUE(os == (std::vector<std::int64_t>{2, 3, 2}));
  EXPECT_TRUE(out[0] == 3.0f && out[1] == 4.0f && out[2] == 8.0f &&
              out[11] == 29.0f);
}

TEST(minimax_h3_context, pcm_trim)
{
  // 124 frames decode to 207 * 800 = 165600 samples (8.3 ms long).
  auto t = h3ctx::plan_pcm_trim(165600, 32000, 24.0, 22, 124, true);
  EXPECT_TRUE(t.drop == 29333 && t.keep == 136000);   // exactly 102 frames
  t = h3ctx::plan_pcm_trim(165600, 32000, 24.0, 22, 0, true);  // derived F
  EXPECT_TRUE(t.drop == 29333 && t.keep == 136000);
  t = h3ctx::plan_pcm_trim(165600, 32000, 24.0, 22, 124, false);
  EXPECT_TRUE(t.keep == 165600 - 29333);
  // 260 frames decode 8.3 ms SHORT (433 * 800): conform pads.
  t = h3ctx::plan_pcm_trim(346400, 32000, 24.0, 22, 260, true);
  EXPECT_TRUE(t.keep == 317333 && t.keep > 346400 - t.drop);
  t = h3ctx::plan_pcm_trim(165600, 32000, 24.0, 0, 124, true);
  EXPECT_TRUE(t.drop == 0 && t.keep == 165333);
}

TEST(minimax_h3_context, luma_match)
{
  // A clip whose body sits flat at 100 and whose first two frames come off
  // the context hot -- the join flash, in miniature.
  const int    kFrames = 12;
  const double kBody   = 100.0;
  std::vector<double> level((std::size_t)kFrames, kBody);
  level[0] = 115.0;
  level[1] = 105.0;
  const auto body = h3ctx::luma_trend(level.data(), kFrames,
                                      h3ctx::kLumaBodyBegin,
                                      h3ctx::kLumaBodyEnd);
  EXPECT_TRUE(body.ok);
  EXPECT_TRUE(std::abs(body.at(0) - kBody) < 1e-6);     // flat fit = the mean
  // Frame 0 lands on the body; the release only weakens what follows.
  const double g0 = h3ctx::luma_gain(body, level[0], 0, 6);
  EXPECT_TRUE(std::abs(level[0] * g0 - kBody) < 0.5);
  const double g1 = h3ctx::luma_gain(body, level[1], 1, 6);
  EXPECT_TRUE(g1 > h3ctx::luma_gain(body, level[1], 2, 6) || g1 < 1.0);
  // A frame already on the settled level is untouched, at any distance.
  for (int k = 0; k < 6; ++k) {
    EXPECT_TRUE(std::abs(h3ctx::luma_gain(body, kBody, k, 6) - 1.0) < 1e-9);
  }
  // Past the window, and with the window off, nothing happens at all.
  EXPECT_TRUE(h3ctx::luma_gain(body, 115.0, 6, 6) == 1.0);
  EXPECT_TRUE(h3ctx::luma_gain(body, 115.0, 0, 0) == 1.0);
  // A shot that is genuinely changing is followed, not flattened: a body
  // falling 3 per frame wants frame 0 ABOVE the window's average.
  std::vector<double> fall((std::size_t)kFrames, 0.0);
  for (int k = 0; k < kFrames; ++k) { fall[(std::size_t)k] = 120.0 - 3.0 * k; }
  const auto trend = h3ctx::luma_trend(fall.data(), kFrames,
                                       h3ctx::kLumaBodyBegin,
                                       h3ctx::kLumaBodyEnd);
  EXPECT_TRUE(trend.ok && std::abs(trend.slope + 3.0) < 1e-6);
  EXPECT_TRUE(std::abs(trend.at(0) - 120.0) < 1e-6);
  // The gain is clamped, so a join too far apart to bridge is not forced.
  EXPECT_TRUE(h3ctx::luma_gain(body, 10.0, 0, 6) <= 1.22 + 1e-9);
  EXPECT_TRUE(h3ctx::luma_gain(body, 1000.0, 0, 6) >= 0.82 - 1e-9);
  // Too little to fit, or a black frame, and there is no trend at all.
  EXPECT_TRUE(!h3ctx::luma_trend(level.data(), kFrames, 3, 4).ok);
  std::vector<double> dark((std::size_t)kFrames, 0.0);
  EXPECT_TRUE(!h3ctx::luma_trend(dark.data(), kFrames, h3ctx::kLumaBodyBegin,
                                 h3ctx::kLumaBodyEnd).ok);

  // And on real pixels: the measurement and the correction agree.
  std::vector<std::uint8_t> px(3 * 4 * 4, 115);
  EXPECT_TRUE(std::abs(h3ctx::frame_luma(px.data(), (std::int64_t)px.size()) -
                       115.0) < 1e-9);
  h3ctx::apply_luma_gain(px.data(), (std::int64_t)px.size(), g0);
  EXPECT_TRUE(std::abs(h3ctx::frame_luma(px.data(), (std::int64_t)px.size()) -
                       kBody) < 1.0);
  std::vector<std::uint8_t> hot(8, 250);
  h3ctx::apply_luma_gain(hot.data(), (std::int64_t)hot.size(), 1.22);
  EXPECT_TRUE(hot[0] == 255);                    // saturates, does not wrap
}

TEST(minimax_h3_context, file_round_trip)
{
  const std::string path =
      (std::filesystem::temp_directory_path() /
       "vpipe-minimax-h3-context-ut.safetensors").string();
  h3ctx::ContextFile f;
  f.video_shape = {24, 7, 4, 6};
  f.video.resize(24 * 7 * 4 * 6);
  for (std::size_t i = 0; i < f.video.size(); ++i) {
    f.video[i] = (float)i * 0.5f - 3.0f;
  }
  f.audio_shape = {2, 32, 37};
  f.audio.resize(2 * 32 * 37);
  for (std::size_t i = 0; i < f.audio.size(); ++i) {
    f.audio[i] = -(float)i;
  }
  f.metadata["format"] = h3ctx::kFormat;
  f.metadata["frames"] = "22";
  f.metadata["prompt"] = "a \"quoted\" prompt\nwith a newline";
  std::string err;
  ASSERT_TRUE(h3ctx::write_context_file(path, f, &err));

  h3ctx::ContextFile g;
  ASSERT_TRUE(h3ctx::read_context_file(path, &g, &err));
  EXPECT_TRUE(g.video_shape == f.video_shape && g.video == f.video);
  EXPECT_TRUE(g.audio_shape == f.audio_shape && g.audio == f.audio);
  EXPECT_TRUE(g.metadata == f.metadata);

  // The header is a real safetensors header: 8-byte length, JSON, data
  // aligned to 8 bytes.
  {
    std::ifstream in(path, std::ios::binary);
    unsigned char len[8];
    in.read((char*)len, 8);
    std::uint64_t n = 0;
    for (int i = 0; i < 8; ++i) { n |= (std::uint64_t)len[i] << (8 * i); }
    EXPECT_TRUE((n + 8) % 8 == 0);
    const auto size = std::filesystem::file_size(path);
    EXPECT_TRUE(size == 8 + n + f.video.size() * 4 + f.audio.size() * 4);
  }

  // Video only.
  h3ctx::ContextFile silent;
  silent.video_shape = f.video_shape;
  silent.video = f.video;
  ASSERT_TRUE(h3ctx::write_context_file(path, silent, &err));
  ASSERT_TRUE(h3ctx::read_context_file(path, &g, &err));
  EXPECT_TRUE(g.audio.empty() && g.video == f.video);

  // A truncated file is refused, not misread.
  {
    const auto size = std::filesystem::file_size(path);
    std::filesystem::resize_file(path, size - 4);
  }
  EXPECT_FALSE(h3ctx::read_context_file(path, &g, &err));
  std::filesystem::remove(path);
  EXPECT_FALSE(h3ctx::read_context_file(path, &g, &err));
}

// The header comes from a FILE, so it is not to be trusted: a shape that
// does not describe the buffer is what turns the tail slicing below into
// an out-of-bounds read.
TEST(minimax_h3_context, a_hostile_header_is_refused)
{
  const std::string path =
      (std::filesystem::temp_directory_path() /
       "vpipe-minimax-h3-context-hostile.safetensors").string();
  auto write_raw = [&](const std::string& json, std::size_t floats) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    const std::uint64_t n = json.size();
    char len[8];
    for (int i = 0; i < 8; ++i) { len[i] = (char)((n >> (8 * i)) & 0xff); }
    f.write(len, 8);
    f.write(json.data(), (std::streamsize)json.size());
    const std::vector<float> data(floats, 1.0f);
    f.write((const char*)data.data(), (std::streamsize)(floats * 4));
  };
  h3ctx::ContextFile g;
  std::string err;

  // A product that OVERFLOWS int64 and lands on a small positive number
  // matching the byte range: 3 * 6148914691236517206 == 2 (mod 2^64).
  write_raw(R"({"video":{"dtype":"F32","shape":[3,6148914691236517206,1,1],)"
            R"("data_offsets":[0,8]}})", 2);
  EXPECT_FALSE(h3ctx::read_context_file(path, &g, &err));

  // Negative and zero dimensions are not counts.
  write_raw(R"({"video":{"dtype":"F32","shape":[-1,-1,4,4],)"
            R"("data_offsets":[0,64]}})", 16);
  EXPECT_FALSE(h3ctx::read_context_file(path, &g, &err));
  write_raw(R"({"video":{"dtype":"F32","shape":[24,0,4,4],)"
            R"("data_offsets":[0,0]}})", 0);
  EXPECT_FALSE(h3ctx::read_context_file(path, &g, &err));

  // A byte range past the end of the data block.
  write_raw(R"({"video":{"dtype":"F32","shape":[24,7,4,6],)"
            R"("data_offsets":[0,16128]}})", 16);
  EXPECT_FALSE(h3ctx::read_context_file(path, &g, &err));

  std::filesystem::remove(path);
}

// The importer reads a tail out of a file that holds a whole clip, so the
// tail it gets must be the one slicing the whole clip would have given.
TEST(minimax_h3_context, header_and_tail_reads)
{
  const std::string path =
      (std::filesystem::temp_directory_path() /
       "vpipe-minimax-h3-context-tail.safetensors").string();
  h3ctx::ContextFile f;
  f.video_shape = {24, 12, 4, 6};
  f.video.resize(24 * 12 * 4 * 6);
  for (std::size_t i = 0; i < f.video.size(); ++i) { f.video[i] = (float)i; }
  f.audio_shape = {2, 32, 40};
  f.audio.resize(2 * 32 * 40);
  for (std::size_t i = 0; i < f.audio.size(); ++i) { f.audio[i] = -(float)i; }
  f.metadata["format"] = h3ctx::kFormat;
  std::string err;
  ASSERT_TRUE(h3ctx::write_context_file(path, f, &err));

  h3ctx::ContextHeader h;
  ASSERT_TRUE(h3ctx::read_context_header(path, &h, &err));
  EXPECT_TRUE(h.video_shape == f.video_shape && h.audio_shape == f.audio_shape);
  EXPECT_TRUE(h.metadata["format"] == h3ctx::kFormat);

  h3ctx::ContextFile tail;
  ASSERT_TRUE(h3ctx::read_context_tail(path, 7, 37, &tail, &err));
  std::vector<float> want_v, want_a;
  std::vector<std::int64_t> want_vs, want_as;
  ASSERT_TRUE(h3ctx::slice_video_tail(f.video.data(), f.video_shape, 7,
                                      &want_v, &want_vs));
  ASSERT_TRUE(h3ctx::slice_audio_tail(f.audio.data(), f.audio_shape, 37,
                                      &want_a, &want_as));
  EXPECT_TRUE(tail.video_shape == want_vs && tail.video == want_v);
  EXPECT_TRUE(tail.audio_shape == want_as && tail.audio == want_a);

  // More than the file holds is an error rather than a short read.
  EXPECT_FALSE(h3ctx::read_context_tail(path, 99, 37, &tail, &err));
  EXPECT_FALSE(h3ctx::read_context_tail(path, 7, 99, &tail, &err));

  // Video only: a tail asked for sound gets none, not a failure.
  h3ctx::ContextFile silent;
  silent.video_shape = f.video_shape;
  silent.video       = f.video;
  ASSERT_TRUE(h3ctx::write_context_file(path, silent, &err));
  ASSERT_TRUE(h3ctx::read_context_tail(path, 7, 37, &tail, &err));
  EXPECT_TRUE(tail.video == want_v && tail.audio.empty());
  std::filesystem::remove(path);
}
