// The MiniMax-H3 packed layout with a CONTINUATION CONTEXT (ContextGuide).
//
// A context is clean conditioning rows placed on the TARGET's own clock,
// so the tests below check the three properties the feature depends on
// rather than goldens (the diffusers reference has no such input):
//
//   * With an empty context both builders are byte-identical to the
//     overloads without one -- existing graphs cannot change.
//   * A head context cut at VAE-cycle phase 0 sits on EXACTLY the
//     rotary coordinates of the first target latents, and its audio on
//     the requested offset from the target origin.
//   * The conditioning rows still LEAD both modalities' index vectors
//     and are counted in num_condition_*_rows, which is what the denoise
//     loop and build_row_timesteps key the clean timestep off.

#include "minitest.h"

#include "generative-models/minimax-h3/minimax-h3-layout.h"

#include <cmath>
#include <cstdio>
#include <vector>

namespace h3 = vpipe::genai::minimax_h3;

namespace {

bool
same_layout_(const h3::PackedLayout& a, const h3::PackedLayout& b)
{
  auto runs_eq = [](const std::vector<h3::RowRun>& x,
                    const std::vector<h3::RowRun>& y) {
    if (x.size() != y.size()) { return false; }
    for (std::size_t i = 0; i < x.size(); ++i) {
      if (x[i].start != y[i].start || x[i].count != y[i].count) {
        return false;
      }
    }
    return true;
  };
  return a.seq_len == b.seq_len && a.num_text_rows == b.num_text_rows &&
         a.condition_start == b.condition_start &&
         a.num_condition_rows == b.num_condition_rows &&
         a.audio_start == b.audio_start &&
         a.num_audio_rows == b.num_audio_rows &&
         a.video_start == b.video_start &&
         a.num_video_rows == b.num_video_rows &&
         a.position_ids == b.position_ids && a.token_tags == b.token_tags &&
         a.video_indices == b.video_indices &&
         a.audio_indices == b.audio_indices &&
         runs_eq(a.video_runs, b.video_runs) &&
         runs_eq(a.audio_runs, b.audio_runs) &&
         a.num_condition_video_rows == b.num_condition_video_rows &&
         a.num_condition_audio_rows == b.num_condition_audio_rows;
}

double
pos_(const h3::PackedLayout& L, int row, int axis)
{
  return L.position_ids[(std::size_t)row * 3 + (std::size_t)axis];
}

bool
row_pos_eq_(const h3::PackedLayout& a, int ra, const h3::PackedLayout& b,
            int rb)
{
  for (int k = 0; k < 3; ++k) {
    if (std::fabs(pos_(a, ra, k) - pos_(b, rb, k)) > 1e-9) { return false; }
  }
  return true;
}

// A 124-frame clip (37 latents, 207 audio latents) on a 16x32 latent
// canvas -- non-square so the width grid is not the height grid -- with
// the 22-frame / 37-audio-latent context the export/import stages carry.
constexpr int kText = 16, kLat = 37, kLh = 16, kLw = 32, kAud = 207;
constexpr int kRowsPerFrame = (kLh / 2) * (kLw / 2);
constexpr int kCtxLat = 7, kCtxAud = 37;
// End-aligned with the 22-frame picture, overhang +1/3: 5/3*22 + 1/3 - 37.
constexpr double kCtxOffset = 5.0 / 3.0 * 22.0 + (207.0 - 5.0 / 3.0 * 124.0) - 37.0;

h3::ContextGuide
head_context_()
{
  h3::ContextGuide g;
  g.num_latent_frames = kCtxLat;
  g.start_frame       = 0;
  g.num_audio_latents = kCtxAud;
  g.audio_offset      = kCtxOffset;
  return g;
}

}  // namespace

TEST(minimax_h3_layout_context, pixel_frames_for_latents)
{
  EXPECT_TRUE(h3::pixel_frames_for_latents(0) == 0);
  EXPECT_TRUE(h3::pixel_frames_for_latents(2) == 5);
  EXPECT_TRUE(h3::pixel_frames_for_latents(7) == 22);
  EXPECT_TRUE(h3::pixel_frames_for_latents(37) == 124);
  EXPECT_TRUE(h3::pixel_frames_for_latents(107) == 362);
}

TEST(minimax_h3_layout_context, empty_context_is_the_legacy_layout)
{
  const std::vector<int> tags((std::size_t)kText, h3::kTextTag);
  for (const auto& anchors :
       {std::vector<h3::Anchor>{},
        std::vector<h3::Anchor>{h3::Anchor::kFirst},
        std::vector<h3::Anchor>{h3::Anchor::kFirst, h3::Anchor::kLast}}) {
    h3::PackedLayout a, b;
    ASSERT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                          h3::kAudioChannels, anchors, &a));
    ASSERT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                          h3::kAudioChannels, anchors,
                                          h3::ContextGuide{}, &b));
    EXPECT_TRUE(same_layout_(a, b));
  }
  const std::vector<h3::Reference> refs = {
      {h3::Reference::Kind::kImage, 1, 32, 32, 0},
      {h3::Reference::Kind::kAudio, 0, 0, 0, 40},
      {h3::Reference::Kind::kVideo, 7, 16, 32, 37}};
  h3::PackedLayout a, b;
  ASSERT_TRUE(h3::build_ref2va_packed_sequence(tags, refs, kLat, kLh, kLw,
                                               kAud, 2, 2, h3::kAudioChannels,
                                               &a));
  ASSERT_TRUE(h3::build_ref2va_packed_sequence(tags, refs, kLat, kLh, kLw,
                                               kAud, 2, 2, h3::kAudioChannels,
                                               h3::ContextGuide{}, &b));
  EXPECT_TRUE(same_layout_(a, b));
}

TEST(minimax_h3_layout_context, fl2va_head_context_continues_the_target)
{
  const std::vector<int> tags((std::size_t)kText, h3::kTextTag);
  h3::PackedLayout L0, L;
  ASSERT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                        h3::kAudioChannels, {}, &L0));
  ASSERT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                        h3::kAudioChannels, {},
                                        head_context_(), &L));
  const int nctxv = kCtxLat * kRowsPerFrame;
  const int nctxa = kCtxAud * h3::kAudioChannels;

  EXPECT_TRUE(L.seq_len == L0.seq_len + nctxv + nctxa);
  EXPECT_TRUE(L.num_condition_video_rows == nctxv);
  EXPECT_TRUE(L.num_condition_rows == nctxv);
  EXPECT_TRUE(L.num_condition_audio_rows == nctxa);
  EXPECT_TRUE(L.num_video_rows == L0.num_video_rows);
  EXPECT_TRUE(L.num_audio_rows == L0.num_audio_rows);
  EXPECT_TRUE((int)L.video_indices.size() == nctxv + L.num_video_rows);
  EXPECT_TRUE((int)L.audio_indices.size() == nctxa + L.num_audio_rows);
  // The generated rows stay the LAST two blocks, which is what Sol's
  // dense sink (everything below video_start) and the unpatchify rely on.
  EXPECT_TRUE(L.video_start + L.num_video_rows == L.seq_len);
  EXPECT_TRUE(L.audio_start + L.num_audio_rows == L.video_start);

  // Runs cover the index vectors in order.
  {
    std::vector<int> vi, ai;
    for (const auto& r : L.video_runs) {
      for (int i = 0; i < r.count; ++i) { vi.push_back(r.start + i); }
    }
    for (const auto& r : L.audio_runs) {
      for (int i = 0; i < r.count; ++i) { ai.push_back(r.start + i); }
    }
    EXPECT_TRUE(vi == L.video_indices);
    EXPECT_TRUE(ai == L.audio_indices);
  }

  // Context video row k sits exactly where target video row k sits: the
  // first seven target latents, same spatial cell.
  bool video_ok = true;
  for (int k = 0; k < nctxv; ++k) {
    const int ctx_row = L.video_indices[(std::size_t)k];
    const int tgt_row = L.video_indices[(std::size_t)(nctxv + k)];
    if (!row_pos_eq_(L, ctx_row, L, tgt_row) ||
        L.token_tags[(std::size_t)ctx_row] != h3::kVideoTag) {
      video_ok = false;
    }
  }
  EXPECT_TRUE(video_ok);

  // Context audio: channel-major, origin + offset + i, the same width
  // extremes as the generated audio of that channel.
  bool audio_ok = true;
  for (int c = 0; c < h3::kAudioChannels; ++c) {
    for (int i = 0; i < kCtxAud; ++i) {
      const int row = L.audio_indices[(std::size_t)(c * kCtxAud + i)];
      const int tgt = L.audio_indices[(std::size_t)(nctxa + c * kAud)];
      if (std::fabs(pos_(L, row, 0) - (kText + kCtxOffset + i)) > 1e-9 ||
          pos_(L, row, 1) != 0.0 || pos_(L, row, 2) != pos_(L, tgt, 2) ||
          L.token_tags[(std::size_t)row] != h3::kAudioTag) {
        audio_ok = false;
      }
    }
  }
  EXPECT_TRUE(audio_ok);
  // The carried audio ends where the carried picture ends, on the
  // generated audio's grid (the +1/3 overhang makes it land on 37).
  EXPECT_TRUE(std::fabs(kCtxOffset) < 1e-9);

  // The generated rows are placed exactly as without a context.
  bool target_ok = true;
  for (int i = 0; i < L.num_video_rows; ++i) {
    if (!row_pos_eq_(L, L.video_start + i, L0, L0.video_start + i)) {
      target_ok = false;
    }
  }
  for (int i = 0; i < L.num_audio_rows; ++i) {
    if (!row_pos_eq_(L, L.audio_start + i, L0, L0.audio_start + i)) {
      target_ok = false;
    }
  }
  EXPECT_TRUE(target_ok);

  // Clean timestep on every context row, the schedule on the rest.
  std::vector<float> uniq;
  std::vector<int> idx;
  h3::build_row_timesteps(L, 0.25f, 0.5f, 1.0f, &uniq, &idx, 1.0f);
  auto t_of = [&](int row) { return uniq[(std::size_t)idx[(std::size_t)row]]; };
  bool ts_ok = true;
  for (int k = 0; k < nctxv; ++k) {
    if (t_of(L.video_indices[(std::size_t)k]) != 1.0f) { ts_ok = false; }
  }
  for (int k = 0; k < nctxa; ++k) {
    if (t_of(L.audio_indices[(std::size_t)k]) != 1.0f) { ts_ok = false; }
  }
  if (t_of(L.video_start) != 0.25f || t_of(L.audio_start) != 0.5f) {
    ts_ok = false;
  }
  EXPECT_TRUE(ts_ok);
  std::printf("[minimax_h3_layout_context] fl2va seq %d -> %d "
              "(+%d video, +%d audio context rows)\n",
              L0.seq_len, L.seq_len, nctxv, nctxa);
}

TEST(minimax_h3_layout_context, fl2va_last_anchor_with_context)
{
  const std::vector<int> tags((std::size_t)kText, h3::kTextTag);
  h3::PackedLayout A, L;
  ASSERT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kLast}, &A));
  ASSERT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                        h3::kAudioChannels,
                                        {h3::Anchor::kLast}, head_context_(),
                                        &L));
  const int nctxv = kCtxLat * kRowsPerFrame;
  EXPECT_TRUE(L.num_condition_video_rows == kRowsPerFrame + nctxv);
  // The anchor block keeps its rows and coordinates; the context follows
  // it inside the same contiguous conditioning run.
  bool anchor_ok = true;
  for (int r = 0; r < kRowsPerFrame; ++r) {
    if (!row_pos_eq_(L, L.condition_start + r, A, A.condition_start + r)) {
      anchor_ok = false;
    }
  }
  EXPECT_TRUE(anchor_ok);
  EXPECT_TRUE(L.video_runs.size() == 2);
  EXPECT_TRUE(L.video_runs[0].start == kText &&
              L.video_runs[0].count == kRowsPerFrame + nctxv);
  EXPECT_TRUE(row_pos_eq_(L, kText + kRowsPerFrame, L, L.video_start));
}

TEST(minimax_h3_layout_context, ref2va_context_starts_at_the_target_origin)
{
  const std::vector<int> tags((std::size_t)kText, h3::kTextTag);
  const std::vector<h3::Reference> refs = {
      {h3::Reference::Kind::kImage, 1, 32, 32, 0},
      {h3::Reference::Kind::kAudio, 0, 0, 0, 40}};
  h3::PackedLayout R0, R;
  ASSERT_TRUE(h3::build_ref2va_packed_sequence(tags, refs, kLat, kLh, kLw,
                                               kAud, 2, 2, h3::kAudioChannels,
                                               &R0));
  ASSERT_TRUE(h3::build_ref2va_packed_sequence(tags, refs, kLat, kLh, kLw,
                                               kAud, 2, 2, h3::kAudioChannels,
                                               head_context_(), &R));
  const int nctxv = kCtxLat * kRowsPerFrame;
  const int nctxa = kCtxAud * h3::kAudioChannels;
  EXPECT_TRUE(R.seq_len == R0.seq_len + nctxv + nctxa);
  EXPECT_TRUE(R.num_condition_video_rows == R0.num_condition_video_rows + nctxv);
  EXPECT_TRUE(R.num_condition_audio_rows == R0.num_condition_audio_rows + nctxa);

  // The references are untouched: same leading index entries, same
  // coordinates, same rows.
  bool refs_ok = true;
  for (int k = 0; k < R0.num_condition_video_rows; ++k) {
    const int a = R.video_indices[(std::size_t)k];
    const int b = R0.video_indices[(std::size_t)k];
    if (a != b || !row_pos_eq_(R, a, R0, b)) { refs_ok = false; }
  }
  for (int k = 0; k < R0.num_condition_audio_rows; ++k) {
    const int a = R.audio_indices[(std::size_t)k];
    const int b = R0.audio_indices[(std::size_t)k];
    if (a != b || !row_pos_eq_(R, a, R0, b)) { refs_ok = false; }
  }
  EXPECT_TRUE(refs_ok);

  // The context follows the references in the index order and sits on the
  // first target latents / the target origin + offset.
  const int v0 = R0.num_condition_video_rows;
  bool ctx_ok = true;
  for (int k = 0; k < nctxv; ++k) {
    const int ctx_row = R.video_indices[(std::size_t)(v0 + k)];
    const int tgt_row = R.video_indices[(std::size_t)(v0 + nctxv + k)];
    if (!row_pos_eq_(R, ctx_row, R, tgt_row)) { ctx_ok = false; }
  }
  const double origin = pos_(R, R.audio_start, 0);   // target audio latent 0
  const int a0 = R0.num_condition_audio_rows;
  for (int i = 0; i < kCtxAud; ++i) {
    const int row = R.audio_indices[(std::size_t)(a0 + i)];
    if (std::fabs(pos_(R, row, 0) - (origin + kCtxOffset + i)) > 1e-9) {
      ctx_ok = false;
    }
  }
  EXPECT_TRUE(ctx_ok);

  // And the generated rows are where they were.
  bool target_ok = true;
  for (int i = 0; i < R.num_video_rows; ++i) {
    if (!row_pos_eq_(R, R.video_start + i, R0, R0.video_start + i)) {
      target_ok = false;
    }
  }
  EXPECT_TRUE(target_ok);
  EXPECT_TRUE(R.video_start + R.num_video_rows == R.seq_len);
}

TEST(minimax_h3_layout_context, rejects_contexts_that_do_not_fit)
{
  const std::vector<int> tags((std::size_t)kText, h3::kTextTag);
  h3::PackedLayout L;
  h3::ContextGuide g = head_context_();

  g.start_frame = 124 - 22;          // ends exactly at the clip's end: fits
  EXPECT_TRUE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                        h3::kAudioChannels, {}, g, &L));
  g.start_frame = 124 - 21;          // one frame past the end
  EXPECT_FALSE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                         h3::kAudioChannels, {}, g, &L));
  g = head_context_();
  g.num_latent_frames = -1;
  EXPECT_FALSE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                         h3::kAudioChannels, {}, g, &L));
  g = head_context_();
  g.audio_offset = std::nan("");
  EXPECT_FALSE(h3::build_packed_sequence(tags, kLat, kLh, kLw, kAud, 2, 2,
                                         h3::kAudioChannels, {}, g, &L));
  const std::vector<h3::Reference> refs = {
      {h3::Reference::Kind::kImage, 1, 32, 32, 0}};
  g = head_context_();
  g.start_frame = -5;
  EXPECT_FALSE(h3::build_ref2va_packed_sequence(tags, refs, kLat, kLh, kLw,
                                                kAud, 2, 2, h3::kAudioChannels,
                                                g, &L));
}
