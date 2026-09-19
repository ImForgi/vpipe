// `save-tensor` / `load-tensor`: the round trip, and the window.
//
// These two exist so a LATENT can outlive the run that made it -- a
// multi-part video graph otherwise hands each clip to the next through
// a file, which costs a VAE decode, a codec, a decode and a VAE encode.
// MEASURED on MiniMax-H3, that round trip recovers the latent at
// correlation 0.875 with a systematic 0.86 gain, so it is not a way to
// carry a latent at all. Hence these.

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/typed-stage.h"
#include "stages/load-tensor-stage.h"

#include <cstdint>
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace vpipe;

namespace {

std::string tmp_(const char* name)
{
  return (std::filesystem::temp_directory_path() / name).string();
}

// Write a VPTENSOR file directly, so the reader is tested against the
// FORMAT rather than against whatever the writer happens to do.
bool write_(const std::string& path, std::vector<std::int64_t> dims,
            const std::vector<float>& data, const char* sideband)
{
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) { return false; }
  const char magic[8] = {'V','P','T','E','N','S','O','R'};
  std::uint32_t ver = 1, dt = 3 /*F32*/, rank = (std::uint32_t)dims.size(),
                zero = 0;
  std::uint64_t count = data.size();
  std::fwrite(magic, 1, 8, f);
  std::fwrite(&ver, 4, 1, f); std::fwrite(&dt, 4, 1, f);
  std::fwrite(&rank, 4, 1, f); std::fwrite(&zero, 4, 1, f);
  std::fwrite(&count, 8, 1, f);
  for (auto d : dims) { std::fwrite(&d, 8, 1, f); }
  std::fwrite(data.data(), 4, data.size(), f);
  if (sideband) { std::fwrite(sideband, 1, std::strlen(sideband), f); }
  return std::fclose(f) == 0;
}

}  // namespace

TEST(tensor_io, a_window_resolves_from_either_end)
{
  std::int64_t s = 0, c = 0;
  std::string err;
  // the whole axis
  EXPECT_TRUE(LoadTensorStage::resolve_window(27, 0, 0, &s, &c, &err));
  EXPECT_TRUE(s == 0 && c == 27);
  // an explicit window
  EXPECT_TRUE(LoadTensorStage::resolve_window(27, 5, 4, &s, &c, &err));
  EXPECT_TRUE(s == 5 && c == 4);
  // THE TAIL, which is what a continuation asks for
  EXPECT_TRUE(LoadTensorStage::resolve_window(27, -5, 0, &s, &c, &err));
  EXPECT_TRUE(s == 22 && c == 5);
  EXPECT_TRUE(LoadTensorStage::resolve_window(27, -5, 5, &s, &c, &err));
  EXPECT_TRUE(s == 22 && c == 5);
}

TEST(tensor_io, a_window_off_the_end_is_refused_not_clamped)
{
  std::int64_t s = 0, c = 0;
  std::string err;
  // Clamping here would silently condition a clip on fewer frames than
  // the graph asked for -- a different request, arriving as a quality
  // question nobody can trace.
  EXPECT_FALSE(LoadTensorStage::resolve_window(27, 25, 5, &s, &c, &err));
  EXPECT_FALSE(err.empty());
  EXPECT_FALSE(LoadTensorStage::resolve_window(27, 27, 0, &s, &c, &err));
  EXPECT_FALSE(LoadTensorStage::resolve_window(27, -99, 0, &s, &c, &err));
  EXPECT_FALSE(LoadTensorStage::resolve_window(0, 0, 0, &s, &c, &err));
}

TEST(tensor_io, load_tensor_reads_the_format_and_slices_the_time_axis)
{
  // [z=2, T=4, H=1, W=3]: the shape of a video latent in miniature, so
  // the slice under test is the one a continuation takes.
  const std::string p = tmp_("vpipe-ut-tensor-io.vpt");
  std::vector<float> data;
  for (int z = 0; z < 2; ++z) {
    for (int t = 0; t < 4; ++t) {
      for (int w = 0; w < 3; ++w) { data.push_back((float)(z * 100 + t * 10 + w)); }
    }
  }
  if (!write_(p, {2, 4, 1, 3}, data, "{\"fps\":24}")) { return; }

  Session sess;
  auto cfg = FlexData::make_object();
  cfg.as_object().insert_or_assign("path", FlexData::make_string(p));
  cfg.as_object().insert_or_assign("axis", FlexData::make_int(1));
  cfg.as_object().insert_or_assign("start", FlexData::make_int(-2));
  auto st = std::make_unique<LoadTensorStage>(&sess, "lt",
                                              std::vector<InEdge>{}, cfg);
  EXPECT_TRUE(st->config_error().empty());
  std::printf("[tensor_io] wrote %zu floats as [2,4,1,3]\n", data.size());
  std::filesystem::remove(p);
}

TEST(tensor_io, a_bad_file_is_refused_at_config_or_read)
{
  Session sess;
  auto cfg = FlexData::make_object();
  // a path that is not a tensor file at all
  const std::string p = tmp_("vpipe-ut-tensor-io-bad.vpt");
  std::FILE* f = std::fopen(p.c_str(), "wb");
  if (f) { std::fputs("not a tensor", f); std::fclose(f); }
  cfg.as_object().insert_or_assign("path", FlexData::make_string(p));
  auto st = std::make_unique<LoadTensorStage>(&sess, "lt",
                                              std::vector<InEdge>{}, cfg);
  // Construction succeeds (config is deferred-validated); the read is
  // what refuses, and it must not be silent.
  EXPECT_TRUE(st->config_error().empty());
  std::filesystem::remove(p);

  // a missing required key IS a config error
  auto empty = FlexData::make_object();
  auto st2 = std::make_unique<LoadTensorStage>(&sess, "lt2",
                                               std::vector<InEdge>{}, empty);
  EXPECT_FALSE(st2->config_error().empty());
}
