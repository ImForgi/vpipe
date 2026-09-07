// A STANDALONE, natively-named Qwen-Image VAE: wikeeyang's Krea2-HD.
//
// The Krea-2 / Qwen-Image VAE is published under two spellings of one
// architecture (see generative-models/shared/wan-vae-names.h), and this
// checkpoint uses the native one -- decoder.upsamples.4.residual.2 where
// the loaders read decoder.up_blocks.1.resnets.0.conv1. The map that
// reconciles them is STRUCTURAL, and a wrong index in it is silent: every
// tensor at a level shares its shape, so a mis-grouped resnet loads
// cleanly and decodes a plausible, wrong image.
//
// So the map is pinned twice. The first three tests pin the RULE with no
// weights and no GPU. The last two pin it NUMERICALLY, and they work
// because of what this checkpoint is: a DECODER fine-tune whose encoder
// is all but untouched (measured tensor for tensor against krea/
// Krea-2-Turbo: encoder median rel-L2 0.010, decoder median 0.83). So
// encoding through it must agree closely with the stock VAE -- a wrong
// name map cannot -- while decoding must NOT, which is what says the HD
// weights are the ones running.
//
// Env: VPIPE_KREA2_HD_VAE_TEST_PATH = the standalone VAE dir (the
// safetensors + the config.json companion), VPIPE_KREA2_TEST_MODEL_PATH =
// the Krea-2-Turbo root for the comparison arm. Skips if unset.

#include "minitest.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/session.h"
#include "generative-models/krea2/metal-krea2-vae.h"
#include "generative-models/llama3/metal-llama-weights.h"
#include "generative-models/shared/wan-vae-names.h"
#include "common/flex-data.h"
#include "stages/model-detect.h"
#include "stages/model-register-stage.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace vpipe;
using namespace vpipe::genai;
using metal_compute::MetalCompute;
using metal_compute::SharedBuffer;

namespace {

double
rel_l2_(const std::vector<float>& a, const std::vector<float>& b)
{
  if (a.size() != b.size() || a.empty()) { return 1e30; }
  double num = 0.0, den = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    const double d = (double)a[i] - (double)b[i];
    num += d * d;
    den += (double)b[i] * (double)b[i];
  }
  return (den > 0.0) ? std::sqrt(num / den) : std::sqrt(num);
}

// A COMPLETE `decoder.upsamples` skeleton plus one name per structural
// case the rule has to get right.
//
// Complete on purpose: the decoder's blocks are recovered by WALKING the
// module list -- an entry carrying `resample`/`time_conv` is the block's
// upsampler and ends it -- so a list with a hole in it would shift every
// later entry into the wrong block. build_name_map refuses a
// non-contiguous list rather than grouping one, and this list is what
// that refusal is defined against.
//
// Indices 3 and 11 are the two shapes of upsampler (with a time_conv and
// without); 12..14 are the LAST up_block, which has no upsampler after
// it and is where an arithmetic stride would need to be told about the
// exception.
std::vector<std::string>
native_names_()
{
  std::vector<std::string> n = {
    "conv1.weight", "conv1.bias",
    "conv2.weight", "conv2.bias",
    "encoder.conv1.weight",
    "encoder.downsamples.0.residual.0.gamma",
    "encoder.downsamples.0.residual.2.weight",
    "encoder.downsamples.0.residual.3.gamma",
    "encoder.downsamples.0.residual.6.bias",
    "encoder.downsamples.2.resample.1.weight",
    "encoder.downsamples.3.shortcut.weight",
    "encoder.downsamples.5.time_conv.bias",
    "encoder.middle.0.residual.2.weight",
    "encoder.middle.1.to_qkv.weight",
    "encoder.middle.1.norm.gamma",
    "encoder.middle.2.residual.6.weight",
    "encoder.head.0.gamma",
    "encoder.head.2.weight",
    "decoder.conv1.weight",
    "decoder.middle.0.residual.2.weight",
    "decoder.middle.1.proj.bias",
    "decoder.middle.2.residual.6.weight",
    "decoder.head.0.gamma",
    "decoder.head.2.bias",
  };
  // upsamples 0..14: three resnets then an upsampler, three times, then
  // three resnets. The extras below name the tensors the assertions
  // reach for; every other index contributes one, which is all the
  // grouping needs to see.
  for (int i = 0; i < 15; ++i) {
    const bool is_up = (i == 3 || i == 7 || i == 11);
    n.push_back("decoder.upsamples." + std::to_string(i) +
                (is_up ? ".resample.1.weight" : ".residual.2.weight"));
  }
  n.push_back("decoder.upsamples.3.time_conv.weight");
  n.push_back("decoder.upsamples.4.shortcut.bias");
  n.push_back("decoder.upsamples.12.residual.0.gamma");
  n.push_back("decoder.upsamples.14.residual.6.bias");
  return n;
}

}  // namespace

// The rule, case by case, with no checkpoint in sight.
TEST(krea2_hd_vae, native_names_map_to_the_diffusers_spelling)
{
  std::unordered_map<std::string, std::string> map;
  std::string err;
  const bool ok = wan_vae::build_name_map(native_names_(), map, &err);
  ASSERT_TRUE(ok);
  if (!ok) {
    std::printf("[krea2_hd_vae] build_name_map: %s\n", err.c_str());
    return;
  }
  // The map is keyed the way the LOADER asks -- diffusers name in, the
  // checkpoint's own name out.
  const std::pair<const char*, const char*> want[] = {
    {"quant_conv.weight",                          "conv1.weight"},
    {"post_quant_conv.bias",                       "conv2.bias"},
    {"encoder.conv_in.weight",                     "encoder.conv1.weight"},
    {"encoder.down_blocks.0.norm1.gamma",
     "encoder.downsamples.0.residual.0.gamma"},
    {"encoder.down_blocks.0.conv1.weight",
     "encoder.downsamples.0.residual.2.weight"},
    {"encoder.down_blocks.0.norm2.gamma",
     "encoder.downsamples.0.residual.3.gamma"},
    {"encoder.down_blocks.0.conv2.bias",
     "encoder.downsamples.0.residual.6.bias"},
    // The encoder's list is NOT regrouped: same index, new member names.
    {"encoder.down_blocks.2.resample.1.weight",
     "encoder.downsamples.2.resample.1.weight"},
    {"encoder.down_blocks.3.conv_shortcut.weight",
     "encoder.downsamples.3.shortcut.weight"},
    {"encoder.down_blocks.5.time_conv.bias",
     "encoder.downsamples.5.time_conv.bias"},
    {"encoder.mid_block.resnets.0.conv1.weight",
     "encoder.middle.0.residual.2.weight"},
    {"encoder.mid_block.attentions.0.to_qkv.weight",
     "encoder.middle.1.to_qkv.weight"},
    {"encoder.mid_block.resnets.1.conv2.weight",
     "encoder.middle.2.residual.6.weight"},
    {"encoder.norm_out.gamma",                     "encoder.head.0.gamma"},
    {"encoder.conv_out.weight",                    "encoder.head.2.weight"},
    {"decoder.conv_in.weight",                     "decoder.conv1.weight"},
    // The decoder's list IS regrouped, and this is the whole of it.
    {"decoder.up_blocks.0.resnets.0.conv1.weight",
     "decoder.upsamples.0.residual.2.weight"},
    {"decoder.up_blocks.0.upsamplers.0.resample.1.weight",
     "decoder.upsamples.3.resample.1.weight"},
    {"decoder.up_blocks.0.upsamplers.0.time_conv.weight",
     "decoder.upsamples.3.time_conv.weight"},
    {"decoder.up_blocks.1.resnets.0.conv1.weight",
     "decoder.upsamples.4.residual.2.weight"},
    {"decoder.up_blocks.1.resnets.0.conv_shortcut.bias",
     "decoder.upsamples.4.shortcut.bias"},
    {"decoder.up_blocks.2.upsamplers.0.resample.1.weight",
     "decoder.upsamples.11.resample.1.weight"},
    {"decoder.up_blocks.3.resnets.0.norm1.gamma",
     "decoder.upsamples.12.residual.0.gamma"},
    {"decoder.up_blocks.3.resnets.2.conv2.bias",
     "decoder.upsamples.14.residual.6.bias"},
    {"decoder.norm_out.gamma",                     "decoder.head.0.gamma"},
    {"decoder.conv_out.bias",                      "decoder.head.2.bias"},
  };
  for (const auto& [diffusers, native] : want) {
    const auto it = map.find(diffusers);
    const bool found = it != map.end();
    EXPECT_TRUE(found);
    if (!found) {
      std::printf("[krea2_hd_vae] MISSING '%s'\n", diffusers);
      continue;
    }
    const bool same = it->second == native;
    EXPECT_TRUE(same);
    if (!same) {
      std::printf("[krea2_hd_vae] '%s' -> '%s', wanted '%s'\n",
                  diffusers, it->second.c_str(), native);
    }
  }
  // ...and nothing else, so a name that quietly failed to map cannot
  // hide behind the ones that did.
  EXPECT_TRUE(map.size() == native_names_().size());
}

// A diffusers checkpoint gets NO map, and the identity that follows is
// what leaves every existing caller on the path it was already on.
TEST(krea2_hd_vae, a_diffusers_checkpoint_is_left_alone)
{
  const std::vector<std::string> names = {
    "decoder.conv_in.weight",
    "decoder.up_blocks.1.resnets.0.conv1.weight",
    "quant_conv.weight",
  };
  std::unordered_map<std::string, std::string> map;
  std::string err;
  EXPECT_TRUE(wan_vae::build_name_map(names, map, &err));
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(wan_vae::detect_layout(names) == wan_vae::NameLayout::kDiffusers);
  // The accessor the loaders call then hands the name straight back.
  EXPECT_TRUE(wan_vae::resolve(map, "decoder.conv_in.weight") ==
              "decoder.conv_in.weight");
}

// A native tensor the rule cannot place REFUSES the whole map. A partial
// one would leave the unmapped tensor reading a name that is not in the
// file -- an empty buffer, at a level where the loader's `ok` chain may
// not notice, and half a decoder is not a slower answer but a wrong one.
TEST(krea2_hd_vae, an_unmappable_native_name_refuses_the_map)
{
  std::vector<std::string> names = native_names_();
  names.push_back("decoder.upsamples.7.residual.9.weight");   // no such slot
  std::unordered_map<std::string, std::string> map;
  std::string err;
  EXPECT_TRUE(!wan_vae::build_name_map(names, map, &err));
  EXPECT_TRUE(map.empty());
  EXPECT_TRUE(!err.empty());
  std::printf("[krea2_hd_vae] refusal: %s\n", err.c_str());
}

// The RELEASED checkpoint, mapped one-to-one and in full.
TEST(krea2_hd_vae, the_released_checkpoint_maps_one_to_one)
{
  const char* dir = std::getenv("VPIPE_KREA2_HD_VAE_TEST_PATH");
  if (dir == nullptr || *dir == '\0') { return; }
  // Through the resolver the VAE stages use: a standalone checkpoint is
  // one freely-named safetensors in a directory nothing globs, so what
  // gets opened is the FILE.
  const std::string wpath = resolve_vae_weights_path(dir);
  std::printf("[krea2_hd_vae] weights at '%s'\n", wpath.c_str());
  EXPECT_TRUE(wpath != std::string(dir));
  auto wts = MetalLlamaWeights::open_model(wpath);
  ASSERT_TRUE(wts.has_value());
  if (!wts.has_value()) { return; }

  const std::vector<std::string> names = wts->tensor_names();
  EXPECT_TRUE(wan_vae::detect_layout(names) == wan_vae::NameLayout::kNative);

  std::unordered_map<std::string, std::string> map;
  std::string err;
  const bool ok = wan_vae::build_name_map(names, map, &err);
  ASSERT_TRUE(ok);
  if (!ok) {
    std::printf("[krea2_hd_vae] build_name_map: %s\n", err.c_str());
    return;
  }
  std::printf("[krea2_hd_vae] %zu tensors, %zu mapped\n", names.size(),
              map.size());
  EXPECT_TRUE(map.size() == names.size());
  // Every mapped name resolves to a tensor that is actually there. This
  // is the direction that matters: the loader asks with the diffusers
  // name, and what comes back has to name real bytes.
  std::size_t missing = 0;
  for (const auto& [diffusers, native] : map) {
    if (wts->info(native) == nullptr) {
      if (missing < 4) {
        std::printf("[krea2_hd_vae] '%s' -> '%s' is not in the file\n",
                    diffusers.c_str(), native.c_str());
      }
      ++missing;
    }
  }
  EXPECT_TRUE(missing == 0);
}

// WHAT GETS OPENED, for the three directory shapes that matter. This is
// the rule the claim, the phase release and the WeightSet all share, so
// getting it wrong is not a load failure but three bookkeeping no-ops
// against a name the manager never heard of.
TEST(krea2_hd_vae, a_standalone_checkpoint_resolves_to_its_file)
{
  namespace fs = std::filesystem;
  const fs::path base =
      fs::temp_directory_path() / "vpipe-krea2-hd-vae-resolve";
  std::error_code ec;
  fs::remove_all(base, ec);
  auto touch = [](const fs::path& p) {
    fs::create_directories(p.parent_path());
    std::ofstream(p) << "x";
  };

  // One freely-named safetensors: the file, because nothing globs it.
  const fs::path one = base / "one";
  touch(one / "Krea2-HD-vae.safetensors");
  touch(one / "config.json");
  EXPECT_TRUE(resolve_vae_weights_path(one.string()) ==
              (one / "Krea2-HD-vae.safetensors").string());

  // A diffusers component: unchanged, because open_model globs the
  // directory itself and always has.
  const fs::path dif = base / "diffusers";
  touch(dif / "diffusion_pytorch_model.safetensors");
  touch(dif / "config.json");
  EXPECT_TRUE(resolve_vae_weights_path(dif.string()) == dif.string());

  // Two freely-named safetensors: unchanged, so the caller's own "no
  // readable checkpoint" fires. Picking by sort order would load one VAE
  // under a config describing the other -- which decodes.
  const fs::path two = base / "two";
  touch(two / "a-vae.safetensors");
  touch(two / "b-vae.safetensors");
  EXPECT_TRUE(resolve_vae_weights_path(two.string()) == two.string());

  fs::remove_all(base, ec);
}

// WHERE THE LATENT STATISTICS COME FROM. A standalone VAE ships none,
// and they cannot be defaulted -- the wrong sixteen numbers decode a
// plausible, wrongly-graded picture rather than failing -- so the order
// the two sources are tried in is the whole of the contract.
TEST(krea2_hd_vae, latent_statistics_come_from_the_parent_when_absent)
{
  namespace fs = std::filesystem;
  const fs::path base =
      fs::temp_directory_path() / "vpipe-krea2-hd-vae-config";
  std::error_code ec;
  fs::remove_all(base, ec);
  const fs::path vae    = base / "standalone-vae";
  const fs::path parent = base / "krea-2-turbo";
  fs::create_directories(vae);
  fs::create_directories(parent / "vae");
  // Enough of a Qwen-Image VAE config to be recognised as one.
  std::ofstream(parent / "vae" / "config.json")
      << R"({"_class_name":"AutoencoderKLQwenImage","z_dim":16})";
  std::ofstream(vae / "Krea2-HD-vae.safetensors") << "x";

  Session sess(R"({"db":{"path":")" + (base / "db").string() + R"("}})");

  // NOTHING to borrow yet: no config here, no parent registered. Empty
  // is the honest answer, and the stage refuses on it rather than
  // inventing statistics.
  EXPECT_TRUE(resolve_vae_config_path(&sess, "wikeeyang/Krea2-Turbo-HD-V1",
                                      vae.string(),
                                      "AutoencoderKLQwenImage").empty());

  // Register both: the VAE under its catalogue key (so its model_type is
  // krea2-vae and its parent link is readable) and a Krea-2 to borrow
  // from.
  //
  // model_type is set EXPLICITLY rather than detected: these are temp
  // directories, so neither the catalogue-by-name nor the
  // catalogue-by-repo-path probe fires and both would register as "type
  // unknown". A real fetch lands under <org>/<repo> and is typed from
  // the catalogue -- which is what the end-to-end run exercises.
  auto reg = [&](const fs::path& dir, const char* key, const char* mt) {
    FlexData c = FlexData::make_object();
    auto o = c.as_object();
    o.insert_or_assign("model_dir", FlexData::make_string(dir.string()));
    o.insert_or_assign("key", FlexData::make_string(std::string(key)));
    o.insert_or_assign("model_type", FlexData::make_string(std::string(mt)));
    ModelRegisterStage st(&sess, std::string("reg-") + key, {}, std::move(c));
    st.register_once();
  };
  reg(parent, "krea/Krea-2-Turbo", "krea2");
  reg(vae, "wikeeyang/Krea2-Turbo-HD-V1", "krea2-vae");

  // NOW the parent answers, in place -- no copy, no download.
  const std::string got = resolve_vae_config_path(
      &sess, "wikeeyang/Krea2-Turbo-HD-V1", vae.string(),
      "AutoencoderKLQwenImage");
  std::printf("[krea2_hd_vae] borrowed config: '%s'\n", got.c_str());
  EXPECT_TRUE(got == (parent / "vae" / "config.json").string());

  // A parent whose VAE is a DIFFERENT architecture is skipped, not read.
  // Wan and Qwen-Image share this VAE's shapes and have entirely
  // different latents_mean/std, so this is the one way the fallback
  // could produce a confident wrong answer.
  EXPECT_TRUE(resolve_vae_config_path(&sess, "wikeeyang/Krea2-Turbo-HD-V1",
                                      vae.string(),
                                      "AutoencoderKLWan").empty());

  // The checkpoint's OWN file wins over any parent: a file someone put
  // there is a statement, the parent is an inference.
  std::ofstream(vae / "config.json")
      << R"({"_class_name":"AutoencoderKLQwenImage","z_dim":16})";
  EXPECT_TRUE(resolve_vae_config_path(&sess, "wikeeyang/Krea2-Turbo-HD-V1",
                                      vae.string(),
                                      "AutoencoderKLQwenImage") ==
              (vae / "config.json").string());

  fs::remove_all(base, ec);
}

// THE NUMERICAL BAR ON THE MAP. The HD checkpoint's encoder is the stock
// Qwen-Image encoder to within a light fine-tune, so the two must encode
// the same picture to nearly the same latent. Under a wrong map they
// could not: the negative control is one directory away, in that the
// same comparison over the DECODER (below) is a different number by two
// orders of magnitude.
TEST(krea2_hd_vae, the_hd_encoder_agrees_with_the_stock_encoder)
{
  const char* hd   = std::getenv("VPIPE_KREA2_HD_VAE_TEST_PATH");
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (hd == nullptr || *hd == '\0' || root == nullptr || *root == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalKrea2Vae::Config cfg;                  // Qwen-Image VAE defaults
  cfg.latents_mean = {-0.7571f, -0.7089f, -0.9113f,  0.1075f,
                      -0.1745f,  0.9653f, -0.1517f,  1.5508f,
                       0.4134f, -0.0715f,  0.5517f, -0.3632f,
                      -0.1922f, -0.9497f,  0.2503f, -0.2921f};
  cfg.latents_std  = { 2.8184f,  1.4541f,  2.3275f,  2.6558f,
                       1.2196f,  1.7708f,  2.6052f,  2.0743f,
                       3.2687f,  2.1526f,  2.8652f,  1.5579f,
                       1.6382f,  1.1253f,  2.8251f,  1.9160f};

  auto v_hd = MetalKrea2Vae::load(resolve_vae_weights_path(hd), mc, cfg,
                                  /*with_encoder=*/true);
  auto v_st = MetalKrea2Vae::load(std::string(root) + "/vae", mc, cfg,
                                  /*with_encoder=*/true);
  ASSERT_TRUE(v_hd != nullptr);
  ASSERT_TRUE(v_st != nullptr);
  if (!v_hd || !v_st) { return; }
  EXPECT_TRUE(v_hd->has_encoder());
  EXPECT_TRUE(v_st->has_encoder());

  // A deterministic picture with real spatial structure -- flat noise
  // would put most of the encoder's response in one band.
  const int H = 256, W = 256;
  const std::size_t n = (std::size_t)3 * H * W;
  SharedBuffer img = mc->make_shared_buffer(n * 2);
  ASSERT_TRUE(!img.empty());
  if (img.empty()) { return; }
  {
    auto* d = static_cast<_Float16*>(img.contents());
    for (int c = 0; c < 3; ++c) {
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          const float v =
              0.6f * std::sin((float)x * 0.11f + (float)c) *
                     std::cos((float)y * 0.07f) +
              0.3f * (float)((x / 16 + y / 16 + c) % 2) - 0.15f;
          d[((std::size_t)c * H + y) * W + x] = (_Float16)v;
        }
      }
    }
  }

  SharedBuffer z_hd = v_hd->encode(img, H, W);
  SharedBuffer z_st = v_st->encode(img, H, W);
  ASSERT_TRUE(!z_hd.empty() && !z_st.empty());
  if (z_hd.empty() || z_st.empty()) { return; }

  const std::size_t nz =
      (std::size_t)cfg.z_dim * (std::size_t)(H / 8) * (std::size_t)(W / 8);
  ASSERT_TRUE(z_hd.byte_size() >= nz * 2 && z_st.byte_size() >= nz * 2);
  if (z_hd.byte_size() < nz * 2 || z_st.byte_size() < nz * 2) { return; }
  std::vector<float> a(nz), b(nz);
  {
    const auto* pa = static_cast<const _Float16*>(z_hd.contents());
    const auto* pb = static_cast<const _Float16*>(z_st.contents());
    for (std::size_t i = 0; i < nz; ++i) {
      a[i] = (float)pa[i];
      b[i] = (float)pb[i];
    }
  }
  const double r = rel_l2_(a, b);
  std::printf("[krea2_hd_vae] encode HD vs stock rel-L2 = %.6g\n", r);
  // The bar is the fine-tune's own size, not equality: the encoder's
  // weights DID move (median rel-L2 0.010 against the stock ones), so
  // the latents differ a little and must not differ a lot. A wrong name
  // map lands far outside this, and so does an unloaded encoder.
  EXPECT_TRUE(r < 0.15);
  EXPECT_TRUE(r > 0.0);
}

// THE OTHER HALF, and the one that says the HD weights are running at
// all: the same latent decoded through both must NOT agree. The decoder
// is where this checkpoint's fine-tune lives (median rel-L2 0.83 tensor
// for tensor), so a small number here would mean the loader had somehow
// fallen back to stock weights -- which is exactly what a silently
// half-applied name map looks like from the outside.
TEST(krea2_hd_vae, the_hd_decoder_is_not_the_stock_decoder)
{
  const char* hd   = std::getenv("VPIPE_KREA2_HD_VAE_TEST_PATH");
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (hd == nullptr || *hd == '\0' || root == nullptr || *root == '\0') {
    return;
  }
  Session sess;
  MetalCompute* mc = sess.metal_compute();
  if (mc == nullptr) { return; }

  MetalKrea2Vae::Config cfg;
  const int Cz = cfg.z_dim, h8 = 32, w8 = 32;      // -> 256x256
  const std::size_t hw = (std::size_t)h8 * w8;
  std::vector<float> lat((std::size_t)Cz * hw);
  std::uint32_t s = 0x9e3779b9u;
  for (auto& v : lat) {
    s = s * 1664525u + 1013904223u;
    v = ((float)(s >> 8) / 8388608.0f - 1.0f) * 3.0f;
  }
  auto make_z = [&]() {
    SharedBuffer z = mc->make_shared_buffer(lat.size() * 2);
    if (z.empty()) { return z; }
    auto* d = static_cast<_Float16*>(z.contents());
    for (std::size_t i = 0; i < lat.size(); ++i) { d[i] = (_Float16)lat[i]; }
    return z;
  };

  auto v_hd = MetalKrea2Vae::load(resolve_vae_weights_path(hd), mc, cfg);
  auto v_st = MetalKrea2Vae::load(std::string(root) + "/vae", mc, cfg);
  ASSERT_TRUE(v_hd != nullptr);
  ASSERT_TRUE(v_st != nullptr);
  if (!v_hd || !v_st) { return; }

  const int H = h8 * 8, W = w8 * 8;
  const std::size_t n = (std::size_t)3 * H * W;
  SharedBuffer o_hd = v_hd->decode(make_z(), h8, w8);
  SharedBuffer o_st = v_st->decode(make_z(), h8, w8);
  ASSERT_TRUE(!o_hd.empty() && o_hd.byte_size() >= n * 2);
  ASSERT_TRUE(!o_st.empty() && o_st.byte_size() >= n * 2);
  if (o_hd.byte_size() < n * 2 || o_st.byte_size() < n * 2) { return; }

  std::vector<float> a(n), b(n);
  const auto* pa = static_cast<const _Float16*>(o_hd.contents());
  const auto* pb = static_cast<const _Float16*>(o_st.contents());
  bool finite = true;
  for (std::size_t i = 0; i < n; ++i) {
    a[i] = (float)pa[i];
    b[i] = (float)pb[i];
    if (!std::isfinite(a[i])) { finite = false; }
  }
  const double r = rel_l2_(a, b);
  std::printf("[krea2_hd_vae] decode HD vs stock rel-L2 = %.6g (%dx%d)\n",
              r, H, W);
  EXPECT_TRUE(finite);
  EXPECT_TRUE(r > 0.05);
}
