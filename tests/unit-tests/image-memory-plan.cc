// Does a streaming image DiT declare a floor the RESOURCE PHASE can see?
//
// There are two ledgers and they take the same number for different
// purposes. StageMemory::hold(source, preload, floor) is what a run
// REPORTS; ResourceClaim::floor_bytes is what phase_footprint_floor()
// and phase_peak() read, and those are the only place a graph is turned
// away. A DiT claimed without one is judged at its full on-disk size
// however carefully the plan describes its floor.
//
// Gated on VPIPE_FLUX2_TEST_MODEL_PATH because it needs a real
// checkpoint's tensor table, but nothing here is FLUX.2-specific: point
// it at any generate-image family and the same assertions hold. It
// loads no weights.
//
// Env: VPIPE_FLUX2_TEST_MODEL_PATH = an image model root (a directory
// holding transformer/).

#include "minitest.h"

#include "common/flex-data.h"
#include "common/session.h"
#include "pipeline/pipeline.h"
#include "generative-models/vosr/metal-vosr-transformer.h"
#include "stages/generate-image-stage.h"
#include "stages/model-memory.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

using namespace vpipe;

// THE DiT'S STREAMING FLOOR HAS TO REACH THE LEDGER THAT REFUSES.
//
// There are two, and they take the same number for different purposes.
// declare_memory()'s floor is what a run REPORTS. The floor on a
// ResourceClaim is what phase_footprint_floor() and phase_peak() read,
// and those are the only place a graph is turned away. A DiT claimed
// without one is judged at its full on-disk size -- so "everything
// streamable at its floor" reads identically to "everything preloaded",
// and a 17 GB checkpoint that runs on a couple of blocks is weighed as
// 17 GB against the box.
//
// Both are asserted here, and asserted EQUAL, because two ledgers fed
// from two copies of a block-stem list is exactly how they drift.
// Neither loads anything: both are answered from the tensor table.
TEST(image_memory_plan, the_dit_is_claimed_at_its_streaming_floor)
{
  const char* root = std::getenv("VPIPE_FLUX2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;
  namespace fs = std::filesystem;
  const std::string dit = (fs::path(root) / "transformer").string();
  const std::size_t whole = model_memory::dir_weights_bytes(dit);
  if (whole == 0) { return; }

  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root));
  GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                           std::move(cfg));
  ASSERT_TRUE(stage.config_error().empty());

  std::size_t claim_floor = 0;
  bool found = false;
  for (const ResourceClaim& c : stage.declare_resources()) {
    if (c.key == dit) { claim_floor = c.floor_bytes; found = true; }
  }
  ASSERT_TRUE(found);
  std::printf("[image_memory_plan] DiT %zu MB on disk, claimed floor %zu MB\n",
              whole >> 20, claim_floor >> 20);
  // A floor of zero is not "unknown" to the planner -- it is "this
  // cannot be reduced", which is the whole bug.
  EXPECT_TRUE(claim_floor > 0);
  EXPECT_TRUE(claim_floor < whole);

  std::size_t plan_floor = 0;
  for (const auto& h : stage.declare_memory().holdings) {
    if (h.source == dit) { plan_floor = h.floor; }
  }
  EXPECT_TRUE(plan_floor == claim_floor);
}

namespace {

// A real safetensors file of F32 zeros: an 8-byte little-endian header
// length, the JSON table, the data. Enough for the tensor table the
// planner reads, and nothing is ever loaded from it.
void
write_safetensors_(const std::filesystem::path& p,
                   const std::vector<std::pair<std::string, std::size_t>>& t)
{
  std::string hdr = "{";
  std::size_t off = 0;
  for (std::size_t i = 0; i < t.size(); ++i) {
    if (i != 0) { hdr += ","; }
    hdr += "\"" + t[i].first + "\":{\"dtype\":\"F32\",\"shape\":[" +
           std::to_string(t[i].second) + "],\"data_offsets\":[" +
           std::to_string(off * 4) + "," +
           std::to_string((off + t[i].second) * 4) + "]}";
    off += t[i].second;
  }
  hdr += "}";
  while (hdr.size() % 8 != 0) { hdr += ' '; }
  std::ofstream f(p, std::ios::binary);
  const std::uint64_t len = hdr.size();
  f.write(reinterpret_cast<const char*>(&len), sizeof(len));
  f.write(hdr.data(), (std::streamsize)hdr.size());
  const std::vector<float> zeros(off, 0.0f);
  f.write(reinterpret_cast<const char*>(zeros.data()),
          (std::streamsize)(zeros.size() * sizeof(float)));
}

}  // namespace

// VOSR IS NOT CLAIMED STREAMABLE, because it cannot stream.
//
// Its checkpoint names its blocks `blocks.N.`, which the image stage's
// single-stack stem matches, so a floor measured from the names says
// "trunk plus two blocks" -- 417 MB for VOSR 2.0 -- while
// MetalVosrTransformer copies all 36 blocks in, ~2659 MB. A floor is a
// promise, and that one would admit graphs the box could not hold. The
// published `ema_model.safetensors` escaped by accident (the directory
// opens nothing, so the floor measured 0 -- see the real-checkpoint arm
// below); a checkpoint named `model.safetensors`, which the loader also
// accepts, did not. Both ledgers must read "cannot be reduced": a claim
// floor of zero and a holding floor of zero, the holding at full size.
//
// No model needed: a two-block VOSR-shaped root in a temp directory is
// enough, and the test first proves the stem WOULD have matched it.
TEST(image_memory_plan, vosr_is_not_claimed_streamable)
{
  namespace fs = std::filesystem;
  const fs::path root = fs::temp_directory_path() / "vpipe-vosr-floor";
  std::error_code ec;
  fs::remove_all(root, ec);
  fs::create_directories(root / "checkpoints", ec);
  {
    std::ofstream a(root / "args.json");
    a << "{\"ae_type\": \"qwen\", \"distill_type\": \"onestep\"}";
  }
  // Named model.safetensors so the tensor table opens: open_model() does
  // not open a directory holding only a freely-named file (the published
  // `ema_model.safetensors`), which is why the real checkpoint measures a
  // floor of 0 today by accident rather than by declaration.
  write_safetensors_(root / "checkpoints" / "model.safetensors",
                     {{"blocks.0.attn.weight", 4096},
                      {"blocks.1.attn.weight", 4096},
                      {"x_embedder.weight", 1024}});
  const std::string dit = (root / "checkpoints").string();
  const std::size_t whole = model_memory::dir_weights_bytes(dit);
  ASSERT_TRUE(whole > 0);
  // Not vacuous: measured by names alone, this checkpoint has a floor.
  const std::vector<std::string_view> stems = {"blocks."};
  EXPECT_TRUE(model_memory::streaming_floor_bytes(dit, stems) > 0);

  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(root.string()));
  GenerateImageStage stage(&sess, "restore", std::vector<InEdge>{},
                           std::move(cfg));

  bool claimed = false;
  std::size_t claim_floor = 1;
  for (const ResourceClaim& c : stage.declare_resources()) {
    if (c.key == dit) { claimed = true; claim_floor = c.floor_bytes; }
  }
  EXPECT_TRUE(claimed);
  EXPECT_TRUE(claim_floor == 0);

  bool held = false;
  std::size_t plan_floor = 1, plan_preload = 0;
  for (const auto& h : stage.declare_memory().holdings) {
    if (h.source == dit) {
      held = true;
      plan_floor = h.floor;
      plan_preload = h.preload;
    }
  }
  EXPECT_TRUE(held);
  EXPECT_TRUE(plan_floor == 0);
  EXPECT_TRUE(plan_preload == whole);
  std::printf("[image_memory_plan] VOSR-shaped root: %zu B on disk, claim "
              "floor %zu, holding floor %zu\n", whole, claim_floor,
              plan_floor);
  fs::remove_all(root, ec);
}

// The same question on the REAL checkpoint, printed rather than assumed.
// VPIPE_VOSR_TEST_MODEL_PATH = the VOSR root (holding VOSR2/).
TEST(image_memory_plan, vosr_real_checkpoint_floor)
{
  const char* root = std::getenv("VPIPE_VOSR_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  namespace fs = std::filesystem;
  std::string vroot = root;
  if (fs::is_directory(fs::path(root) / "VOSR2")) {
    vroot = (fs::path(root) / "VOSR2").string();
  }
  const std::string dit = genai::MetalVosrTransformer::weights_dir(vroot);
  ASSERT_TRUE(!dit.empty());
  if (dit.empty()) { return; }
  const std::vector<std::string_view> stems = {"blocks."};
  const std::size_t by_names = model_memory::streaming_floor_bytes(dit, stems);
  const std::size_t whole = model_memory::dir_weights_bytes(dit);

  Session sess;
  FlexData cfg = FlexData::make_object();
  cfg.as_object().insert("hf_dir", FlexData::make_string(vroot));
  GenerateImageStage stage(&sess, "restore", std::vector<InEdge>{},
                           std::move(cfg));
  std::size_t claim_floor = 1, plan_floor = 1;
  for (const ResourceClaim& c : stage.declare_resources()) {
    if (c.key == dit) { claim_floor = c.floor_bytes; }
  }
  for (const auto& h : stage.declare_memory().holdings) {
    if (h.source == dit) { plan_floor = h.floor; }
  }
  std::printf("[image_memory_plan] real VOSR %s: %zu MB on disk, floor by "
              "block names %zu MB, declared claim floor %zu, holding floor "
              "%zu\n", dit.c_str(), whole >> 20, by_names >> 20, claim_floor,
              plan_floor);
  EXPECT_TRUE(claim_floor == 0);
  EXPECT_TRUE(plan_floor == 0);
}
