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
#include <iterator>
#include "pipeline/resource-plan.h"
#include "generative-models/generative-model-manager.h"

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

// The ANE tier's memory is CoreML's, so the only thing that can keep it
// honest is the claim. Two halves, and both have to hold: the stage
// claims when the tier is on, and does NOT when it is off -- a claim for
// bytes nothing allocates is the same lie pointing the other way.
TEST(image_memory_plan, the_ane_tier_claims_coreml_residency)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  Session sess;

  auto claims_of = [&](bool ane) {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    if (ane) {
      cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    }
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    std::size_t bytes = 0;
    int units = 0;
    for (const ResourceClaim& c : stage.declare_resources()) {
      if (c.kind != model_memory::kCoreMLKind) { continue; }
      // "<label>|<unit_bytes>|<units>"
      const std::size_t b2 = c.key.rfind('|');
      const std::size_t b1 = c.key.rfind('|', b2 - 1);
      bytes = (std::size_t)std::stoull(c.key.substr(b1 + 1, b2 - b1 - 1));
      units = std::stoi(c.key.substr(b2 + 1));
      // Charged to the denoise: the modules go when the DiT does, so a
      // claim held through the decode would put them in a moment they
      // are not in.
      EXPECT_TRUE(c.phase == model_memory::kPhaseDenoise);
    }
    return std::make_pair(bytes, units);
  };

  const auto off = claims_of(false);
  EXPECT_TRUE(off.second == 0);

  const auto on = claims_of(true);
  std::printf("[image_memory_plan] ANE claim: %d unit(s) x %zu MB\n",
              on.second, on.first >> 20);
  // ONE unit: the feed-forward runs on a single shared runtime-weight
  // module, so its cost does not scale with the number of blocks.
  EXPECT_TRUE(on.second == 1);
  const genai::MetalKrea2Transformer::Config kc;
  EXPECT_TRUE(on.first ==
              genai::MetalKrea2Transformer::ane_runtime_bytes(
                  kc.hidden, kc.ffn,
                  genai::MetalKrea2Transformer::ane_plan_seq(0, 0)));
  // At least the IOSurface weight slots themselves, which hold a full
  // fp16 copy of one block's feed-forward.
  const std::size_t slots =
      3ull * (std::size_t)kc.hidden * (std::size_t)kc.ffn * 2ull;
  EXPECT_TRUE(on.first > slots);
}

// A QUANTIZED checkpoint claims the ANE module exactly as a dense one does:
// its feed-forward is dequantized into the module's fp16 inputs per block,
// so the module costs the same. Built as a copy of the real checkpoint's
// layout that differs only in transformer/config.json carrying a
// quantization block -- what the loader keys on -- so the claim is checked
// against the same geometry either way.
TEST(image_memory_plan, a_quantized_checkpoint_claims_the_ane_module)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  namespace fs = std::filesystem;
  const fs::path src(root);
  const fs::path q = fs::temp_directory_path() / "vpipe-krea2-quantized-plan";
  std::error_code ec;
  fs::remove_all(q, ec);
  fs::create_directories(q / "transformer", ec);
  for (const auto& e : fs::directory_iterator(src, ec)) {
    if (e.path().filename() == "transformer") { continue; }
    fs::create_symlink(e.path(), q / e.path().filename(), ec);
  }
  for (const auto& e : fs::directory_iterator(src / "transformer", ec)) {
    if (e.path().filename() == "config.json") { continue; }
    fs::create_symlink(e.path(), q / "transformer" / e.path().filename(), ec);
  }
  {
    std::ifstream in(src / "transformer" / "config.json");
    std::string j((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());
    const std::size_t brace = j.find('{');
    ASSERT_TRUE(brace != std::string::npos);
    j.insert(brace + 1,
             "\n  \"quantization\": {\"bits\": 4, \"group_size\": 64},");
    std::ofstream out(q / "transformer" / "config.json");
    out << j;
  }

  Session sess;
  auto ane_units = [&](const std::string& r) {
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(r));
    cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    int n = 0;
    for (const ResourceClaim& c : stage.declare_resources()) {
      if (c.kind == model_memory::kCoreMLKind) { ++n; }
    }
    return n;
  };
  EXPECT_TRUE(ane_units(src.string()) == 1);
  EXPECT_TRUE(ane_units(q.string()) == 1);
  fs::remove_all(q, ec);
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


// Does a real Krea-2 graph get its ANE module, box by box?
//
// The claims come from the actual stage at 1024x1024 and go through the
// actual registered planners, with VPIPE_RAM_LIMIT_MB standing in for the
// box. The generate-image stage's claims are most of the graph -- the DiT
// floor, the text encoder, the decode arena and the ANE unit; a VAE
// decode stage would add its ~250 MB of weights -- so this is a slight
// UNDER-estimate of the peak, and a refusal here is a refusal for real.
TEST(image_memory_plan, ane_module_grant_by_box)
{
  const char* root = std::getenv("VPIPE_KREA2_TEST_MODEL_PATH");
  if (root == nullptr || *root == '\0') { return; }
  for (int gb : {16, 24, 32, 64}) {
    ::setenv("VPIPE_RAM_LIMIT_MB", std::to_string(gb * 1024).c_str(), 1);
    Session sess;
    auto* mgr = sess.generative_model_manager();
    if (mgr == nullptr) { break; }
    mgr->clear_declarations();
    mgr->clear_scratch();
    FlexData cfg = FlexData::make_object();
    cfg.as_object().insert("hf_dir", FlexData::make_string(root));
    cfg.as_object().insert("ane_ffn", FlexData::make_bool(true));
    cfg.as_object().insert("width", FlexData::make_int(1024));
    cfg.as_object().insert("height", FlexData::make_int(1024));
    GenerateImageStage stage(&sess, "t2i", std::vector<InEdge>{},
                             std::move(cfg));
    const std::vector<ResourceClaim> claims = stage.declare_resources();
    const auto planners = ResourcePlannerRegistry::get().all();
    for (auto* p : planners) { p->begin_plan(&sess); }
    mgr->set_phase_order({std::string(model_memory::kPhaseCondition),
                          std::string(model_memory::kPhaseDenoise),
                          std::string(model_memory::kPhaseDecodeAudio),
                          std::string(model_memory::kPhaseDecode)});
    std::string label;
    std::size_t unit = 0;
    for (const ResourceClaim& c : claims) {
      if (auto* p = ResourcePlannerRegistry::get().find(c.kind)) {
        p->claim(&sess, c.key, c.phase, c.last_phase, c.floor_bytes);
      }
      if (c.kind == model_memory::kCoreMLKind) {
        const std::size_t b2 = c.key.rfind('|');
        const std::size_t b1 = c.key.rfind('|', b2 - 1);
        label = c.key.substr(0, b1);
        unit = (std::size_t)std::stoull(c.key.substr(b1 + 1, b2 - b1 - 1));
      }
    }
    const std::size_t peak = mgr->phase_peak();
    const std::size_t flat = mgr->phase_footprint(std::string()) +
                             mgr->scratch_bytes(std::string());
    for (auto* p : planners) { p->end_plan(&sess); }
    const int granted = model_memory::coreml_grant(&sess, label, unit, 1);
    std::vector<std::pair<std::string, std::size_t>> by;
    const std::size_t peak_after = mgr->phase_peak(&by);
    std::string phases;
    for (const auto& ph : by) {
      phases += " " + ph.first + "=" + std::to_string(ph.second >> 20);
    }
    std::printf("[image_memory_plan] %2d GB box: module %zu MB -> %s | "
                "per phase after grant%s | peak %zu -> %zu MB (full-size "
                "flat %zu)\n", gb,
                unit >> 20, granted > 0 ? "GRANTED" : "refused",
                phases.c_str(), peak >> 20, peak_after >> 20, flat >> 20);
    // Granted on every box, 16 GB included: the module lives in the
    // denoise, which stays under the decode's peak.
    EXPECT_TRUE(granted == 1);
    EXPECT_TRUE(peak_after == peak);
    mgr->clear_scratch();
    mgr->clear_declarations();
  }
  ::unsetenv("VPIPE_RAM_LIMIT_MB");
}
