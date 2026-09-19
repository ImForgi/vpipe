#include "stages/save-tensor-stage.h"

#include "apple-silicon/tensor-beat.h"
#include "common/flex-data.h"
#include "common/job.h"
#include "common/vpipe-format.h"
#include "interfaces/session-context-intf.h"
#include "pipeline/runtime-context.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <utility>

using namespace std;

namespace vpipe {

namespace {

constexpr char        kMagic[8]  = {'V','P','T','E','N','S','O','R'};
constexpr std::uint32_t kVersion = 1;
constexpr std::uint32_t kMaxRank = 8;

const PortSpec kIports[] = {
  {.name = "tensor", .doc = "any TensorBeat; written verbatim",
   .type = &typeid(TensorBeatPayload), .clock_group = 0},
};

constexpr ConfigKey kAttrs[] = {
  {.key = "path", .type = ConfigType::String, .required = true,
   .doc = "output file. With `numbered` it is a template: the beat's "
          "index is inserted before the extension",
   .is_path = true, .path_write = true},
  {.key = "numbered", .type = ConfigType::Bool, .required = false,
   .doc = "write ONE FILE PER BEAT, numbered from 1 -- `x.vpt` becomes "
          "`x-000001.vpt`. Off (the default) writes every beat to the "
          "same path, so the last one wins, which is what a graph "
          "emitting a single tensor wants",
   .def_bool = false},
};

const StageSpec kSpec = {
  .type_name = "save-tensor",
  .doc       = "Sink: writes TensorBeats to disk verbatim, so a tensor "
               "outlives the run that made it. The companion of "
               "load-tensor. Carries dtype, shape and the beat's "
               "sideband, so a reader can refuse a mismatch rather than "
               "reinterpret the bytes.",
  .display_name = "Save Tensor",
  .category  = StageCategory::Generic,
  .iports    = kIports,
  .oports    = {},
  .attrs     = kAttrs,
};

string
numbered_path_(const string& path, int n)
{
  const auto dot = path.find_last_of('.');
  char suffix[16];
  std::snprintf(suffix, sizeof suffix, "-%06d", n);
  if (dot == string::npos) { return path + suffix; }
  return path.substr(0, dot) + suffix + path.substr(dot);
}

}  // namespace

SaveTensorStage::SaveTensorStage(const SessionContextIntf* session,
                                 string                    id,
                                 vector<InEdge>            iports,
                                 FlexData                  config)
  : TypedStage<SaveTensorStage>(session, std::move(id), std::move(iports),
                                std::move(config))
{
  _path   = attr_path("path", /*for_write=*/true);
  _number = attr_bool("numbered");
  if (_path.empty()) {
    fail_config(fmt("SaveTensorStage('{}'): config.path is required",
                    this->id()));
  }
}

const StageSpec&
SaveTensorStage::spec() const noexcept
{
  return kSpec;
}

Job
SaveTensorStage::process(RuntimeContext& ctx)
{
  auto in0 = co_await ctx.read(0);
  if (!in0) { ctx.signal_done(); co_return; }

  const auto* tin = dynamic_cast<const TensorBeatPayload*>(in0.get());
  if (tin == nullptr) {
    session()->warn(fmt("SaveTensorStage('{}'): not a TensorBeat; dropping",
                        this->id()));
    co_return;
  }
  if (tin->shape.size() > kMaxRank) {
    session()->warn(fmt(
        "SaveTensorStage('{}'): rank {} exceeds the {} this format "
        "carries; dropping", this->id(), tin->shape.size(), kMaxRank));
    co_return;
  }

  const string path = _number ? numbered_path_(_path, _written + 1) : _path;
  std::FILE* f = std::fopen(path.c_str(), "wb");
  if (f == nullptr) {
    session()->error(fmt("SaveTensorStage('{}'): cannot open '{}' for "
                         "writing", this->id(), path));
    co_return;
  }

  // Strided or Shared-backed beats are laid out row-major first, so the
  // file is always the tensor a reader expects rather than a view that
  // needs the producer's strides to interpret.
  const auto bytes = tin->materialize_contiguous();
  const std::uint32_t rank  = (std::uint32_t)tin->shape.size();
  const std::uint64_t count = (std::uint64_t)tin->element_count();
  const std::uint32_t dt    = (std::uint32_t)tin->dtype;
  const std::uint32_t zero  = 0;

  bool ok = std::fwrite(kMagic, 1, sizeof kMagic, f) == sizeof kMagic;
  auto u32 = [&](std::uint32_t v) {
    ok = ok && std::fwrite(&v, sizeof v, 1, f) == 1;
  };
  u32(kVersion); u32(dt); u32(rank); u32(zero);
  ok = ok && std::fwrite(&count, sizeof count, 1, f) == 1;
  for (std::uint32_t i = 0; i < rank; ++i) {
    const std::int64_t d = tin->shape[i];
    ok = ok && std::fwrite(&d, sizeof d, 1, f) == 1;
  }
  if (ok && !bytes.empty()) {
    ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
  }
  // The sideband rides along so a reader can put the beat back as it
  // was; a beat without one writes nothing and the reader sees EOF.
  if (ok && !tin->sideband.is_null()) {
    const string js = tin->sideband.to_json();
    if (!js.empty()) {
      ok = std::fwrite(js.data(), 1, js.size(), f) == js.size();
    }
  }
  const bool closed = std::fclose(f) == 0;
  if (!ok || !closed) {
    session()->error(fmt("SaveTensorStage('{}'): writing '{}' failed",
                         this->id(), path));
    co_return;
  }
  ++_written;

  string dims;
  for (std::size_t i = 0; i < tin->shape.size(); ++i) {
    dims += (i ? "x" : "") + std::to_string(tin->shape[i]);
  }
  session()->info(fmt("SaveTensorStage('{}'): wrote '{}' -- [{}] dtype {}, "
                      "{} bytes", this->id(), path, dims, dt,
                      bytes.size()));
}

VPIPE_REGISTER_STAGE(SaveTensorStage)

}  // namespace vpipe
