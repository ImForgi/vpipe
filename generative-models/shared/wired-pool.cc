#include "generative-models/shared/wired-pool.h"

#include "apple-silicon/metal-compute/metal-compute.h"
#include "apple-silicon/metal-compute/shared-buffer.h"
#include "common/vpipe-format.h"
#include "generative-models/generative-model-manager.h"
#include "generative-models/shared/accel-settings.h"
#include "interfaces/session-context-intf.h"
#include "interfaces/session-services-intf.h"

#include <cstdlib>
#include <string>

namespace vpipe::genai {
namespace {

GenerativeModelManager*
manager_(metal_compute::MetalCompute* mc)
{
  const auto* sess = mc != nullptr ? mc->session() : nullptr;
  if (sess == nullptr || sess->services() == nullptr) { return nullptr; }
  return sess->services()->generative_model_manager();
}

}  // namespace

struct WiredPool::Impl {
  metal_compute::MetalCompute* mc = nullptr;
  std::string tag = "wired pool";
  bool        on = false;
  std::size_t wired = 0;
  // Set when this model met a refusal, cleared when the ceiling is
  // reopened -- so a box busy at forward 2 is still asked at forward 5.
  bool        retry = false;
  // available_physical at the refusal: the retry waits for the box to have
  // freed a block's worth SINCE, not for time to pass.
  std::size_t retry_at = 0;
  // Whether this run already reopened a ceiling it did not collapse.
  bool        inherited_asked = false;
  long long   refusals = 0;
  long long   reopens = 0;
  std::size_t last_refused = 0;

  void arm_retry(metal_compute::MetalCompute* m)
  {
    ++refusals;
    if (retry) { return; }
    retry    = true;
    retry_at = m != nullptr ? m->memory_budget().available_physical : 0;
    if (m != nullptr && m->session() != nullptr) {
      m->session()->log_debug(fmt(
          "{}: the pool refused more with {} MB booked; holding there and "
          "retrying once the box has freed a block's worth", tag,
          wired >> 20));
    }
  }
};

WiredPool::WiredPool() : _p(new Impl) {}

WiredPool::~WiredPool() { delete _p; }

WiredPool::WiredPool(WiredPool&& o) noexcept : _p(o._p) { o._p = nullptr; }

WiredPool&
WiredPool::operator=(WiredPool&& o) noexcept
{
  if (this != &o) {
    delete _p;
    _p = o._p;
    o._p = nullptr;
  }
  return *this;
}

void
WiredPool::open(metal_compute::MetalCompute* mc)
{
  open(mc, FlexData::make_object());
}

void
WiredPool::open(metal_compute::MetalCompute* mc, const FlexData& options)
{
  if (_p == nullptr) { _p = new Impl; }
  namespace acc = accel;
  _p->mc = mc;
  _p->tag = acc::text(&options, wired_pool::kTag, _p->tag);
  auto* mgr = manager_(mc);
  _p->on = mgr != nullptr && mgr->wired_pool_limit() > 0 &&
           acc::flag(&options, wired_pool::kEnabled, true);
  if (const char* e = std::getenv("VPIPE_WIRE_RESIDENT")) {
    _p->on = std::atoi(e) != 0;
  }
}

bool
WiredPool::on() const
{
  return _p != nullptr && _p->on;
}

std::size_t
WiredPool::wired_bytes() const
{
  return _p != nullptr ? _p->wired : 0;
}

std::size_t
WiredPool::budget() const
{
  if (!on()) { return 0; }
  auto* mgr = manager_(_p->mc);
  if (mgr == nullptr) { return _p->wired; }
  const std::size_t lim  = mgr->wired_pool_limit();
  const std::size_t used = mgr->wired_pool_used();
  return _p->wired + (lim > used ? lim - used : 0);
}

bool
WiredPool::wirable(std::size_t nb) const
{
  if (!on()) { return true; }
  auto* mgr = manager_(_p->mc);
  return mgr != nullptr && mgr->wired_pool_can_take(nb);
}

std::size_t
WiredPool::wire_one(metal_compute::MetalCompute* mc,
                    metal_compute::SharedBuffer& b, bool on)
{
  if (b.byte_size() == 0 || b.is_wired() == on) { return 0; }
  auto* mgr = manager_(mc);
  if (mgr == nullptr) { return 0; }
  if (!on) {
    const std::size_t n = b.byte_size();
    mgr->unwire_from_pool(b);
    return n;
  }
  return mgr->wire_into_pool(b);
}

bool
WiredPool::refused(metal_compute::MetalCompute* mc,
                   const metal_compute::SharedBuffer& b) const
{
  if (!GenerativeModelManager::pool_wirable(b)) { return false; }
  auto* mgr = manager_(mc);
  return mgr != nullptr && !mgr->wired_pool_can_take(b.byte_size());
}

std::size_t
WiredPool::wire_set(std::span<metal_compute::SharedBuffer* const> bufs,
                    bool on)
{
  if (_p == nullptr) { return 0; }
  auto* m = _p->mc;
  std::size_t changed = 0;
  std::size_t left = 0;
  bool stopped = false;
  for (metal_compute::SharedBuffer* b : bufs) {
    if (b == nullptr || b->byte_size() == 0) { continue; }
    if (stopped) {
      if (GenerativeModelManager::pool_wirable(*b) && !b->is_wired()) {
        left += b->byte_size();
      }
      continue;
    }
    const std::size_t n = wire_one(m, *b, on);
    if (on && n == 0 && !b->is_wired() && refused(m, *b)) {
      stopped = true;
      left += b->byte_size();
      continue;
    }
    changed += n;
  }
  if (on) {
    _p->last_refused = left;
    if (stopped) { _p->arm_retry(m); }
  }
  return changed;
}

void
WiredPool::note_wired(metal_compute::MetalCompute* mc, std::size_t got,
                      std::size_t want)
{
  if (_p == nullptr) { return; }
  _p->wired += got;
  if (got >= want || !_p->on) { return; }
  // A SHORTFALL IS NOT ALWAYS A REFUSAL. The pool never wires buffers
  // below GenerativeModelManager::kMinWiredBytes, and every block carries
  // some, so `want` always exceeds what a fully wired block reports. Only a
  // pool that can no longer take the missing bytes has said no; otherwise
  // this block is as wired as the pool will make it, and growth goes on.
  auto* mgr = manager_(mc);
  if (mgr != nullptr && mgr->wired_pool_can_take(want - got)) { return; }
  // The pool itself stops the next admission -- wirable() reads it -- so
  // all that is left is to remember to ask again.
  _p->arm_retry(mc);
}

void
WiredPool::note_unwired(std::size_t n)
{
  if (_p == nullptr) { return; }
  _p->wired -= (n > _p->wired) ? _p->wired : n;
}

void
WiredPool::new_run()
{
  if (_p != nullptr) { _p->inherited_asked = false; }
}

bool
WiredPool::retry(metal_compute::MetalCompute* mc, std::size_t block_bytes)
{
  if (!on() || mc == nullptr) { return false; }
  auto* mgr = manager_(mc);
  if (mgr == nullptr) { return false; }
  const std::size_t lim = mgr->wired_pool_limit();
  const std::size_t ask = mgr->wired_pool_ask();
  // NOT COLLAPSED: the ceiling is the configured one, so a pool that ran
  // out is simply full -- reopening would change nothing.
  if (lim >= ask) {
    _p->retry = false;
    return false;
  }
  const std::size_t now = mc->memory_budget().available_physical;
  const bool own = _p->retry;
  if (own) {
    if (now <= _p->retry_at + block_bytes) { return false; }
  } else {
    if (_p->inherited_asked) { return false; }
    _p->inherited_asked = true;
  }
  mgr->reopen_wired_pool();
  const std::size_t after = mgr->wired_pool_limit();
  _p->retry    = false;
  _p->retry_at = now;
  if (after <= lim) { return false; }
  ++_p->reopens;
  if (mc->session() != nullptr) {
    mc->session()->log_debug(fmt(
        "{}: reopening a ceiling {} collapsed -- {} -> {} MB", _p->tag,
        own ? "this model's refusal" : "an earlier refusal",
        lim >> 20, after >> 20));
  }
  return true;
}

FlexData
WiredPool::info() const
{
  namespace acc = accel;
  FlexData out = FlexData::make_object();
  acc::set_flag(&out, wired_pool::kInfoOn, on());
  acc::set_integer(&out, wired_pool::kInfoWiredBytes, (long long)wired_bytes());
  acc::set_integer(&out, wired_pool::kInfoBudget, (long long)budget());
  auto* mgr = _p != nullptr ? manager_(_p->mc) : nullptr;
  acc::set_integer(&out, wired_pool::kInfoPoolUsed,
                   mgr != nullptr ? (long long)mgr->wired_pool_used() : 0);
  acc::set_integer(&out, wired_pool::kInfoPoolLimit,
                   mgr != nullptr ? (long long)mgr->wired_pool_limit() : 0);
  acc::set_integer(&out, wired_pool::kInfoPoolAsk,
                   mgr != nullptr ? (long long)mgr->wired_pool_ask() : 0);
  acc::set_flag(&out, wired_pool::kInfoRetryArmed,
                _p != nullptr && _p->retry);
  acc::set_integer(&out, wired_pool::kInfoRefusals,
                   _p != nullptr ? _p->refusals : 0);
  acc::set_integer(&out, wired_pool::kInfoReopens,
                   _p != nullptr ? _p->reopens : 0);
  acc::set_integer(&out, wired_pool::kInfoLastRefused,
                   _p != nullptr ? (long long)_p->last_refused : 0);
  return out;
}

}  // namespace vpipe::genai
