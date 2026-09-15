#include "apple-silicon/coreml/ane-worker.h"

#include <algorithm>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

namespace vpipe {


struct AneWorker::Impl {
  // Dispatch side: one thread, one outstanding job.
  std::thread             disp;
  std::mutex              dm;
  std::condition_variable dcv;
  std::function<void()>   job;
  bool                    job_pending = false;
  bool                    job_running = false;
  bool                    stop = false;

  // Convert side: a small fixed pool driven as a barrier per call.
  std::vector<std::thread>                              conv;
  std::mutex                                            cm;
  std::condition_variable                               ccv, cdone;
  const std::function<void(std::size_t, std::size_t)>*  body = nullptr;
  std::size_t                                           count = 0;
  unsigned                                              gen = 0;
  unsigned                                              left = 0;
};

AneWorker::AneWorker() : _impl(new Impl())
{
  Impl* p = _impl.get();
  p->disp = std::thread([p]() {
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock<std::mutex> lk(p->dm);
        p->dcv.wait(lk, [p] { return p->job_pending || p->stop; });
        if (p->stop && !p->job_pending) { return; }
        f = std::move(p->job);
        p->job_pending = false;
        p->job_running = true;
      }
      if (f) { f(); }
      {
        std::lock_guard<std::mutex> lk(p->dm);
        p->job_running = false;
      }
      p->dcv.notify_all();
    }
  });

  // Half the cores, capped: the conversions are memory-bound, so past a
  // handful of threads there is nothing left to win, and the GPU thread
  // is doing real work beside them.
  unsigned n = std::thread::hardware_concurrency() / 2;
  if (n < 1) { n = 1; }
  if (n > 4) { n = 4; }
  for (unsigned i = 0; i < n; ++i) {
    p->conv.emplace_back([p, i, n]() {
      unsigned seen = 0;
      for (;;) {
        std::unique_lock<std::mutex> lk(p->cm);
        p->ccv.wait(lk, [p, &seen] { return p->gen != seen || p->stop; });
        if (p->stop) { return; }
        seen = p->gen;
        const std::size_t cnt = p->count;
        const auto* b = p->body;
        lk.unlock();
        const std::size_t per = (cnt + n - 1) / n;
        const std::size_t lo = std::min((std::size_t)i * per, cnt);
        const std::size_t hi = std::min(lo + per, cnt);
        if (b != nullptr && hi > lo) { (*b)(lo, hi); }
        {
          std::lock_guard<std::mutex> lk2(p->cm);
          if (--p->left == 0) { p->cdone.notify_all(); }
        }
      }
    });
  }
}

AneWorker::~AneWorker()
{
  Impl* p = _impl.get();
  {
    std::lock_guard<std::mutex> lk(p->dm);
    p->stop = true;
  }
  {
    std::lock_guard<std::mutex> lk(p->cm);
    p->stop = true;
  }
  p->dcv.notify_all();
  p->ccv.notify_all();
  if (p->disp.joinable()) { p->disp.join(); }
  for (auto& t : p->conv) {
    if (t.joinable()) { t.join(); }
  }
}

void
AneWorker::dispatch(std::function<void()> fn)
{
  Impl* p = _impl.get();
  std::unique_lock<std::mutex> lk(p->dm);
  p->dcv.wait(lk, [p] { return !p->job_pending && !p->job_running; });
  p->job = std::move(fn);
  p->job_pending = true;
  lk.unlock();
  p->dcv.notify_all();
}

void
AneWorker::join()
{
  Impl* p = _impl.get();
  std::unique_lock<std::mutex> lk(p->dm);
  p->dcv.wait(lk, [p] { return !p->job_pending && !p->job_running; });
}

void
AneWorker::parallel_for(
    std::size_t count,
    const std::function<void(std::size_t, std::size_t)>& body)
{
  Impl* p = _impl.get();
  // Below this the handoff costs more than the work. Also the path when
  // the pool could not start.
  if (count < 65536 || p->conv.empty()) {
    body(0, count);
    return;
  }
  std::unique_lock<std::mutex> lk(p->cm);
  p->body  = &body;
  p->count = count;
  p->left  = (unsigned)p->conv.size();
  ++p->gen;
  p->ccv.notify_all();
  p->cdone.wait(lk, [p] { return p->left == 0; });
  p->body = nullptr;
}


}  // namespace vpipe
