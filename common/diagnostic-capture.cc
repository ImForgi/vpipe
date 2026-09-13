#include "common/diagnostic-capture.h"

using namespace std;

namespace vpipe {

namespace {

// The innermost capture open on this thread, or null. A raw pointer
// chain rather than a container: the scopes nest by construction, so
// the list IS the stack and pushing costs one store.
thread_local DiagnosticCapture* g_innermost = nullptr;

}

DiagnosticCapture::DiagnosticCapture()
    : _outer(g_innermost)
{
  g_innermost = this;
}

DiagnosticCapture::~DiagnosticCapture()
{
  // Pop only if we are still the top. An out-of-order destruction
  // would mean a capture outlived a scope nested inside it, which the
  // type cannot produce (no copy, no move, no heap use in the tree) --
  // but restoring blindly there would resurrect a dead pointer, and a
  // stale thread-local is a use-after-free on the next warn().
  if (g_innermost == this) {
    g_innermost = _outer;
  }
}

bool
DiagnosticCapture::active() noexcept
{
  return g_innermost != nullptr;
}

void
DiagnosticCapture::note(string_view msg)
{
  for (DiagnosticCapture* c = g_innermost; c != nullptr; c = c->_outer) {
    c->note_(msg);
  }
}

void
DiagnosticCapture::note_(string_view msg)
{
  if (msg.empty()) {
    return;
  }
  if (_lines.size() >= kMaxLines || _bytes >= kMaxBytes) {
    _truncated = true;
    return;
  }
  _bytes += msg.size();
  _lines.emplace_back(msg);
}

string
DiagnosticCapture::text() const
{
  string out;
  for (const string& l : _lines) {
    if (!out.empty()) {
      out.push_back('\n');
    }
    out += l;
  }
  return out;
}

}
