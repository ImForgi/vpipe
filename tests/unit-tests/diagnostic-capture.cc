// What a refused operation reported, and who gets to hear it.
//
// The feature these cover is one sentence long -- a caller can find out
// why its own load was refused -- but it rests on three properties that
// are easy to lose: the reasons a real loader produces actually land in
// the capture, an unrelated thread's reports do NOT, and a capture that
// is open changes nothing about what the session reports.

#include "minitest.h"
#include "common/diagnostic-capture.h"
#include "common/flex-data.h"
#include "common/session.h"
#include "common/vpipe-format.h"
#include "pipeline/pipeline-spec.h"
#include "pipeline/pipeline.h"

#include <atomic>
#include <memory>
#include <string>
#include <thread>

using namespace std;
using namespace vpipe;

namespace {

bool
mentions_(const DiagnosticCapture& c, string_view needle)
{
  return c.text().find(needle) != string::npos;
}

// A spec that names a stage its own iport cannot reach. The loader
// refuses it and says which stage, which iport and which missing
// upstream -- exactly the text an operator needs and never saw.
FlexData
forward_reference_spec_()
{
  FlexData edge = FlexData::make_object();
  edge.as_object().insert_or_assign("src", FlexData::make_string("later"));
  edge.as_object().insert_or_assign("oport", FlexData::make_int(0));
  FlexData iports = FlexData::make_array();
  iports.as_array().push_back(std::move(edge));

  FlexData stage = FlexData::make_object();
  stage.as_object().insert_or_assign("id", FlexData::make_string("a"));
  stage.as_object().insert_or_assign("type", FlexData::make_string("chrono"));
  stage.as_object().insert_or_assign("iports", std::move(iports));
  stage.as_object().insert_or_assign("config", FlexData::make_object());

  FlexData stages = FlexData::make_array();
  stages.as_array().push_back(std::move(stage));

  FlexData spec = FlexData::make_object();
  spec.as_object().insert_or_assign("id", FlexData::make_string("fwd"));
  spec.as_object().insert_or_assign("stages", std::move(stages));
  return spec;
}

}

TEST(diagnostic_capture, a_refused_spec_leaves_its_reason_in_the_scope) {
  Session sess;
  FlexData spec = forward_reference_spec_();

  DiagnosticCapture why;
  auto pl = pipeline_from_spec(spec, &sess);

  ASSERT_TRUE(pl == nullptr);
  ASSERT_TRUE(!why.empty());
  // The three things that make the message worth showing: which
  // pipeline, which stage, and what it could not resolve.
  EXPECT_TRUE(mentions_(why, "fwd"));
  EXPECT_TRUE(mentions_(why, "'a'"));
  EXPECT_TRUE(mentions_(why, "later"));
}

TEST(diagnostic_capture, an_unknown_stage_type_is_named) {
  Session sess;
  FlexData stage = FlexData::make_object();
  stage.as_object().insert_or_assign("id", FlexData::make_string("s"));
  stage.as_object().insert_or_assign(
      "type", FlexData::make_string("no-such-stage-type"));
  FlexData stages = FlexData::make_array();
  stages.as_array().push_back(std::move(stage));
  FlexData spec = FlexData::make_object();
  spec.as_object().insert_or_assign("id", FlexData::make_string("p"));
  spec.as_object().insert_or_assign("stages", std::move(stages));

  DiagnosticCapture why;
  auto pl = pipeline_from_spec(spec, &sess);

  ASSERT_TRUE(pl == nullptr);
  // The registry refuses behind a virtual, several frames below the
  // loader. Its reason reaches the caller all the same -- that is the
  // whole argument for capturing rather than threading an out-param.
  EXPECT_TRUE(mentions_(why, "no-such-stage-type"));
}

TEST(diagnostic_capture, nothing_is_captured_outside_a_scope) {
  Session sess;
  {
    DiagnosticCapture why;
    sess.warn(fmt("inside"));
    ASSERT_TRUE(!why.empty());
  }
  // The thread-local must be popped on the way out, or the next warn
  // writes into a destroyed object.
  sess.warn(fmt("outside"));
  DiagnosticCapture after;
  EXPECT_TRUE(after.empty());
}

TEST(diagnostic_capture, an_inner_scope_does_not_blind_the_outer_one) {
  Session sess;
  DiagnosticCapture outer;
  {
    DiagnosticCapture inner;
    sess.warn(fmt("the reason"));
    EXPECT_TRUE(mentions_(inner, "the reason"));
  }
  sess.warn(fmt("and another"));
  // The outer scope owns the verdict; a sub-step that looked at its own
  // reasons must not consume them.
  EXPECT_TRUE(mentions_(outer, "the reason"));
  EXPECT_TRUE(mentions_(outer, "and another"));
}

TEST(diagnostic_capture, another_thread_is_not_this_operations_business) {
  Session sess;
  DiagnosticCapture why;
  std::thread t([&sess] { sess.warn(fmt("an unrelated stage")); });
  t.join();
  sess.warn(fmt("this operation"));

  // A session serves many pipelines at once. Attaching whatever else
  // happened to be logged would make the reason actively misleading.
  EXPECT_TRUE(mentions_(why, "this operation"));
  EXPECT_TRUE(!mentions_(why, "an unrelated stage"));
}

TEST(diagnostic_capture, the_line_cap_reports_itself) {
  Session sess;
  DiagnosticCapture why;
  for (int i = 0; i < 200; ++i) {
    sess.warn(fmt("line {}", i));
  }
  // Bounded, and it says it is bounded -- a truncated reason that
  // claims to be the whole one is worse than a long one.
  EXPECT_TRUE(why.lines().size() <= 32);
  EXPECT_TRUE(why.truncated());
}
