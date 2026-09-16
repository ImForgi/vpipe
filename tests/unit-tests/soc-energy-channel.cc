// Which IOReport "Energy Model" channels belong to a block's total.
//
// This is a four-line predicate with two expensive failure modes, and
// both have already happened in this tree:
//
//   TOO NARROW -- an exact-name match. macOS 27 renamed every block with
//   a tile index (ANE -> ANE0), so an exact match reports 0.00 W on a
//   busy machine. That is what broke macmon on macOS 27.
//
//   TOO WIDE -- a prefix match. The same group carries "GPU SRAM" and
//   "GPU Energy", the latter being the same energy republished in
//   nanojoules, so a prefix match reported close to DOUBLE the real GPU
//   power -- an idle box read ~48 W.
//
// The channel names below are transcribed from live dumps: an M4 Pro on
// macOS 26, an M5 on macOS 26.6.2, and an M5 Pro on macOS 27.

#include "minitest.h"
#include "common/soc-energy-channel.h"

using namespace vpipe;

TEST(soc_energy_channel, bare_stem_matches) {
  // macOS 26 and earlier, both on M4 and on M5 silicon.
  EXPECT_TRUE(soc_block_energy_channel("ANE", "ANE"));
  EXPECT_TRUE(soc_block_energy_channel("GPU", "GPU"));
}

TEST(soc_energy_channel, indexed_stem_matches) {
  // macOS 27 indexes every block.
  EXPECT_TRUE(soc_block_energy_channel("ANE0", "ANE"));
  EXPECT_TRUE(soc_block_energy_channel("GPU0", "GPU"));
  // The index is a TILE NUMBER, so a part with two ANEs publishes both
  // and the block's power is their sum. Nothing in the tree has shipped
  // with two yet; this is the case the suffix exists for.
  EXPECT_TRUE(soc_block_energy_channel("ANE1", "ANE"));
  EXPECT_TRUE(soc_block_energy_channel("ANE10", "ANE"));
}

// THE DOUBLE-COUNT CASES. Each of these is a real channel sitting in
// "Energy Model" beside the block's own, and each was summed into the
// GPU total by the prefix match this replaced.
TEST(soc_energy_channel, companions_are_rejected) {
  // The same energy, republished in nanojoules.
  EXPECT_FALSE(soc_block_energy_channel("GPU Energy", "GPU"));
  // A component already inside the block's figure.
  EXPECT_FALSE(soc_block_energy_channel("GPU SRAM", "GPU"));
  // Real macOS 27 channel names from other groups, which a reader that
  // matched loosely could pick up if it ever stopped filtering by group.
  EXPECT_FALSE(soc_block_energy_channel("ANEXL U", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("ANE L0 RD", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("THROT-ANE-SUM", "ANE"));
}

TEST(soc_energy_channel, near_misses_are_rejected) {
  // Shorter than the stem, and a stem that is a prefix of a longer word.
  EXPECT_FALSE(soc_block_energy_channel("AN", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("", "ANE"));
  EXPECT_FALSE(soc_block_energy_channel("ANEX", "ANE"));
  // Anchored at the front: a block whose name merely ends in the stem is
  // a different block.
  EXPECT_FALSE(soc_block_energy_channel("DISPEXT0", "DISP"));
  // Case-sensitive -- these are kernel names, not user input.
  EXPECT_FALSE(soc_block_energy_channel("ane0", "ANE"));
}

// The other blocks renamed by macOS 27, so the helper is exercised as
// the general rule it is rather than only on ANE and GPU.
TEST(soc_energy_channel, every_renamed_block) {
  EXPECT_TRUE(soc_block_energy_channel("ISP0", "ISP"));
  EXPECT_TRUE(soc_block_energy_channel("DRAM0", "DRAM"));
  EXPECT_TRUE(soc_block_energy_channel("AVE0", "AVE"));
  EXPECT_TRUE(soc_block_energy_channel("AMCC0", "AMCC"));
  EXPECT_TRUE(soc_block_energy_channel("DCS0", "DCS"));
  EXPECT_TRUE(soc_block_energy_channel("MSR0", "MSR"));
  EXPECT_TRUE(soc_block_energy_channel("DISP0", "DISP"));
  EXPECT_TRUE(soc_block_energy_channel("DISPEXT0", "DISPEXT"));
  // New on macOS 27; no unsuffixed spelling ever existed for these.
  EXPECT_TRUE(soc_block_energy_channel("AFR0", "AFR"));
  EXPECT_TRUE(soc_block_energy_channel("FAB0", "FAB"));
}
