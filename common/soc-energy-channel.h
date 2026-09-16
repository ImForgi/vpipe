// soc-energy-channel.h -- which IOReport "Energy Model" channels ARE a
// SoC block's energy, and which merely look like it.
//
// The Energy Model group names one channel per on-die block -- ANE, GPU,
// ISP, DRAM -- and a reader wanting "the ANE's power" sums the channels
// belonging to that block over a sample delta. Getting the membership
// test wrong is silent in both directions: too narrow and the block
// reads 0.00 W, too wide and it reads double.
//
// THE INDEX IS A TILE NUMBER. macOS 27 renamed every block with a `0`
// suffix -- ANE -> ANE0, GPU -> GPU0, ISP -> ISP0, and so on -- which is
// not decoration: it numbers the instance. A part carrying two ANEs
// publishes ANE0 and ANE1, and the block's power is their SUM. So the
// match accepts a bare stem and a stem followed by digits, and a caller
// that also counts the matches learns how many units the box has
// without a per-chip table.
//
// MEASURED, three boxes, one run each:
//
//   M4 Pro, macOS 26      ANE   GPU   GPU SRAM  GPU Energy(nJ)
//   M5,     macOS 26.6.2  ANE   GPU   ...
//   M5 Pro, macOS 27      ANE0  GPU0  ...       GPU Energy(nJ)
//
// The M5 on macOS 26 is what makes this an OS change rather than a
// silicon one: same family as the M5 Pro, unsuffixed names.
//
// "AND NOTHING ELSE" IS LOAD-BEARING. The same group carries `GPU SRAM`
// (a component of the block, already inside its figure) and `GPU Energy`
// -- the SAME energy published a second time in nanojoules. MEASURED on
// the M4 Pro: GPU 376284 J against GPU Energy 375695 J, 0.16% apart. A
// prefix match over "GPU" sums all three and reports roughly twice the
// real power, which is how an idle box came to read ~48 W.

#ifndef VPIPE_SOC_ENERGY_CHANNEL_H
#define VPIPE_SOC_ENERGY_CHANNEL_H

#include <cstddef>
#include <string_view>

namespace vpipe {

// True when `name` is `stem`, or `stem` followed only by digits.
//
// Case-sensitive and anchored at both ends on purpose: these are kernel
// channel names, not user input, and every near-miss this rejects
// ("GPU SRAM", "GPU Energy", "ANEXL U") is a channel a reader must not
// add into the block's total.
inline bool
soc_block_energy_channel(std::string_view name, std::string_view stem)
{
  if (name.size() < stem.size()) { return false; }
  if (name.compare(0, stem.size(), stem) != 0) { return false; }
  if (name.size() == stem.size()) { return true; }
  for (std::size_t i = stem.size(); i < name.size(); ++i) {
    if (name[i] < '0' || name[i] > '9') { return false; }
  }
  return true;
}

}

#endif
