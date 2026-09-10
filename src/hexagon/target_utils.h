/*!
 * \file tl/hexagon/target_utils.h
 * \brief Hexagon target helpers.
 */

#ifndef TVM_TL_HEXAGON_TARGET_UTILS_H_
#define TVM_TL_HEXAGON_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

inline bool TargetIsHexagon(Target target) {
  return target.defined() && target->kind.defined() &&
         target->kind->name == "hexagon";
}

} // namespace tl
} // namespace tvm

#endif // TVM_TL_HEXAGON_TARGET_UTILS_H_
