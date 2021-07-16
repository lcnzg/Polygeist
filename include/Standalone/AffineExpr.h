#ifndef MLIR_IR_AFFINE_EXPR_MATCHER_H_
#define MLIR_IR_AFFINE_EXPR_MATCHER_H_

#include "mlir/IR/AffineExpr.h"

namespace mlir {

namespace detail {

template <typename AffineExprTy>
void bindDims(MLIRContext *ctx, AffineExprTy &e, int N) {
  e = getAffineDimExpr(N, ctx);
}

} // end namespace detail

template <typename AffineExprTy>
void bindDims(MLIRContext *ctx, AffineExprTy &expr, int pos) {
  detail::bindDims(ctx, expr, pos);
}

} // end namespace mlir

#endif
