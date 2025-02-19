// RUN: mlir-clang %s --function=* -fopenmp -S | FileCheck %s

void square(double* x) {
    int i;
    #pragma omp parallel for private(i)
    for(i=3; i < 10; i+= 2) {
        x[i] = i;
        i++;
        x[i] = i;
    }
}

// CHECK:   func @square(%arg0: memref<?xf64>)
// CHECK-NEXT:     %c1 = arith.constant 1 : index
// CHECK-NEXT:     %c2 = arith.constant 2 : index
// CHECK-NEXT:     %c11 = arith.constant 11 : index
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c3 = arith.constant 3 : index
// CHECK-NEXT:     scf.parallel (%arg1) = (%c3) to (%c11) step (%c2) {
// CHECK-NEXT:       %0 = arith.index_cast %arg1 : index to i32
// CHECK-NEXT:       %1 = arith.sitofp %0 : i32 to f64
// CHECK-NEXT:       memref.store %1, %arg0[%arg1] : memref<?xf64>
// CHECK-NEXT:       %2 = arith.addi %0, %c1_i32 : i32
// CHECK-NEXT:       %3 = arith.addi %arg1, %c1 : index
// CHECK-NEXT:       %4 = arith.sitofp %2 : i32 to f64
// CHECK-NEXT:       memref.store %4, %arg0[%3] : memref<?xf64>
// CHECK-NEXT:       scf.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
