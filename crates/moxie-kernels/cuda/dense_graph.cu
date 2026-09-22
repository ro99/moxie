// The selected dense graph image reuses the already qualified generic kernels
// and adds the missing dense semantic operations.  This keeps one module alive
// for a graph whose nodes include both existing and new operations.
#include "bf16_chain.cu"
#include "paged_attention.cu"
#include "dense_ops.cu"
