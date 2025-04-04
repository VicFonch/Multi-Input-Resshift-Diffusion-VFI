import cupy

#@cupy.memoize(for_each_device=True)
def cupy_launch(strFunction, strKernel):
    # return cupy.cuda.compile_with_cache(strKernel).get_function(strFunction)
    return cupy.RawKernel(strKernel, strFunction)
# end