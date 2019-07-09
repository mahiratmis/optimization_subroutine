from distutils.core import setup 
from distutils.extension import Extension 
from Cython.Build import cythonize 
#setup(
#    ext_modules = cythonize(["optimizatin_subroutine.pyx"],annotate=True, )
#)

ext_modules = [
    Extension(
        "optimization_subroutine",
        ["optimization_subroutine.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        libraries=["m"]
    )
]

ext_options = {"compiler_directives": {"profile": True}, "annotate": True} 

setup(
    name='optimization_subroutine',
    ext_modules=cythonize(ext_modules,
    **ext_options
))
