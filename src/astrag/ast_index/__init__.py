from astrag.ast_index.clang_env import configure_libclang

configure_libclang()

from astrag.ast_index.index import AstIndex

__all__ = ["AstIndex", "configure_libclang"]
