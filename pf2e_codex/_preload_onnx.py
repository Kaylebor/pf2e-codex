"""Pre-load bundled shared libraries needed by onnxruntime providers.

On systems with protobuf 35+ (e.g. CachyOS 2026+), onnxruntime-migraphx
fails because its provider .so links against libprotobuf.so.34 which isn't
on the system. We bundle the v34 .so files next to the onnxruntime capi
directory, then pre-load them with RTLD_GLOBAL before onnxruntime imports.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _preload_protobuf34() -> bool:
    """Find and pre-load bundled protobuf 34 shared libraries.

    Returns True if at least one library was pre-loaded.
    """
    # Locate the onnxruntime capi directory where we bundle protobuf 34
    # Search paths (in priority order):
    candidates = []

    # 1. Next to this module (venv layout)
    #    pf2e_codex/_preload_onnx.py → sibling pf2e_codex/
    #    onnxruntime/capi/ is a sibling
    try:
        our_dir = Path(__file__).resolve().parent.parent
        candidates.append(our_dir / "onnxruntime" / "capi")
    except NameError:
        pass

    # 2. PYTHONPATH-style layout (PKGBUILD target)
    for p in sys.path:
        p_path = Path(p)
        candidates.append(p_path / "onnxruntime" / "capi")

    seen = set()
    for capi_dir in candidates:
        capi = capi_dir.resolve()
        if not capi.is_dir() or str(capi) in seen:
            continue
        seen.add(str(capi))

        pb = capi / "libprotobuf.so.34.1.0"
        if not pb.is_file():
            continue

        # Load with RTLD_GLOBAL so all transitively loaded libraries can find it
        try:
            ctypes.CDLL(str(pb), mode=ctypes.RTLD_GLOBAL)
            # Also pre-load utf8_validity (another protobuf 34-era dep)
            for lib_name in ("libutf8_validity.so.34.1.0", "libutf8_range.so.34.1.0"):
                lib_path = capi / lib_name
                if lib_path.is_file():
                    ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_GLOBAL)
            return True
        except OSError:
            continue

    return False


_preloaded = _preload_protobuf34()
