#!/usr/bin/env bash
# Build ChezScheme from vendor/ChezScheme and install it into vendor/chez.
#
# The install prefix contains everything a Rust host needs to embed Chez:
#   vendor/chez/bin/scheme                          REPL / compiler
#   vendor/chez/lib/csv<ver>/<machine>/scheme.h     C API header
#   vendor/chez/lib/csv<ver>/<machine>/libkernel.a  runtime to link against
#   vendor/chez/lib/csv<ver>/<machine>/*.boot       petite.boot + scheme.boot
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
src="$root/vendor/ChezScheme"
prefix="$root/vendor/chez"

if [ ! -f "$src/configure" ]; then
    # Blob-filtered clone keeps the checkout small (see CLAUDE.md).
    git -C "$root" submodule update --init --filter=blob:none vendor/ChezScheme
fi

cd "$src"
./configure --threads --disable-x11 --installprefix="$prefix" "$@"
make -j"$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
make install

echo
echo "ChezScheme installed to $prefix"
echo "Run it with: $prefix/bin/scheme"
