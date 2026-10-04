#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
deps="$repo/.deps"
prefix=${GNUSTEP_PREFIX:-"$deps/gnustep"}
jobs=${JOBS:-4}

if [[ $(uname -s) != Linux ]]; then
    echo "This bootstrap is for Linux; macOS uses the system Foundation SDK." >&2
    exit 1
fi
mkdir -p "$deps/src" "$prefix"
prefix=$(cd "$prefix" && pwd)
if [[ -f "$prefix/.mil-hwx-toolchain-v1" ]]; then
    echo "GNUstep toolchain already installed in $prefix"
    exit 0
fi
for command in clang clang++ git make cmake pkg-config; do
    command -v "$command" >/dev/null || {
        echo "Missing build prerequisite: $command" >&2; exit 1;
    }
done
for package in libffi icu-uc libxml-2.0 openssl libcurl; do
    pkg-config --exists "$package" || {
        echo "Missing development package: $package (see README Linux setup)" >&2
        exit 1
    }
done

# Pin the source releases used for the Asahi verification.
fetch() {
    local project=$1 tag=$2 commit=$3
    local source="$deps/src/$project"
    if [[ ! -d "$source/.git" ]]; then
        git clone --depth 1 --branch "$tag" "https://github.com/gnustep/$project.git" "$source"
    fi
    if [[ $(git -C "$source" rev-parse HEAD) != "$commit" ]]; then
        echo "Unexpected $project revision in $source; expected $commit" >&2
        exit 1
    fi
}
fetch libobjc2 v2.3 e877e782fb965c4e870d2ecf6aff58dddc6290ae
fetch tools-make make-2_9_3 214a57eb7a6d2e0590a8aece4970a72561beb8d5
fetch libs-base base-1_31_1 6307e474dd39e36eb2c5f6ee15ade8b51e337970

export CC=clang CXX=clang++
cmake -S "$deps/src/libobjc2" -B "$deps/objc-build" \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
    -DCMAKE_OBJC_COMPILER=clang -DCMAKE_OBJCXX_COMPILER=clang++ \
    -DCMAKE_INSTALL_PREFIX="$prefix" -DCMAKE_INSTALL_LIBDIR=lib \
    -DGNUSTEP_INSTALL_TYPE=NONE -DTESTS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build "$deps/objc-build" -j "$jobs"
cmake --install "$deps/objc-build"

export PATH="$prefix/bin:$PATH"
export CPPFLAGS="-I$prefix/include ${CPPFLAGS:-}"
# Base checks curl headers before applying the pkg-config flags.
export CPPFLAGS="$CPPFLAGS $(pkg-config --cflags libcurl)"
export LDFLAGS="-L$prefix/lib -Wl,-rpath,$prefix/lib ${LDFLAGS:-}"
export LD_LIBRARY_PATH="$prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
(
    cd "$deps/src/tools-make"
    ./configure --prefix="$prefix" --with-layout=fhs \
        --with-library-combo=ng-gnu-gnu --with-runtime-abi=gnustep-2.2 \
        --enable-objc-arc
    make -j "$jobs"
    make install
)
(
    cd "$deps/src/libs-base"
    # The compiler needs no networking or GCD; use C++ call_once instead.
    ./configure --prefix="$prefix" --disable-tls --disable-libdispatch
    make -j "$jobs"
    make install
)
touch "$prefix/.mil-hwx-toolchain-v1"
echo "GNUstep installed in $prefix. Run: make test -j$jobs"
