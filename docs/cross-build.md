# Building for the Pi Zero: cross toolchain and ARMv6 sysroot

The target is a BCM2835 / ARM1176: the binary must carry
`Tag_CPU_arch: v6` and `Tag_FP_arch: VFPv2`, or it faults on the board.
Nothing else in this repository is as easy to get *quietly* wrong, and the
mistake is invisible until the board reboots into a crash.

This page is host-agnostic.  Machine-specific paths and versions belong in
`AGENTS.local.md` / `AGENTS.override.md`, not here.

- [Why it goes wrong](#why-it-goes-wrong)
- [1. The cross compiler](#1-the-cross-compiler)
- [2. The sysroot](#2-the-sysroot)
- [3. Point the build at them](#3-point-the-build-at-them)
- [4. Verification](#4-verification)
- [Troubleshooting](#troubleshooting)

## Why it goes wrong

Debian's armhf cross toolchain is **ARMv7** (`-march=armv7-a+fp`), and so are its
`crt*.o`, `libgcc.a` and `libstdc++`.  Building for the Pi therefore has to
overwrite the compiler's own start files with the sysroot's ARMv6 ones, which the
Makefile does with `-B`:

```make
-B<sysroot>/usr/lib/gcc/arm-linux-gnueabihf/<runtime-major>/   # crtbegin, libgcc
-B<sysroot>/usr/lib/arm-linux-gnueabihf/                       # crt1, libc, libstdc++
```

**gcc silently ignores a `-B` directory that does not exist.**  It then finds its
own ARMv7 start files, the link succeeds with exit status 0 and no warning, and
the resulting binary reports:

| `-B` passed | `Tag_CPU_arch` | `.comment` shows the runtime linked |
|---|---|---|
| the sysroot's real runtime dir | `v6` / `VFPv2` | the sysroot's Raspbian gcc |
| a version that is not present | **`v7` / `VFPv3-D16`** | the host's cross gcc |
| nothing at all | **`v7` / `VFPv3-D16`** | the host's cross gcc |

That is measured, not theorised.  It is also why this repository no longer
hardcodes the runtime version: a sysroot built by a different Raspbian release
has a different `gcc/<major>` directory, and hardcoding it converts a correct
sysroot into a silently wrong binary.

## 1. The cross compiler

Any Debian/Ubuntu armhf cross `g++` works; its version does **not** have to match
the sysroot's compiler runtime (see below).

```bash
sudo apt install g++-arm-linux-gnueabihf    # pulls binutils-arm-linux-gnueabihf
arm-linux-gnueabihf-g++-12 --version        # the versioned name is what gets installed
```

The Makefile discovers the driver: `arm-linux-gnueabihf-g++` first, then the
versioned names from newest to oldest.  Two things are worth knowing:

- **The headers come from this package, not from the sysroot.**  libc headers are
  in the host's `libc6-dev-armhf-cross`, C++ headers in the host's
  `include/c++/<driver-major>`.  EGL/GLES headers and the libraries do come from
  the sysroot.  So a sysroot alone is not enough to build with.
- **No root?**  Installing packages is the only privileged step.  On a machine
  where you cannot use `apt`, unpack the `.deb`s into a prefix and put
  `<prefix>/usr/bin` on `PATH`; the Debian cross packages are ordinary `.deb`s and
  the driver finds its libexec directory relative to itself.  If the packaging is
  unusual, `CROSS_CXX=/path/to/arm-linux-gnueabihf-g++` overrides the discovery.

## 2. The sysroot

It must be a **Raspbian** root filesystem — Debian's armhf userland is ARMv7 and
will not run on a Pi Zero.  A stock image is not enough on its own: the *linker*
needs the `-dev` files (`crt1.o`, `crti.o`, `crtn.o`, the `libc.so` linker
script, `libgcc-*-dev`, and the `libEGL`/`libGLESv2` symlinks), which Lite and
desktop images do not ship.  Install them on the Pi first, then copy that root:

```bash
# on the Pi, once (trixie names; bookworm and older also accept
# libegl1-mesa-dev / libgles2-mesa-dev)
sudo apt install libc6-dev libstdc++-14-dev libegl-dev libgles-dev

# from the build host: reading the whole rootfs needs root on the Pi
rsync -a --delete --numeric-ids --rsync-path="sudo rsync" \
  --exclude=/proc --exclude=/sys --exclude=/dev --exclude=/boot \
  --exclude=/home --exclude=/tmp --exclude=/var/log \
  <pi-host>:/  "$HOME/pi-rootfs/"
```

A bootstrap works too, but it needs two things an x86 host does not have by
default: `qemu-user-static` plus `binfmt-support` (mmdebstrap refuses to build a
foreign architecture without them) and the Raspbian keyring, which is a Raspbian
package and is not in Debian/Ubuntu archives:

```bash
sudo apt install qemu-user-static binfmt-support
curl -fsSL https://archive.raspbian.org/raspbian.public.key \
  | sudo gpg --dearmor -o /usr/share/keyrings/raspbian-archive-keyring.gpg

sudo mmdebstrap --arch=armhf --variant=apt \
  --include=libc6-dev,libegl1,libgles2,libegl-dev,libgles-dev,libstdc++-14-dev \
  --keyring=/usr/share/keyrings/raspbian-archive-keyring.gpg \
  trixie "$HOME/pi-rootfs" http://raspbian.raspbian.org/raspbian
```

A mirror and suite name are checks against that mirror's release file; the
packages this build needs are exactly the ones in the `--include` line, so
compare the two lists if the suite moved on.

Check that what the link actually needs is present — `check-cross` tests these
two, but a sysroot can be broken in ways only the linker sees:

```bash
S=$HOME/pi-rootfs
readelf -A "$S/usr/lib/arm-linux-gnueabihf/libc.so.6" | grep -E 'Tag_CPU_arch|Tag_FP_arch'
#   want: Tag_CPU_arch: v6        Tag_FP_arch: VFPv2
ls -d "$S"/usr/lib/gcc/arm-linux-gnueabihf/*/          # the runtime dir(s); >=1 required
ls    "$S"/usr/lib/arm-linux-gnueabihf/{crt1.o,crti.o,crtn.o,libc.so}   # start files
ls    "$S"/usr/lib/arm-linux-gnueabihf/libEGL.so* "$S"/usr/lib/arm-linux-gnueabihf/libGLESv2.so*
```

Keep the merged-`/usr` layout intact: if `$S/lib` is not the usual symlink to
`usr/lib`, the sysroot's own linker scripts (`GROUP ( /lib/arm-linux-gnueabihf/… )`)
stop resolving and the link fails with "cannot find /lib/arm-linux-gnueabihf/libc.so.6".

## 3. Point the build at them

```bash
make -C engine -B pi CROSS_SYSROOT=$HOME/pi-rootfs
```

There is no default sysroot: a default is wrong on every host but one.  The
recipe first runs `check-cross`, which prints what it resolved and fails on any
missing piece:

```
cross: arm-linux-gnueabihf-g++-12 (12)
  sysroot /home/you/pi-rootfs
  runtime /home/you/pi-rootfs/usr/lib/gcc/arm-linux-gnueabihf/14
  libs    /home/you/pi-rootfs/usr/lib/arm-linux-gnueabihf
```

Overrides, in the order you are likely to need them:

| variable | when |
|---|---|
| `CROSS_SYSROOT` | always; the rootfs path |
| `CROSS_CXX` | the driver is named differently, or you have several |
| `CROSS_GCC_DIR` | the sysroot contains several runtime dirs and the glob picked the wrong one |
| `CROSS_CXXFLAGS`, `CROSS_LDLIBS` | a genuinely different target (not needed for the Pi Zero) |
| `CROSS_READELF` | verification needs a working ELF reader (any host `readelf` reads ARM ELF) |

Runtime selection: the directory matching the *driver's* major wins, otherwise
the highest version present.  A driver/runtime major mismatch is fine and is what
this repository has always shipped: the driver only compiles, while `crtbegin.o`,
`libgcc`, `libstdc++` and glibc all come from the sysroot as one consistent set.
The old `CROSS_GCCLIB` variable is gone — no `-B` path is written down anywhere.

## 4. Verification

`make pi` refuses to leave an invalid artifact:

```
  engine_mm3_pi: ARMv6/VFPv2, GCC: (Raspbian 14.2.0-19+rpi1)
```

The `GCC:` field is the sysroot runtime that was actually linked, read from
`.comment`; it is the cheapest positive proof that the `-B` substitution
happened.  A host-only compiler version there means the sysroot was not used.
To check by hand, on any machine:

```bash
readelf -A engine/engine_mm3_pi | grep -E 'Tag_CPU_arch:|Tag_FP_arch:'
readelf -p .comment engine/engine_mm3_pi | grep GCC      # look for the Raspbian entry
```

Also worth checking, because the board's glibc is fixed:

```bash
readelf --version-info engine/engine_mm3_pi | grep -o 'GLIBC_[0-9.]*' | sort -Vu | tail -1
# must not exceed the glibc on the Pi (2.41 for trixie-era Raspbian)
```

On the board itself, `make -C engine -B armv6` builds natively with the same
flag set and no sysroot, and needs no verification — the host compiler already
targets ARMv6.  It uses `ARMV6_CXXFLAGS`, not `CXXFLAGS`, so a script left over
from the old `make … CXXFLAGS='-march=armv6 …'` habit silently builds with the
default flags; override `ARMV6_CXXFLAGS` instead.

## Troubleshooting

| symptom | cause |
|---|---|
| `Tag_CPU_arch: v7` in the artifact | the `-B` runtime dir was wrong or missing; `check-cross` now fails earlier, so this means `CROSS_GCC_DIR` was overridden by hand |
| `.comment` lists only the host gcc | same as above — the sysroot was not used |
| `cannot find /lib/arm-linux-gnueabihf/libc.so.6` | `$SYSROOT/lib` is missing (merged-`/usr` broken), or the sysroot was assembled file by file |
| `undefined reference to std::__throw_bad_array_new_length` and friends | the *runtime* `libstdc++6` in the sysroot is older than the host driver's headers expect — use a driver no newer than the sysroot's compiler runtime, or update the sysroot |
| `fatal error: EGL/egl.h: No such file` | the sysroot lacks the EGL/GLES dev packages |
| build works, board faults immediately | the binary is ARMv7; run the verification above |
| `check-cross` cannot find a runtime, but `ls` shows one | `CROSS_SYSROOT` points at the wrong level (it must contain `usr/lib/arm-linux-gnueabihf`) |

`docs/vc4-1536-repro.md` and `docs/benchmarks.md` contain historical build
lines for specific hosts; those are records of runs, not the supported commands.
