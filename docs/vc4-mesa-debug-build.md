# Instrumenting Mesa's vc4 driver for the dropped-draw bug

How the debug Mesa on the Pi was built (2026-11-08), what it showed, and how to
rebuild it.  The bug it was built for: a single 1536-wide draw into a 1536×1
target leaves the target unwritten (see `docs/engine-internals.md` §2).

## What was built, and why it is not a small patch

The Pi runs Mesa `26.2.2-1~bpo13+0~rpt1` from Raspbian trixie-backports.  Its
`dri/*_dri.so` are symlinks to `libdril_dri.so`, and the vc4 driver lives inside
`libgallium-26.2.2-1~bpo13+0~rpt1.so`.  There is no per-driver `.so` to patch, so
the whole Gallium library has to be rebuilt (vc4-only).

The exact source is on the USTC Raspbian mirror:

```
http://mirrors.ustc.edu.cn/raspberrypi/debian/pool/main/m/mesa/
  mesa_26.2.2.orig.tar.xz
  mesa_26.2.2-1~bpo13+0~rpt1.debian.tar.xz
  mesa_26.2.2-1~bpo13+0~rpt1.dsc
```

## Build host (107)

Job containers have no compiler; everything is vendored under `$HOME`.  The
parts that were needed beyond `AGENTS.local.md`'s toolchain:

```bash
# host build tools (Debian debs, extracted into ~/tools/cross; no root)
cd ~/mesa/debs
apt-get download meson ninja-build python3-mako python3-yaml \
                 python3-packaging python3-markupsafe bison m4 flex patchelf
for d in *.deb; do dpkg-deb -x "$d" ~/tools/cross; done
```

Two pieces of the vendored tools need environment help because their data dirs
are compiled in as `/usr/share/...`:

```bash
export BISON_PKGDATADIR=$HOME/tools/cross/usr/share/bison
export FLEX_SKELETON=$HOME/tools/cross/usr/share/flex/flex.skl
export M4=$HOME/tools/cross/usr/bin/m4
```

Target dependency missing from the sysroot: `libexpat1-dev` (Mesa's driconf).
The sysroot is root-owned, so:

```bash
~/tools/fetch-armhf.sh libexpat1-dev
sudo dpkg-deb -x ~/debs-armhf/libexpat1-dev_*.deb ~/pi-rootfs
```

## The two things that make it a drop-in replacement

1. **The soname carries the Debian revision.**  Mesa's dri target names the
   library `libgallium-<meson.project_version()>.so` and uses
   `-Wl,--default-symver`, so the *symbol version node* is that soname too.  The
   distro's `libEGL_mesa` and `libdril` reference
   `libgallium-26.2.2-1~bpo13+0~rpt1.so`; an upstream `26.2.2` build produces
   `libgallium-26.2.2.so` and fails at load with "version ... not found".
   Fix: write the full package version into Mesa's `VERSION` file before
   configuring.
2. **Do not mix our `libEGL_mesa` with the distro's.**  The distro builds with
   X11 enabled, so its `libEGL_mesa` needs `loader_dri3_*` symbols from
   `libgallium`.  A `-Dplatforms=` build has no X11 and therefore no dri3
   loader, and the distro `libEGL_mesa` fails to initialise ("eglInitialize
   failed", `ldd -r` shows the undefined `loader_dri3_*`).  Ship **both** our
   `libgallium` and our `libEGL_mesa`; the engine uses the surfaceless platform
   explicitly (`eglGetPlatformDisplayEXT(EGL_PLATFORM_SURFACELESS_MESA, ...)`),
   so no window system is needed.

## Commands

Configure (can run on the login host, ~15 s):

```bash
echo '26.2.2-1~bpo13+0~rpt1' > ~/mesa/src/mesa-26.2.2/VERSION
source ~/tools/armv6env.sh
export PATH=$HOME/tools/cross/usr/bin:$PATH \
       PYTHONPATH=$HOME/tools/cross/usr/lib/python3/dist-packages \
       BISON_PKGDATADIR=$HOME/tools/cross/usr/share/bison \
       FLEX_SKELETON=$HOME/tools/cross/usr/share/flex/flex.skl \
       M4=$HOME/tools/cross/usr/bin/m4
cd ~/mesa/src/mesa-26.2.2
meson setup ~/mesa/build --cross-file ~/mesa/armv6.txt --prefix ~/mesa/prefix \
  -Dbuildtype=debug -Dplatforms= -Dgallium-drivers=vc4 -Dvulkan-drivers= \
  -Dglx=disabled -Degl=enabled -Dgles2=enabled -Dgbm=disabled -Dllvm=disabled \
  -Dexpat=enabled -Dtools= -Dvideo-codecs= -Dwerror=false
```

`armv6.txt` is the usual cross file: `arm-linux-gnueabihf-{gcc,g++}-14`,
`--sysroot=$HOME/pi-rootfs`, the sysroot's `-B` paths, `-march=armv6 -mfpu=vfp
-mfloat-abi=hard`, and `pkg-config` from `~/tools/cross`.

Build **in Slurm**, never on the login host:

```bash
sbatch --wrap='source ~/tools/armv6env.sh; export PATH=$HOME/tools/cross/usr/bin:$PATH PYTHONPATH=$HOME/tools/cross/usr/lib/python3/dist-packages BISON_PKGDATADIR=$HOME/tools/cross/usr/share/bison FLEX_SKELETON=$HOME/tools/cross/usr/share/flex/flex.skl M4=$HOME/tools/cross/usr/bin/m4; ninja -C ~/mesa/build -j16 && ninja -C ~/mesa/build install' -c 16
```

A clean build is ~1.5 min at `-j16`; `-Dbuildtype=debug` is needed because
`MESA_DEBUG` (and therefore `debug_warn_once`, which reports a skipped draw) is
only defined for `buildtype == 'debug'`.

## Deploy

```bash
strip --strip-unneeded ~/mesa/prefix/lib/libgallium-26.2.2-1~bpo13+0~rpt1.so
# two hops: 107 -> laptop -> raspi
#   /home/david/mesa-dbg/lib/libgallium-26.2.2-1~bpo13+0~rpt1.so
#   /home/david/mesa-dbg/lib/libEGL_mesa.so.0.0.0  (+ .so.0 / .so symlinks)
```

Run anything GL with:

```bash
LD_LIBRARY_PATH=/home/david/mesa-dbg/lib <command>
```

The system `libglvnd` loads our `libEGL_mesa.so.0` (vendor library by soname),
which loads our `libgallium`; `libdril_dri.so` from the system `dri/` directory
dlopens the same file, so one copy is used throughout.

## Instrumentation that was added

In `src/gallium/drivers/vc4/vc4_draw.c`:

- at the end of `vc4_draw_vbo()`, just before `vc4_start_draw()`: log
  `framebuffer size, dirty, viewport translate/scale, job pointer, job
  draw_width, needs_flush` when the framebuffer is 1536 wide;
- in `vc4_start_draw()`: log when it fixes `job->draw_width`;
- in the `vc4_update_compiled_shaders()` failure branch: log
  `DROP=shader` with the fs/vs/cs `failed` flags (the `debug_warn_once` message
  is compiled out in release builds).

In `src/gallium/drivers/vc4/vc4_job.c`, at `vc4_job_submit()`: log the job's
`draw_width/height`, `min_x_tile..max_x_tile`, `draw_calls_queued` and
`draw_min_x..draw_max_x`.

`DRAW_CLEAR=1` in the engine (see `docs/test-hooks.md`) fills a draw target
with 0x33/0x66/0x99/0xff before the draw, so a readback tells "the draw did not
execute" apart from "the draw read stale input".

## What the instrumented driver showed

For the qkv pass at `QKV_PAIR=0 MATVEC_STRIP=1536` (one 1536-wide draw):

```
VCDBG draw fb=1536x1 dirty=ffffffff vp t=768.0,0.5 s=768.0,0.5 job=0x1d3af98 jw=0 nf=0
VCDBG start fb=1536x1 job=0x1d3af98
VCDBG submit jw=1536 jh=1 tiles x=0..23 draws=1 bounds=0..1536
```

and the same pass split in two (`MATVEC_STRIP` default):

```
VCDBG draw fb=1536x1 ... job=0x23e1488 jw=0 nf=0
VCDBG start fb=1536x1 job=0x23e1488
VCDBG draw fb=1536x1 dirty=00001000 vp t=1280.0,0.5 s=256.0,0.5 job=0x23e1488 jw=1536 nf=1
VCDBG submit jw=1536 jh=1 tiles x=0..23 draws=2 bounds=0..1536
```

So in both cases the driver emits the right viewport, the right clip and a
24-tile job, and no `DROP=shader` fires - yet only the two-draw job lands.  The
remaining candidate is below the driver's command list (the kernel's RCL
generation or the VC4 binner handling a single draw that covers every tile of
the job), not a Mesa state bug reachable from the app.

## If this is picked up again

- The instrumented build is on 107 at `~/mesa/` (`src/`, `build/`, `prefix/`,
  `logs/`), and the two libraries are on the Pi in `/home/david/mesa-dbg/lib/`.
- To test a change: patch, `sbatch` the ninja build, strip, copy the single
  `libgallium` file, run the engine with `LD_LIBRARY_PATH` and
  `QKV_DUMP=1 QKV_PAIR=0 MATVEC_STRIP=1536`.
- The next instrumentation to add is on the kernel side (`drivers/gpu/drm/v3d`):
  log the RCL/BIN sizes and the ioctl result per submit, and whether the binner
  list for the 24-tile job differs between the one-draw and two-draw cases.
