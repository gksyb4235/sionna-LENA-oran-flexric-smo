# RET downtilt sweep results

Run from the repository root after generating matching ray-tracing caches:

```bash
./scenarios/khu-real/tools/run_ret_sweep.sh
```

The default sweep applies 1, 3, 5, 7, 9, and 12 degrees of downtilt to all
three gNBs simultaneously. Bearings remain 20, 0, and 10 degrees for gNB_5G,
gNB_4G_1, and gNB_4G_2. Every tilt needs both a 1.8 GHz and 3.5 GHz cache.

The default cache layout is:

```text
scenarios/khu-real/ret_caches/
  tilt_1deg/
    sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5
    sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5
  tilt_5deg/
    ...
```

The runner checks every cache before starting, so a partial sweep cannot start
with mismatched geometry. Override the layout with `RET_CACHE_ROOT`, or with
`RET_CACHE18_TEMPLATE` and `RET_CACHE35_TEMPLATE`. Templates accept `{value}`
and `{label}` placeholders.
