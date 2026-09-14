#!/usr/bin/env python3
"""Does the reference safetensors implementation accept what Moxie publishes?

[ADR 0022]'s 2026-09-14 amendment asks for exactly this: "Test shards with the
reference safetensors reader as well as the production Moxie reader". A format
decision whose only evidence is our own reader has not been checked against the
format.

This publishes a selection with `moxie-repack`, opens every shard with the
reference implementation -- `safetensors.deserialize`, which is the Rust crate
through its Python binding, returning raw bytes without dtype conversion -- and
compares, per component:

  * the dtype and shape the reference reports against the manifest's claim;
  * the bytes the reference returns against the SHA-256 the manifest records.

Usage:  python3 tools/experiments/0007-reference-reader.py <artifact-dir>

Exit status is zero only if every component matched.

[ADR 0022]: ../../decisions/adr/0022-user-programs-and-canonical-write-authority.md
"""

import hashlib
import json
import os
import sys
import tomllib

EXPECTED_DTYPE = {
    "codes": {"U8", "I8"},
    "scales": {"F16", "BF16", "F32"},
    "zero_points": {"I16"},
    "weights": {"BF16"},
}

SCALE_BYTES = {"F16": 2, "BF16": 2, "F32": 4}
GROUP_SIZE = {"contiguous-32": 32, "contiguous-128": 128}


def expected(tensor):
    """The physical components ADR 0025 derives from a manifest tensor.

    Independent review changed a manifest's logical shape from `[8, 8]` to
    `[8, 7]` and this driver still reported that every component matched: it
    compared each component against the shard's own header, and the shard's
    header against nothing. A dtype-and-membership check is not a schema check.
    The descriptor is recomputed here, from the manifest's logical shape and
    affine fields, and the components are held against **that**.
    """
    shape = tensor["shape"]
    precision = tensor["precision"]
    role = tensor["role"]
    if precision == "bf16-v1":
        n = 1
        for d in shape:
            n *= d
        return [("weights", role, {"BF16"}, list(shape), n * 2)]

    out = 1
    for d in shape[:-1]:
        out *= d
    in_features = shape[-1]
    rule = tensor["group_rule"]
    if rule == "per-channel":
        groups = 1
    else:
        size = GROUP_SIZE[rule]
        if in_features % size:
            raise SystemExit(
                f"FAIL: {role}: {in_features} input(s) is not a whole number of {size}-groups"
            )
        groups = in_features // size
    scale_dtype = tensor["scale_dtype"].upper()
    if precision == "affine-int4-v1":
        code_dtype, width = {"U8"}, (in_features + 1) // 2
    elif precision == "affine-int8-v1":
        code_dtype, width = {"I8"}, in_features
    else:
        raise SystemExit(f"FAIL: {role}: unknown precision {precision}")

    want = [
        ("codes", f"{role}.codes", code_dtype, [out, width], out * width),
        (
            "scales",
            f"{role}.scales",
            {scale_dtype},
            [out, groups],
            out * groups * SCALE_BYTES[scale_dtype],
        ),
    ]
    if tensor["zero_point"] == "per-group":
        want.append(
            ("zero_points", f"{role}.zero_points", {"I16"}, [out, groups], out * groups * 2)
        )
    return want


def main(artifact):
    import safetensors
    from safetensors import deserialize

    manifest = tomllib.load(open(os.path.join(artifact, "manifest.toml"), "rb"))
    if manifest["schema_version"] != 2:
        print(f"FAIL: schema_version is {manifest['schema_version']}, not 2")
        return 1

    # Every shard, through the reference implementation.
    shards = {}
    for name in sorted(os.listdir(artifact)):
        if not name.endswith(".safetensors"):
            continue
        blob = open(os.path.join(artifact, name), "rb").read()
        try:
            shards[name] = dict(deserialize(blob))
        except Exception as e:  # the format's own verdict on our file
            print(f"FAIL: {name} was REFUSED by the reference reader: {e}")
            return 1
        print(f"ok   {name}: reference reader accepted {len(blob)} bytes, "
              f"{len(shards[name])} tensors")

    checked = 0
    for tensor in manifest["tensors"]:
        role = tensor["role"]
        want = expected(tensor)
        got = tensor["components"]
        if len(got) != len(want):
            print(f"FAIL: {role}: manifest carries {len(got)} component(s), "
                  f"its descriptor implies {len(want)}")
            return 1
        for component, (kind, name, dtypes, shape, length) in zip(got, want):
            # Order is part of the contract: the components concatenate into the
            # canonical payload, codes then scales then zero points.
            if component["kind"] != kind or component["name"] != name:
                print(f"FAIL: {role}: component {component['kind']} '{component['name']}' "
                      f"is not the '{name}' the descriptor implies at that position")
                return 1
            shard = shards.get(component["file"])
            if shard is None:
                print(f"FAIL: {role}: no shard {component['file']}")
                return 1
            info = shard.get(component["name"])
            if info is None:
                print(f"FAIL: {role}: shard carries no tensor {component['name']}")
                return 1
            if info["dtype"] not in EXPECTED_DTYPE[kind]:
                print(f"FAIL: {component['name']}: dtype {info['dtype']} is not "
                      f"one of {sorted(EXPECTED_DTYPE[kind])} for a {kind} component")
                return 1
            # Against the descriptor, not against the shard's own claim.
            if info["dtype"] not in dtypes:
                print(f"FAIL: {component['name']}: dtype {info['dtype']} is not "
                      f"{sorted(dtypes)}, which this descriptor implies")
                return 1
            if list(info["shape"]) != shape:
                print(f"FAIL: {component['name']}: shape {list(info['shape'])} is not "
                      f"the {shape} this descriptor implies")
                return 1
            if len(info["data"]) != length:
                print(f"FAIL: {component['name']}: {len(info['data'])} byte(s) is not "
                      f"the {length} this descriptor implies")
                return 1
            digest = hashlib.sha256(info["data"]).hexdigest()
            if digest != component["sha256"]:
                print(f"FAIL: {component['name']}: the reference reader's bytes hash "
                      f"to {digest}, the manifest records {component['sha256']}")
                return 1
            print(f"ok   {component['name']}: dtype={info['dtype']} "
                  f"shape={info['shape']} bytes={len(info['data'])} sha256 matches")
            checked += 1

    print(f"\n{checked} component(s) matched the manifest, read by "
          f"safetensors {safetensors.__version__}")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
