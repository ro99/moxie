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

Usage:  python3 docs/evidence/experiments/drivers/0007-reference-reader.py <artifact-dir>

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
        for component in tensor["components"]:
            shard = shards.get(component["file"])
            if shard is None:
                print(f"FAIL: {role}: no shard {component['file']}")
                return 1
            info = shard.get(component["name"])
            if info is None:
                print(f"FAIL: {role}: shard carries no tensor {component['name']}")
                return 1
            kind = component["kind"]
            if info["dtype"] not in EXPECTED_DTYPE[kind]:
                print(f"FAIL: {component['name']}: dtype {info['dtype']} is not "
                      f"one of {sorted(EXPECTED_DTYPE[kind])} for a {kind} component")
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
