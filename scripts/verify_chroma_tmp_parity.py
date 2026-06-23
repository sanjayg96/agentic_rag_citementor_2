"""Phase 2 parity check: bundled Chroma vs. the /tmp relocated copy.

Lambda copies the read-only Chroma bundle to /tmp on cold start (see
``src/core/storage.resolve_chroma_path``). This script proves that relocation is
loss-less: it runs a fixed set of eval queries through the retriever twice — once
reading ``storage/chroma_db`` directly, once forcing the /tmp copy — and asserts
the retrieved chunk IDs (and order) match exactly.

Each mode runs in its own subprocess because ``resolve_chroma_path`` memoises the
resolved path per process. Requires OPENAI_API_KEY (openai embedding mode).

Usage:  .venv/bin/python scripts/verify_chroma_tmp_parity.py
"""

import json
import os
import subprocess
import sys

EVAL_QUERIES = [
    "What does it mean to live a virtuous life?",
    "How should a leader treat their followers?",
    "What is the nature of true friendship?",
    "How do we find meaning in suffering?",
    "What is the role of reason in human happiness?",
]


def _retrieve_ids() -> list[list[str]]:
    """Run the eval queries through a fresh retriever; print chunk IDs as JSON.

    Invoked as a subprocess so the CHROMA_COPY_TO_TMP env (and the memoised path)
    is isolated per mode.
    """
    from src.core.retriever import HybridRetriever

    retriever = HybridRetriever()
    out = []
    for q in EVAL_QUERIES:
        results = retriever.retrieve([q])
        out.append([r["id"] for r in results])
    print(json.dumps(out))
    return out


def _run_mode(copy_to_tmp: bool) -> list[list[str]]:
    env = dict(os.environ)
    # Run from repo root so the core's relative paths (config/, storage/) resolve,
    # and make the repo importable as ``src`` in the child.
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = repo_root + os.pathsep + env.get("PYTHONPATH", "")
    env["CHROMA_COPY_TO_TMP"] = "1" if copy_to_tmp else "0"
    # Use a clean /tmp target each run so we exercise the actual copy.
    if copy_to_tmp:
        env["CHROMA_TMP_DIR"] = "/tmp/chroma_db_parity"
        subprocess.run(["rm", "-rf", "/tmp/chroma_db_parity"], check=True)
    proc = subprocess.run(
        [sys.executable, __file__, "--child"],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise SystemExit(f"child run (copy_to_tmp={copy_to_tmp}) failed")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main() -> None:
    if "--child" in sys.argv:
        _retrieve_ids()
        return

    print("Running retrieval against bundled storage/chroma_db ...")
    bundled = _run_mode(copy_to_tmp=False)
    print("Running retrieval against /tmp relocated copy ...")
    relocated = _run_mode(copy_to_tmp=True)

    ok = True
    for q, a, b in zip(EVAL_QUERIES, bundled, relocated):
        match = a == b
        ok = ok and match
        print(f"[{'OK ' if match else 'MISMATCH'}] {q}")
        if not match:
            print(f"    bundled : {a}")
            print(f"    /tmp    : {b}")

    print("\nPARITY:", "PASS ✅" if ok else "FAIL ❌")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
