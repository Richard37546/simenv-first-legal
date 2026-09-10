# P2K-G7R Ray Evidence Shadow

This directory is independent of production L3V and is not referenced by any production launch script. It is default-off and audit-only. It subscribes to filtered L1S, gated odom and the existing local grid; it publishes only `/audit/p2kg7r/ray_evidence_status` and writes under `debug/odom_accuracy_audit_v1/p2kg7r_ray_evidence_shadow_closure_045/`.

It has no portal, target, state-transition or control publisher. Each source cloud is atomically persisted as a compressed sparse NPZ containing direct measured rays only. Stopping its process is the complete runtime rollback.

Manual-only startup after the formal stack is already running:

```bash
bash scripts/l3v_ray_evidence_shadow/run_ray_evidence_shadow.sh
```
