# P2K-G9 Current-Door Passive Portal Shadow

This independent audit component is default-off and is not referenced by a production launch file. It reads only `/team/livox/scan_cloud_filtered` plus TF, reuses the G7R Ray Evidence Core and frozen G8 METHOD_3/METHOD_4, and publishes only `/audit/p2kg9/portal_candidate`, `/audit/p2kg9/portal_status`, and `/audit/p2kg9/event_journal`.

It has no Truth/world input, entry pose, target, state-machine event, or velocity publisher. Its complete rollback is stopping its process and removing this isolated directory. Manual-only start after the production stack is already running:

```bash
bash scripts/p2kg9_portal_shadow/run_portal_shadow.sh
```
