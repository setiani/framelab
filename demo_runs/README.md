# Demo runs

Cached finished runs behind the "Try a sample" cards, created by `scripts/build_demo_cache.py`
(see `docs/plan/07-backend-api-and-frontend.md` §3). They are read-only and are viewed or replayed
without any API call, so they keep working when the server has no API key.

Rebuild after recording (the run ids come from `runs/`, newest first in the app):

```
uv run python scripts/build_demo_cache.py \
    --sample free_fall=r_... --sample pendulum=r_... --sample trap=r_... \
    [--series s_...]
```

`samples.json` lists the cards in order. Keep the folder under 50 MB so it can be committed.
