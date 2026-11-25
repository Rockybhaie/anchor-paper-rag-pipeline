# Runbook

How to run, redeploy and roll back the Anchor Paper RAG API. Anyone on the team should be able to follow it without asking.

## The service

| Item | Value |
|---|---|
| Live URL | See the [Deployment](README.md#deployment) section of the README |
| Platform | Render, free web service, Docker runtime, deployed from the `main` branch of this repository |
| Health check | `GET /health` returns `{"status": "ok", ...}` and calls no outside service |
| Monitoring | `GET /stats` returns the request count, average latency and the live judge's scores since the last start |
| Secrets | `OPENAI_API_KEY`, `PINECONE_API_KEY`, `GEMINI_API_KEY`, set in Render under Environment. Never in the repository |

| Endpoint | Use |
|---|---|
| `GET /health` | Is the service up, and which anchors can it answer? |
| `GET /stats` | Request count, average latency in ms, counts by path, live judge scores |
| `POST /predict` | Answer a question with sources |
| `POST /explain` | The passages behind an answer |
| `GET /docs` | Interactive API documentation |

## Before a demo: wake the service

The free plan sleeps after about 15 minutes without traffic, and the first request after that can take a minute or more. Open `/health` a few minutes before any demo and wait until it answers.

On every start, the API rebuilds its passages from Pinecone and checks them against `data/index_manifests/`, which takes a few more seconds. `/health` only answers once that has finished.

## Redeploy

- **Normal change:** merge a pull request into `main`. Render builds the Dockerfile and deploys automatically. Watch the deploy under **Events** in the Render dashboard.
- **Same code again,** for example after changing a secret: in the Render dashboard, open the service and choose **Manual Deploy**, then **Deploy latest commit**.
- **Check it worked:** `/health` returns `ok`, then send one `/predict` request and confirm that `/stats` counts it.

## Roll back

If a deploy breaks the service:

1. **Fastest, on Render:** open the service, go to **Events**, find the last deploy that worked, and choose **Rollback** on it. Render redeploys that exact build.
2. **In Git, so `main` matches what is running:** revert the bad commit and merge it through a pull request. Render then deploys the reverted code.

   ```bash
   git checkout main && git pull
   git revert <bad commit>
   git push origin HEAD:fix/revert-bad-deploy    # Then open and merge the pull request
   ```

3. **Confirm:** `/health` returns `ok` and a `/predict` request succeeds.

If the cause is a secret, such as an expired key, fix it under **Environment** and redeploy. No rollback is needed.

## Run a version locally

Every image is tagged with a version, so an older one can always be run again.

```bash
docker build -t anchor-rag:v2.0 .                          # Build the current version
docker images anchor-rag                                   # List the versions you have
docker run --env-file .env -p 8000:8000 anchor-rag:v2.0    # Run one
docker ps                                                  # STATUS shows (healthy) after about a minute
docker logs <container id>
docker stop <container id>
```

To run an older release that you have not built yet, check out its Git tag first, for example `git checkout v2.0-milestone2`, then build it with that tag as the image version.

If port 8000 is taken, map another one: `-p 8001:8000`.

On a slow connection, the first start can take a couple of minutes while the passages are rebuilt from Pinecone. If this machine already has the built index, add `-v "${PWD}/artifacts:/app/artifacts"` to `docker run` to skip that.

## Common problems

| Symptom | Likely cause | Fix |
|---|---|---|
| `/health` never answers after a deploy | The start failed. Read the deploy log in Render | Usually a missing secret. Check all three keys under **Environment** |
| Start fails with "carry no passage text" | The Pinecone vectors were uploaded before they carried their passages | Run `python -m src.build_index --attach-text` once, from a machine with the built index |
| Start fails with "do not match" | The passages in Pinecone differ from the committed manifest | Rebuild the index with `python -m src.build_index` and commit the new `data/index_manifests/` files |
| `/predict` returns 503 | A model or Pinecone call failed | Check the provider's status and the key. The answer explains which service failed |
| First request takes a minute | The free service was asleep | Expected. Wake it before demos |
| `/stats` counts went back to zero | The service restarted or slept | Expected. Counts live in memory |

## Rotate a secret

1. Create a new key with the provider.
2. Replace it under **Environment** in Render. Render redeploys with the new value.
3. Delete the old key with the provider.
4. Update your local `.env` if you use the same key there.
