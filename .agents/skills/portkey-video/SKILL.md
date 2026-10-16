---
name: portkey-video
description: |
  Generate AI videos through the Portkey AI gateway with a single API key.
  Use when: (1) Generating videos routed via Portkey (@provider-slug/model),
  (2) Using one gateway key for Gemini/Veo, Seedance, Sora, Kling, MiniMax
  instead of per-vendor keys, (3) Applying gateway-level routing, fallbacks,
  or virtual keys, (4) Debugging Portkey video task failures.
metadata:
  openclaw:
    requires:
      env:
        - PORTKEY_API_KEY
---

# Video Generation via Portkey Gateway

The `portkey_video` tool routes video generation through Portkey
(https://portkey.ai). One gateway key (`PORTKEY_API_KEY`) covers every
video model configured in the Portkey dashboard — OpenMontage needs no
per-vendor env vars for models reached through this path.

## How routing works

Models are addressed as `@provider-slug/model`. The `@slug` tells the
gateway which vendor integration (or virtual key) serves the request:

| Model string | Vendor | Notes |
|---|---|---|
| `@google/veo-3.1` | Google (Veo) | text/image-to-video, native audio |
| `@openai/sora-2` | OpenAI (Sora) | native audio; `-pro` variant costs more |
| `@fal-ai/...` | fal.ai | Seedance, Kling, MiniMax catalog |
| `@byteplus/seedance-...` | BytePlus/Seedance | Seedance via BytePlus routing |

Slugs are configured in the Portkey dashboard (Integrations / Virtual
Keys) — the exact slug shown there is what goes in `model`. If a slug is
uncertain, check `GET /v1/providers` on the gateway or the dashboard's
model catalog rather than guessing.

## Requests

- `POST {base}/videos` — create; body: `prompt`, `model`, optional
  `seconds`, `size`. Image-to-video posts multipart form with
  `input_reference` as a file field.
- `GET {base}/videos/{id}` — poll; terminal when `status` is
  `completed`/`succeeded`/`failed`/`cancelled`.
- `GET {base}/videos/{id}/content` — stream the finished MP4.

Auth header is `x-portkey-api-key: <PORTKEY_API_KEY>`. Optional headers:
`x-portkey-virtual-key` (per-request credential selection) and
`x-portkey-config` (saved routing config with fallbacks).

## Prompting

The underlying vendor model sees the prompt — write for the vendor, not
for Portkey. Follow the vendor's Layer 3 skill: `seedance-2-0` for
Seedance (camera direction, multi-shot, quoted dialogue for lip-sync),
`ai-video-gen` for general video prompt structure. Portkey adds no
prompt transformation.

## Cost and governance

Portkey bills per vendor usage plus any gateway fees. `estimate_cost`
uses per-second planning rates (`veo` ≈ $0.40/s, `sora-2-pro` ≈ $0.50/s,
`seedance` ≈ $0.05/s); pass `cost_per_second_usd` when the dashboard
shows different pricing. Gateway-level budgets and rate limits are
configured in the Portkey dashboard, not in OpenMontage.

## Failure handling

- `401/403` — gateway key wrong, or the vendor integration/virtual key
  lacks the model. Fix in the Portkey dashboard.
- `404` on create — the model slug doesn't exist for the account; check
  the dashboard catalog.
- Task `failed` with vendor error — retry per the vendor skill (often a
  prompt policy issue), not a Portkey problem.
- Poll timeout — rerun with `task_action: "status"` and the same
  `task_id`; long jobs (Veo 4k, Seedance multi-shot) can exceed 15 min.
