# Task: restyle the SeraphByte frontend (website/) to the new design

Design reference (open in browser): https://claude.ai/artifact/LNMgCEmD6MsJ5zXhbzzSd9
Also attached: seraphbyte_design_reference.html (the design source, static mock) and
seraphbyte-logo-transparent.svg (logo with no background, for the header mark, hero and faint watermark).

Keep it plain HTML/CSS/JS. No build step, no Node server. Keep working offline:
bundle fonts and libs locally in website/ (fonts/, vendor/), no CDN links.

## Look
Illuminated-manuscript structure, modern restraint. Hairline rules instead of filled cards, square corners, no pills or bubbles.
- Ground #0e0a0b, ivory text #f3e9d8, muted #cdb996, gold #d9b66a (bright #f2d58a), ember #ff9a4a only as a flame accent.
- Hairlines: rgba(217,182,106,0.2 to 0.35). Thin gold frame inset 14px around the page.
- Fonts (woff2 latin, via @fontsource npm, copy files into website/fonts/): Cormorant Garamond 500/600/700 + 500 italic (titles, big numerals),
  Newsreader 400/500 + 400 italic (chat body 19px/1.7), JetBrains Mono 400/500 (code, data).
- Labels: 11px uppercase, letter-spacing 0.22em. Sessions numbered with roman numerals (I., II., ...).
- Gold diamond divider between turns. Assistant label "Seraph" in gold, user label "You", user text italic.
- Right panel is a ledger: dotted leaders between label and value, 3px bars, large serif numerals.
- Lineage: six hairline bars I to VI, gen VI glowing gold-ember. Decorative.
- Responsive: three columns (index / chat / engine) that stack on narrow screens. Touch targets 44px. Text contrast 4.5:1.

## Layout (see reference)
Left: logo mark + "SERAPHBYTE / Engine · Gen VI", New conversation, session list (Today / Earlier), loaded model at bottom.
Center: header (conversation title, model + ws url, Export, Clear), chat log, composer (rule above, textarea, Stop, Send).
Right: Engine status, tok/s and first-token numerals, context bar, Sampling sliders, System prompt textarea, Lineage.
Empty state: big faint logo and "Speak to the engine."

## What the engine really provides (read seraph_engine/src/main.rs)
- ws://127.0.0.1:8543. On connect it sends one JSON frame {"type":"model_info","model":...}. Then every later text frame is raw token text.
- Client sends {prompt, temperature, top_p, max_tokens, system_prompt}. Each prompt is stateless (system + one user turn), no history.
- It does NOT send: end-of-stream, VRAM, tok/s, a model list. One model is picked at startup.
So wire for real what exists, and do not fake the rest:
- Real: connection state, model name, streaming Markdown, the four per-message params, system prompt.
- Measure in the browser: first-token latency, tok/s (frames / time since first frame), est. context (chars/4, CTX_MAX 8192 as in the old app).
- Sessions: localStorage only (display transcripts; the model does not see history yet). Keep the existing pinned-memory localStorage key seraph_memory_v1.
- Drop from the mock: the VRAM bar and the second "not loaded" model. Show the one loaded model.
- Stop button: close the socket and reconnect. The engine notices the dead client on its next token send and stops.

## Fixes to carry along
1. The old app only finalizes a reply when the socket closes. Optional engine patch: after the `while let Ok(token) = rx_tokens.recv()` loop in main.rs, count frames
   and send {"type":"done","tokens":n}. Frontend: finalize on "done"; if never seen, fall back to ~2.5s idle after the last token.
2. Only treat a frame as a control frame if it parses as an object with type in {"model_info","done"}. The old check (startsWith('{')) silently drops a token like `{}`.
3. Render Markdown with marked, then sanitize with DOMPurify before innerHTML (marked does not sanitize). Throttle re-render with requestAnimationFrame.
   Keep the unclosed-fence patch while streaming. Use event delegation for code-block Copy buttons since the DOM is rebuilt each frame.
4. Possible follow-up (not part of this task): make the engine accept a messages array so sessions can carry history.
