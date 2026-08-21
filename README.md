## Approach: Invisible Content Hash via Token-Level Watermarking

The core idea is to encode a secret hash into generated text by subtly biasing
token choices, without making the output look unnatural. The hash is _invisible_
(statistical, not visible markers) and _split-invariant_ (detectable on any
sub-span of the text).

### High-Level Algorithm

**1. Token Coloring (Deterministic Hashing)**

Each token ID is independently classified as "green" or "red" by hashing the
token ID with a secret seed. The hash of `(seed, token_id)` produces a single
bit — 1 means green, 0 means red. Crucially, this classification depends _only_
on the token ID itself, not on its position in the text or the surrounding
context.

**2. Seeded Generation (Encoding the Hash)**

During text generation, at each step the model produces a distribution over the
vocabulary. Among the top candidates (filtered via nucleus sampling), those with
near-equal logit scores are identified. For these ambiguous choices, green
tokens receive a score boost, making them more likely to be sampled. Tokens that
are clearly best (far above the competition) are chosen regardless of color.

This means the watermark only influences decisions the model was already
uncertain about — the output remains natural-looking.

**3. Statistical Detection (Reading the Hash)**

To verify a passage contains the watermark, count the fraction of green tokens
and compare it against the expected 50% baseline using a z-score test. A
significantly elevated green fraction (z ≥ 1.65 for ~95% confidence) indicates
the secret key was used during generation.

**4. Split Invariance**

Because token coloring depends only on the token ID (not position or context),
any contiguous sub-span of text can be independently checked. The z-score
detection works on sentence fragments, paragraphs, or any portion of the passage
— the watermark signal persists regardless of where you start reading.

### Key Properties

- **Invisible**: No visible markers, delimiters, or artifacts — the text reads
  normally
- **Keyed**: Only someone with the secret seed can detect the watermark; wrong
  keys show no signal
- **Split-invariant**: Detection works on any sub-span, enabling paragraph-level
  verification
- **Reproducible**: Same seed + same prompt always produces the same output
- **Natural**: Watermark bias only applies to near-equal token choices,
  preserving fluency

### Sample Output

Colored text output from the notebooks in [`notebooks/`](notebooks/), rendered
with ANSI colors (shown on GitHub and in most markdown viewers). Every token is
colored by watermark match against the detection seed.

#### Token-level watermark — [`notebook_watermark.ipynb`](notebooks/notebook_watermark.ipynb)

Prompt: *"What is a rainbow?"* · seed `default-seed` · top-k 20 · top-p 0.95

```ansi
[1;38;5;28mgreen[0m = green-list (key match)   [38;5;203mred[0m = red-list (no match)
```

**🟩 Watermarked output**

```ansi
[38;5;203mA[0m[38;5;203m rainbow[0m[1;38;5;28m is[0m[38;5;203m a[0m[1;38;5;28m natural[0m[1;38;5;28m phenomenon[0m[38;5;203m caused[0m[38;5;203m by[0m[1;38;5;28m the[0m[38;5;203m ref[0m[38;5;203mraction[0m[38;5;203m,[0m[38;5;203m reflection[0m[38;5;203m,[0m[38;5;203m and[0m[1;38;5;28m dispersion[0m[1;38;5;28m of[0m
[1;38;5;28msunlight[0m[1;38;5;28m within[0m[38;5;203m water[0m[1;38;5;28m droplets[0m[1;38;5;28m in[0m[1;38;5;28m the[0m[1;38;5;28m air[0m[38;5;203m.[0m[1;38;5;28m When[0m[1;38;5;28m sunlight[0m[1;38;5;28m enters[0m[38;5;203m a[0m[38;5;203m ra[0m[38;5;203mind[0m[38;5;203mrop[0m[38;5;203m,[0m[1;38;5;28m it[0m[1;38;5;28m b[0m[38;5;203mends[0m[38;5;203m and[0m
[1;38;5;28mspreads[0m[1;38;5;28m across[0m[1;38;5;28m the[0m[1;38;5;28m surface[0m[38;5;203m,[0m[38;5;203m creating[0m[38;5;203m a[0m[1;38;5;28m circular[0m[1;38;5;28m pattern[0m[1;38;5;28m of[0m[1;38;5;28m colors[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[1;38;5;28m bottom[0m[1;38;5;28m of[0m[1;38;5;28m the[0m[1;38;5;28m drop[0m[38;5;203m.[0m[38;5;203m As[0m
[1;38;5;28mthe[0m[1;38;5;28m sunlight[0m[1;38;5;28m ex[0m[38;5;203mits[0m[1;38;5;28m the[0m[1;38;5;28m drop[0m[38;5;203m,[0m[1;38;5;28m it[0m[1;38;5;28m dispers[0m[38;5;203mes[0m[38;5;203m into[0m[1;38;5;28m the[0m[1;38;5;28m air[0m[38;5;203m,[0m[38;5;203m creating[0m[1;38;5;28m the[0m[1;38;5;28m colors[0m[1;38;5;28m that[0m[1;38;5;28m are[0m[38;5;203m visible[0m[1;38;5;28m to[0m
[1;38;5;28mthe[0m[1;38;5;28m human[0m[1;38;5;28m eye[0m[38;5;203m—[0m[38;5;203mtyp[0m[1;38;5;28mically[0m[1;38;5;28m red[0m[38;5;203m,[0m[1;38;5;28m orange[0m[38;5;203m,[0m[38;5;203m yellow[0m[38;5;203m,[0m[1;38;5;28m green[0m[38;5;203m,[0m[38;5;203m blue[0m[38;5;203m,[0m[1;38;5;28m ind[0m[1;38;5;28migo[0m[38;5;203m,[0m[38;5;203m and[0m[38;5;203m violet[0m[38;5;203m.[0m

[38;5;203mR[0m[38;5;203main[0m[1;38;5;28mbows[0m[1;38;5;28m usually[0m[1;38;5;28m appear[0m[1;38;5;28m at[0m[1;38;5;28m certain[0m[1;38;5;28m angles[0m[1;38;5;28m relative[0m[1;38;5;28m to[0m[1;38;5;28m the[0m[1;38;5;28m horizon[0m[38;5;203m,[0m[38;5;203m most[0m[38;5;203m commonly[0m[1;38;5;28m at[0m[1;38;5;28m around[0m[1;38;5;28m [0m[38;5;203m42[0m
[1;38;5;28mdegrees[0m[1;38;5;28m to[0m[1;38;5;28m the[0m[1;38;5;28m left[0m[1;38;5;28m of[0m[1;38;5;28m the[0m[38;5;203m sun[0m[38;5;203m and[0m[1;38;5;28m [0m[1;38;5;28m30[0m[1;38;5;28m degrees[0m[1;38;5;28m to[0m[1;38;5;28m the[0m[38;5;203m right[0m[1;38;5;28m of[0m[1;38;5;28m the[0m[1;38;5;28m horizon[0m[38;5;203m.[0m[38;5;203m The[0m[1;38;5;28m colors[0m[1;38;5;28m that[0m
[1;38;5;28mdominate[0m[38;5;203m a[0m[38;5;203m rainbow[0m[1;38;5;28m are[0m[1;38;5;28m usually[0m[1;38;5;28m red[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[1;38;5;28m longest[0m[1;38;5;28m wavelengths[0m[38;5;203m,[0m[1;38;5;28m but[0m[1;38;5;28m there[0m[1;38;5;28m are[0m[1;38;5;28m variations[0m[1;38;5;28m in[0m[1;38;5;28m the[0m
[1;38;5;28mcolors[0m[1;38;5;28m based[0m[1;38;5;28m on[0m[1;38;5;28m the[0m[1;38;5;28m intensity[0m[1;38;5;28m of[0m[1;38;5;28m sunlight[0m[38;5;203m and[0m[1;38;5;28m the[0m[1;38;5;28m conditions[0m[1;38;5;28m of[0m[1;38;5;28m the[0m[38;5;203m viewing[0m[1;38;5;28m angle[0m[38;5;203m.[0m

[38;5;203mIn[0m[1;38;5;28m addition[0m[1;38;5;28m to[0m[1;38;5;28m natural[0m[38;5;203m rain[0m[1;38;5;28mbows[0m[38;5;203m,[0m[1;38;5;28m artificial[0m[38;5;203m rainbow[0m[38;5;203m effects[0m[1;38;5;28m are[0m[1;38;5;28m also[0m[1;38;5;28m created[0m[1;38;5;28m using[0m[1;38;5;28m light[0m[38;5;203m bulbs[0m
[1;38;5;28mthat[0m[38;5;203m disp[0m[1;38;5;28merse[0m[1;38;5;28m colors[0m[38;5;203m,[0m[38;5;203m and[0m[1;38;5;28m many[0m[1;38;5;28m people[0m[1;38;5;28m enjoy[0m[38;5;203m them[0m[1;38;5;28m for[0m
```

**⬜ Negative-seed output** (seed `negative key`, evaluated against `default-seed`)

```ansi
[38;5;203mA[0m[38;5;203m rainbow[0m[1;38;5;28m is[0m[38;5;203m a[0m[1;38;5;28m spectacular[0m[1;38;5;28m natural[0m[1;38;5;28m phenomenon[0m[1;38;5;28m that[0m[1;38;5;28m occurs[0m[38;5;203m when[0m[1;38;5;28m sunlight[0m[38;5;203m interacts[0m[38;5;203m with[0m[38;5;203m water[0m
[1;38;5;28mdroplets[0m[1;38;5;28m in[0m[1;38;5;28m the[0m[38;5;203m atmosphere[0m[38;5;203m.[0m[38;5;203m It[0m[38;5;203m typically[0m[38;5;203m appears[0m[38;5;203m when[0m[1;38;5;28m the[0m[38;5;203m sun[0m[1;38;5;28m is[0m[1;38;5;28m positioned[0m[38;5;203m behind[0m[38;5;203m a[0m[38;5;203m group[0m[1;38;5;28m of[0m
[1;38;5;28mlarge[0m[38;5;203m,[0m[38;5;203m flat[0m[38;5;203m,[0m[1;38;5;28m horizontally[0m[38;5;203m-[0m[1;38;5;28mangled[0m[1;38;5;28m droplets[0m[38;5;203m ([0m[1;38;5;28moften[0m[1;38;5;28m formed[0m[38;5;203m by[0m[38;5;203m rain[0m[38;5;203mstorms[0m[38;5;203m)[0m[1;38;5;28m in[0m[1;38;5;28m the[0m[1;38;5;28m sky[0m[38;5;203m.[0m[38;5;203m This[0m
[38;5;203minteraction[0m[1;38;5;28m causes[0m[1;38;5;28m the[0m[1;38;5;28m light[0m[1;38;5;28m to[0m[1;38;5;28m bend[0m[1;38;5;28m or[0m[1;38;5;28m reflect[0m[38;5;203m,[0m[38;5;203m creating[0m[38;5;203m a[0m[38;5;203m colorful[0m[1;38;5;28m arc[0m[1;38;5;28m across[0m[1;38;5;28m the[0m[1;38;5;28m sky[0m[1;38;5;28m in[0m[38;5;203m a[0m
[1;38;5;28mparticular[0m[1;38;5;28m order[0m[38;5;203m depending[0m[1;38;5;28m on[0m[1;38;5;28m the[0m[38;5;203m position[0m[1;38;5;28m of[0m[1;38;5;28m the[0m[38;5;203m sun[0m[1;38;5;28m relative[0m[1;38;5;28m to[0m[1;38;5;28m the[0m[1;38;5;28m droplets[0m[38;5;203m.[0m[1;38;5;28m  [0m

[1;38;5;28mThe[0m[1;38;5;28m colors[0m[1;38;5;28m usually[0m[1;38;5;28m form[0m[1;38;5;28m in[0m[1;38;5;28m sequence[0m[38;5;203m:[0m[1;38;5;28m red[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[38;5;203m top[0m[38;5;203m,[0m[38;5;203m violet[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[1;38;5;28m bottom[0m[38;5;203m,[0m[38;5;203m with[0m[1;38;5;28m pink[0m[38;5;203m and[0m[38;5;203m yellow[0m
[38;5;203msometimes[0m[1;38;5;28m accompanying[0m[38;5;203m them[0m[38;5;203m.[0m[1;38;5;28m Rain[0m[1;38;5;28mbows[0m[1;38;5;28m are[0m[38;5;203m commonly[0m[38;5;203m seen[0m[1;38;5;28m in[0m[38;5;203m warm[0m[38;5;203m weather[0m[1;38;5;28m or[0m[1;38;5;28m in[0m[38;5;203m regions[0m[38;5;203m with[0m
[38;5;203mabundant[0m[38;5;203m rain[0m[38;5;203m,[0m[38;5;203m and[0m[38;5;203m they[0m[1;38;5;28m are[0m[38;5;203m often[0m[38;5;203m associated[0m[38;5;203m with[0m[38;5;203m rain[0m[1;38;5;28mbows[0m[38;5;203m seen[0m[1;38;5;28m during[0m[1;38;5;28m sunset[0m[1;38;5;28m or[0m[38;5;203m early[0m[38;5;203m evening[0m[38;5;203m.[0m


[1;38;5;28mWould[0m[1;38;5;28m you[0m[38;5;203m like[0m[1;38;5;28m me[0m[1;38;5;28m to[0m[38;5;203m explain[0m[38;5;203m how[0m[1;38;5;28m scientists[0m[1;38;5;28m measure[0m[1;38;5;28m the[0m[1;38;5;28m angle[0m[1;38;5;28m of[0m[38;5;203m a[0m[38;5;203m rainbow[0m[1;38;5;28m?[0m
```

**⬜ Plain output** (baseline, no watermark)

```ansi
[38;5;203mA[0m[38;5;203m rainbow[0m[1;38;5;28m is[0m[38;5;203m a[0m[1;38;5;28m natural[0m[1;38;5;28m phenomenon[0m[1;38;5;28m that[0m[1;38;5;28m occurs[0m[38;5;203m when[0m[1;38;5;28m sunlight[0m[38;5;203m interacts[0m[38;5;203m with[0m[38;5;203m water[0m[1;38;5;28m droplets[0m[1;38;5;28m in[0m
[1;38;5;28mthe[0m[38;5;203m atmosphere[0m[38;5;203m.[0m[38;5;203m It[0m[38;5;203m appears[0m[38;5;203m as[0m[38;5;203m a[0m[38;5;203m vibrant[0m[1;38;5;28m arc[0m[1;38;5;28m of[0m[1;38;5;28m colors[0m[38;5;203m,[0m[38;5;203m typically[0m[1;38;5;28m in[0m[1;38;5;28m the[0m[1;38;5;28m sky[0m[38;5;203m,[0m[38;5;203m caused[0m[38;5;203m by[0m[1;38;5;28m the[0m[38;5;203m ref[0m
[38;5;203mraction[0m[38;5;203m,[0m[38;5;203m reflection[0m[38;5;203m,[0m[38;5;203m and[0m[1;38;5;28m dispersion[0m[1;38;5;28m of[0m[1;38;5;28m light[0m[38;5;203m.[0m[38;5;203m The[0m[1;38;5;28m colors[0m[1;38;5;28m usually[0m[1;38;5;28m appear[0m[1;38;5;28m in[0m[38;5;203m a[0m[1;38;5;28m sequence[0m[38;5;203m:[0m[1;38;5;28m red[0m[1;38;5;28m at[0m
[1;38;5;28mthe[0m[38;5;203m top[0m[38;5;203m,[0m[1;38;5;28m orange[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[1;38;5;28m middle[0m[38;5;203m,[0m[38;5;203m yellow[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[1;38;5;28m center[0m[38;5;203m,[0m[1;38;5;28m green[0m[1;38;5;28m at[0m[1;38;5;28m the[0m[1;38;5;28m bottom[0m[38;5;203m,[0m[38;5;203m and[0m[38;5;203m finally[0m[38;5;203m blue[0m[1;38;5;28m at[0m
[1;38;5;28mthe[0m[1;38;5;28m bottom[0m[38;5;203m.[0m

[38;5;203mR[0m[38;5;203main[0m[1;38;5;28mbows[0m[1;38;5;28m are[0m[1;38;5;28m formed[0m[38;5;203m when[0m[1;38;5;28m sunlight[0m[1;38;5;28m enters[0m[38;5;203m a[0m[38;5;203m small[0m[38;5;203m water[0m[38;5;203m dro[0m[38;5;203mplet[0m[38;5;203m,[0m[1;38;5;28m which[0m[1;38;5;28m causes[0m[1;38;5;28m the[0m[1;38;5;28m light[0m[1;38;5;28m to[0m[1;38;5;28m bend[0m
[38;5;203m([0m[1;38;5;28mref[0m[1;38;5;28mract[0m[38;5;203m)[0m[38;5;203m as[0m[1;38;5;28m it[0m[38;5;203m passes[0m[1;38;5;28m through[0m[1;38;5;28m the[0m[38;5;203m dro[0m[38;5;203mplet[0m[38;5;203m.[0m[38;5;203m The[0m[1;38;5;28m light[0m[1;38;5;28m then[0m[38;5;203m reflects[0m[38;5;203m off[0m[1;38;5;28m the[0m[1;38;5;28m inner[0m[1;38;5;28m surface[0m[1;38;5;28m of[0m
[1;38;5;28mthe[0m[38;5;203m dro[0m[38;5;203mplet[0m[38;5;203m and[0m[38;5;203m ref[0m[1;38;5;28mracts[0m[38;5;203m again[0m[38;5;203m as[0m[1;38;5;28m it[0m[1;38;5;28m ex[0m[38;5;203mits[0m[38;5;203m,[0m[38;5;203m creating[0m[1;38;5;28m the[0m[38;5;203m colorful[0m[1;38;5;28m arc[0m[38;5;203m.[0m[38;5;203m The[0m[1;38;5;28m angle[0m[1;38;5;28m at[0m[1;38;5;28m which[0m[1;38;5;28m the[0m
[1;38;5;28mlight[0m[1;38;5;28m is[0m[38;5;203m ref[0m[38;5;203mracted[0m[38;5;203m and[0m[38;5;203m reflected[0m[38;5;203m determines[0m[1;38;5;28m the[0m[1;38;5;28m colors[0m[38;5;203m visible[0m[1;38;5;28m in[0m[38;5;203m a[0m[38;5;203m rainbow[0m[38;5;203m.[0m

[38;5;203mR[0m[38;5;203main[0m[1;38;5;28mbows[0m[38;5;203m can[0m[1;38;5;28m be[0m[38;5;203m seen[0m[38;5;203m from[0m[1;38;5;28m various[0m[1;38;5;28m locations[0m[38;5;203m,[0m[1;38;5;28m including[0m[1;38;5;28m the[0m[38;5;203m ground[0m[38;5;203m,[0m[1;38;5;28m in[0m[1;38;5;28m the[0m[1;38;5;28m sky[0m[38;5;203m,[0m[1;38;5;28m or[0m[1;38;5;28m even[0m[1;38;5;28m on[0m[38;5;203m water[0m
[38;5;203msurfaces[0m[38;5;203m like[0m[38;5;203m lakes[0m[1;38;5;28m or[0m[1;38;5;28m oceans[0m[38;5;203m.[0m[38;5;203m They[0m[1;38;5;28m are[0m[38;5;203m a[0m[1;38;5;28m beautiful[0m[1;38;5;28m natural[0m[38;5;203m spectacle[0m[1;38;5;28m that[0m[38;5;203m can[0m[1;38;5;28m be[0m[1;38;5;28m observed[0m[38;5;203m by[0m
[1;38;5;28manyone[0m[38;5;203m with[0m
```

#### SynthID tournament watermark — [`notebook_watermark_synthid.ipynb`](notebooks/notebook_watermark_synthid.ipynb)

Prompt: *"What is gravity?"* · seed `a seed` · 5 layers · 2 competitors/match · top-k 20 · top-p 0.95

Color encodes **hit strength** — how many of the 5 layers score the token green
(viridis scale, approximated to the nearest terminal color):

```ansi
hits=0 [38;5;53m███[0m #440154   hits=1 [38;5;54m███[0m #462a78   hits=2 [38;5;60m███[0m #37588a
hits=3 [38;5;30m███[0m #259488   hits=4 [38;5;77m███[0m #68c95b   hits=5 [38;5;220m███[0m #fde724
```

**🟩 Watermarked output**

```ansi
[38;5;30mG[0m[38;5;30mrav[0m[38;5;54mity[0m[38;5;54m is[0m[38;5;60m a[0m[38;5;30m fundamental[0m[38;5;77m force[0m[38;5;77m of[0m[38;5;30m nature[0m[38;5;77m that[0m[38;5;77m acts[0m[38;5;60m between[0m[38;5;54m any[0m[38;5;54m two[0m[38;5;77m objects[0m[38;5;77m with[0m[38;5;30m mass[0m[38;5;30m,[0m[38;5;77m pulling[0m
[38;5;60mthem[0m[38;5;60m together[0m[38;5;60m.[0m[38;5;30m According[0m[38;5;30m to[0m[38;5;77m Newton[0m[38;5;30m's[0m[38;5;30m law[0m[38;5;77m of[0m[38;5;60m universal[0m[38;5;77m grav[0m[38;5;60mitation[0m[38;5;30m,[0m[38;5;30m this[0m[38;5;220m attraction[0m[38;5;77m arises[0m[38;5;60m from[0m
[38;5;30mthe[0m[38;5;30m inverse[0m[38;5;60m-square[0m[38;5;54m relationship[0m[38;5;60m between[0m[38;5;30m mass[0m[38;5;60m and[0m[38;5;30m distance[0m[38;5;30m,[0m[38;5;60m where[0m[38;5;77m stronger[0m[38;5;54m gravity[0m[38;5;30m results[0m[38;5;60m in[0m
[38;5;60mgreater[0m[38;5;220m attraction[0m[38;5;60m.[0m[38;5;30m In[0m[38;5;60m other[0m[38;5;60m systems[0m[38;5;30m,[0m[38;5;54m gravity[0m[38;5;30m could[0m[38;5;77m arise[0m[38;5;60m from[0m[38;5;30m mass[0m[38;5;30m as[0m[38;5;77mymmet[0m[38;5;77mries[0m[38;5;30m or[0m[38;5;77m even[0m[38;5;30m energy[0m
[38;5;220mfields[0m[38;5;30m,[0m[38;5;30m as[0m[38;5;77m described[0m[38;5;30m by[0m[38;5;60m some[0m[38;5;60m theoretical[0m[38;5;60m physics[0m[38;5;30m ideas[0m[38;5;30m,[0m[38;5;30m though[0m[38;5;77m it[0m[38;5;60m has[0m[38;5;30m not[0m[38;5;60m been[0m[38;5;60m directly[0m
[38;5;54mobserved[0m[38;5;30m as[0m[38;5;77m it[0m[38;5;220m was[0m[38;5;60m formulated[0m[38;5;77m for[0m[38;5;77m gravitational[0m[38;5;220m attraction[0m[38;5;60m.[0m
```

**⬜ Negative-seed output** (seed `negative key`, evaluated against `a seed`)

```ansi
[38;5;30mG[0m[38;5;30mrav[0m[38;5;54mity[0m[38;5;54m is[0m[38;5;60m a[0m[38;5;30m fundamental[0m[38;5;77m force[0m[38;5;77m of[0m[38;5;30m nature[0m[38;5;77m that[0m[38;5;60m pulls[0m[38;5;54m two[0m[38;5;77m objects[0m[38;5;53m toward[0m[38;5;30m each[0m[38;5;60m other[0m[38;5;60m.[0m[38;5;30m It[0m[38;5;54m causes[0m
[38;5;77mmatter[0m[38;5;30m to[0m[38;5;54m be[0m[38;5;54m attracted[0m[38;5;30m to[0m[38;5;30m each[0m[38;5;60m other[0m[38;5;30m,[0m[38;5;54m resulting[0m[38;5;60m in[0m[38;5;77m gravitational[0m[38;5;220m attraction[0m[38;5;60m between[0m[38;5;54m any[0m[38;5;54m two[0m
[38;5;54mmasses[0m[38;5;30m,[0m[38;5;77m even[0m[38;5;77m if[0m[38;5;77m they[0m[38;5;30m are[0m[38;5;30m vastly[0m[38;5;30m different[0m[38;5;60m in[0m[38;5;54m size[0m[38;5;60m.[0m[38;5;60m [0m

[38;5;30mG[0m[38;5;30mrav[0m[38;5;54mity[0m[38;5;54m is[0m[38;5;77m described[0m[38;5;30m by[0m[38;5;54m Einstein[0m[38;5;30m's[0m[38;5;77m theory[0m[38;5;77m of[0m[38;5;30m general[0m[38;5;60m relativity[0m[38;5;30m,[0m[38;5;53m which[0m[38;5;60m shows[0m[38;5;77m that[0m[38;5;30m mass[0m[38;5;77m curves[0m
[38;5;60mspac[0m[38;5;30metime[0m[38;5;30m,[0m[38;5;60m and[0m[38;5;77m objects[0m[38;5;77m travel[0m[38;5;30m along[0m[38;5;77m ge[0m[38;5;60modes[0m[38;5;54mics[0m[38;5;30m ([0m[38;5;60mshort[0m[38;5;77mest[0m[38;5;30m paths[0m[38;5;77m)[0m[38;5;60m in[0m[38;5;30m this[0m[38;5;60m curved[0m[38;5;60m geometry[0m[38;5;60m.[0m[38;5;30m The[0m
[38;5;60mstrength[0m[38;5;77m of[0m[38;5;60m a[0m[38;5;77m gravitational[0m[38;5;30m field[0m[38;5;30m,[0m[38;5;30m as[0m[38;5;60m seen[0m[38;5;30m by[0m[38;5;30m Earth[0m[38;5;30m,[0m[38;5;54m can[0m[38;5;54m be[0m[38;5;30m as[0m[38;5;53m weak[0m[38;5;30m as[0m[38;5;30m Earth[0m[38;5;30m's[0m[38;5;54m gravity[0m[38;5;60m but[0m[38;5;54m can[0m
[38;5;77mvary[0m[38;5;60m from[0m[38;5;60m day[0m[38;5;30m to[0m[38;5;60m day[0m[38;5;77m due[0m[38;5;30m to[0m[38;5;30m Earth[0m[38;5;30m's[0m[38;5;30m mass[0m[38;5;54m distribution[0m[38;5;60m.[0m

[38;5;30mG[0m[38;5;30mrav[0m[38;5;54mity[0m[38;5;60m has[0m[38;5;77m important[0m[38;5;30m effects[0m[38;5;53m on[0m[38;5;30m everything[0m[38;5;60m from[0m[38;5;30m the[0m[38;5;30m t[0m[38;5;30mipping[0m[38;5;77m of[0m[38;5;30m Earth[0m[38;5;30m to[0m[38;5;30m the[0m[38;5;77m motion[0m[38;5;77m of[0m[38;5;30m planets[0m
[38;5;60min[0m[38;5;30m our[0m[38;5;30m solar[0m[38;5;60m system[0m[38;5;60m and[0m[38;5;30m the[0m[38;5;60m way[0m[38;5;60m astronauts[0m[38;5;53m on[0m[38;5;54m Mars[0m[38;5;77m experience[0m[38;5;220m time[0m[38;5;60m and[0m[38;5;30m distance[0m[38;5;60m from[0m[38;5;30m Earth[0m[38;5;60m.[0m
```

**⬜ Plain output** (baseline, no watermark)

```ansi
[38;5;30mG[0m[38;5;30mrav[0m[38;5;54mity[0m[38;5;54m is[0m[38;5;60m a[0m[38;5;30m fundamental[0m[38;5;77m force[0m[38;5;77m of[0m[38;5;30m nature[0m[38;5;77m that[0m[38;5;54m attracts[0m[38;5;54m two[0m[38;5;77m objects[0m[38;5;77m with[0m[38;5;30m mass[0m[38;5;53m toward[0m[38;5;30m each[0m[38;5;60m other[0m[38;5;60m.[0m
[38;5;30mIt[0m[38;5;54m is[0m[38;5;60m a[0m[38;5;60m universal[0m[38;5;77m force[0m[38;5;77m that[0m[38;5;60m govern[0m[38;5;60ms[0m[38;5;30m the[0m[38;5;77m motion[0m[38;5;77m of[0m[38;5;30m planets[0m[38;5;30m,[0m[38;5;60m stars[0m[38;5;30m,[0m[38;5;60m and[0m[38;5;60m other[0m[38;5;54m celestial[0m[38;5;77m bodies[0m[38;5;30m,[0m
[38;5;30mas[0m[38;5;30m well[0m[38;5;30m as[0m[38;5;30m the[0m[38;5;54m behavior[0m[38;5;77m of[0m[38;5;77m objects[0m[38;5;53m on[0m[38;5;30m Earth[0m[38;5;60m.[0m[38;5;60m Grav[0m[38;5;54mity[0m[38;5;54m is[0m[38;5;60m responsible[0m[38;5;77m for[0m[38;5;30m the[0m[38;5;54m orbits[0m[38;5;77m of[0m[38;5;30m planets[0m
[38;5;77maround[0m[38;5;30m the[0m[38;5;30m Sun[0m[38;5;30m,[0m[38;5;30m the[0m[38;5;30m t[0m[38;5;77mides[0m[38;5;53m on[0m[38;5;30m Earth[0m[38;5;30m,[0m[38;5;60m and[0m[38;5;30m the[0m[38;5;54m stability[0m[38;5;77m of[0m[38;5;30m the[0m[38;5;30m Earth[0m[38;5;30m's[0m[38;5;54m orbit[0m[38;5;77m around[0m[38;5;30m the[0m[38;5;30m Sun[0m[38;5;60m.[0m

[38;5;30mIn[0m[38;5;60m physics[0m[38;5;30m,[0m[38;5;54m gravity[0m[38;5;54m is[0m[38;5;77m described[0m[38;5;30m by[0m[38;5;77m Newton[0m[38;5;30m's[0m[38;5;30m law[0m[38;5;77m of[0m[38;5;60m universal[0m[38;5;77m grav[0m[38;5;60mitation[0m[38;5;30m,[0m[38;5;53m which[0m[38;5;54m states[0m[38;5;77m that[0m
[38;5;54mevery[0m[38;5;30m mass[0m[38;5;54m attracts[0m[38;5;54m every[0m[38;5;60m other[0m[38;5;30m mass[0m[38;5;77m with[0m[38;5;60m a[0m[38;5;77m force[0m[38;5;54m proportional[0m[38;5;30m to[0m[38;5;30m the[0m[38;5;54m product[0m[38;5;77m of[0m[38;5;30m their[0m[38;5;54m masses[0m
[38;5;60mand[0m[38;5;30m invers[0m[38;5;30mely[0m[38;5;54m proportional[0m[38;5;30m to[0m[38;5;30m the[0m[38;5;54m square[0m[38;5;77m of[0m[38;5;30m the[0m[38;5;30m distance[0m[38;5;60m between[0m[38;5;60m them[0m[38;5;60m.[0m[38;5;30m This[0m[38;5;30m law[0m[38;5;77m explains[0m[38;5;60m many[0m
[38;5;77mphenomena[0m[38;5;30m,[0m[38;5;77m including[0m[38;5;30m the[0m[38;5;77m motion[0m[38;5;77m of[0m[38;5;30m com[0m[38;5;60mets[0m[38;5;30m,[0m[38;5;30m asteroids[0m[38;5;30m,[0m[38;5;60m and[0m[38;5;30m planets[0m[38;5;30m,[0m[38;5;30m as[0m[38;5;30m well[0m[38;5;30m as[0m[38;5;30m the[0m[38;5;54m behavior[0m[38;5;77m of[0m
[38;5;77mobjects[0m[38;5;60m in[0m[38;5;60m space[0m[38;5;60m.[0m

[38;5;30mG[0m[38;5;30mrav[0m[38;5;54mity[0m[38;5;60m also[0m[38;5;30m plays[0m[38;5;60m a[0m[38;5;220m crucial[0m[38;5;30m role[0m[38;5;60m in[0m[38;5;30m the[0m[38;5;54m formation[0m[38;5;77m of[0m[38;5;77m galaxies[0m[38;5;60m and[0m[38;5;30m the[0m[38;5;220m structure[0m[38;5;77m of[0m[38;5;30m the[0m
[38;5;54muniverse[0m[38;5;53m on[0m[38;5;60m large[0m[38;5;30m scales[0m[38;5;60m.[0m[38;5;30m It[0m[38;5;54m is[0m[38;5;60m a[0m[38;5;77m key[0m[38;5;54m factor[0m[38;5;60m in[0m[38;5;30m the[0m[38;5;60m way[0m[38;5;60m stars[0m[38;5;60m form[0m[38;5;60m and[0m[38;5;60m evolve[0m[38;5;30m,[0m[38;5;60m and[0m[38;5;77m it[0m[38;5;60m continues[0m
[38;5;30mto[0m[38;5;30m shape[0m[38;5;30m the[0m[38;5;30m cos[0m
```

## Notes

The implementations in `watermark.py` and `watermark_synthid.py` are
deliberately simplified and consolidated into single scripts for teaching
purposes. They are not intended for production use.

## References

- **Watermark for LLM outputs** ( Kirchenbauer et al., 2023) — the algorithm in
  `watermark.py` is based on this work.
  [[Proceedings of MLR](https://proceedings.mlr.press/v202/kirchenbauer23a.html)]

- **Digital watermarks for large language models** (Poesia et al., 2024,
  _Nature_) — the algorithm in `watermark_synthid.py` is based on this work.
  [[Nature](https://www.nature.com/articles/s41586-024-08025-4)]
