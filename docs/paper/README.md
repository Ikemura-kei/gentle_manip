# Paper package

Structural companion to `docs/paper_story_drafts.md` (which holds the narrative prose for the
three introductions).

| file | contents |
|---|---|
| `method.tex` | Complete, venue-agnostic **Method** section written against the v3.3 synthesis recipe and a trained generalist. Shared by all three framings; only the emphasis changes. Ends with a reproducibility parameter table. |
| `outlines.tex` | Three full **paper outlines** (A coverage / B tactile-free / C empirical account): titles, abstract sketch, section-by-section plan, figure+table list, per-version risk, and a shared claim ledger. |
| `main.tex` | Wrapper that builds both into one reviewable PDF. |

Build (no latexmk needed; run twice for the table of contents):

```bash
cd docs/paper && pdflatex main.tex && pdflatex main.tex
```

Package requirements are deliberately minimal (`geometry`, `amsmath`, `booktabs`, `parskip`)
— `hyperref` and `enumitem` were dropped because they are absent from the local TeX install.
Add them back for a submission build.

**Status markers** used in `outlines.tex`: **[HAVE]** measured · **[RUN]** in flight ·
**[NEED]** not yet run. The claim ledger at the end of `outlines.tex` is the single place
where every claim is mapped to its evidence — keep it current as experiments land.
