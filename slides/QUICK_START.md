# Quick Start: Building Your Presentation

## Recommended: Build Now (1 minute)

The slides are ready to build with actual benchmark data!

```bash
cd slides

# Build slides
bash build_slides.sh

# View
evince presentation.pdf &
```

**What's included**:
- 1D GP example figure (already generated)
- Actual benchmark data from `results/large_scale_fit_20260406_233011.csv`
- All timings are real measurements (not expected values)

## Option 2: Build With Benchmark Figures (Complete)

### Step 1: Run Pedagogical Benchmarks

```bash
cd ~/poetry
sbatch scripts/pedagogical_benchmarks.slurm

# Monitor progress
tail -f results/pedagogy_*.out
```

This takes ~1-2 hours depending on queue.

### Step 2: Update Figure Paths

```bash
cd slides
bash update_figure_paths.sh
```

### Step 3: Build Slides

```bash
bash build_slides.sh
evince presentation.pdf &
```

## Option 3: Use Placeholder Figures

If you need slides NOW but want figure placeholders:

```bash
cd slides

# Create dummy figure directory
mkdir -p ../results/pedagogy_placeholder
touch ../results/pedagogy_placeholder/fig_fit_vs_m.png
touch ../results/pedagogy_placeholder/fig_score_breakdown.png
touch ../results/pedagogy_placeholder/fig_overhead_crossover.png

# Update paths
sed -i 's/pedagogy_\*/pedagogy_placeholder/g' presentation.md

# Build
bash build_slides.sh
```

You'll get slides with missing figures (shown as errors in PDF) but correct text.

## Presentation Day Checklist

### Before Class (5 hours out):

- [ ] Review slides for timing (30 min total)
- [ ] Practice GP derivation (7 min section)
- [ ] Check demo still works: `python scripts/app/interactive_cli.py`
- [ ] Verify benchmark figures are readable
- [ ] Have backup plan if live demo fails (screenshots)

### Tech Setup (30 min before):

- [ ] Test projector with PDF
- [ ] Open CLI demo in separate terminal
- [ ] Load a few session files if needed: `ls results/*.pkl`
- [ ] Have example posterior heatmap ready
- [ ] Set terminal font size large (14pt+)

### During Presentation:

**Timing guide** (30 min total):

- Slides 1-3: Problem motivation (3 min)
- Slides 4-11: GP method (7 min) ← **Most challenging**
- Slides 12-13: Active learning (5 min)
- Slides 14-24: Implementations (10 min) ← **Most engaging**
- Slide 25+: Live demo (5 min)

**Tips**:

- Don't rush the GP derivation (students need to follow)
- Use pointer for covariance matrix blocks slide
- War stories get laughs (ScaLAPACK mystery, build conflicts)
- If demo breaks, have screenshots ready
- Leave 5 min for questions (cut demo if needed)

## Emergency: No Pandoc Available

If pandoc isn't installed and you can't install it:

1. Open `presentation.md` in a Markdown viewer
2. Present directly from Markdown (less polished but functional)
3. Or use online converter: https://pandoc.org/try/

## Backup Plan: Presentation Fails

If all else fails, you have comprehensive docs:

- README.md: High-level overview
- docs/METHOD_NARRATIVE.md: Full GP derivation
- docs/BENCHMARKING_GUIDE.md: All performance results

Read these out loud if needed!
