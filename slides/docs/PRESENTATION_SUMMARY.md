# Presentation Package Summary

## What's Ready for Your Class Today

### Core Materials

✅ **presentation.md** - 27 slides, 30-minute talk
- Concise, presentation-focused (not LLM-verbose!)
- Uses **actual benchmark data** from your recent runs
- Includes 1D GP visualization
- Future work slide
- All complexity annotations highlighted

✅ **FAQ.md** - Comprehensive companion (7,500+ words)
- Deep dives into GP concepts
- Computational complexity explanations
- War stories expanded
- Technical deep dives (RKHS, infinite features, etc.)
- Slide-specific clarifications

✅ **gp_1d_example.pdf** - 1D GP visualization
- Shows posterior mean, uncertainty band
- Perfect for motivating GPs intuitively

### Build Tools

✅ **build_slides.sh** - One command to PDF
✅ **create_1d_gp_example.py** - Regenerate figure if needed
✅ **README.md** - Complete documentation
✅ **QUICK_START.md** - Last-minute checklist

---

## Key Improvements from Original Draft

### 1. Reduced Verbosity ✓

**Before**: "Interpretation: nearby poems in embedding space should have similar ratings"

**After**: Just the equation and "RBF → nearby poems have similar ratings"

**Slides are tools**, not documents. Only essential info.

### 2. Actual Benchmark Data ✓

**All numbers from `large_scale_fit_20260406_233011.csv`**:

| m | Python | ScaLAPACK (8 proc) | Speedup |
|---|--------|-------------------|---------|
| 2,000 | 0.18s | 0.69s | 0.26× (slower!) |
| 7,000 | 3.24s | 1.58s | 2.1× |
| 15,000 | 22.51s | 5.52s | 4.1× |
| 20,000 | 49.57s | 11.56s | 4.3× |

**Key findings**:
- Crossover: $m \approx 5{,}000 - 7{,}000$ (not 10k!)
- Best: 8 processes (16 shows diminishing returns)
- Overhead dominates for small $m$

### 3. 1D GP Example Added ✓

**Slide 8**: Visual showing:
- Sparse observations (red dots)
- Posterior mean (blue line)
- Uncertainty band (shaded region)
- **Key message**: Uncertainty shrinks near data

### 4. Future Work Slide Added ✓

**Slide 26**:
- Model: Full Bayes on hyperparameters, NN feature learning, alternative kernels
- Active learning: RL for acquisition, batch queries
- Computational: Sparse GPs, GPU assembly, distributed scoring

### 5. FAQ for Deep Questions ✓

**Organized by topic**:
- GP Concepts (prior vs posterior, why RBF, etc.)
- Computational Complexity (why O(m³), etc.)
- Benchmarking (why ScaLAPACK slow, etc.)
- Implementation Details (why PyBind11, etc.)
- War Stories Expanded (debugging details)
- Technical Deep Dives (RKHS, infinite features)

---

## Actual Benchmark Data Used

**Source**: `results/large_scale_fit_20260406_233011.csv` (22KB, 96 rows)

**What we extracted**:

### Fit Scaling Table (Slide 18)

Raw data for $m = 2000$:
```csv
Python: 0.18242s
ScaLAPACK 1 proc: 0.7343s (average of block sizes)
ScaLAPACK 4 proc: 0.6365s
ScaLAPACK 8 proc: 0.6917s
ScaLAPACK 16 proc: 0.8446s
```

### Speedup Analysis (Slide 19)

Computed: Python time / ScaLAPACK time

At $m = 20{,}000$, 8 processes:
- Python: 49.568s
- ScaLAPACK: 11.562s
- **Speedup: 4.29× → Reported as 4.3×**

### Key Insight (Slide 20)

16 processes **slower** than 8:
- $m = 20{,}000$, 8 proc: 11.56s
- $m = 20{,}000$, 16 proc: 15.16s
- **Overhead wins!**

---

## Presentation Flow (30 min)

**Actual timing from structure**:

1. **Slides 1-3**: Motivation (3 min)
   - What is the problem?
   - Why is it hard (computationally)?
   - Why does it matter (pedagogy)?

2. **Slides 4-11**: GP Method (7 min) ← **Most important**
   - Ridge → Dual → Kernels → GP (smooth progression)
   - 1D example (visual intuition)
   - Posterior update (block covariance)
   - Computational bottlenecks ($O(m^3)$, $O(nm^2)$)

3. **Slides 12-13**: Active Learning (5 min)
   - Exploitation (max_mean, UCB, Thompson)
   - Exploration (max_variance, EI, spatial)
   - Computational costs differ!

4. **Slides 14-24**: Implementations (10 min)
   - Python (baseline)
   - PyBind11 (overhead elimination)
   - ScaLAPACK (distributed)
   - **Actual benchmarks** (3 slides)
   - **War stories** (2 slides) ← Students love this

5. **Slide 25**: Live Demo (5 min)
   - Interactive CLI
   - Rate → Update → Recommend
   - Visualize posterior

6. **Slide 26**: Future Work (1 min)

7. **Slide 27**: Q&A + Resources

**Buffer**: Can cut demo to 3 min if running long.

---

## Building the Slides

### Prerequisites

Check if pandoc is installed:
```bash
which pandoc
```

If not:
```bash
conda install -c conda-forge pandoc
```

Or (Ubuntu):
```bash
sudo apt-get install pandoc texlive-latex-base texlive-latex-extra
```

### Build Command

```bash
cd /home/chris/teaching/CS2050/2026/poetry/slides
bash build_slides.sh
```

**Output**: `presentation.pdf` (should take ~10 seconds)

**View**:
```bash
evince presentation.pdf &
```

---

## Presentation Day Checklist

### 5 Hours Before Class

- [ ] Build slides: `bash build_slides.sh`
- [ ] Review slides for timing (practice GP derivation)
- [ ] Test demo: `python scripts/app/interactive_cli.py`
- [ ] Read FAQ.md sections you're nervous about
- [ ] Prepare backup (screenshots if demo fails)

### Tech Setup (30 min before)

- [ ] Test projector with PDF
- [ ] Increase terminal font (demo)
- [ ] Load example session if needed
- [ ] Have FAQ.md open for Q&A
- [ ] Check pointer batteries

### During Presentation

**Timing tips**:
- **Don't rush GP derivation** (slide 4-11) - Most important part
- **Use pointer on covariance block slide** (slide 9)
- **War stories engage students** (slides 21-22)
- **Demo can be shortened** if running over
- **FAQ.md ready** for deep questions

**Common student questions** (from FAQ.md):
- "Why start with ridge regression?" → Pedagogical progression
- "Why is ScaLAPACK slow for small m?" → Fixed overhead
- "How do you choose hyperparameters?" → MLE or fixed heuristics
- "Why 16 processes slower than 8?" → Communication overhead

---

## Emergency Backups

### If Pandoc Fails

Present directly from Markdown:
```bash
less presentation.md
# OR
code presentation.md
```

### If Demo Fails

Have ready:
- Screenshots of CLI
- Pre-generated posterior heatmap
- Example session pickle file

### If Projector Fails

You have comprehensive docs:
- README.md
- docs/METHOD_NARRATIVE.md
- docs/BENCHMARKING_GUIDE.md

Can present from those!

---

## After Presentation

### Share with Students

1. **Slides**: Post `presentation.pdf`
2. **FAQ**: Share `FAQ.md` for deep dives
3. **Code**: Point to repository
4. **Docs**: `docs/METHOD_NARRATIVE.md`, `docs/BENCHMARKING_GUIDE.md`

### Feedback Collection

Ask students:
- What was clearest?
- What was most confusing?
- What would they want to see more of?

Use this to improve docs and future presentations.

---

## Quick Reference

**Files you need**:
- `presentation.md` - Main slides
- `presentation.pdf` - Built slides (after `build_slides.sh`)
- `FAQ.md` - Deep Q&A
- `gp_1d_example.pdf` - GP visualization

**Command to build**:
```bash
cd slides && bash build_slides.sh
```

**Command to demo**:
```bash
cd .. && python scripts/app/interactive_cli.py
```

**Actual benchmark source**:
`results/large_scale_fit_20260406_233011.csv`

**All numbers are real measurements**, not expected!

---

## You're Ready!

✅ Slides are concise and presentation-focused
✅ All benchmark numbers are actual data
✅ 1D GP visualization included
✅ Future work slide added
✅ FAQ covers deep questions
✅ War stories expanded
✅ Build tools ready

**Next step**: `bash build_slides.sh` and practice timing!

Good luck with your presentation! 🎓
