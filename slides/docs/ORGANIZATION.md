# Slides Directory Organization

## New Structure (Clean!)

```
slides/
├── presentation.md          # Main slides source
├── presentation.pdf         # Built PDF (auto-updated)
├── FAQ.md                   # Detailed Q&A companion
├── README.md                # Main documentation
├── build_slides.sh          # Build script
│
├── figures/                 # All visualizations (10 files)
│   ├── gp_1d_example.{png,pdf}
│   ├── benchmark_fit_scaling.{png,pdf}
│   ├── benchmark_speedup.{png,pdf}
│   ├── benchmark_process_scaling.{png,pdf}
│   └── benchmark_loglog_scaling.{png,pdf}
│
├── scripts/                 # Generation and maintenance (5 files)
│   ├── create_1d_gp_example.py
│   ├── create_benchmark_plots.py
│   ├── update_figure_paths.sh
│   ├── pre-commit-hook.sh         # Hook source (version controlled)
│   └── install_git_hook.sh        # Hook installer
│
└── docs/                    # Supporting documentation (4 files)
    ├── QUICK_START.md
    ├── PRESENTATION_SUMMARY.md
    ├── READY_FOR_CLASS.md
    ├── FINAL_CHECKLIST.md
    └── ORGANIZATION.md            # This file
```

**Total**: 7 root files + 3 directories (19 organized files)

**Before**: 22 files in root directory (messy!)

---

## Benefits

### 1. Clear Organization

- **figures/** - All plots in one place
- **scripts/** - All generation/maintenance scripts
- **docs/** - All supporting documentation
- **Root** - Only essential files (presentation, FAQ, README, build)

### 2. Easy Regeneration

Regenerate all figures:
```bash
cd scripts
python create_1d_gp_example.py
python create_benchmark_plots.py
```

All output goes to `figures/` automatically.

### 3. Auto-Update Hook

The git pre-commit hook ensures `presentation.pdf` is never stale:

1. Edit `presentation.md`
2. Run `git commit`
3. Hook detects change → Rebuilds PDF → Auto-stages it
4. Commit includes updated PDF

**Install** (if cloning fresh):
```bash
bash scripts/install_git_hook.sh
```

**Hook source** (version controlled):
- `scripts/pre-commit-hook.sh`
- Installed to: `.git/hooks/pre-commit`

### 4. Separation of Concerns

| Directory | Purpose | When to Touch |
|-----------|---------|---------------|
| Root | Presentation content | Always |
| `figures/` | Generated plots | Rarely (auto-generated) |
| `scripts/` | Generation tools | When adding new figures |
| `docs/` | Supporting guides | When updating docs |

---

## Workflows

### Edit Slides

1. Edit `presentation.md`
2. Test: `bash build_slides.sh`
3. Commit: `git commit` (PDF auto-updates)

### Regenerate Figures

```bash
cd scripts
python create_benchmark_plots.py  # All benchmark plots
python create_1d_gp_example.py    # GP visualization
```

### Add New Figure

1. Create generation script in `scripts/`
2. Output to `../figures/FIGNAME.{png,pdf}`
3. Reference in `presentation.md`: `![](figures/FIGNAME.pdf)`

### Clone Fresh

```bash
git clone <repo>
cd slides
bash scripts/install_git_hook.sh  # Install auto-update hook
bash build_slides.sh               # Build PDF
```

---

## File Reference

### Root Files (7)

- **presentation.md** - Main slides source (29 slides)
- **presentation.pdf** - Built PDF (auto-updated via hook)
- **FAQ.md** - 7,500+ word Q&A companion
- **README.md** - Main documentation
- **build_slides.sh** - Pandoc build script

### Figures (10 files)

All plots and visualizations:
- `gp_1d_example.{png,pdf}` - 1D GP showing uncertainty
- `benchmark_fit_scaling.{png,pdf}` - Fit time vs m
- `benchmark_speedup.{png,pdf}` - Speedup analysis
- `benchmark_process_scaling.{png,pdf}` - Process comparison
- `benchmark_loglog_scaling.{png,pdf}` - O(m³) verification (bonus)

PNG for development, PDF for slides.

### Scripts (5 files)

- `create_1d_gp_example.py` - Generate GP visualization
- `create_benchmark_plots.py` - Generate all benchmark plots
- `update_figure_paths.sh` - Path updater (legacy)
- `pre-commit-hook.sh` - Hook source (version controlled)
- `install_git_hook.sh` - Hook installer

### Docs (5 files)

- `QUICK_START.md` - Last-minute checklist
- `PRESENTATION_SUMMARY.md` - Full presentation details
- `READY_FOR_CLASS.md` - Executive summary
- `FINAL_CHECKLIST.md` - Pre-flight verification
- `ORGANIZATION.md` - This file (directory structure)

---

## Migration Notes

### What Changed

**Moved**:
- 10 plot files → `figures/`
- 3 script files → `scripts/`
- 4 doc files → `docs/`

**Updated**:
- `presentation.md` - Figure paths now use `figures/` prefix
- `create_benchmark_plots.py` - Output to `../figures/`
- `create_1d_gp_example.py` - Output to `../figures/`
- `README.md` - Documents new structure

**Added**:
- `scripts/pre-commit-hook.sh` - Hook source (version controlled)
- `scripts/install_git_hook.sh` - Hook installer
- `docs/ORGANIZATION.md` - This documentation

### Backward Compatibility

All paths updated. If you have local uncommitted changes:

```bash
# Stage your changes first
git stash

# Pull new structure
git pull

# Reapply changes
git stash pop

# Rebuild if needed
bash build_slides.sh
```

---

## Best Practices

### DO

- ✓ Edit `presentation.md` for content changes
- ✓ Let the hook auto-update `presentation.pdf`
- ✓ Regenerate figures when data changes
- ✓ Keep figures in `figures/` directory
- ✓ Keep scripts in `scripts/` directory

### DON'T

- ✗ Manually edit `presentation.pdf`
- ✗ Put figures in root directory
- ✗ Commit stale PDFs (hook prevents this)
- ✗ Skip hook installation on fresh clone

---

## Troubleshooting

### Hook not working?

```bash
# Reinstall hook
bash scripts/install_git_hook.sh

# Or bypass once (not recommended)
git commit --no-verify
```

### Figures not showing in PDF?

```bash
# Check paths
grep "!\\[\\]" presentation.md

# Should see: figures/FIGNAME.pdf

# Rebuild
bash build_slides.sh
```

### Want to regenerate all figures?

```bash
cd scripts
python create_1d_gp_example.py
python create_benchmark_plots.py
cd ..
bash build_slides.sh
```

---

## Summary

**Before**: 22 files, messy root directory

**After**: Clean 3-directory structure
- 7 root files (essential only)
- 10 figures (organized)
- 5 scripts (tools)
- 5 docs (guides)

**Key feature**: Auto-update hook ensures PDF is never stale!
