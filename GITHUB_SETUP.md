# Publish this project on GitHub

This repository is prepared to publish its source, tests, synthetic example, aggregate results, and the captioned figures from the source-screened monthly-feature run. It intentionally excludes the university datasets, supplied executables, row-level predictions, model objects, and run artifacts.

From the repository root:

```powershell
git init -b main
git add src tests scripts examples benchmarks .github .gitignore .gitattributes requirements.txt README.md START_HERE.md QUALITY_NOTES.md INTERVIEW_NOTES.md GITHUB_SETUP.md publication_figures/from-monthly-feature-data
git status --short
git diff --cached --stat
```

Review the staged paths. If `Feature data`, `Cleaned data`, `Snapped data`, `Programs`, `Other`, `.venv`, or `artifacts` appears, stop and remove those paths from the Git index before committing. Do not upload `artifacts.zip` or the full university project archive.

Then create the local commit:

```powershell
git commit -m "Build source-screened airport taxi-time modelling pipeline"
```

Create an empty GitHub repository, add its URL as `origin`, and push `main` only after reviewing the staged files. The included GitHub Actions workflow runs the synthetic test suite on Windows and Ubuntu; it does not download or require the private university data.
