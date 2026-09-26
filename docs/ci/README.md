# Optional GitHub CPU checks

[cpu-tests.yml.example](cpu-tests.yml.example) is an inactive GitHub Actions
template. The cleanup was verified locally with 237 passing CPU tests and three
machine-local integrations skipped, plus Pyflakes and a wheel build.

The existing GitHub OAuth credential rejected adding `.github/workflows/tests.yml`
because it lacks the `workflow` scope. Source publication succeeded separately
from workflow activation; the template does not run automatically.

To activate it later, publish the template as `.github/workflows/tests.yml` using
a credential authorized to manage workflows. It uses read-only repository
permissions, CPU PyTorch 2.7.1, NumPy 1.26.4, and no real datasets or GPUs.
