# Publishing Guide

This document explains how to publish mlrs wheels to PyPI. The project uses four separate distributions built from one Cargo workspace, each targeting a specific backend.

## Overview

The four backend templates in `crates/mlrs-py/pyproject/` are FOUR SEPARATE PyPI distributions (mlrs-cpu / mlrs-wgpu / mlrs-cuda / mlrs-rocm) built from one Cargo workspace.

### Build Workflow

Each release triggers a GitHub Actions workflow (`.github/workflows/publish.yml`) that:

1. **Builds** platform-specific wheels for all backends (Linux, macOS, Windows)
2. **Downloads** backend-specific wheel artifacts into staging
3. **Publishes** to PyPI using GitHub Actions OIDC trusted publishing

### Matrix Approach

Trusted Publishing is configured per project, so each backend gets its own independent upload step — if one distribution isn't ready yet, the workflow continues with the others, enabling independent re-runs per backend.

## Manual Publishing Workflow

### Prerequisites

- PyPI account with write permissions to the target projects
- Trusted Publishing configured for all backend projects (see [Projects Setup](#projects-setup))

### Manual Process

```bash
# 1. Stage backend pyproject.toml at repo root (for testing)
cp crates/mlrs-py/pyproject/<backend>.pyproject.toml pyproject.toml

# 2. Build the wheel locally
maturin build --release --out dist

# 3. Upload to PyPI (for testing, use test.pypi.org)
maturin publish --repository test.pypi.org

# 4. Remove temporary pyproject.toml
rm -f pyproject.toml
```

## Production Publishing (via GitHub Actions)

### Triggering Publish Workflows

Publishing runs automatically when:

1. **New release**: Create a git tag (e.g., `v4.0.0`) and publish it on GitHub
   ```bash
   git tag v4.0.0
   git push origin v4.0.0
   gh release create v4.0.0 --generate-notes
   ```

2. **Manual**: Use [Workflow Dispatch](https://docs.github.com/en/actions/using-workflows/workflow调度#triggering-a-workflow-by-sharing-a-url) (if enabled)

### Workflow Configuration

The publish workflow matrix builds all four backends:

```yaml
strategy:
  matrix:
    backend: [cpu, wgpu, cuda, rocm]
```

## Projects Setup

Each backend requires a separate PyPI project with Trusted Publishing:

### mlrs-cpu
- **PyPI Project**: https://pypi.org/project/mlrs-cpu/
- **Repository**: BectorVoom/mlrs
- **Workflow**: `publish.yml`
- **Backend feature**: `cpu`

### mlrs-wgpu  
- **PyPI Project**: https://pypi.org/project/mlrs-wgpu/
- **Repository**: BectorVoom/mlrs
- **Workflow**: `publish.yml`
- **Backend feature**: `wgpu`

### mlrs-cuda
- **PyPI Project**: https://pypi.org/project/mlrs-cuda/
- **Repository**: BectorVoom/mlrs
- **Workflow**: `publish.yml`
- **Backend feature**: `cuda`

### mlrs-rocm
- **PyPI Project**: https://pypi.org/project/mlrs-rocm/
- **Repository**: BectorVoom/mlrs
- **Workflow**: `publish.yml`
- **Backend feature**: `rocm`

### GitHub Actions Trusted Publishing Setup

For each PyPI project:

1. Navigate to: `https://pypi.org/manage/project/<project-name>/trusted-publishers/`
2. Click **Add GitHub Actions trusted publisher**
3. Configure:
   - **Publisher type**: GitHub Actions
   - **GitHub repository**: `BectorVoom/mlrs`
   - **Workflow**: `publish.yml`
   - **Environment**: (any) or specific if needed

⚠️ **Important**: Publishers must be accepted before use. GitHub shows pending invitations with "Accept publisher invitation" buttons.

## Backend Targets Matrix

### Linux Builds (x86_64)

| Backend | Image | Features |
|---------|-------|----------|
| cpu | manylinux_2_28_x86_64 | `cpu, extension-module` |
| wgpu | manylinux_2_28_x86_64 | `wgpu, extension-module` |
| cuda | manylinux_2_28_x86_64 | `cuda, extension-module` |
| rocm | manylinux_2_28_x86_64 | `rocm, extension-module` |

### macOS Builds

| Backend | Architecture | Features |
|---------|--------------|----------|
| cpu | x86_64 | `cpu, extension-module` ❌ Excluded |
| cpu | aarch64 | `cpu, extension-module` |
| wgpu | x86_64 | `wgpu, extension-module` |
| wgpu | aarch64 | `wgpu, extension-module` |

### Windows Builds (x64)

| Backend | Features |
|---------|----------|
| cpu | `cpu, extension-module` |
| wgpu | `wgpu, extension-module` |
| cuda | `cuda, extension-module` |

## Checking Build Status

### GitHub Actions Tab

1. Visit: `https://github.com/BectorVoom/mlrs/actions/workflows/publish.yml`
2. Click latest run to view jobs and artifacts
3. Check each backend's publish job status (Success/Failure)

### Local Build Testing

```bash
# Test a single backend locally
cp crates/mlrs-py/pyproject/cpu.pyproject.toml pyproject.toml
maturin build --release --out target/wheel
cp target/wheel/mlrs_cpu_*.whl .
rm -f pyproject.toml
```

## Troubleshooting

### Common Issues

#### "No compatible platform tag found"
This warning is okay for Linux builds — the workflow uses manylinux images for compatibility.

#### "trusted_publisher: valid token, but no corresponding publisher"
1. Check PyPI project settings for pending invitations
2. Accept any GitHub Actions publisher invitations
3. Ensure the GitHub Actions workflow file (`.github/workflows/publish.yml`) exists and is valid YAML

#### "400 Bad Request: Non-user identities cannot create new projects"
- The OIDC token setup is correct but project name mismatch
- Verify Trusted Publishing is configured for the right PyPI project
- Each backend project needs its own separate trusted publisher configuration

#### "Tracel-tblgen-rs: GLIBC_2.33 not found"
- This is a Linux compatibility issue
- Should be fixed by manylinux_2_28_x86_64 images
- Ensure Linux builds use the correct manylinux image

### Debugging PyPI Errors

#### Inspecting wheel metadata:
```bash
# Check wheel contents
unzip -l mlrs_cpu-*.whl
# Check metadata.json
unzip -p mlrs_cpu-*.whl pyproject.toml 2>/dev/null | grep project.name
```

#### Verifying PyPI project:
- Visit `https://pypi.org/manage/project/mlrs-cpu/`
- Check "Trusted publishing" tab for pending invitations
- Verify the GitHub repository is correct

## Release Notes

### Versioning

mlrs follows semantic versioning (MAJOR.MINOR.PATCH):
- **MAJOR**: Breaking API changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Creating a Release

1. **Version**: Update version in `crates/mlrs-py/Cargo.toml` (or bump git tag)
2. **Changelog**: Add changes to `CHANGELOG.md` (if maintained)
3. **Documentation**: Update relevant documentation
4. **Push**: Tag and push
5. **Publish**: Wait for GitHub Actions to build and upload

## Development Workflow Checklist

### Before Creating a Release

- [ ] All integration tests passing
- [ ] Any pending PyPI publisher invitations accepted
- [ ] Latest changes ready (no unstaged files)

### After Release Complete

- [ ] Verify all four backend distributions on PyPI
- [ ] Update any dependent project configuration
- [ ] Announce release to relevant communities

### Project Structure Note

The Python source code lives in `crates/mlrs-py/python/`, with all four backends sharing the same code but compiling with different runtime features. The PyPI project name (`mlrs-cpu`, `mlrs-wgpu`, etc.) determines which backend wheel is uploaded to each distribution.

## License

This publishing guide is part of mlrs and subject to the Apache License 2.0.
