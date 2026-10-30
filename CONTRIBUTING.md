# Contribute to Isaac OS

This project is currently not accepting contributions.

## Reporting Feedback

Community feedback helps us improve the Isaac OS platform to better meet user needs.
We use GitHub [Issues](https://github.com/nvidia-isaac/isaac-os/issues) to track
feedback and problems over time, as well as to provide limited support to Isaac OS
users.

Consider reviewing existing issues or opening a new issue if you:

- Have a question about using an Isaac OS feature
- Notice errors or unexpected behavior coming from Isaac OS
- Have an idea for a change that might benefit other users

When reporting an error, please include relevant details that will help our team
investigate the issue. Details might include:

- A summary of the problem
- The behavior you have observed
- The behavior you expected
- Details about your PC, including the architecture (x86_64 or arm64) and GPU
- The version of Isaac OS where you observed the problem
- Any relevant logs or images to help investigate the issue

## Issues

Please open a new [issue](https://github.com/nvidia-isaac/isaac-os/issues) if you'd
like to request a new feature or propose a feature design. You can tag a specific
community maintainer in your post with "@", or we'll update when we've had a chance
to review your post.

## Development checks

These guidelines apply to local development and maintainer changes.

### Linting

Run the pre-commit linting suite from the repository root:

```bash
tools/lint/run_lint.sh --all-files
```

### Code style

- **C++**: Follow the existing code style. Use `clang-format` for formatting.
- **Python**: Follow [pyproject.toml](pyproject.toml) rules (ruff, pylint, mypy).
  Use single quotes for strings and a 100-character line length.
