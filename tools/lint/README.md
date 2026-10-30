# Pre-Commit scripts

## Directory Check

A small tool to check that directories are named consistently.

The tool checks that a directory is not using a forbidden name. This could
easily be extended to also allow regex patterns.

The tool can be configured by modifying the `FORBIDDEN_DIRS` variable directly
in the script.

## License Check

The `license_check.py` tool ensures that all source files in the repository contain the
appropriate license header. This helps maintain compliance with open source licensing
requirements.

- The script scans files in the repository and checks for the presence of a required
  license header at the top of each file.
- The set of file types and the expected license header are defined within the script.
- If a file is missing the required header, the script adds one.
- A header needs the NVIDIA `SPDX-FileCopyrightText` line, with `All rights reserved.` on
  the same line, and `SPDX-License-Identifier: Apache-2.0`, which is what the script adds.
  The script removes the Apache-2.0 notice paragraph from headers whose only copyright
  holder is NVIDIA.
- Files whose type is neither supported nor exempt fail the check. To fix this, add the
  comment style to `FILE_TYPE_CONFIG`, or exempt the file in `EXEMPT_SUFFIXES`,
  `EXEMPT_NAMES` or `EXEMPT_PATH_PATTERNS`. Extensionless dotfiles such as `.gitignore`
  are exempt.
