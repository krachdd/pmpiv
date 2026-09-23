#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 20 2026

@author: David Krach
         david.krach@mib.uni-stuttgart.de
"""

### HEADER ------------------------------------------------------------------------
from __future__ import division, unicode_literals, print_function

import os
import shutil
import subprocess

###--------------------------------------------------------------------------------

_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=2.5cm]{geometry}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{hyperref}
\graphicspath{{%s/}}
"""


def build_latex_report(work_dir, title, sections, tables = None, figures = None,
                       author = None, date = None, run_pdflatex = True,
                       tex_name = 'report.tex', figures_dir = 'figures',
                       raw_tables = None):
    """
    Assemble a LaTeX report and (optionally) compile it to PDF.

    work_dir    : base output directory; the report is written to
                  work_dir/report/{tex_name} with figures under
                  work_dir/report/{figures_dir}/.
    sections    : ordered list of (heading, body_latex) pairs. body_latex is raw
                  LaTeX; a placeholder '{{table:NAME}}' is replaced with
                  tables['NAME'].to_latex(index=False) before writing.
    tables      : dict {name: pandas.DataFrame}, referenced from sections via
                  '{{table:name}}' placeholders. Column names always have '_'
                  escaped (a bare '_' outside math mode is a LaTeX error);
                  cell values are LaTeX-escaped too unless the table's name
                  is listed in raw_tables.
    raw_tables  : iterable of table names (from `tables`) whose CELL VALUES
                  should be inserted as raw LaTeX instead of escaped (e.g. a
                  column already containing '$...$' math).
    figures     : dict {published_filename: source_path}; each source file is
                  copied into work_dir/report/{figures_dir}/{published_filename}
                  (source and destination may already coincide, in which case
                  nothing is copied). Reference them from sections with
                  \\includegraphics{published_filename} (no path prefix needed,
                  \\graphicspath already points at the figures directory).
    run_pdflatex: compile twice (for cross-reference resolution) with
                  `pdflatex -interaction=nonstopmode`. On failure raises
                  RuntimeError with the last ~40 lines of the .log file.

    Returns the path to the compiled PDF (run_pdflatex=True) or to the
    written .tex file (run_pdflatex=False).
    """
    report_dir = os.path.join(work_dir, 'report')
    fig_dir = os.path.join(report_dir, figures_dir)
    os.makedirs(fig_dir, exist_ok = True)

    for published_name, src in (figures or {}).items():
        dst = os.path.join(fig_dir, published_name)
        if os.path.abspath(src) != os.path.abspath(dst):
            shutil.copyfile(src, dst)

    tables = tables or {}

    body = []
    for heading, section_body in sections:
        text = section_body
        for name, df in tables.items():
            placeholder = '{{table:%s}}' % name
            if placeholder in text:
                # Column names are rarely meant to contain LaTeX; a raw '_'
                # (e.g. from a snake_case DataFrame column) is read as a
                # math-mode subscript outside math mode and breaks the
                # compile. Replaced with a space (not an escaped '\_') so it
                # doesn't get double-escaped by pandas' own escape=True below
                # (which would otherwise also escape that literal backslash).
                # Cell values are escaped by pandas itself unless the caller
                # opted out per-table via `raw_tables`.
                safe_df = df.rename(columns = {c: str(c).replace('_', ' ') for c in df.columns})
                escape = name not in (raw_tables or ())
                text = text.replace(placeholder, safe_df.to_latex(index = False, escape = escape))
        body.append('\\section{%s}\n%s\n' % (heading, text))

    date_line = date if date is not None else r'\today'
    author_line = ('\\author{%s}\n' % author) if author else ''

    tex = (
        (_PREAMBLE % figures_dir)
        + '\n\\title{%s}\n' % title
        + author_line
        + '\\date{%s}\n' % date_line
        + '\n\\begin{document}\n\\maketitle\n\n'
        + '\n'.join(body)
        + '\n\\end{document}\n'
    )

    tex_path = os.path.join(report_dir, tex_name)
    with open(tex_path, 'w') as fh:
        fh.write(tex)

    if not run_pdflatex:
        return tex_path

    proc = None
    for _ in range(2):
        proc = subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_name],
            cwd = report_dir, capture_output = True, text = True,
        )

    pdf_path = os.path.join(report_dir, tex_name.replace('.tex', '.pdf'))
    log_path = os.path.join(report_dir, tex_name.replace('.tex', '.log'))

    if proc.returncode != 0 or not os.path.isfile(pdf_path):
        tail = ''
        if os.path.isfile(log_path):
            with open(log_path, 'r', errors = 'replace') as fh:
                tail = ''.join(fh.readlines()[-40:])
        raise RuntimeError(f'pdflatex failed (returncode={proc.returncode}):\n{tail}')

    return pdf_path
