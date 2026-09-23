#!/usr/bin/env python3
"""Tests for pmpiv.report.build_latex_report."""
import shutil

import pandas as pd
import pytest

from pmpiv import report

if shutil.which('pdflatex') is None:
    pytest.skip('pdflatex not installed', allow_module_level=True)


class TestBuildLatexReport:
    def test_tex_only(self, tmp_path):
        path = report.build_latex_report(
            str(tmp_path), title='Test Report',
            sections=[('Intro', 'Hello world.')],
            run_pdflatex=False,
        )
        assert path.endswith('report.tex')
        text = open(path).read()
        assert 'Hello world.' in text
        assert '\\section{Intro}' in text

    def test_table_placeholder_substitution(self, tmp_path):
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        path = report.build_latex_report(
            str(tmp_path), title='Test Report',
            sections=[('Results', 'See table below.\n\n{{table:mytable}}')],
            tables={'mytable': df},
            run_pdflatex=False,
        )
        text = open(path).read()
        assert '{{table:mytable}}' not in text
        assert 'begin{tabular}' in text

    def test_underscore_column_names_not_double_escaped(self, tmp_path):
        # Regression: a pre-escaped '\_' fed into pandas' own escape=True
        # gets re-escaped into a literal visible backslash in the PDF.
        df = pd.DataFrame({'n_eddies': [1, 2], 'dominant_radius_px': [10.0, 20.0]})
        path = report.build_latex_report(
            str(tmp_path), title='Test Report',
            sections=[('Results', '{{table:mytable}}')],
            tables={'mytable': df},
            run_pdflatex=False,
        )
        text = open(path).read()
        assert '\\_' not in text
        assert 'n eddies' in text
        assert 'dominant radius px' in text

    def test_compiles_to_pdf(self, tmp_path):
        pdf_path = report.build_latex_report(
            str(tmp_path), title='Smoke Test',
            sections=[('Section', 'Some text with $x^2$ math.')],
        )
        assert pdf_path.endswith('report.pdf')
        with open(pdf_path, 'rb') as fh:
            content = fh.read()
        assert len(content) > 0
        assert content.startswith(b'%PDF')

    def test_compile_failure_raises_with_log_tail(self, tmp_path):
        with pytest.raises(RuntimeError):
            report.build_latex_report(
                str(tmp_path), title='Broken',
                sections=[('Section', '\\undefinedcommandthatdoesnotexist{}')],
            )

    def test_figures_are_copied(self, tmp_path):
        src_dir = tmp_path / 'src'
        src_dir.mkdir()
        fig = src_dir / 'plot.png'
        fig.write_bytes(b'\x89PNG\r\n\x1a\n' + b'0' * 20)

        path = report.build_latex_report(
            str(tmp_path), title='Figs',
            sections=[('Results', '\\includegraphics{plot.png}')],
            figures={'plot.png': str(fig)},
            run_pdflatex=False,
        )
        copied = tmp_path / 'report' / 'figures' / 'plot.png'
        assert copied.exists()
        assert copied.read_bytes() == fig.read_bytes()
