# Configuration file for the Sphinx documentation builder.
import os
import sys

from pathlib import Path

# -- Project information
project = "REaLTabFormer"
copyright = "2023, World Bank"
author = "Aivin Solatorio"

# -- General configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_nb",
    "sphinx_autodoc_typehints",
]
autoapi_dirs = ["../src"]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "sphinx": ("https://www.sphinx-doc.org/en/master/", None),
}
source_suffix = [".rst", ".md"]
templates_path = ["_templates"]

html_logo = "../img/REalTabFormer_Final_EQ.png"
html_title = "REaLTabFormer"

html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/worldbank/realtabformer",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "logo": {
        "text": f"REaLTabFormer {(Path(__file__).parent.parent / 'src' / 'realtabformer'/ 'VERSION').read_text().strip()}"
    },
}
