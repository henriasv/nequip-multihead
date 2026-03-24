project = "nequip-multihead"
copyright = "2026, Henrik Andersen Sveinsson"
author = "Henrik Andersen Sveinsson"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": True,
}

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
]
myst_heading_anchors = 3

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
    "nequip": ("https://nequip.readthedocs.io/en/latest/", None),
    "ase": ("https://wiki.fysik.dtu.dk/ase/", None),
}
