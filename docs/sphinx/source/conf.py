from plotly.io._sg_scraper import plotly_sg_scraper

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Supporting code for the LDS for Neuro lecture at ITBA 2026'
copyright = '2026, Joaquin Rapela'
author = 'Joaquin Rapela'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx_gallery.gen_gallery",
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

sphinx_gallery_conf = {
  # Path to your lecture example scripts relative to conf.py
  "examples_dirs": "../../../examples",
  # Path where Sphinx-Gallery will output generated rst/images
  "gallery_dirs": "auto_examples",
  # Use Plotly scraper for interactive figures (add 'matplotlib' if mixing both)
  "image_scrapers": ("plotly.io._sg_scraper.plotly_sg_scraper",),
  # "image_scrapers": (plotly_sg_scraper,),
  # Automatically re-run stale examples on build
  "run_stale_examples": True,
}
