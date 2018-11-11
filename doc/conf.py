
# -- Project information -----------------------------------------------------

project = 'l3py'
copyright = '2018, Andreas Kvas'
author = 'Andreas Kvas'

version = '0.1'
release = '0.1.1'

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinx.ext.napoleon',
    'sphinx.ext.autosummary',
    'numpydoc'
]

autosummary_generate = True
templates_path = ['_templates']


source_suffix = '.rst'

master_doc = 'index'

language = None

exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

pygments_style = None

html_theme = 'sphinxdoc'

html_static_path = ['_static']

htmlhelp_basename = 'l3pydoc'
