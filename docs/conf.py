from datetime import date
import os
import shutil

import toml

import audeer


config = toml.load(audeer.path('..', 'pyproject.toml'))


# Project -----------------------------------------------------------------
author = ', '.join(author['name'] for author in config['project']['authors'])
copyright = f'2021-{date.today().year} audEERING GmbH'
project = config['project']['name']
version = audeer.git_repo_version()
title = 'Documentation'


# General -----------------------------------------------------------------
master_doc = 'index'
source_suffix = '.rst'
exclude_patterns = [
    'api-src',
    'build',
    'tests',
    'Thumbs.db',
    '.DS_Store',
]
templates_path = ['_templates']
pygments_style = None
extensions = [
    'jupyter_sphinx',  # executing code blocks
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # support for Google-style docstrings
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',  # for "copy to clipboard" buttons
]

# Ignore package dependencies during building the docs
autodoc_mock_imports = [
    'tqdm',
]

# Reference with :ref:`data-header:Database`
autosectionlabel_prefix_document = True
autosectionlabel_maxdepth = 2

# Do not copy prompot output
copybutton_prompt_text = r'>>> |\.\.\. '
copybutton_prompt_is_regexp = True

# Mapping to external documentation
intersphinx_mapping = {
    'audobject': ('https://audeering.github.io/audobject/', None),
    'python': ('https://docs.python.org/3/', None),
    'onnxruntime': ('https://onnxruntime.ai/docs/api/python/', None),
    'opensmile': ('https://audeering.github.io/opensmile-python/', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
}

# Disable Gitlab as we need to sign in
linkcheck_ignore = [
    'https://gitlab.audeering.com',
    'https://sphinx-doc.org',
]

# Disable auto-generation of TOC entries in the API
# https://github.com/sphinx-doc/sphinx/issues/6316
toc_object_entries = False


# HTML --------------------------------------------------------------------
html_theme = 'sphinx_audeering_theme'
html_theme_options = {
    'display_version': True,
    'logo_only': False,
    'footer_links': False,
}
html_context = {
    'display_github': True,
}
html_title = title
html_static_path = ['_static']


# Copy API (sub-)module RST files to docs/api/ folder ---------------------
audeer.rmdir('api')
audeer.mkdir('api')
api_src_files = audeer.list_file_names('api-src')
api_dst_files = [
    audeer.path('api', os.path.basename(src_file))
    for src_file in api_src_files
]
for src_file, dst_file in zip(api_src_files, api_dst_files):
    shutil.copyfile(src_file, dst_file)
