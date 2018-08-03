from setuptools import setup

setup(name='btr',
      version='0.1',
      description='better than random comparisons made easy',
      url='http://github.com/pvtodorov/btr',
      author='Petar Todorov, Artem Sokolov',
      author_email='petar.v.todorov@gmail.com',
      license='MIT',
      packages=['btr'],
      zip_safe=False,
      entry_points={
          'console_scripts': ['btr-predict=btr.cli:predict',
                              'btr-score=btr.cli:score',
                              'btr-stats=btr.cli:stats',
                              'btr-md5=btr.cli:print_settings_md5']},
      install_requires=[
          'numpy', 'pandas', 'patsy', 'python-dateutil', 'pytz',
          'scikit-learn', 'scipy', 'six', 'scikit-learn', 'statsmodels',
          'tqdm', 'morph', 'xgboost', 'mord'
      ]
      )
