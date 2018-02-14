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
          'console_scripts': ['btr-predict=btr.command_line:predict_main',
                              'btr-score=btr.command_line:score_main',
                              'btr-stats=btr.command_line:stats_main']},
      install_requires=[
          'mord', 'numpy', 'pandas', 'patsy', 'python-dateutil', 'pytz',
          'scikit-learn', 'scipy', 'six', 'sklearn', 'statsmodels', 'tqdm'
      ]
      )
