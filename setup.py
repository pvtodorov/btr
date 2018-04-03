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
          'console_scripts': ['btr-predict=btr.predict:main',
                              'btr-score=btr.score:score_main',
                              'btr-stats=btr.score:stats_main']},
      install_requires=[
          'scipy', 'numpy', 'pandas', 'patsy', 'python-dateutil', 'pytz',
          'scikit-learn', 'six', 'sklearn', 'statsmodels', 'tqdm',
          'nose2', 'synapseclient', 'mord'
      ]
      )
