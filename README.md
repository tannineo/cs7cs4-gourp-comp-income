# cs7cs4-group-comp

Group Competition CS7CS4

## best result

[main_best_code_2.py](./main_best_code_2.py)
[tcd-ml-1920-group-income-submission.csv](./tcd-ml-1920-group-income-submission.csv)

## environment

- use `pipenv` as the virtual env manager to manage dependencies

## non-ASCII processing for lightgbm

regex: `[^\x00-\x7f]` to find non-ASCII characters.

Côte d'Ivoire => CDI

## notes

- Don't use `KFold` from `sklearn`, the memory management is terrible when using this to deal large datasets.

## authors

- [Mary McDonald](https://github.com/mcdonam7)
- [Jagadish Ramamurthy](https://github.com/jagadishr12)
- [Chao Chen](https://github.com/tannineo)
