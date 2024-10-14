#### Required packages
Provided in `requirements.txt`.

#### Reproduce the experimental results
- Generate the training data by `generate_data/generate_dataset.ipynb` and move it into `data` folder 

- Run sample bash scripts `train.sh`, the path is needed to add manully.

- The training code is based on `LLaMA-Factory` Git Repo.

- Run `infer.sh` for evaluate on the test set.
 

***Attention: We replaced all absolute paths in the experimental code with relative paths, requiring users to modify them according to their own directory structure.***
