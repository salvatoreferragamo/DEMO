#### Required packages
Provided in `requirements.txt`.

#### Install huggingface library
Follow instructions in https://huggingface.co/docs/transformers/installation and do <b>Editable Install</b>

#### Add demand-oriented context-aware decoding
Replace `transformers/src/transformers/generation/utils.py` with `generation_utils.py`

#### Reproduce the experimental results
- The required datasets are processed before using the `data_process.ipynb`.

- Change the `datasets_file_path` in `./src/utils.py` according to the path of `test_plos.jsonl` and  `test_elife.jsonl`.

- Run sample bash scripts `test_decoder.sh`, the path is needed to add manully.

- A detailed list of arguments can be found at `src/test_performance_decoder.py`.
 

***Attention: To comply with the anonymity policy, we replaced all original paths in the experimental code with relative paths or manually added paths to prevent the disclosure of the author's name or the name of their institution.***