# CoKG(Chain of Knowledge Graph)


## About

CoKG(Chain_of_Knowledge_Graph) Prompting Method for Text Summarization

## Getting Started


### Prerequisites
```shell
$ pip install -r requirements.txt
```

```shell
## .env
OPENAI_API = "sk-xxx"
```

### Run Scripts


#### 1. **Run Summarization with CoKG**
```shell
$ python main.py
```

#### (optional)
```shell
"-t", "--type"          : Summarization Type(default : CoKG, ['Base', 'CoD', 'CoE', 'CoKG'])
"-m", "--model"         : chat model(default : gpt-4o)
"-n", "--num_samples"   : num_sampels(default : 100)
"-o", "--output"        : output_path(default : results/results.json)
```

#### 2. **Run Evaluate**
```shell
$ python evaluate.py -t 'all'
```

#### (optional)
```shell
"-t", "--type"          : Evaluation Type(default : All, ['rouge', 'geval', 'all']])
"-m", "--model"         : chat model(default : gpt-4o-mini)
"-r", "--results"       : result file path(default : results/results.json)
"-s", "--save_fp"       : save_path(default : results/)
```

## Results

### ROUGE-1
| precision | recall | fmeasure |
|-----------|--------|----------|
| 0.9286    | 0.084  | 0.1497   |
### ROUGE-2
| precision | recall | fmeasure |
|-----------|--------|----------|
| 0.5239    | 0.0479 | 0.0852   |
### ROUGE-L
| precision | recall | fmeasure |
|-----------|--------|----------|
| 0.6589    | 0.0599 | 0.1066   |
### GEval
| coherence | consistency | fluency | relevance |
|-----------|-------------|---------|-----------|
| 4.05      | 4.15        | 2.75    | 4.4       |
