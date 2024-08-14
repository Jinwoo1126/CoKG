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

| Metric          |       | Base  | CoD   | CoE   | CoKG  |
|-----------------|-------|-------|-------|-------|-------|
| **ROUGE-1**     |       | 0.417 | 0.226 | 0.360 | **0.418** |
| **ROUGE-2**     |       | **0.114** | 0.055 | 0.099 | **0.114** |
| **ROUGE-L**     |       | **0.189** | 0.123 | 0.178 | **0.189** |
| **METEOR**      |       | 0.266 | 0.111 | 0.195 | **0.269** |
| **G-eval**      | Coherence   | 4.301 | 3.990 | 4.467 | **4.555** |
|                 | Consistency | 4.630 | 4.450 | 4.644 | **4.685** |
|                 | Fluency     | 2.731 | 2.587 | 2.680 | **2.743** |
|                 | Relevance   | 4.722 | 4.537 | 4.781 | **4.818** |