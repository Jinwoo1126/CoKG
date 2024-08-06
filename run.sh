#!/bin/bash
python3 main.py -t 'CoD' -n 100 -o 'results/CoD_results.json'
python3 evaluate.py -t 'rouge' -r 'results/CoD_results.json'
python3 main.py -t 'Base' -n 100 -o 'results/Base_results.json'
python3 evaluate.py -t 'rouge' -r 'results/Base_results.json'
python3 main.py -t 'CoE' -n 100 -o 'results/CoE_results.json'
python3 evaluate.py -t 'rouge' -r 'results/CoE_results.json'
python3 main.py -t 'CoKG' -n 100 -o 'results/CoKG_results.json'
python3 evaluate.py -t 'rouge' -r 'results/CoKG_results.json'
