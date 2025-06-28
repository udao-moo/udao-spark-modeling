# Index
1. [Installation](#installation)
2. [Execution](#execution)
2. [Download VLDB queries](#download-link-for-queries-vldb)


# Installation
We use two tools
1. [Pixi](#pixi)
2. [Ollama](#ollama)

## Pixi
We use [pixi](https://pixi.sh/latest/#installation) to install 
the necessary packages to install all the necessary packages. Once 
installed you can activate a shell with 
`pixi shell -e dev`
to activate the shell that has all the packages for python. 


To access the python binary you can run `which python`

With pixi you are also installing our linter, formatter and our libraries
for testing. You can access this tasks as a pixi task. 
```bash
pixi run format # Formats src and test files
pixi run check  # Checks lint rules in src
pixi run typing # Uses mypy to validate types
pixi run lint   # Runs format, check and typing
pixi run test   # Runs tests
pixi run main   # Runs main endpoint
pixi run commit # Commits changes after checking lint and test
```
## Ollama
This library you need to install outside of pixi since the python wrapper for ollama
is only querying the ollama server that runs. 
To install Ollama you can follow [the ollama install](https://ollama.com/download) depending 
on your platform. We highly recommend the use of GPU to run LLM models. 
The command `pixi run main complex_queries -c path/to/config` is the only one that
will use ollama, and we recommend adjusting the LLM in the config to one 
that adjusts to your computing resources.

# Execution
## Main program
For documentation you may run
```bash
cd udao_queries
pixi run main --help
```
This will show several endpoints that you may understand by running

```bash
pixi run main param-search --help
```

## Generating Snowflake TPCDS and LLM TPCDS
To generate the snowflake and the LLM dataset we follow the following diagram
![image](https://github.com/user-attachments/assets/36009caa-f2bf-421a-8526-3018c3dcfc23)

```bash
cd udao_queries
pixi run main param-search -c path/to/config
pixi run main --csv path/to/generated_queries/TPCDS_batches.csv --dataset TPCDS --destination some/folder
pixi run main complex_queries -c path/to/config
```

You can find the configs used on `udao_queries/params_config`

# Download link for queries VLDB
You may download the queries by following [This link](https://www.dropbox.com/scl/fi/686eh4fuub8uky7jybk4o/2025_06_26_final_vldb_queries.tar.gz?rlkey=3rmymygnnno2xwjq1ngzwf993&st=9ajey4jy&dl=0)




