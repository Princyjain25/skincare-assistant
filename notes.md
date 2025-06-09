## Reference
https://saurav-samantray.medium.com/unlock-the-power-of-local-ai-build-your-first-agent-with-ollama-rag-and-langchain-f8d3bb171ea1


## Ollama
docker pull ollama/ollama

docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama

Library - https://ollama.com/library

ollama rm <model-name>

## Virtual env
.\vpyrag\Scripts\activate 

.\vpyrag\Scripts\deactivate

## Pip

pip install -r requirements.txt

python -m pip install langchain-openai
