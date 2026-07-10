# LangChain UI 
A simple connection between langchain deepagents app and LibreChat frontend 

## Installation

### Install langchain-ui

To install langchain_ui from GitHub repository, do:

```console
git clone git@github.com:surf_research_agent/langchain_ui.git
cd langchain_ui
python -m pip install .
```

### Install LibreChat

Clone the LibreChat repo on your machine

```console
git clone https://github.com/danny-avila/librechat
```

## Using LangChainUI natively

1. Start the langchain ui server

```bash
cd langchain_ui/app
python app.pu
```

2. Update your [librechat.yaml](https://www.librechat.ai/docs/configuration/librechat_yaml) to include your server as a custom endpoint.

If not already done copy `librechat.example.yaml` in `librechat.yaml` Then add the following lines in the yaml file:

```yaml filename="librechat.yaml"
endpoints:
  custom:
    - name: "SURF Agents" # Could be anything
      apiKey: "super-secret" # Could be anything
      baseURL: "http://host.docker.internal:8000/"
      models:
        default: ["SURF AI"] # Could be anything
      titleConvo: true
      titleModel: "current_model"
      summarize: false
      summaryModel: "current_model"
      forcePrompt: false
      modelDisplayLabel: "SURF Research Agent" # Could be anything
```

LibreChat will call your app using the OpenAI-style route, typically `/v1/chat/completions`, so the Flask server accepts both `/chat/completions` and `/v1/chat/completions`.

3. Create and start Librechat containers.

```bash
cd Librechat
docker compose up -d
```

4. Open UI and choose your endpoint from the dropdown menu at the top left.
5. Start conversing with the model!


## Using LangChainUI in Docker

1. Build an image using the provided dockerfile.
2. Reference the Docker image in the [docker-compose.override.yml](https://www.librechat.ai/docs/configuration/docker_override).
3. Create and start Librechat containers.

```bash
cd Librechat
docker compose up -d
```

## Resources

- [OpenAI API Chat Docs](https://platform.openai.com/docs/api-reference/chat/create)


## Documentation

Include a link to your project's full documentation here.



## Credits

This package was created with [Copier](https://github.com/copier-org/copier) and the [NLeSC/python-template](https://github.com/NLeSC/python-template).
