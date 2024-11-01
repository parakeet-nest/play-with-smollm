# Install Ollama on a Pi

## Install Ollama
```bash
# install
curl -fsSL https://ollama.com/install.sh | sh
```

## Make Ollama reachable from outside

```bash
sudo systemctl edit ollama.service
```

Add this:
```text
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Then:
```bash
systemctl daemon-reload
systemctl restart ollama
```

Check `curl http://dns-name-or-ip:11434`

## Load and run a LLM

```bash
ollama run smollm2:360m
```
