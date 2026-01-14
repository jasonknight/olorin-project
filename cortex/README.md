# Cortex - Exo Kafka Consumer

Cortex is a Kafka consumer that bridges prompts from a Kafka topic to [exo](https://github.com/exo-explore/exo), a distributed AI inference system, and sends the model's responses back to Kafka for text-to-speech processing.

## Architecture

```
[Producer] -> [Kafka: prompts] -> [Cortex Consumer] -> [Exo API]
                                                           |
                                                           v
[TTS Consumer] <- [Kafka: ai_out] <-----------------------+
```

## Prerequisites

1. **Kafka**: Running at `localhost:9092`
2. **Exo**: Running with OpenAI-compatible API at `http://localhost:52415/v1`
3. **Python 3.8+**: For running the consumer

## Installation

1. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create the Kafka topics:
```bash
./create-topic prompts
./create-topic ai_out  # If not already created by broca
```

## Configuration

Edit `.env` to configure the consumer:

```bash
# Kafka Configuration
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_INPUT_TOPIC=prompts
KAFKA_OUTPUT_TOPIC=ai_out
KAFKA_CONSUMER_GROUP=exo-consumer-group
KAFKA_AUTO_OFFSET_RESET=earliest

# Exo Configuration
EXO_BASE_URL=http://localhost:52415/v1
EXO_API_KEY=dummy-key

# Model Configuration
MODEL_NAME=llama-3.2-1b
TEMPERATURE=0.7
MAX_TOKENS=1000

# Logging
LOG_LEVEL=INFO
```

### Key Configuration Options

- **MODEL_NAME**: The model loaded in exo (check with `exo` CLI)
- **TEMPERATURE**: Controls response randomness (0.0-2.0)
- **MAX_TOKENS**: Maximum response length
- **EXO_BASE_URL**: Exo API endpoint (default: `http://localhost:52415/v1`)

## Usage

### Start the Consumer

```bash
./run
```

Or manually:
```bash
source venv/bin/activate
python3 consumer.py
```

### Send Prompts

You can send prompts to the `prompts` topic using:

1. **Kafka Console Producer**:
```bash
kafka-console-producer --topic prompts --bootstrap-server localhost:9092
> Hello, tell me a joke
```

2. **Python Script**:
```python
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

producer.send('prompts', {'text': 'Tell me a joke'})
```

3. **Plain Text**:
The consumer also accepts plain text messages (not just JSON).

### Message Format

The consumer accepts messages in two formats:

**JSON Format** (recommended):
```json
{
  "text": "Your prompt here",
  "id": "optional-message-id"
}
```

**Plain Text Format**:
```
Your prompt here
```

### Output Format

Responses are sent to the `ai_out` topic in this format:
```json
{
  "text": "The model's response",
  "id": "message_id_response",
  "prompt_id": "original_message_id",
  "model": "llama-3.2-1b",
  "timestamp": "2026-01-13T09:30:00.123456"
}
```

## Integration with Broca

The `ai_out` topic is consumed by the [broca](../broca) TTS consumer, which:
1. Receives the text response
2. Converts it to speech using Coqui TTS
3. Plays the audio

This creates a complete pipeline:
```
Text Prompt -> Exo (AI) -> Text Response -> TTS -> Audio Output
```

## Dynamic Configuration Reloading

The consumer automatically detects changes to `.env` and reloads configuration without restarting. This allows you to:
- Change model parameters on the fly
- Switch models
- Adjust temperature/max_tokens
- Update logging levels

## Troubleshooting

### Exo Not Running
```
Error: Connection refused to localhost:52415
```
**Solution**: Start exo before running the consumer:
```bash
exo  # or however you start your exo instance
```

### Kafka Connection Issues
```
Error: NoBrokersAvailable
```
**Solution**: Ensure Kafka is running:
```bash
# Check if Kafka is running
kafka-topics --list --bootstrap-server localhost:9092
```

### Model Not Found
```
Error: Model 'llama-3.2-1b' not found
```
**Solution**: Check available models in exo and update `MODEL_NAME` in `.env`

## Utility Scripts

- `./create-topic [topic-name]` - Create a Kafka topic
- `./list-topics` - List all Kafka topics
- `./run` - Start the consumer with automatic venv setup

## Development

The consumer is structured similarly to the broca TTS consumer:
- Configuration via environment variables
- Hot-reload on `.env` changes
- Structured logging
- Error handling with error messages sent to output topic

## Related Projects

- **broca**: TTS consumer (`../broca`)
- **hippocampus**: Vector database for RAG (`../hippocampus`)
- **exo**: Distributed AI inference ([GitHub](https://github.com/exo-explore/exo))
