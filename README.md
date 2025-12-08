# SmartWoody

Multi-agent customer support system for construction store sellers using LangChain agents and LangGraph.

## Project Structure

```
SmartWoody/
├── app.py                    # Main entry point
├── .env.example              # Environment variables template
├── requirements.txt          # Python dependencies
├── CLAUDE.md                # Development guidelines
├── README.md                # This file
│
├── src/                     # Source code
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   │
│   ├── agents/              # Agent implementations
│   │   ├── __init__.py
│   │   ├── classifier_agent.py  # Problem category classifier
│   │   ├── dialog_agent.py      # Information collection agent
│   │   ├── judge_agent.py       # Dialog quality evaluator
│   │   └── prompts.py           # Agent prompts
│   │
│   ├── tools/               # LangChain tools
│   │   ├── __init__.py
│   │   ├── order_validator.py   # Order number validation
│   │   ├── data_checker.py      # Data completeness checker
│   │   └── problem_saver.py     # JSON data saver
│   │
│   ├── services/            # Business logic services
│   │   ├── __init__.py
│   │   ├── llm_factory.py       # LLM instance factory
│   │   └── session_manager.py   # Session state management
│   │
│   ├── models/              # Data models
│   │   ├── __init__.py
│   │   ├── problem_data.py      # Problem data model
│   │   └── session_state.py     # Session state model
│   │
│   └── ui/                  # User interface
│       ├── __init__.py
│       └── gradio_interface.py  # Gradio web interface
│
└── appeals/                 # Output directory for problem data (auto-created)
```

## Features

- **DialogAgent**: Collects detailed problem information from sellers
  - Validates order numbers (8 digits starting with 7 or 8)
  - Gathers comprehensive problem description
  - Collects required actions from logistician
  - Guides seller through structured data collection

- **ClassifierAgent**: Automatically determines problem category after collection
  - Analyzes collected information (order number, description, actions)
  - Determines category: доставка (delivery), комплектация (packaging), брак (defect), or возврат (return)
  - Classifies based on complete context, not just keywords

- **JudgeAgent**: Evaluates dialog quality based on:
  - Completeness of collected information
  - Relevance of questions asked
  - Efficiency of the conversation

- **OOP Architecture**: Clean separation of concerns
  - Models: Data structures
  - Services: Business logic
  - Agents: LangChain agent implementations
  - Tools: LangChain tools
  - UI: Gradio interface

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd SmartWoody
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env and set your OPENAI_API_KEY
```

## Configuration

Configuration is managed through environment variables in `.env` file:

```bash
# Required
OPENAI_API_KEY=your-api-key-here

# LLM Configuration (optional, has defaults)
LLM_BASE_URL=https://gpt.sdvor.com/api/v1
LLM_MODEL=cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit
LLM_TEMPERATURE=0.3
LLM_MAX_TOKENS=1000
LLM_TIMEOUT=30

# Application Configuration (optional, has defaults)
APPEALS_DIR=appeals
SERVER_HOST=0.0.0.0
SERVER_PORT=7860
DEBUG_MODE=True
```

## Usage

### Running the Application

```bash
python app.py
```

The application will start on `http://0.0.0.0:7860` (or configured host/port).

### Using the Interface

1. **Initialize**: Enter your API key and click "Инициализировать"
2. **Collect Information**: DialogAgent will guide you through:
   - Providing order number
   - Describing the problem in detail
   - Specifying required actions from logistician
3. **Automatic Classification**: Once data is collected, ClassifierAgent determines the category
4. **New Session**: Click "🔄 Новая сессия" to start fresh
5. **End Session**: Click "✅ Завершить" to evaluate the dialog

### Workflow

The system follows a two-stage workflow:

**Stage 1: Information Collection** (DialogAgent)
- Seller starts conversation
- DialogAgent requests order number and validates it
- DialogAgent asks for detailed problem description
- DialogAgent asks for required actions from logistician
- When complete, DialogAgent signals: "Информация собрана. Готово к классификации."

**Stage 2: Automatic Classification** (ClassifierAgent)
- System extracts collected data from conversation
- ClassifierAgent analyzes all information together
- Determines category: доставка, комплектация, брак, or возврат
- System displays the determined category
- Data can be saved with correct category

### Order Number Format

Valid order numbers:
- 8 digits starting with 7 or 8: `87954612`
- 10 digits with leading zeros: `0087954612`

Regex: `^0{2}[7|8]\d{7}|[7|8]\d{7}$`

### Problem Categories

- `доставка` - Delivery issues
- `комплектация` - Packaging/completeness issues
- `брак` - Defective products
- `возврат` - Returns

## Output Format

Problem data is saved to `appeals/` directory in JSON format:

```json
{
  "timestamp": "2025-12-08T12:34:56.789012",
  "order_number": "87954612",
  "category": "доставка",
  "description": "Заказ не доставлен в срок",
  "required_action": "Связаться с курьерской службой"
}
```

Filename pattern: `problem_{order_number}_{timestamp}.json`

## Development

### Code Style

- Follow PEP 8 style guide
- Use type hints where applicable
- Document classes and methods with docstrings
- Keep modules focused and cohesive

### Adding New Tools

1. Create tool function in `src/tools/`
2. Use `@tool` decorator from `langchain.tools`
3. Implement tool logic with clear docstring
4. Register in `src/tools/__init__.py`
5. Import and add to agent's tool list

### Adding New Agents

1. Create agent class in `src/agents/`
2. Implement agent logic using LLMFactory
3. Add prompts to `src/agents/prompts.py`
4. Register in `src/agents/__init__.py`
5. Integrate with UI workflow in `src/ui/gradio_interface.py`

### Customizing ClassifierAgent

The ClassifierAgent prompt is located at `src/agents/prompts.py:3-12`. To customize:

1. Open `src/agents/prompts.py`
2. Edit `CLASSIFIER_AGENT_PROMPT` variable
3. Define clear classification criteria
4. Specify output format expectations
5. Test with various problem descriptions

## Logging

Application uses Python's built-in logging. Logs include:
- INFO: Application flow and state changes
- WARNING: Validation failures and issues
- ERROR: Exceptions and failures

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
