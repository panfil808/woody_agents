# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SmartWoody is a multi-agent customer support system for construction store sellers. It uses LangChain agents with LangGraph to help sellers collect and validate customer problem information through an interactive dialog interface built with Gradio.

## Project Architecture

### Core Components

**DialogAgent**: Information collection agent - the FIRST step in the workflow:
1. Validates order numbers (8 digits starting with 7 or 8)
2. Collects comprehensive problem details
3. Gathers required actions from logistician
4. When done, signals: "Информация собрана. Готово к классификации."

Note: DialogAgent does NOT determine problem category - it only collects information.

**ClassifierAgent**: Category determination agent - runs AFTER DialogAgent completes:
- Receives collected information (order number, problem description, required actions)
- Analyzes complete context to determine category
- Returns one of: доставка, комплектация, брак, возврат
- Classification based on full information, not just keywords

**JudgeAgent**: Evaluates the quality of DialogAgent's interaction based on completeness, relevance, and efficiency. Returns JSON evaluation with score (1-10), completeness status, identified issues, and summary.

### Key Technical Details

- **LLM Backend**: Uses custom endpoint at `https://gpt.sdvor.com/api/v1` with model `cpatonn/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit`
- **Session Management**: Each conversation has a unique thread_id with InMemorySaver checkpoint for conversation state
- **Tools**: DialogAgent has access to three simple functions decorated with `@tool`:
  - `validate_order_number`: Regex validation (10 digits with leading zeros OR 8 digits, starting with 7/8)
  - `check_data_completeness`: Validates pipe-delimited data format (order_num|category|description|action)
  - `save_problem_data`: Persists problem data to JSON in `appeals/` directory
  - All tools are simple coroutines, not classes, for simplicity

### Data Flow

1. User initializes system with API key (or uses environment variable OPENAI_API_KEY)
2. System creates DialogAgent and ClassifierAgent
3. **Collection Stage**: DialogAgent guides seller through information gathering
   - Requests order number (validates with tool)
   - Asks for problem description
   - Asks for required actions from logistician
4. **Classification Trigger**: DialogAgent signals completion ("Информация собрана")
5. **Classification Stage**: System invokes ClassifierAgent
   - Extracts data from conversation history
   - Analyzes: order number + description + actions
   - Determines category
6. Problem data saved to `appeals/problem_{order_num}_{timestamp}.json` with category
7. On session end, JudgeAgent evaluates the dialog quality

### Workflow Stages

**Stage 1: Information Collection**
- DialogAgent is active from session start
- Collects: order number, problem description, required actions
- Uses validation tools
- When done, says: "Информация собрана. Готово к классификации."

**Stage 2: Automatic Classification** (triggered by Stage 1 completion)
- System detects DialogAgent completion signal
- Extracts collected data from chat history
- ClassifierAgent analyzes all data together
- Category stored in `session_state.category`
- Valid categories: доставка, комплектация, брак, возврат

## Common Commands

### Run Application
```bash
python app.py
```
Server starts on `0.0.0.0:7860` with Gradio interface in debug mode.

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Set API Key
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Development Notes

### Problem Data Format
JSON files in `appeals/` directory follow this schema:
```json
{
  "timestamp": "ISO 8601 datetime",
  "order_number": "8-digit string",
  "category": "доставка|комплектация|брак|возврат",
  "description": "free text problem description",
  "required_action": "free text action required"
}
```

### Order Number Validation
Valid formats: `0087954612` (10 digits) OR `87954612` (8 digits). Must start with 7 or 8.
Regex: `^0{2}[7|8]\d{7}|[7|8]\d{7}$`

### Agent State Management
The `session_state` dictionary maintains:
- `api_key`: API key for LLM calls
- `active`: Boolean session status
- `chat_history`: List of message dicts with role/content
- `agent_graph`: LangChain agent graph instance
- `thread_id`: UUID for conversation checkpointing

### Prompt Engineering
- DialogAgent uses system prompt with strict 5-step algorithm
- JudgeAgent requires strict JSON output format (without markdown code blocks)
- All prompts are in Russian for Russian-speaking sellers