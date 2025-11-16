```mermaid
graph TB
    User[👤 User: Competition URL] --> Orch[🎯 Orchestrator Agent<br/>ADK Core]
    
    Orch --> CR[📖 Competition Reader<br/>Gemini 2.0 Flash]
    CR --> |Strategy Document| Orch
    
    Orch --> DE[🔍 Data Explorer<br/>ADK + Tools]
    DE --> |EDA Results| Orch
    
    Orch --> FE[⚙️ Feature Engineer<br/>ADK + Memory]
    FE --> |Features + Importance| Orch
    
    Orch --> MS[🤖 Model Selection<br/>ADK + A2A]
    MS --> |Model Results| Orch
    
    MS -.->|A2A: Request Better Features| FE
    FE -.->|A2A: New Features| MS
    
    Orch --> RG[📊 Report Generator<br/>Gemini 2.0 + Tools]
    RG --> |Final Deliverables| Output
    
    Output[📦 Outputs:<br/>- EDA Notebooks<br/>- Features Documentation<br/>- Trained Models<br/>- Submission CSV<br/>- Executive Summary]
    
    subgraph Memory["💾 Persistent Memory"]
        DB[(SQLite:<br/>Feature Performance<br/>Experiment History)]
    end
    
    FE <-.-> DB
    
    subgraph Tools["🛠️ Python Tools"]
        T1[analyze_dataframe]
        T2[create_visualization]
        T3[test_feature_importance]
        T4[train_model]
        T5[cross_validate]
    end
    
    DE --> Tools
    FE --> Tools
    MS --> Tools
    
    style Orch fill:#4285f4,stroke:#1967d2,stroke-width:3px,color:#fff
    style CR fill:#fbbc04,stroke:#f9ab00,stroke-width:2px
    style DE fill:#34a853,stroke:#0f9d58,stroke-width:2px
    style FE fill:#ea4335,stroke:#c5221f,stroke-width:2px
    style MS fill:#9334e6,stroke:#7627bb,stroke-width:2px
    style RG fill:#ff6d01,stroke:#e65100,stroke-width:2px
    style Output fill:#5f6368,stroke:#3c4043,stroke-width:2px,color:#fff
    style DB fill:#185abc,stroke:#1967d2,stroke-width:2px,color:#fff
    style Memory fill:#e8f0fe,stroke:#d2e3fc,stroke-width:2px
    style Tools fill:#f1f3f4,stroke:#dadce0,stroke-width:2px
```

## Architecture Diagram

### System Flow:
1. **User Input** → Competition URL or dataset
2. **Orchestrator** coordinates all agents via ADK
3. **Competition Reader** (Gemini) → Extracts problem requirements
4. **Data Explorer** → Generates EDA with tools
5. **Feature Engineer** → Creates features (stores in memory)
6. **Model Selection** → Trains models, requests features via A2A
7. **Report Generator** (Gemini) → Synthesizes outputs
8. **Final Deliverables** → Notebooks, models, submission

### Key Components:

**Agent Communication:**
- Solid arrows (→): Sequential workflow
- Dotted arrows (⇢): A2A protocol (async communication)

**Technology Stack:**
- 🎯 ADK Core: Multi-agent orchestration
- 🤖 Gemini 2.0: Natural language processing
- 🛠️ Python Tools: Deterministic execution
- 💾 SQLite: Persistent memory

**Agent Specialization:**
- 📖 Competition Reader: NLU for requirements
- 🔍 Data Explorer: Statistical analysis
- ⚙️ Feature Engineer: Feature creation + learning
- 🤖 Model Selection: ML training + A2A requests
- 📊 Report Generator: Human-readable synthesis
