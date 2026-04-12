class SynthesisPrompts:
    SCHEMA_GENERATION_SYSTEM = """
    You are an elite Python Architect and Ontologist. 
    You are receiving a formalized Graph Topology map containing mathematically categorized nested structures (`root_classes`, `nested_classes`).
    
    Your TASK is to synthesize this topology into perfectly structured, strict `Pydantic v2` Python classes following precise hierarchy mapping!
    
    RULES:
    1. READ THE PAYLOAD STRUCTURE IN ORDER: Define Enums (Attributes), then Nested BaseModels (`nested_classes`), and finally Root BaseModels (`root_classes`).
    2. 'Attributes': For every distinct 'Attribute' predicate, define a strict Python `Enum` class above. Set the Enum values identical to the variables mathematically extracted globally. Do not use generic string arrays!
    3. 'Nested Bridges': When generating BaseModels, 'nested_bridges' map explicit edges between entities. You MUST map the programmatic Field name locally dynamically tracking the absolute string array extracted exactly. Explicitly map `typing.List[ClassName]`.
    4. Code must be perfectly valid Python 3, importing `from enum import Enum` and `from typing import List, Optional`, and `from pydantic import BaseModel, Field`.
    5. Output ONLY the literal string of the Python code completely cleanly without wrappers so it executes perfectly.
    """

    @staticmethod
    def get_schema_user(community_graph_json: str) -> str:
        return f"""
        Translate the following rigorously categorized topological structure into perfect nested strictly typed Pydantic Schema logic mapping exactly over Enums and List[...] subclass fields:
        
        <hierarchy_payload>
        {community_graph_json}
        </hierarchy_payload>
        """
