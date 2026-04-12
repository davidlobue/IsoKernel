from typing import List
from pydantic import BaseModel, Field, field_validator

class Theme(BaseModel):
    title: str = Field(description="The short, overarching title of the theme or categorical class.")
    description: str = Field(description="A brief description of what this theme encompasses.")
    importance_reasoning: str = Field(description="Why this theme is one of the most critical classes of information in the text.")

class ThemeDiscoveryResult(BaseModel):
    themes: List[Theme] = Field(description="The most critical overarching themes/classes of information discovered in the text. The exact number of themes is dictated dynamically by the text's core logic.")

class RawTriple(BaseModel):
    subject: str = Field(description="The source entity node.")
    predicate: str = Field(description="The relationship or action linking Subject to Object.")
    object: str = Field(description="The target entity node or literal value.")
    theme_association: str = Field(default="", description="The specific overarching Theme title this relationship falls under, if any.")
    quote: str = Field(description="The exact source quote from the text that justifies this relationship.")
    certainty_score: float = Field(description="The certainty score between 0.0 and 1.0.", ge=0.0, le=1.0)

    @field_validator('subject', 'predicate', 'object', 'theme_association', 'quote', mode='before')
    def _coerce_null(cls, v):
        return "" if v is None else str(v)
        
    @field_validator('certainty_score', mode='before')
    def _coerce_float(cls, v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return 0.0

class TripleExtractionResult(BaseModel):
    reasoning: str = Field(description="Step-by-step reasoning explaining why these exact triples were selected.")
    triples: List[RawTriple] = Field(default_factory=list, description="List of extracted open triples.")

class DocumentSource(BaseModel):
    id: str = Field(description="Unique identifier for the document.")
    text_content: str = Field(description="Raw text content of the document.")

class ClusterHypernym(BaseModel):
    cluster_id: str = Field(description="The unique identifier for the mapped cluster.")
    hypernym: str = Field(description="The standardized general semantic label that best represents the cluster variants.")

class TaxonomicVerification(BaseModel):
    cluster_id: str = Field(description="The unique identifier for the mapped cluster.")
    centroid: str = Field(description="The anchor term mathematically centered inside the cluster matrix.")
    formal_hypernym: str = Field(description="The formal categorical target the node logically belongs to.")
    members_verified: bool = Field(description="True if ALL members structurally satisfy the strict Is-A categorical taxonomy mapping against the formal_hypernym, otherwise False.")
    reasoning: str = Field(description="A single sentence deduction structurally validating the Is-A constraints across all members.")

class TaxonomicLiftingResult(BaseModel):
    resolutions: List[TaxonomicVerification] = Field(description="List of structurally verified taxonomic classifications mapped natively to their parent group.")

class ClusterContextualValidation(BaseModel):
    cluster_id: str = Field(description="The internal ID of the cluster being formally validated.")
    accuracy_destroyed: bool = Field(description="True if merging these mathematical entities mechanically destroys critical distinguishing accuracy in the context of the corpus. False if they are contextually safe to merge.")
    reasoning: str = Field(description="A single sentence explaining strictly why merging these exact subjects mathematically either destroys or preserves critical accuracy down the line.")

class GeneratedSchema(BaseModel):
    class_name: str = Field(description="The formal Python class name structurally aligning to the graph community hypernym.")
    python_code: str = Field(description="The 100% syntactically valid raw Python 3 execution code text mapping the Enum fields natively securely out across Pydantic inheritance bounds.")
