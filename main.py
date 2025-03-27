import os
import io
import re
import json
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import nltk
from nltk.tokenize import sent_tokenize
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from PyPDF2 import PdfReader

from transformers import pipeline

# Initialize the zero-shot classification pipeline
# We use a model fine-tuned on natural language inference such as facebook/bart-large-mnli
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Initialize sentence transformer model for embeddings
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

app = FastAPI(title="Enhanced O-1A Visa Eligibility Evaluator")

# Define weights for different criteria (based on USCIS emphasis)
criteria_weights = {
    "awards": 1.5,
    "original_contributions": 1.5,
    "scholarly_articles": 1.2,
    "critical_employment": 1.2,
    "high_remuneration": 1.0,
    "press": 1.0,
    "judging": 0.8,
    "memberships": 0.8
}

# Define the response models
class EvidenceItem(BaseModel):
    text: str
    confidence: float
    section: str
    boosted: Optional[bool] = None
    uscis_reference: Optional[str] = None

class EnhancedEvaluationResponse(BaseModel):
    awards: List[EvidenceItem]
    memberships: List[EvidenceItem]
    press: List[EvidenceItem]
    judging: List[EvidenceItem]
    original_contributions: List[EvidenceItem]
    scholarly_articles: List[EvidenceItem]
    critical_employment: List[EvidenceItem]
    high_remuneration: List[EvidenceItem]
    overall_rating: str
    score: float
    criterion_scores: Dict[str, float]
    field: Optional[str] = None
    explanations: Dict[str, str]

# Load and index USCIS policy information
uscis_policy_text = """
In order to qualify as an alien of "extraordinary ability" in the sciences, education, business, or athletics (commonly referred to as O-1A), or in arts (commonly referred to as O-1B (Arts)), a beneficiary must have "sustained national or international acclaim." With regard to classification to work in motion picture and television productions (commonly referred to as O-1B (MPTV)), a beneficiary must have a demonstrated record of extraordinary achievement. In all cases, an O-1 beneficiary's achievements must have been recognized in the field through extensive documentation.

The regulations define "extraordinary ability" as applied to the O-1 classification as follows:
* In the field of science, education, business, or athletics: a level of expertise indicating that the person is one of the small percentage who have arisen to the very top of the field of endeavor.
* In the field of arts: distinction, defined as a high level of achievement in the field of arts, as evidenced by a degree of skill and recognition substantially above that ordinarily encountered to the extent that a person described as prominent is renowned, leading, or well-known in the field of arts.

The supporting documentation for an O-1A petition must include evidence that the beneficiary has received a major internationally recognized award (such as the Nobel Prize) or satisfies at least three of the following evidentiary criteria:

Criterion 1: Documentation of the beneficiary's receipt of nationally or internationally recognized prizes or awards for excellence in the field of endeavor.

Criterion 2: Documentation of the beneficiary's membership in associations in the field for which classification is sought, which require outstanding achievements of their members, as judged by recognized national or international experts in their disciplines or fields.

Criterion 3: Published material in professional or major trade publications or major media about the beneficiary, relating to the beneficiary's work in the field for which classification is sought. This evidence must include the title, date, and author of such published material and any necessary translation.

Criterion 4: Evidence of the beneficiary's participation on a panel, or individually, as a judge of the work of others in the same or in an allied field of specialization for which classification is sought.

Criterion 5: Evidence of the beneficiary's original scientific, scholarly, or business-related contributions of major significance in the field.

Criterion 6: Evidence of the beneficiary's authorship of scholarly articles in the field, in professional or major trade publications or other major media.

Criterion 7: Evidence that the beneficiary has been employed in a critical or essential capacity for organizations and establishments that have a distinguished reputation.

Criterion 8: Evidence that the beneficiary has commanded a high salary or other significantly high remuneration for services, in relation to others in the field.

If the criteria are not readily applicable to the beneficiary's occupation, the petitioner may submit comparable evidence to establish the beneficiary's eligibility.
"""

def process_uscis_policy():
    """"Process USCIS policy text into chunks"""
    chunks = []
    sentences = sent_tokenize(uscis_policy_text)
    
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < 512:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def create_uscis_index():
    """"Create embeddings and index for USCIS policy chunks"""
    chunks = process_uscis_policy()
    embeddings = embedding_model.encode(chunks)
    
    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    
    return index, chunks

# Initialize USCIS policy index
uscis_index, uscis_chunks = create_uscis_index()

def retrieve_uscis_info(query, top_k=3):
    """Retrieve relevant USCIS policy information"""
    query_embedding = embedding_model.encode([query])
    distances, indices = uscis_index.search(np.array(query_embedding).astype('float32'), top_k)
    
    results = []
    for idx in indices[0]:
        if idx < len(uscis_chunks):
            results.append(uscis_chunks[idx])
    
    return results

def extract_text_from_pdf(file_content):
    pdf_reader = PdfReader(io.BytesIO(file_content))
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def improved_text_segmentation(text: str) -> List[str]:
    """Segmentation for CV/resume text using an LLM to handle various formats."""

    import openai
    
    # Check if API key is set
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    # Create a client with the API key
    client = openai.OpenAI(api_key=openai_api_key)
    
    # Create a prompt for the LLM
    prompt = f"""
    I have a CV/resume text that I need to segment into meaningful chunks for analysis.
    Please extract each section, bullet point, and important sentence as separate items.
    For each item, include the section name as context (e.g., "Education: Bachelor's degree in Computer Science").
    
    Format your response as a JSON array of strings, where each string is a segment.
    
    Here's the CV/resume text:
    
    {text}
    """
    
    # Call the OpenAI API with the new client format
    response = client.chat.completions.create(
        model="gpt-4o",  # Use GPT-4o for better understanding of document structure
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # Low temperature for more consistent output
        response_format={"type": "json_object"}  # Request JSON format
    )
    
    # Extract the segments from the response
    result = json.loads(response.choices[0].message.content)
    segments = result.get("segments", [])
    
    # If we got valid segments, return them
    if segments and isinstance(segments, list) and len(segments) > 0:
        return segments
    else:
        raise ValueError("Failed to segment CV text - LLM segmentation failed")

def detect_field_from_evidence(evidence: Dict[str, List[Dict]]) -> str:
    """
    Detect the applicant's field based on evidence content.
    """
    # Combine all evidence text
    all_text = ""
    for criterion, items in evidence.items():
        for item in items:
            all_text += item["text"] + " "
    
    # Define field keywords
    field_keywords = {
        "Computer Science": ["algorithm", "software", "programming", "computer", "data", "AI", "machine learning"],
        "Engineering": ["engineering", "mechanical", "electrical", "civil", "structural", "design"],
        "Medicine": ["medical", "doctor", "physician", "clinical", "patient", "healthcare", "treatment"],
        "Business": ["business", "entrepreneur", "startup", "CEO", "executive", "management", "finance"],
        "Arts": ["artist", "creative", "design", "exhibition", "gallery", "performance"],
        "Sciences": ["research", "scientific", "laboratory", "experiment", "discovery", "innovation"],
        "Education": ["professor", "teaching", "education", "academic", "university", "college", "school"],
        "Athletics": ["athlete", "sports", "olympic", "championship", "competition", "tournament"]
    }
    
    # Count occurrences of field keywords
    field_scores = {}
    for field, keywords in field_keywords.items():
        score = 0
        for keyword in keywords:
            score += len(re.findall(r'\b' + re.escape(keyword) + r'\b', all_text.lower()))
        field_scores[field] = score
    
    # Return the field with the highest score
    if not field_scores or max(field_scores.values()) == 0:
        return "Unspecified Field"
    
    return max(field_scores.items(), key=lambda x: x[1])[0]

def apply_domain_knowledge(evidence: Dict[str, List[Dict]], field: str = None) -> Dict[str, List[Dict]]:
    """
    Apply domain-specific knowledge to refine evidence classification.
    """
    # Detect the applicant's field if not provided
    if not field:
        field = detect_field_from_evidence(evidence)
    
    # Field-specific keywords and phrases
    field_keywords = {
        "Computer Science": {
            "awards": ["ACM", "IEEE", "Turing", "SIGKDD", "best paper", "outstanding contribution"],
            "original_contributions": ["algorithm", "framework", "system", "novel approach", "innovation"],
            "scholarly_articles": ["conference", "journal", "proceedings", "publication", "citation"],
            "judging": ["program committee", "reviewer", "editorial board", "judge", "evaluator"],
            "memberships": ["ACM", "IEEE", "fellow", "senior member"]
        },
        "Engineering": {
            "awards": ["IEEE", "ASME", "outstanding", "innovation", "design", "excellence"],
            "original_contributions": ["patent", "design", "invention", "system", "method", "process"],
            "scholarly_articles": ["journal", "transactions", "proceedings", "publication"],
            "judging": ["review", "committee", "board", "panel", "examiner"],
            "memberships": ["IEEE", "ASME", "society", "institution", "fellow"]
        }
    }
    
    # Default keywords for any field
    default_keywords = {
        "awards": ["award", "prize", "recognition", "honor", "medal", "certificate", "grant"],
        "memberships": ["member", "fellow", "association", "society", "organization", "committee"],
        "press": ["featured", "article", "interview", "media", "publication", "news", "press"],
        "judging": ["judge", "reviewer", "evaluator", "examiner", "panel", "committee", "board"],
        "original_contributions": ["contribution", "innovation", "development", "creation", "discovery", "impact"],
        "scholarly_articles": ["author", "publication", "journal", "article", "paper", "research", "study"],
        "critical_employment": ["lead", "chief", "head", "director", "manager", "key role", "essential"],
        "high_remuneration": ["salary", "compensation", "remuneration", "income", "earnings", "paid", "contract"]
    }
    
    # Get keywords for the detected field or use default
    keywords = field_keywords.get(field, default_keywords)
    
    # Fill in any missing criteria with default keywords
    for criterion in default_keywords:
        if criterion not in keywords:
            keywords[criterion] = default_keywords[criterion]
    
    # Boost confidence for items containing field-specific keywords
    for criterion, items in evidence.items():
        criterion_keywords = keywords.get(criterion, [])
        for item in items:
            # Check if any keyword is in the text
            if any(keyword.lower() in item["text"].lower() for keyword in criterion_keywords):
                # Boost confidence but cap at 0.95
                item["confidence"] = min(item["confidence"] * 1.2, 0.95)
                item["boosted"] = True
    
    return evidence

def enhanced_extract_evidence(text: str, threshold: float = 0.5) -> Dict[str, List[Dict]]:
    """
    Enhanced evidence extraction with context awareness and confidence scores.
    """
    # Define candidate labels with more detailed descriptions
    candidate_labels = [
        "awards and prizes for excellence in the field",
        "membership in prestigious professional associations",
        "significant media coverage or press about the individual",
        "experience judging the work of others in the field",
        "original scientific, scholarly, or business contributions of major significance",
        "authorship of scholarly articles in professional publications",
        "employment in a critical or essential capacity for distinguished organizations",
        "high salary or remuneration compared to others in the field"
    ]
    
    # Map labels to criteria keys
    label_to_key = {
        "awards and prizes for excellence in the field": "awards",
        "membership in prestigious professional associations": "memberships",
        "significant media coverage or press about the individual": "press",
        "experience judging the work of others in the field": "judging",
        "original scientific, scholarly, or business contributions of major significance": "original_contributions",
        "authorship of scholarly articles in professional publications": "scholarly_articles",
        "employment in a critical or essential capacity for distinguished organizations": "critical_employment",
        "high salary or remuneration compared to others in the field": "high_remuneration"
    }
    
    # Initialize evidence dictionary
    evidence = {
        "awards": [],
        "memberships": [],
        "press": [],
        "judging": [],
        "original_contributions": [],
        "scholarly_articles": [],
        "critical_employment": [],
        "high_remuneration": []
    }
    
    # Segment the text
    segments = improved_text_segmentation(text)
    
    # Process each segment
    for segment in segments:
        # Skip very short segments
        if len(segment.split()) < 5:
            continue
        
        # Classify the segment
        result = classifier(segment, candidate_labels, multi_label=True)
        
        # Extract section if available
        section = "General"
        if ":" in segment:
            section, content = segment.split(":", 1)
            section = section.strip()
        
        # Process each label with confidence above threshold
        for label, score in zip(result["labels"], result["scores"]):
            if score > threshold:
                key = label_to_key[label]
                
                # Retrieve relevant USCIS policy information
                uscis_info = retrieve_uscis_info(f"{key} for O-1A visa", top_k=1)
                uscis_reference = uscis_info[0] if uscis_info else None
                
                # Create evidence item
                evidence_item = {
                    "text": segment,
                    "confidence": score,
                    "section": section,
                    "uscis_reference": uscis_reference
                }
                
                # Avoid duplicates
                if not any(e["text"] == segment for e in evidence[key]):
                    evidence[key].append(evidence_item)
    
    # Sort evidence by confidence score
    for key in evidence:
        evidence[key] = sorted(evidence[key], key=lambda x: x["confidence"], reverse=True)
    
    return evidence

def compute_weighted_rating(evidence: Dict[str, List[Dict]]) -> Dict:
    """
    Compute a more nuanced rating that considers evidence quality and quantity.
    """
    # Initialize criterion scores
    criterion_scores = {criterion: 0.0 for criterion in criteria_weights}
    
    # Calculate score for each criterion
    for criterion, items in evidence.items():
        # Skip if no evidence
        if not items:
            continue
        
        # Base score on number of items and their confidence
        base_score = min(len(items), 3) * 0.5  # Cap at 3 items
        
        # Add confidence bonus from top items
        confidence_bonus = sum(item["confidence"] for item in items[:3]) / 3 if items else 0
        
        # Calculate criterion score
        criterion_scores[criterion] = base_score + confidence_bonus
        
        # Apply weight
        criterion_scores[criterion] *= criteria_weights.get(criterion, 1.0)
    
    # Calculate total weighted score
    total_weighted_score = sum(criterion_scores.values())
    
    # Calculate maximum possible score (if all criteria had max score)
    max_possible_score = sum(3.0 * weight for weight in criteria_weights.values())
    
    # Normalize score to 0-10 scale
    normalized_score = (total_weighted_score / max_possible_score) * 10
    
    # Determine rating based on normalized score
    if normalized_score >= 8:
        rating = "Excellent"
    elif normalized_score >= 6:
        rating = "Very Good"
    elif normalized_score >= 4:
        rating = "Good"
    elif normalized_score >= 2:
        rating = "Medium"
    elif normalized_score >= 1:
        rating = "Low"
    else:
        rating = "Very Low"
    
    return {
        "overall_rating": rating,
        "normalized_score": round(normalized_score, 1),
        "criterion_scores": {k: round(v, 2) for k, v in criterion_scores.items()}
    }

def generate_explanations(evidence: Dict[str, List[Dict]], rating_data: Dict) -> Dict[str, str]:
    """
    Generate human-readable explanations for the evaluation results.
    """
    explanations = {}
    
    # Overall assessment
    total_evidence = sum(len(items) for items in evidence.values())
    criteria_with_evidence = sum(1 for items in evidence.values() if items)
    
    if total_evidence == 0:
        explanations["overall"] = "No clear evidence was found to support O-1A eligibility criteria."
    else:
        if rating_data["overall_rating"] in ["Excellent", "Very Good"]:
            explanations["overall"] = f"Strong evidence found for {criteria_with_evidence} out of 8 criteria. The application demonstrates substantial achievements that align well with O-1A visa requirements."
        elif rating_data["overall_rating"] in ["Good", "Medium"]:
            explanations["overall"] = f"Moderate evidence found for {criteria_with_evidence} out of 8 criteria. The application shows some achievements that may support O-1A eligibility, but could benefit from strengthening in certain areas."
        else:
            explanations["overall"] = f"Limited evidence found for only {criteria_with_evidence} out of 8 criteria. The application may need significant strengthening."
    
    # Explanations for each criterion
    for criterion, items in evidence.items():
        if not items:
            explanations[criterion] = f"No clear evidence found for the {criterion.replace('_', ' ')} criterion."
        else:
            top_item = items[0]
            uscis_reference = top_item.get("uscis_reference", "")
            
            if len(items) == 1:
                explanations[criterion] = f"Found 1 piece of evidence for the {criterion.replace('_', ' ')} criterion: \"{top_item['text']}\" (confidence: {top_item['confidence']:.2f})."
            else:
                explanations[criterion] = f"Found {len(items)} potential pieces of evidence for the {criterion.replace('_', ' ')} criterion. Most confident match: \"{top_item['text']}\" (confidence: {top_item['confidence']:.2f})."
            
            if uscis_reference:
                explanations[criterion] += f" According to USCIS policy: \"{uscis_reference}\""
    
    return explanations

@app.post("/evaluate", response_model=EnhancedEvaluationResponse)
async def evaluate_cv(
    file: UploadFile = File(...),
    field: Optional[str] = Form(None)
):
    """
    Evaluate a CV/resume for O-1A visa eligibility.
    """
    # Read file content
    file_content = await file.read()
    
    # Extract text based on file type
    file_extension = file.filename.split(".")[-1].lower()
    
    try:
        if file_extension == "pdf":
            text = extract_text_from_pdf(file_content)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {file_extension}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting text: {str(e)}")
    
    # Extract evidence
    evidence = enhanced_extract_evidence(text)
    
    # Apply domain knowledge
    evidence = apply_domain_knowledge(evidence, field)
    
    # Compute the weighted rating
    rating_data = compute_weighted_rating(evidence)
    
    # Generate explanations
    explanations = generate_explanations(evidence, rating_data)
    
    # Detect field if not provided
    detected_field = field or detect_field_from_evidence(evidence)
    
    # Prepare the response
    return EnhancedEvaluationResponse(
        awards=evidence["awards"],
        memberships=evidence["memberships"],
        press=evidence["press"],
        judging=evidence["judging"],
        original_contributions=evidence["original_contributions"],
        scholarly_articles=evidence["scholarly_articles"],
        critical_employment=evidence["critical_employment"],
        high_remuneration=evidence["high_remuneration"],
        overall_rating=rating_data["overall_rating"],
        score=rating_data["normalized_score"],
        criterion_scores=rating_data["criterion_scores"],
        field=detected_field,
        explanations=explanations
    )

@app.get("/criteria-info/{criterion}")
async def get_criterion_info(criterion: str):
    """
    Get detailed information about a specific O-1A criterion from USCIS policy.
    """
    if criterion not in criteria_weights:
        raise HTTPException(status_code=404, detail=f"Criterion '{criterion}' not found")
    
    # Retrieve USCIS policy information for the criterion
    uscis_info = retrieve_uscis_info(f"{criterion} for O-1A visa", top_k=3)
    
    return {
        "criterion": criterion,
        "weight": criteria_weights[criterion],
        "uscis_policy": uscis_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
    # Run with: python main.py
    # Then visit http://127.0.0.1:8000/docs for the interactive API documentation