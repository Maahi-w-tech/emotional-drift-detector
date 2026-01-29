from transformers import pipeline
import ruptures as rpt
import numpy as np
from utils import preprocess_and_chunk, generate_explanation

# Global variables to cache models
_emotion_classifier = None
_nli_model = None
_regenerator = None

def get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        _emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return _emotion_classifier

def get_nli_model():
    global _nli_model
    if _nli_model is None:
        _nli_model = pipeline("text-classification", model="roberta-large-mnli")
    return _nli_model

def get_regenerator():
    global _regenerator
    if _regenerator is None:
        _regenerator = pipeline(model="google/flan-t5-base")
    return _regenerator

def regenerate_text(text, target_emotion):
    if not target_emotion or target_emotion.startswith("None"):
        return None
    
    regenerator = get_regenerator()
    prompt = f"Rewrite the following text to have a {target_emotion} tone: {text}"
    try:
        result = regenerator(prompt, max_length=len(text.split()) * 2 + 50, do_sample=True)[0]['generated_text']
        return result
    except Exception as e:
        print(f"Error in text regeneration: {str(e)}")
        return None

# Map emotions to our labels (approximate for demo)
emotion_map = {
    "joy": "Inspirational", "optimism": "Inspirational", "anger": "Aggressive", "disappointment": "Defensive",
    "sadness": "Empathetic", "fear": "Defensive", "surprise": "Neutral", "love": "Empathetic",
    "admiration": "Inspirational", "gratitude": "Empathetic", "annoyance": "Assertive", "disapproval": "Defensive",
    "neutral": "Neutral", "curiosity": "Informative", "confusion": "Neutral"  # Add more as needed
}

def classify_emotions(chunks):
    emotion_vectors = []
    emotion_dicts = []
    emotion_classifier = get_emotion_classifier()
    
    for idx, chunk in enumerate(chunks):
        try:
            raw_result = emotion_classifier(chunk)
            
            # Handle different return formats: 
            # - List of lists of dicts: [[{'label': '...', 'score': ...}, ...]] (standard with top_k=None)
            # - List of dicts: [{'label': '...', 'score': ...}, ...] (some versions or one chunk)
            if isinstance(raw_result, list) and len(raw_result) > 0:
                if isinstance(raw_result[0], list):
                    scores = raw_result[0]
                else:
                    scores = raw_result
            else:
                print(f"ERROR: Unexpected raw result format {type(raw_result)}: {raw_result}")
                scores = []
            
            mapped_scores = {}
            for s in scores:
                if isinstance(s, dict) and 'label' in s and 'score' in s:
                    emotion_label = emotion_map.get(s['label'], 'Neutral')
                    mapped_scores[emotion_label] = s['score']
                else:
                    print(f"ERROR: Invalid score format: {s}")
            
            # Ensure all labels are present
            full_scores = {label: mapped_scores.get(label, 0.0) for label in ["Inspirational", "Informative", "Neutral", "Empathetic", "Assertive", "Aggressive", "Defensive"]}
            emotion_vectors.append(list(full_scores.values()))
            emotion_dicts.append(full_scores)
            
        except Exception as e:
            print(f"ERROR in chunk {idx}: {str(e)}")
            print(f"Chunk text: {chunk[:100]}...")
            import traceback
            traceback.print_exc()
            raise
    
    return emotion_vectors, emotion_dicts

def detect_drift(emotion_vectors):
    # Use change-point detection
    signal = np.array(emotion_vectors)
    algo = rpt.Pelt(model="rbf").fit(signal)
    change_points = algo.predict(pen=10)  # Penalty for fewer points
    drifts = []
    for i in range(1, len(change_points)):
        start = change_points[i-1]
        end = change_points[i]
        if end < len(emotion_vectors):
            drifts.append((start, end))
    return drifts

def detect_contradictions(chunks):
    contradictions = []
    nli_model = get_nli_model()
    for i in range(len(chunks)):
        for j in range(i+1, len(chunks)):
            premise = chunks[i]
            hypothesis = chunks[j]
            try:
                result = nli_model(f"{premise} [SEP] {hypothesis}")
                
                # Handle different output formats
                if isinstance(result, list) and len(result) > 0:
                    first_result = result[0]
                    if isinstance(first_result, dict) and 'label' in first_result and 'score' in first_result:
                        if first_result['label'] == 'CONTRADICTION' and first_result['score'] > 0.5:
                            contradictions.append((i, j, f"{premise[:50]}... contradicts {hypothesis[:50]}..."))
                    else:
                        print(f"Unexpected NLI result format: {first_result}")
            except Exception as e:
                print(f"Error in contradiction detection between chunks {i} and {j}: {str(e)}")
                continue
    return contradictions
    return contradictions

def detect_confusion(emotion_vectors, threshold=0.8):
    confusions = []
    for idx, vec in enumerate(emotion_vectors):
        entropy = -sum(p * np.log(p + 1e-9) for p in vec if p > 0)  # Shannon entropy
        if entropy > threshold:
            confusions.append(idx)
    return confusions

def run_pipeline(text, target_emotion=None):
    chunks = preprocess_and_chunk(text)
    emotion_vectors, emotion_dicts = classify_emotions(chunks)
    drifts = detect_drift(emotion_vectors)
    contradictions = detect_contradictions(chunks)
    confusions = detect_confusion(emotion_vectors)
    
    # Generate explanations
    explanations = {}
    
    for start, end in drifts:
        explanations[f"drift_{start}_{end}"] = generate_explanation("drift", start, emotion_dicts[start], emotion_dicts[end])
             
    for i, j, details in contradictions:
        explanations[f"contradict_{i}_{j}"] = generate_explanation("contradiction", i, {}, {}, details)
        
    for idx in confusions:
        explanations[f"confusion_{idx}"] = generate_explanation("confusion", idx, {}, {})
            
    return chunks, emotion_vectors, drifts, contradictions, confusions, explanations