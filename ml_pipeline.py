from transformers import pipeline
import ruptures as rpt
import numpy as np
from utils import preprocess_and_chunk, generate_explanation

# Global variables to cache models
_emotion_classifier = None
_regenerator = None

def get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        _emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return _emotion_classifier

def get_regenerator():
    global _regenerator
    if _regenerator is None:
        # Using T5-small for speed and lower memory usage
        _regenerator = pipeline(model="google/flan-t5-small")
    return _regenerator

def regenerate_text(text, target_emotion):
    if not target_emotion or target_emotion.startswith("None"):
        return None
    
    regenerator = get_regenerator()
    prompt = f"Rewrite the following text to have a {target_emotion} tone: {text}"
    try:
        # Limit max_length to prevent AI load
        result = regenerator(prompt, max_length=150, do_sample=True, temperature=0.7)[0]['generated_text']
        return result
    except Exception as e:
        print(f"Error in text regeneration: {str(e)}")
        return None

# Map Hartmann's 7 emotions to user labels
emotion_map = {
    "joy": "Inspirational", 
    "anger": "Aggressive", 
    "disgust": "Aggressive",
    "sadness": "Empathetic", 
    "fear": "Defensive", 
    "surprise": "Neutral", 
    "neutral": "Neutral"
}

def classify_emotions(chunks):
    emotion_vectors = []
    emotion_dicts = []
    emotion_classifier = get_emotion_classifier()
    
    # Define our labels
    labels = ["Inspirational", "Informative", "Neutral", "Empathetic", "Assertive", "Aggressive", "Defensive"]

    for idx, chunk in enumerate(chunks):
        try:
            raw_result = emotion_classifier(chunk)
            
            # Extract scores
            if isinstance(raw_result, list) and len(raw_result) > 0:
                scores = raw_result[0] if isinstance(raw_result[0], list) else raw_result
            else:
                scores = []
            
            # Map Hartmann results to our labels
            mapped_scores = {label: 0.0 for label in labels}
            
            for s in scores:
                if isinstance(s, dict) and 'label' in s and 'score' in s:
                    label = emotion_map.get(s['label'], 'Neutral')
                    mapped_scores[label] += s['score']
            
            # Bias Correction: If Neutral is dominant but low intensity
            if mapped_scores["Neutral"] > 0.4 and mapped_scores["Neutral"] < 0.8:
                other_max = max([v for k, v in mapped_scores.items() if k != "Neutral"])
                if other_max > 0.1:
                    # Slightly boost the more "active" emotions to reduce neutral bias
                    mapped_scores["Neutral"] *= 0.8
            
            emotion_vectors.append(list(mapped_scores.values()))
            emotion_dicts.append(mapped_scores)
            
        except Exception as e:
            print(f"ERROR in chunk {idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    return emotion_vectors, emotion_dicts

def detect_drift(emotion_vectors):
    # Use change-point detection
    signal = np.array(emotion_vectors)
    if len(signal) < 2:
        return []
    try:
        algo = rpt.Pelt(model="rbf").fit(signal)
        change_points = algo.predict(pen=10)  # Penalty for fewer points
        drifts = []
        for i in range(1, len(change_points)):
            start = change_points[i-1]
            end = change_points[i]
            if end < len(emotion_vectors):
                drifts.append((start, end))
        return drifts
    except:
        return []

def detect_confusion(emotion_vectors, threshold=0.75):
    confusions = []
    for idx, vec in enumerate(emotion_vectors):
        p = np.array(vec)
        p = p / (p.sum() + 1e-9)
        entropy = -sum(x * np.log(x + 1e-9) for x in p if x > 0)
        if entropy > threshold:
            confusions.append(idx)
    return confusions

def run_pipeline(text, target_emotion=None):
    chunks = preprocess_and_chunk(text)
    emotion_vectors, emotion_dicts = classify_emotions(chunks)
    drifts = detect_drift(emotion_vectors)
    confusions = detect_confusion(emotion_vectors)
    
    # Generate explanations
    explanations = {}
    for start, end in drifts:
        explanations[f"drift_{start}_{end}"] = generate_explanation("drift", start, emotion_dicts[start], emotion_dicts[end])
    for idx in confusions:
        explanations[f"confusion_{idx}"] = generate_explanation("confusion", idx, {}, {})
            
    return chunks, emotion_vectors, drifts, confusions, explanations