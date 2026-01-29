import streamlit as st
import plotly.graph_objects as go
from ml_pipeline import run_pipeline, regenerate_text
import os

# Page Configuration
st.set_page_config(
    page_title="Emotional Drift Detector",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load Custom CSS
def load_css():
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# Session state for page navigation
if 'page' not in st.session_state:
    st.session_state.page = 'landing'

if 'recommendations' not in st.session_state:
    st.session_state.recommendations = {}

def show_landing_page():
    """Display the landing page with marketing content"""
    
    # Hero Section
    st.markdown("""
        <div style='text-align: center; padding: 3rem 0 2rem 0;'>
            <h1 style='font-size: 4rem; margin-bottom: 1rem;'>🌊 Emotional Drift Detector</h1>
            <p style='font-size: 1.8rem; color: #4A3267; font-weight: 500; margin-bottom: 0.5rem;'>
                See how your content <em>feels</em> — not just what it says
            </p>
            <p style='font-size: 1.2rem; color: #4A3267; opacity: 0.8; max-width: 900px; margin: 0 auto; line-height: 1.8;'>
                Emotional Drift Detector is an AI-powered tool that analyzes long-form content to detect emotional tone shifts, 
                messaging contradictions, and audience confusion points — <strong>before your audience feels them</strong>.
            </p>
            <p style='font-size: 1.1rem; color: #4A3267; opacity: 0.8; max-width: 900px; margin: 2rem auto 3rem auto; line-height: 1.8;'>
                Whether you're a creator, brand, educator, or organization, our AI helps you maintain emotional consistency, 
                clarity, and trust across your content.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # CTA Button
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("🚀 Try It Now", key="cta_button", use_container_width=True):
            st.session_state.page = 'analyzer'
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Problem Section
    st.markdown("""
        <div style='background: rgba(222, 99, 138, 0.15); padding: 2.5rem; border-radius: 20px; border-left: 5px solid #DE638A; margin: 3rem 0;'>
            <h2 style='color: #DE638A; font-size: 2.2rem; margin-bottom: 1.5rem;'>🚨 The Problem</h2>
            <p style='font-size: 1.15rem; color: #4A3267; line-height: 1.8; margin-bottom: 1rem;'>
                Most content tools focus on grammar, SEO, or sentiment averages.
            </p>
            <p style='font-size: 1.15rem; color: #4A3267; line-height: 1.8; margin-bottom: 1rem;'>
                <strong>But real engagement drops when content:</strong>
            </p>
            <ul style='font-size: 1.1rem; color: #4A3267; line-height: 2; margin-left: 2rem;'>
                <li>✗ unintentionally shifts tone</li>
                <li>✗ contradicts its own message</li>
                <li>✗ confuses or alienates readers</li>
            </ul>
            <p style='font-size: 1.15rem; color: #4A3267; line-height: 1.8; margin-top: 1rem; font-weight: 600;'>
                These issues are invisible until it's too late.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Solution Section
    st.markdown("""
        <div style='background: rgba(198, 186, 222, 0.2); padding: 2.5rem; border-radius: 20px; border-left: 5px solid #C6BADE; margin: 3rem 0;'>
            <h2 style='color: #4A3267; font-size: 2.2rem; margin-bottom: 1.5rem;'>✨ Our Solution</h2>
            <p style='font-size: 1.15rem; color: #4A3267; line-height: 1.8; margin-bottom: 1rem;'>
                Emotional Drift Detector goes beyond basic sentiment analysis.
            </p>
            <p style='font-size: 1.15rem; color: #4A3267; line-height: 1.8; margin-bottom: 1rem;'>
                <strong>It tracks emotional flow over time, revealing where your content:</strong>
            </p>
            <ul style='font-size: 1.1rem; color: #4A3267; line-height: 2; margin-left: 2rem;'>
                <li>✓ drifts from inspirational to aggressive</li>
                <li>✓ loses emotional coherence</li>
                <li>✓ sends mixed or conflicting signals</li>
            </ul>
            <p style='font-size: 1.15rem; color: #4A3267; line-height: 1.8; margin-top: 1rem; font-weight: 600;'>
                You don't just get alerts — you get clear explanations and actionable guidance.
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Features Grid
    st.markdown("<h2 style='text-align: center; color: #4A3267; font-size: 2.5rem; margin: 4rem 0 2rem 0;'>🧠 Why It's Different</h2>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.8); padding: 2rem; border-radius: 15px; height: 100%; border: 2px solid #C6BADE;'>
                <h3 style='color: #DE638A; margin-bottom: 1rem;'>📊 Tracks emotional consistency</h3>
                <p style='color: #4A3267; line-height: 1.8;'>Not just sentiment — we analyze how emotions evolve throughout your content</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.8); padding: 2rem; border-radius: 15px; height: 100%; border: 2px solid #C6BADE;'>
                <h3 style='color: #DE638A; margin-bottom: 1rem;'>🔍 Explainable AI</h3>
                <p style='color: #4A3267; line-height: 1.8;'>No black-box decisions — understand exactly why issues were flagged</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.8); padding: 2rem; border-radius: 15px; height: 100%; border: 2px solid #C6BADE;'>
                <h3 style='color: #DE638A; margin-bottom: 1rem;'>📝 Designed for long-form content</h3>
                <p style='color: #4A3267; line-height: 1.8;'>Perfect for blogs, articles, transcripts, and course materials</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("""
            <div style='background: rgba(255, 255, 255, 0.8); padding: 2rem; border-radius: 15px; height: 100%; border: 2px solid #C6BADE;'>
                <h3 style='color: #DE638A; margin-bottom: 1rem;'>🏢 Built for creators and enterprises</h3>
                <p style='color: #4A3267; line-height: 1.8;'>Scales from individual creators to large organizations</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Target Audience Section
    st.markdown("<h2 style='text-align: center; color: #4A3267; font-size: 2.5rem; margin: 4rem 0 2rem 0;'>🎯 Who It's For</h2>", unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    audiences = [
        ("👨‍💻", "Content creators & influencers"),
        ("🎨", "Brands & marketing teams"),
        ("📚", "Educators & course creators"),
        ("🎤", "Public speakers & podcasters"),
        ("🏛️", "Organizations managing public trust")
    ]
    
    for col, (icon, text) in zip([col1, col2, col3, col4, col5], audiences):
        with col:
            st.markdown(f"""
                <div style='background: rgba(247, 185, 196, 0.3); padding: 1.5rem 1rem; border-radius: 15px; text-align: center; height: 150px; display: flex; flex-direction: column; justify-content: center; border: 2px solid #F7B9C4;'>
                    <div style='font-size: 2.5rem; margin-bottom: 0.5rem;'>{icon}</div>
                    <p style='color: #4A3267; font-size: 0.9rem; font-weight: 500; margin: 0;'>{text}</p>
                </div>
            """, unsafe_allow_html=True)
    
    # Final CTA
    st.markdown("<br><br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2, 2, 2])
    with col2:
        if st.button("🚀 Start Analyzing Now", key="cta_button_bottom", use_container_width=True):
            st.session_state.page = 'analyzer'
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #4A3267; opacity: 0.7;'>
            <p>Powered by AI • Built with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

def show_analyzer_page():
    """Display the analyzer page"""
    
    # Back button
    if st.button("← Back to Home", key="back_button"):
        st.session_state.page = 'landing'
        st.rerun()
    
    # Header Section
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1>🎭 Emotional Drift Detector</h1>
            <p style='font-size: 1.2rem; color: #4A3267; opacity: 0.9;'>
                Analyze emotional tone drift, contradictions, and confusion in long-form content with AI-powered insights
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Main Content
    col1, col2, col3 = st.columns([1, 6, 1])

    with col2:
        # Target Emotion Section (Optional)
        st.markdown("### 🎯 Target Emotion (Optional)")
        st.markdown("*Select the emotional tone you're aiming for in your content*")
        
        target_emotion = st.selectbox(
            "Target Emotional Tone",
            ["None - Just analyze my content", "Inspirational", "Informative", "Neutral", "Empathetic", "Assertive", "Aggressive", "Defensive"],
            label_visibility="collapsed"
        )
        
        # Input Section
        st.markdown("### 📝 Input Your Text")
        text_input = st.text_area(
            "Content to Analyze",
            height=250,
            placeholder="Paste your long-form text here (blog post, transcript, article, etc.)...",
            label_visibility="collapsed"
        )
        
        # Analyze Button
        col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 2])
        with col_btn2:
            analyze_button = st.button("🔍 Analyze Content", use_container_width=True)

        if analyze_button:
            if text_input and len(text_input.strip()) > 50:
                # Labels for emotions
                labels = ["Inspirational", "Informative", "Neutral", "Empathetic", "Assertive", "Aggressive", "Defensive"]
                
                # Show loading message
                loading_placeholder = st.empty()
                
                with st.spinner("🧠 Processing your content with AI models..."):
                    try:
                        chunks, emotion_vectors, drifts, contradictions, confusions, explanations = run_pipeline(text_input, target_emotion)
                        # Clear old recommendations on new analysis
                        st.session_state.recommendations = {}
                        
                        # Clear loading message
                        loading_placeholder.empty()
                        
                        
                        # Success Message
                        st.success("✅ Analysis Complete!")
                        
                        # Target Emotion Matching (if selected)
                        if target_emotion != "None - Just analyze my content":
                            st.markdown("---")
                            st.markdown("## 🎯 Target Emotion Match")
                            
                            # Calculate match with target emotion
                            avg_emotions = [sum(vec[i] for vec in emotion_vectors) / len(emotion_vectors) for i in range(len(labels))]
                            target_idx = labels.index(target_emotion)
                            target_score = avg_emotions[target_idx]
                            overall_dominant_idx = avg_emotions.index(max(avg_emotions))
                            overall_dominant = labels[overall_dominant_idx]
                            
                            # Calculate match percentage
                            match_percentage = target_score * 100
                            
                            # Determine if it's a good match
                            if match_percentage >= 60:
                                match_status = "success"
                                match_icon = "✅"
                                match_message = f"**Great match!** Your content strongly aligns with your target **{target_emotion}** tone ({match_percentage:.1f}% intensity)."
                            elif match_percentage >= 40:
                                match_status = "warning"
                                match_icon = "⚠️"
                                match_message = f"**Partial match.** Your content shows some **{target_emotion}** tone ({match_percentage:.1f}% intensity), but could be stronger."
                            else:
                                match_status = "error"
                                match_icon = "❌"
                                match_message = f"**Low match.** Your content is predominantly **{overall_dominant}** ({max(avg_emotions):.1%}), not **{target_emotion}** ({match_percentage:.1f}%)."
                            
                            # Display match result
                            if match_status == "success":
                                st.success(f"{match_icon} {match_message}")
                            elif match_status == "warning":
                                st.warning(f"{match_icon} {match_message}")
                            else:
                                st.error(f"{match_icon} {match_message}")
                            
                            # Show emotion breakdown
                            col_match1, col_match2 = st.columns(2)
                            with col_match1:
                                st.markdown(f"""
                                    <div style='background: rgba(155, 126, 189, 0.15); padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #9B7EBD;'>
                                        <h3 style='color: #4A3267; margin: 0;'>{match_percentage:.1f}%</h3>
                                        <p style='color: #4A3267; margin: 0.5rem 0 0 0;'>Target Match</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            with col_match2:
                                st.markdown(f"""
                                    <div style='background: rgba(155, 126, 189, 0.15); padding: 1.5rem; border-radius: 15px; text-align: center; border: 2px solid #9B7EBD;'>
                                        <h3 style='color: #4A3267; margin: 0;'>{overall_dominant}</h3>
                                        <p style='color: #4A3267; margin: 0.5rem 0 0 0;'>Actual Dominant Tone</p>
                                    </div>
                                """, unsafe_allow_html=True)
                            
                            # Provide specific suggestions to achieve target emotion
                            if match_percentage < 60:
                                st.markdown("### 💡 How to Achieve Your Target Tone")
                                
                                target_guidance = {
                                    "Inspirational": [
                                        "Use uplifting language and positive future-focused statements",
                                        "Include success stories, transformations, or aspirational examples",
                                        "Employ power words like 'achieve', 'transform', 'breakthrough', 'empower'",
                                        "Paint a vivid picture of possibilities and potential",
                                        "End sections with motivational calls-to-action"
                                    ],
                                    "Informative": [
                                        "Lead with facts, data, and concrete information",
                                        "Use clear, objective language without emotional embellishment",
                                        "Structure content with headings, lists, and logical flow",
                                        "Include specific examples, case studies, or research",
                                        "Define terms and explain concepts thoroughly"
                                    ],
                                    "Neutral": [
                                        "Use balanced, objective language",
                                        "Present multiple perspectives without bias",
                                        "Avoid emotionally charged words or phrases",
                                        "Focus on facts rather than opinions",
                                        "Maintain professional, measured tone throughout"
                                    ],
                                    "Empathetic": [
                                        "Acknowledge and validate feelings and experiences",
                                        "Use phrases like 'I understand', 'You're not alone', 'It's okay to feel...'",
                                        "Share relatable stories or vulnerable moments",
                                        "Use warm, compassionate language",
                                        "Offer support and understanding before solutions"
                                    ],
                                    "Assertive": [
                                        "Use confident, direct language",
                                        "Make clear statements without hedging (avoid 'maybe', 'perhaps')",
                                        "Back claims with evidence and reasoning",
                                        "Use active voice and strong verbs",
                                        "State your position clearly and stand by it"
                                    ],
                                    "Aggressive": [
                                        "Use strong, forceful language (use cautiously)",
                                        "Challenge opposing views directly",
                                        "Employ urgent, pressing tone",
                                        "Note: Consider if assertive tone would be more effective",
                                        "Be aware this may alienate some readers"
                                    ],
                                    "Defensive": [
                                        "Address concerns and objections proactively",
                                        "Provide justifications and explanations",
                                        "Use qualifying language when appropriate",
                                        "Note: Consider if confident assertive tone would be stronger",
                                        "Balance defense with positive positioning"
                                    ]
                                }
                                
                                if target_emotion in target_guidance:
                                    for tip in target_guidance[target_emotion]:
                                        st.markdown(f"• {tip}")
                        
                        # Results Section
                        st.markdown("---")
                        st.markdown("## 📊 Emotional Timeline")
                        
                        # Timeline Visualization
                        fig = go.Figure()
                        colors = ['#DE638A', '#4A3267', '#C6BADE', '#F7B9C4', '#F3D9E5', '#8B5A8E', '#6B4668']
                        
                        for i, label in enumerate(labels):
                            fig.add_trace(go.Scatter(
                                x=list(range(len(chunks))),
                                y=[vec[i] for vec in emotion_vectors],
                                mode='lines+markers',
                                name=label,
                                line=dict(width=3, color=colors[i % len(colors)]),
                                marker=dict(size=8)
                            ))
                        
                        fig.update_layout(
                            title={
                                'text': "Emotional Tone Across Content Segments",
                                'x': 0.5,
                                'xanchor': 'center',
                                'font': {'size': 20, 'color': '#4A3267', 'family': 'Inter'}
                            },
                            xaxis_title="Content Segment",
                            yaxis_title="Emotional Intensity",
                            hovermode='x unified',
                            plot_bgcolor='rgba(255, 255, 255, 0.5)',
                            paper_bgcolor='rgba(255, 255, 255, 0)',
                            font=dict(family='Inter', color='#4A3267'),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            ),
                            height=500
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistics
                        st.markdown("---")
                        st.markdown("## 📈 Analysis Summary")
                        
                        stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                        
                        with stat_col1:
                            st.markdown(f"""
                                <div style='background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); 
                                            padding: 1.5rem; border-radius: 15px; text-align: center; 
                                            border: 2px solid #C6BADE; box-shadow: 0 4px 6px rgba(74, 50, 103, 0.1);'>
                                    <h3 style='color: #DE638A; margin: 0;'>{len(chunks)}</h3>
                                    <p style='color: #4A3267; margin: 0.5rem 0 0 0;'>Segments</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col2:
                            st.markdown(f"""
                                <div style='background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); 
                                            padding: 1.5rem; border-radius: 15px; text-align: center; 
                                            border: 2px solid #C6BADE; box-shadow: 0 4px 6px rgba(74, 50, 103, 0.1);'>
                                    <h3 style='color: #DE638A; margin: 0;'>{len(drifts)}</h3>
                                    <p style='color: #4A3267; margin: 0.5rem 0 0 0;'>Tone Drifts</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col3:
                            st.markdown(f"""
                                <div style='background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); 
                                            padding: 1.5rem; border-radius: 15px; text-align: center; 
                                            border: 2px solid #C6BADE; box-shadow: 0 4px 6px rgba(74, 50, 103, 0.1);'>
                                    <h3 style='color: #DE638A; margin: 0;'>{len(contradictions)}</h3>
                                    <p style='color: #4A3267; margin: 0.5rem 0 0 0;'>Contradictions</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with stat_col4:
                            st.markdown(f"""
                                <div style='background: rgba(255, 255, 255, 0.7); backdrop-filter: blur(10px); 
                                            padding: 1.5rem; border-radius: 15px; text-align: center; 
                                            border: 2px solid #C6BADE; box-shadow: 0 4px 6px rgba(74, 50, 103, 0.1);'>
                                    <h3 style='color: #DE638A; margin: 0;'>{len(confusions)}</h3>
                                    <p style='color: #4A3267; margin: 0.5rem 0 0 0;'>Confusions</p>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Detailed Observations Section
                        st.markdown("---")
                        st.markdown("## 🔍 Detailed Observations")
                        
                        # Emotional Journey
                        st.markdown("### 🎭 Emotional Journey")
                        journey_text = ""
                        for idx, vec in enumerate(emotion_vectors):
                            dominant_idx = vec.index(max(vec))
                            dominant_emotion = labels[dominant_idx]
                            intensity = max(vec)
                            
                            emoji_map = {
                                "Inspirational": "✨",
                                "Informative": "📚",
                                "Neutral": "😐",
                                "Empathetic": "💙",
                                "Assertive": "💪",
                                "Aggressive": "⚡",
                                "Defensive": "🛡️"
                            }
                            
                            journey_text += f"**Segment {idx+1}**: {emoji_map.get(dominant_emotion, '•')} {dominant_emotion} ({intensity:.1%} intensity)\n\n"
                        
                        st.markdown(f"""
                            <div style='background: rgba(198, 186, 222, 0.15); padding: 1.5rem; border-radius: 15px; border-left: 4px solid #C6BADE;'>
                                {journey_text}
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Key Insights
                        st.markdown("### 💡 Key Insights")
                        
                        # Calculate overall emotional consistency
                        emotion_variances = []
                        for i in range(len(labels)):
                            values = [vec[i] for vec in emotion_vectors]
                            variance = sum((x - sum(values)/len(values))**2 for x in values) / len(values)
                            emotion_variances.append(variance)
                        
                        avg_variance = sum(emotion_variances) / len(emotion_variances)
                        consistency_score = max(0, 100 - (avg_variance * 1000))
                        
                        insights = []
                        
                        # Overall consistency
                        if consistency_score > 80:
                            insights.append("✅ **High Emotional Consistency**: Your content maintains a stable emotional tone throughout.")
                        elif consistency_score > 60:
                            insights.append("⚠️ **Moderate Emotional Consistency**: Some emotional fluctuations detected, but generally stable.")
                        else:
                            insights.append("❌ **Low Emotional Consistency**: Significant emotional shifts may confuse your audience.")
                        
                        # Dominant emotion overall
                        avg_emotions = [sum(vec[i] for vec in emotion_vectors) / len(emotion_vectors) for i in range(len(labels))]
                        overall_dominant = labels[avg_emotions.index(max(avg_emotions))]
                        insights.append(f"🎯 **Overall Tone**: Your content is predominantly **{overall_dominant}** ({max(avg_emotions):.1%} average intensity)")
                        
                        # Drift analysis
                        if len(drifts) > 0:
                            insights.append(f"🌊 **Tone Shifts Detected**: {len(drifts)} significant emotional drift(s) found - review these sections for consistency")
                        else:
                            insights.append("✨ **No Major Drifts**: Your emotional tone flows smoothly from start to finish")
                        
                        # Contradiction analysis
                        if len(contradictions) > 0:
                            insights.append(f"⚠️ **Contradictions Found**: {len(contradictions)} contradictory statement(s) detected - these may confuse readers")
                        
                        # Confusion analysis
                        if len(confusions) > 0:
                            insights.append(f"😕 **Confusion Points**: {len(confusions)} segment(s) with mixed emotional signals - consider simplifying")
                        
                        for insight in insights:
                            st.markdown(f"""
                                <div style='background: rgba(247, 185, 196, 0.15); padding: 1rem 1.5rem; border-radius: 10px; margin: 0.5rem 0; border-left: 3px solid #F7B9C4;'>
                                    {insight}
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Segment-by-Segment Breakdown
                        st.markdown("---")
                        st.markdown("### 📊 Segment-by-Segment Breakdown")
                        
                        for idx, (chunk, vec) in enumerate(zip(chunks, emotion_vectors)):
                            dominant_idx = vec.index(max(vec))
                            dominant_emotion = labels[dominant_idx]
                            
                            # Get top 3 emotions
                            emotion_scores = list(zip(labels, vec))
                            emotion_scores.sort(key=lambda x: x[1], reverse=True)
                            top_3 = emotion_scores[:3]
                            
                            is_flagged = any(start <= idx < end for start, end in drifts) or idx in confusions
                            border_color = "#DE638A" if is_flagged else "#C6BADE"
                            
                            with st.expander(f"{'🚩' if is_flagged else '✓'} Segment {idx+1}: {dominant_emotion} - {chunk[:60]}...", expanded=False):
                                st.markdown(f"**Full Text:**")
                                st.info(chunk)
                                
                                st.markdown(f"**Emotional Composition:**")
                                for emotion, score in top_3:
                                    percentage = score * 100
                                    st.markdown(f"- **{emotion}**: {percentage:.1f}%")
                                
                                # Show explanations and recommendations if flagged
                                if is_flagged:
                                    for key, exp in explanations.items():
                                        if str(idx) in key:
                                            st.warning(f"**⚠️ Issue Detected:** {exp}")
                                            
                                    # On-demand recommendation button
                                    if target_emotion and not target_emotion.startswith("None"):
                                        rec_key = f"rec_{idx}"
                                        if rec_key in st.session_state.recommendations:
                                            st.markdown(f"**✨ Suggested Revision (to match {target_emotion}):**")
                                            st.success(st.session_state.recommendations[rec_key])
                                        else:
                                            if st.button(f"✨ Generate AI Recommendation", key=f"btn_{idx}"):
                                                with st.spinner("Rewriting..."):
                                                    suggestion = regenerate_text(chunk, target_emotion)
                                                    st.session_state.recommendations[rec_key] = suggestion
                                                    st.rerun()
                        
                        # Flagged Sections Summary
                        if drifts or contradictions or confusions:
                            st.markdown("---")
                            st.markdown("## 🚩 Flagged Sections")
                            st.markdown("*Click on any section below to view details and recommendations*")
                            
                            for idx, chunk in enumerate(chunks):
                                is_flagged = any(start <= idx < end for start, end in drifts) or idx in confusions
                                
                                if is_flagged:
                                    with st.expander(f"📍 Segment {idx+1}: {chunk[:80]}{'...' if len(chunk) > 80 else ''}", expanded=False):
                                        st.markdown(f"**Full Text:**")
                                        st.info(chunk)
                                        
                                        # Show explanations and recommendations
                                        for key, exp in explanations.items():
                                            if str(idx) in key:
                                                st.warning(f"**⚠️ Issue Detected:** {exp}")
                                                
                                        # Show recommendation if exists in session state
                                        rec_key = f"rec_{idx}"
                                        if rec_key in st.session_state.recommendations:
                                            st.markdown(f"**✨ Suggested Revision (to match {target_emotion}):**")
                                            st.success(st.session_state.recommendations[rec_key])
                            
                            # Show contradictions separately
                            if contradictions:
                                st.markdown("### 🔄 Contradictory Statements")
                                for i, j, details in contradictions:
                                    st.error(f"**Segments {i+1} and {j+1}:** {details}")
                        else:
                            st.markdown("---")
                            st.success("🎉 **Great news!** No significant issues detected in your content. Your emotional tone is consistent throughout!")
                        
                        # Suggestions Section
                        st.markdown("---")
                        st.markdown("## 💡 Suggestions to Improve Your Content")
                        
                        suggestions = []
                        
                        # Tone consistency suggestions
                        if len(drifts) > 0:
                            suggestions.append({
                                "title": "🎯 Maintain Emotional Consistency",
                                "tips": [
                                    "Review the flagged drift sections and align them with your opening tone",
                                    "Use consistent language patterns throughout (e.g., if you start inspirational, maintain that energy)",
                                    "Create an emotional outline before writing to plan your tone journey",
                                    "Read your content aloud to catch unintentional tone shifts"
                                ]
                            })
                        
                        # Contradiction suggestions
                        if len(contradictions) > 0:
                            suggestions.append({
                                "title": "🔄 Resolve Contradictions",
                                "tips": [
                                    "Review contradictory statements and choose the message you want to emphasize",
                                    "If presenting multiple perspectives, use clear transitions (e.g., 'On the other hand...')",
                                    "Ensure your conclusion aligns with your main argument",
                                    "Consider removing or rephrasing statements that conflict with your core message"
                                ]
                            })
                        
                        # Confusion suggestions
                        if len(confusions) > 0:
                            suggestions.append({
                                "title": "✨ Improve Clarity",
                                "tips": [
                                    "Simplify complex sections with mixed emotional signals",
                                    "Use shorter sentences and clearer language in confusing areas",
                                    "Add subheadings to break up dense content",
                                    "Focus each paragraph on one main idea or emotion"
                                ]
                            })
                        
                        # General suggestions based on dominant emotion
                        avg_emotions = [sum(vec[i] for vec in emotion_vectors) / len(emotion_vectors) for i in range(len(labels))]
                        overall_dominant = labels[avg_emotions.index(max(avg_emotions))]
                        
                        emotion_suggestions = {
                            "Inspirational": {
                                "title": "✨ Enhance Your Inspirational Message",
                                "tips": [
                                    "Use vivid imagery and metaphors to paint a compelling vision",
                                    "Include personal stories or examples that resonate emotionally",
                                    "End with a powerful call-to-action that motivates readers",
                                    "Balance optimism with authenticity to maintain credibility"
                                ]
                            },
                            "Informative": {
                                "title": "📚 Strengthen Your Educational Content",
                                "tips": [
                                    "Use bullet points and lists for easy scanning",
                                    "Include examples or case studies to illustrate key points",
                                    "Add visual elements (charts, diagrams) if possible",
                                    "Summarize key takeaways at the end of each section"
                                ]
                            },
                            "Empathetic": {
                                "title": "💙 Deepen Emotional Connection",
                                "tips": [
                                    "Validate your audience's feelings and experiences",
                                    "Share vulnerable moments to build trust",
                                    "Use inclusive language ('we', 'us') to create community",
                                    "Offer practical support or resources alongside empathy"
                                ]
                            },
                            "Assertive": {
                                "title": "💪 Strengthen Your Assertive Voice",
                                "tips": [
                                    "Back up strong statements with evidence or examples",
                                    "Use confident language while remaining respectful",
                                    "Anticipate and address counterarguments",
                                    "Balance assertiveness with openness to dialogue"
                                ]
                            },
                            "Aggressive": {
                                "title": "⚠️ Soften Aggressive Tone",
                                "tips": [
                                    "Replace aggressive language with assertive but respectful phrasing",
                                    "Focus on solutions rather than blame or criticism",
                                    "Use 'I' statements instead of accusatory 'you' statements",
                                    "Consider your audience's emotional response to strong language"
                                ]
                            },
                            "Defensive": {
                                "title": "🛡️ Reduce Defensive Language",
                                "tips": [
                                    "Lead with confidence rather than justification",
                                    "Address concerns proactively without over-explaining",
                                    "Focus on your strengths rather than defending weaknesses",
                                    "Use positive framing instead of defensive qualifiers"
                                ]
                            },
                            "Neutral": {
                                "title": "🎨 Add Emotional Engagement",
                                "tips": [
                                    "Inject personality and voice into your writing",
                                    "Use storytelling to make dry topics more engaging",
                                    "Add relevant examples that evoke emotion",
                                    "Consider your audience's emotional needs and address them"
                                ]
                            }
                        }
                        
                        if overall_dominant in emotion_suggestions:
                            suggestions.append(emotion_suggestions[overall_dominant])
                        
                        # Always add general best practices
                        suggestions.append({
                            "title": "📝 General Best Practices",
                            "tips": [
                                "Know your audience and write for their emotional state",
                                "Use the 'inverted pyramid' - put key messages first",
                                "Vary sentence length to create rhythm and maintain interest",
                                "Edit ruthlessly - remove content that doesn't serve your core message",
                                "Get feedback from others before publishing important content"
                            ]
                        })
                        
                        # Display suggestions
                        for idx, suggestion in enumerate(suggestions):
                            with st.expander(f"{suggestion['title']}", expanded=(idx == 0)):
                                for tip in suggestion['tips']:
                                    st.markdown(f"• {tip}")
                        
                    except Exception as e:
                        st.error(f"❌ An error occurred during analysis: {str(e)}")
                        st.info("Please try again with different text or check your input.")
            
            elif text_input:
                st.warning("⚠️ Please enter at least 50 characters for meaningful analysis.")
            else:
                st.error("❌ Please enter some text to analyze.")

    # Footer
    st.markdown("---")
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0; color: #4A3267; opacity: 0.7;'>
            <p>Powered by AI • Built with ❤️ using Streamlit</p>
        </div>
    """, unsafe_allow_html=True)

# Main app logic
if st.session_state.page == 'landing':
    show_landing_page()
else:
    show_analyzer_page()