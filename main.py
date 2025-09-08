import streamlit as st
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ollama
from datasets import load_dataset
from pydantic import BaseModel, Field
import os
import numpy as np

# Pure alias trope classifier
try:
    from pure_alias_classifier import PureAliasCharacterClassifier
except Exception:
    PureAliasCharacterClassifier = None

# Import your existing models
class CulpritPredictionOutput(BaseModel):
    """Structured output model for culprit predictions"""
    culprits: List[str] = Field(description="List of names of the culprits who committed the crime")
    reasoning: str = Field(
        description="Detailed explanation of why these people are guilty, including key evidence and clues from the story")
    confidence: int = Field(description="Confidence level in the prediction (0-100)", ge=0, le=100)
    additional_suspects: Optional[List[str]] = Field(default=None, description="Other potential suspects considered")
    key_evidence: Optional[List[str]] = Field(default=None,
                                              description="Key pieces of evidence that support the conclusion")


class JudgingOutput(BaseModel):
    """Structured output model for LLM-based judging"""
    is_correct: bool = Field(description="Whether the prediction matches the actual culprits")
    match_score: int = Field(description="Score from 0 to 100 indicating how well the prediction matches", ge=0,
                               le=100)
    reasoning: str = Field(description="Explanation of why the prediction is considered correct or incorrect")
    matched_culprits: List[str] = Field(description="List of predicted culprits that match actual culprits")
    missed_culprits: List[str] = Field(description="List of actual culprits that were missed in the prediction")
    extra_culprits: List[str] = Field(description="List of predicted culprits that are not in the actual culprits")


@dataclass
class CulpritPrediction:
    """Structure for storing culprit predictions with evaluation data"""
    story_title: str
    predicted_culprits: List[str]
    reasoning: str
    confidence: int
    actual_culprits: List[str]
    is_correct: bool
    match_score: int
    judging_reasoning: str
    matched_culprits: List[str]
    missed_culprits: List[str]
    extra_culprits: List[str]
    additional_suspects: Optional[List[str]] = None
    key_evidence: Optional[List[str]] = None


class StreamingWhoDunItDetector:
    """Streamlit-adapted WhoDunIt detector with streaming support"""

    def __init__(self, model_name: str = "gemma3:12b", max_tokens: int = 4000):
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.available_models = self._get_available_models()

    def _get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = ollama.list()
            models = [model['model'] for model in response['models']]
            return models
        except Exception as e:
            st.error(f"❌ Error getting models: {e}")
            return []

    def _check_ollama_connection(self):
        """Check if Ollama is running and accessible"""
        try:
            response = ollama.list()
            return True, [model['model'] for model in response['models']]
        except Exception as e:
            return False, str(e)

    def update_model(self, model_name: str):
        """Update the model being used"""
        self.model_name = model_name

    def truncate_text(self, text: str, max_tokens: int = None) -> str:
        """Truncate text to fit within token limits"""
        if max_tokens is None:
            max_tokens = self.max_tokens

        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text

        truncated = text[:max_chars]
        last_para = truncated.rfind('\n\n')
        if last_para > max_chars * 0.8:
            return truncated[:last_para]

        last_sentence = truncated.rfind('. ')
        if last_sentence > max_chars * 0.8:
            return truncated[:last_sentence + 1]

        return truncated

    def create_prompt(self, story_text: str, story_title: str, story_author: str, custom_prompt: str = None) -> str:
        """Create a prompt for culprit detection"""
        if custom_prompt:
            # Replace placeholders in custom prompt
            prompt = custom_prompt.replace("{story_title}", story_title)
            prompt = prompt.replace("{story_author}", story_author)
            prompt = prompt.replace("{story_text}", story_text)
            return prompt

        # Default prompt
        prompt = f"""You are an expert detective analyzing mystery stories. Read the following story carefully and identify who committed the crime.

Story Title: {story_title}
Author: {story_author}

Story Text:
{story_text}

Your task is to analyze this mystery story and identify ALL the culprits involved in the crime. Consider:

1. **Motive**: Who had reasons to commit the crime?
2. **Opportunity**: Who had access and the chance to commit the crime?
3. **Evidence**: What clues point to specific individuals?
4. **Behavior**: Who acted suspiciously or inconsistently?
5. **Revelations**: What key information is revealed about the perpetrator(s)?

Focus on finding the actual perpetrator(s) based on evidence presented in the story. Look for the resolution where the detective or narrator reveals who committed the crime.

IMPORTANT: 
- If there are multiple people involved in the crime, list ALL of them
- If only one person committed the crime, return a list with just that person
- Be thorough in identifying all culprits, including accomplices and co-conspirators

Provide your analysis in the structured format requested."""

        return prompt

    def create_judging_prompt(self, predicted_culprits: List[str], actual_culprits: List[str], story_title: str,
                              custom_judging_prompt: str = None) -> str:
        """Create a prompt for LLM-based judging"""
        if custom_judging_prompt:
            # Replace placeholders in custom judging prompt
            prompt = custom_judging_prompt.replace("{story_title}", story_title)
            prompt = prompt.replace("{predicted_culprits}", str(predicted_culprits))
            prompt = prompt.replace("{actual_culprits}", str(actual_culprits))
            return prompt

        # Default judging prompt
        prompt = f"""You are an expert judge evaluating culprit predictions for mystery stories. Your task is to determine if the predicted culprits match the actual culprits.

Story Title: {story_title}

Predicted Culprits: {predicted_culprits}
Actual Culprits: {actual_culprits}

Your task is to evaluate whether the prediction is correct by considering:

1. **Exact Matches**: Do the predicted names exactly match the actual culprit names?
2. **Partial Matches**: Are there variations in naming (e.g., "John Smith" vs "John", "Dr. Watson" vs "Watson")?
3. **Character References**: Do the predictions refer to the same characters using different names or titles?
4. **Completeness**: Are all actual culprits covered in the prediction?
5. **Extra Predictions**: Are there predicted culprits that are not actual culprits?

Evaluation Guidelines:
- A prediction is CORRECT if it identifies all actual culprits, even with minor name variations
- A prediction is PARTIALLY CORRECT if it identifies some but not all culprits
- A prediction is INCORRECT if it misses major culprits or identifies wrong people
- Consider common name variations (nicknames, titles, full names vs partial names)

Provide:
- is_correct: True if the prediction substantially matches the actual culprits
- match_score: 0 to 100 score (100 = perfect match, 0 = no match)
- reasoning: Detailed explanation of your judgment
- matched_culprits: List of predicted culprits that match actual ones
- missed_culprits: List of actual culprits not predicted
- extra_culprits: List of predicted culprits that are not actual culprits

Be generous with partial name matches but strict about identifying the right characters."""

        return prompt

    def stream_ollama_response(self, prompt: str, output_model: BaseModel, placeholder, temperature: float = 0.1,
                               max_tokens: int = 800):
        """Stream response from Ollama with structured output"""
        try:
            full_response = ""

            # Use Ollama's streaming capability
            response_stream = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                format=output_model.model_json_schema(),
                stream=True,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                }
            )

            # Stream the response
            for chunk in response_stream:
                if chunk.get('response'):
                    full_response += chunk['response']
                    placeholder.markdown(f"**Generating response...**\n\n{full_response}")
                    time.sleep(0.01)  # Small delay for visual effect

            # Try to parse the structured response
            try:
                response_data = json.loads(full_response)
                return output_model(**response_data)
            except json.JSONDecodeError:
                # If JSON parsing fails, return a fallback response
                if output_model == CulpritPredictionOutput:
                    return CulpritPredictionOutput(
                        culprits=["Parse Error"],
                        reasoning="Failed to parse structured response",
                        confidence=0
                    )
                else:
                    return JudgingOutput(
                        is_correct=False,
                        match_score=0,
                        reasoning="Failed to parse structured response",
                        matched_culprits=[],
                        missed_culprits=[],
                        extra_culprits=[]
                    )

        except Exception as e:
            st.error(f"Error during streaming: {e}")
            if output_model == CulpritPredictionOutput:
                return CulpritPredictionOutput(
                    culprits=["Error"],
                    reasoning=f"Error occurred: {str(e)}",
                    confidence=0
                )
            else:
                return JudgingOutput(
                    is_correct=False,
                    match_score=0,
                    reasoning=f"Error occurred during judging: {str(e)}",
                    matched_culprits=[],
                    missed_culprits=[],
                    extra_culprits=[]
                )

    def process_story_streaming(self, story_data: Dict[str, Any], prediction_placeholder, judging_placeholder,
                                custom_prompt: str = None, custom_judging_prompt: str = None,
                                temperature: float = 0.1, max_tokens: int = 800) -> CulpritPrediction:
        """Process a single story with streaming output"""
        title = story_data['title']
        author = story_data['author']
        text = story_data['text']
        actual_culprits = story_data['culprit_ids']

        # Truncate text if too long
        truncated_text = self.truncate_text(text)

        # Create prompt and stream prediction
        prompt = self.create_prompt(truncated_text, title, author, custom_prompt)
        result = self.stream_ollama_response(prompt, CulpritPredictionOutput, prediction_placeholder, temperature,
                                             max_tokens)

        # Update prediction display
        prediction_placeholder.markdown(f"""
        **🔍 Model Prediction:**
        - **Predicted Culprits:** {result.culprits}
        - **Confidence:** {result.confidence}%
        - **Reasoning:** {result.reasoning}
        - **Additional Suspects:** {result.additional_suspects or 'None'}
        - **Key Evidence:** {result.key_evidence or 'None'}
        """)

        # Stream judging
        judging_result = self.stream_ollama_response(
            self.create_judging_prompt(result.culprits, actual_culprits, title, custom_judging_prompt),
            JudgingOutput,
            judging_placeholder,
            temperature,
            max_tokens
        )

        # Update judging display
        judging_placeholder.markdown(f"""
        **⚖️ Evaluation:**
        - **Is Correct:** {'✅ Yes' if judging_result.is_correct else '❌ No'}
        - **Match Score:** {judging_result.match_score}%
        - **Matched Culprits:** {judging_result.matched_culprits}
        - **Missed Culprits:** {judging_result.missed_culprits}
        - **Extra Culprits:** {judging_result.extra_culprits}
        - **Reasoning:** {judging_result.reasoning}
        """)

        return CulpritPrediction(
            story_title=title,
            predicted_culprits=result.culprits,
            reasoning=result.reasoning,
            confidence=result.confidence,
            actual_culprits=actual_culprits,
            is_correct=judging_result.is_correct,
            match_score=judging_result.match_score,
            judging_reasoning=judging_result.reasoning,
            matched_culprits=judging_result.matched_culprits,
            missed_culprits=judging_result.missed_culprits,
            extra_culprits=judging_result.extra_culprits,
            additional_suspects=result.additional_suspects,
            key_evidence=result.key_evidence
        )


def get_default_prompts():
    """Get default prompt templates"""
    default_detection_prompt = """You are an expert detective analyzing mystery stories. Read the following story carefully and identify who committed the crime.

Story Title: {story_title}
Author: {story_author}

Story Text:
{story_text}

Your task is to analyze this mystery story and identify ALL the culprits involved in the crime. Consider:

1. **Motive**: Who had reasons to commit the crime?
2. **Opportunity**: Who had access and the chance to commit the crime?
3. **Evidence**: What clues point to specific individuals?
4. **Behavior**: Who acted suspiciously or inconsistently?
5. **Revelations**: What key information is revealed about the perpetrator(s)?

Focus on finding the actual perpetrator(s) based on evidence presented in the story. Look for the resolution where the detective or narrator reveals who committed the crime.

IMPORTANT: 
- If there are multiple people involved in the crime, list ALL of them
- If only one person committed the crime, return a list with just that person
- Be thorough in identifying all culprits, including accomplices and co-conspirators

Provide your analysis in the structured format requested."""

    default_judging_prompt = """You are an expert judge evaluating culprit predictions for mystery stories. Your task is to determine if the predicted culprits match the actual culprits.

Story Title: {story_title}

Predicted Culprits: {predicted_culprits}
Actual Culprits: {actual_culprits}

Your task is to evaluate whether the prediction is correct by considering:

1. **Exact Matches**: Do the predicted names exactly match the actual culprit names?
2. **Partial Matches**: Are there variations in naming (e.g., "John Smith" vs "John", "Dr. Watson" vs "Watson")?
3. **Character References**: Do the predictions refer to the same characters using different names or titles?
4. **Completeness**: Are all actual culprits covered in the prediction?
5. **Extra Predictions**: Are there predicted culprits that are not actual culprits?

Evaluation Guidelines:
- A prediction is CORRECT if it identifies all actual culprits, even with minor name variations
- A prediction is PARTIALLY CORRECT if it identifies some but not all culprits
- A prediction is INCORRECT if it misses major culprits or identifies wrong people
- Consider common name variations (nicknames, titles, full names vs partial names)

Provide:
- is_correct: True if the prediction substantially matches the actual culprits
- match_score: 00 to 1.0 score (100 = perfect match, 0 = no match)
- reasoning: Detailed explanation of your judgment
- matched_culprits: List of predicted culprits that match actual ones
- missed_culprits: List of actual culprits not predicted
- extra_culprits: List of predicted culprits that are not actual culprits

Be generous with partial name matches but strict about identifying the right characters."""

    return default_detection_prompt, default_judging_prompt


def main():
    st.set_page_config(page_title="WhoDunIt Detective", layout="wide")

    st.title("🔍 WhoDunIt Mystery Detective")
    st.markdown("Analyze mystery stories and identify culprits using AI")

    # Initialize detector
    if 'detector' not in st.session_state:
        st.session_state.detector = StreamingWhoDunItDetector()

    # Check Ollama connection
    is_connected, models_or_error = st.session_state.detector._check_ollama_connection()
    available_models = []  # Initialize to avoid UnboundLocalError

    if not is_connected:
        st.error(f"❌ Ollama connection failed: {models_or_error}")
        st.stop()
    else:
        available_models = models_or_error
        if available_models:
            st.success(f"✅ Ollama connected. Available models: {', '.join(available_models)}")
        else:
            st.warning("⚠️ No models found. Please ensure you have models installed.")

    # Load dataset
    if 'dataset' not in st.session_state:
        with st.spinner("Loading dataset..."):
            st.session_state.dataset = load_dataset("kjgpta/WhoDunIt", split="train")
            st.success(f"Loaded {len(st.session_state.dataset)} stories")

    # Sidebar for configuration
    st.sidebar.header("⚙️ Configuration")

    # Model selection
    current_model = st.session_state.detector.model_name
    if available_models:
        current_model = st.sidebar.selectbox(
            "Select Model:",
            available_models,
            index=available_models.index(
                st.session_state.detector.model_name) if st.session_state.detector.model_name in available_models else 0
        )
        st.session_state.detector.update_model(current_model)

    # Model parameters
    st.sidebar.subheader("Model Parameters")
    temperature = st.sidebar.slider("Temperature", 0.0, 2.0, 0.1, 0.1)
    max_tokens = st.sidebar.slider("Max Tokens", 100, 120000, 800, 50)

    # Prompt configuration
    st.sidebar.subheader("Prompt Configuration")

    # Get default prompts
    default_detection_prompt, default_judging_prompt = get_default_prompts()

    # Initialize prompts in session state
    if 'custom_detection_prompt' not in st.session_state:
        st.session_state.custom_detection_prompt = default_detection_prompt
    if 'custom_judging_prompt' not in st.session_state:
        st.session_state.custom_judging_prompt = default_judging_prompt

    # Prompt editor toggle
    edit_prompts = st.sidebar.checkbox("Edit Prompts", value=False)

    if edit_prompts:
        with st.sidebar.expander("Detection Prompt", expanded=True):
            st.session_state.custom_detection_prompt = st.text_area(
                "Detection Prompt Template",
                value=st.session_state.custom_detection_prompt,
                height=200,
                help="Available placeholders: {story_title}, {story_author}, {story_text}"
            )

            if st.button("Reset to Default", key="reset_detection"):
                st.session_state.custom_detection_prompt = default_detection_prompt
                st.rerun()

        with st.sidebar.expander("Judging Prompt", expanded=False):
            st.session_state.custom_judging_prompt = st.text_area(
                "Judging Prompt Template",
                value=st.session_state.custom_judging_prompt,
                height=200,
                help="Available placeholders: {story_title}, {predicted_culprits}, {actual_culprits}"
            )

            if st.button("Reset to Default", key="reset_judging"):
                st.session_state.custom_judging_prompt = default_judging_prompt
                st.rerun()

    # Story selection
    st.sidebar.subheader("Story Selection")
    story_index = st.sidebar.selectbox(
        "Choose a story:",
        range(len(st.session_state.dataset)),
        format_func=lambda x: f"{x + 1}. {st.session_state.dataset[x]['title']}"
    )

    # Metadata-based trope analysis controls
    st.sidebar.subheader("Pure Alias Trope Analysis")
    enable_trope_analysis = st.sidebar.checkbox("Enable pure alias-based trope analysis", value=True)
    trope_model_path = st.sidebar.text_input("Model file (pkl)", value="pure_alias_classifier_last_word_with_weights.pkl")
    load_trope_model = st.sidebar.button("Load trope model")
    
    # Trope Dictionary Reference
    if st.sidebar.checkbox("Show Trope Dictionary", value=False):
        with st.sidebar.expander("📚 Complete Trope Dictionary", expanded=True):
            try:
                import json
                with open('trope_categories_filtered.json', 'r') as f:
                    trope_dict = json.load(f)
                
                st.write(f"**Total Categories:** {len(trope_dict)}")
                total_tropes = sum(len(tropes) for tropes in trope_dict.values())
                st.write(f"**Total Tropes:** {total_tropes}")
                
                # Show categories in alphabetical order
                for category in sorted(trope_dict.keys()):
                    tropes = trope_dict[category]
                    with st.expander(f"{category.replace('_', ' ').title()} ({len(tropes)} tropes)", expanded=False):
                        for trope in tropes:
                            st.write(f"• {trope}")
                            
            except Exception as e:
                st.sidebar.error(f"Error loading trope dictionary: {e}")

    if 'trope_classifier' not in st.session_state:
        st.session_state.trope_classifier = None
    if 'trope_error' not in st.session_state:
        st.session_state.trope_error = None

    if load_trope_model and enable_trope_analysis:
        st.session_state.trope_error = None
        try:
            if not os.path.exists(trope_model_path):
                st.session_state.trope_error = f"File not found: {trope_model_path}. Train it via pure_alias_classifier.py."
            else:
                import pickle
                with open(trope_model_path, 'rb') as f:
                    clf = pickle.load(f)
                # Basic API sanity checks
                if not hasattr(clf, 'predict_all_candidates'):
                    st.session_state.trope_error = "Loaded object is missing predict_all_candidates()"
                else:
                    st.session_state.trope_classifier = clf
                    st.success("✅ Pure alias trope model loaded")
                    
                    # Display category weights if available
                    if hasattr(clf, 'category_weights') and clf.category_weights:
                        st.sidebar.subheader("🏷️ Category Weights")
                        
                        # Load trope dictionary
                        trope_dict = {}
                        try:
                            import json
                            with open('trope_categories_filtered.json', 'r') as f:
                                trope_dict = json.load(f)
                        except Exception as e:
                            st.sidebar.warning(f"Could not load trope dictionary: {e}")
                        
                        sorted_weights = sorted(clf.category_weights.items(), key=lambda x: x[1], reverse=True)
                        for category, weight in sorted_weights[:10]:  # Show top 10
                            with st.sidebar.expander(f"{category.replace('_', ' ').title()} ({weight:.2f})", expanded=False):
                                st.write(f"**Discriminative Weight:** {weight:.2f}")
                                
                                # Show tropes in this category
                                if category in trope_dict:
                                    st.write(f"**Tropes ({len(trope_dict[category])}):**")
                                    tropes = trope_dict[category]
                                    # Display tropes in a nice format
                                    for i, trope in enumerate(tropes):
                                        st.write(f"• {trope}")
                                        if i >= 19:  # Show max 20 tropes
                                            remaining = len(tropes) - 20
                                            if remaining > 0:
                                                st.write(f"*... and {remaining} more*")
                                            break
                                else:
                                    st.write("*Tropes not found in dictionary*")
        except Exception as e:
            st.session_state.trope_error = str(e)
    if st.session_state.trope_error:
        st.sidebar.error(f"Trope model error: {st.session_state.trope_error}")

    # Get current story
    current_story = st.session_state.dataset[story_index]

    # Main layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📚 Story")
        st.subheader(f"Title: {current_story['title']}")
        st.markdown(f"**Author:** {current_story['author']}")
        st.markdown(f"**Actual Culprits:** {current_story['culprit_ids']}")

        # Display story text in scrollable container
        st.markdown("**Story Text:**")
        story_container = st.container()
        with story_container:
            st.markdown(
                f'<div style="height: 600px; overflow-y: auto; padding: 10px; border: 1px solid #ddd; border-radius: 5px; background-color: #f8f9fa;">'
                f'{current_story["text"].replace(chr(10), "<br>")}'
                f'</div>',
                unsafe_allow_html=True
            )

    with col2:
        st.header("🤖 AI Analysis")

        # Current configuration display
        with st.expander("Current Configuration", expanded=False):
            st.markdown(f"""
            **Model:** {current_model}
            **Temperature:** {temperature}
            **Max Tokens:** {max_tokens}
            **Custom Prompts:** {'Yes' if edit_prompts else 'No (Using defaults)'}
            """)

        # Generate button
        if st.button("🔍 Generate Analysis", type="primary", use_container_width=True):
            # Clear previous results
            st.session_state.pop('current_prediction', None)

            # Create placeholders for streaming
            prediction_placeholder = st.empty()
            judging_placeholder = st.empty()

            # Show loading message
            prediction_placeholder.markdown("**🔄 Generating prediction...**")
            judging_placeholder.markdown("**🔄 Preparing evaluation...**")

            # Process story with streaming
            try:
                prediction = st.session_state.detector.process_story_streaming(
                    current_story,
                    prediction_placeholder,
                    judging_placeholder,
                    custom_prompt=st.session_state.custom_detection_prompt if edit_prompts else None,
                    custom_judging_prompt=st.session_state.custom_judging_prompt if edit_prompts else None,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                st.session_state.current_prediction = prediction

                # Success message
                st.success("✅ Analysis complete!")

            except Exception as e:
                st.error(f"❌ Error during analysis: {e}")

        # Display golden answer
        st.markdown("---")
        st.markdown(f"""
        **🎯 Golden Answer:**
        - **Actual Culprits:** {current_story['culprit_ids']}
        """)

        # Pure Alias-Based Trope Analysis UI - CULPRIT-FOCUSED
        st.markdown("---")
        st.subheader("📋 Pure Alias Trope Analysis (Culprit-Focused)")
        if not enable_trope_analysis:
            st.info("Enable in the sidebar to view how well tropes identify the actual culprits.")
        else:
            if st.session_state.trope_classifier is None:
                st.warning("Load a trained pure alias trope model (pkl) from the sidebar.")
            else:
                try:
                    # Get actual culprits from the dataset
                    culprits_raw = current_story.get('culprit_ids', [])
                    if isinstance(culprits_raw, str):
                        try:
                            import ast
                            culprits_raw = ast.literal_eval(culprits_raw)
                        except:
                            culprits_raw = []
                    
                    if not culprits_raw:
                        st.info("No culprits specified for this story.")
                    else:
                        st.markdown("**🎯 Pure Alias Trope-based Scores for Actual Culprits:**")
                        st.markdown("*How well does the pure alias-based trope classifier identify the known culprits?*")
                        
                        # Show loading message while processing
                        with st.spinner("🔍 Analyzing trope patterns for culprits..."):
                            # Get all candidate scores to have context
                            cand_scores, cand_details = st.session_state.trope_classifier.predict_all_candidates(current_story)
                        
                        culprit_results = []
                        for culprit in culprits_raw:
                            culprit_name = str(culprit).strip()
                            
                            # Find the best matching candidate for this culprit
                            best_score = 0.0
                            best_match = None
                            best_details = None
                            
                            for cand_name, score in cand_scores.items():
                                # Check if candidate matches culprit (exact or contains)
                                if (culprit_name.lower() in cand_name.lower() or 
                                    cand_name.lower() in culprit_name.lower() or
                                    culprit_name.lower() == cand_name.lower()):
                                    if score > best_score:
                                        best_score = score
                                        best_match = cand_name
                                        best_details = cand_details.get(cand_name)
                            
                            culprit_results.append({
                                'culprit': culprit_name,
                                'score': best_score,
                                'matched_candidate': best_match,
                                'details': best_details
                            })
                        
                        # Display culprit scores
                        for i, result in enumerate(culprit_results, 1):
                            if result['score'] > 0:
                                confidence_level = "🔥 HIGH" if result['score'] > 0.8 else "🟡 MEDIUM" if result['score'] > 0.5 else "🔵 LOW"
                                st.markdown(f"{i}. **{result['culprit']}** → {result['score']:.3f} {confidence_level}")
                                if result['matched_candidate'] != result['culprit']:
                                    st.markdown(f"   *Matched via: {result['matched_candidate']}*")
                            else:
                                st.markdown(f"{i}. **{result['culprit']}** → ❌ Not found as candidate")
                        
                        # Show average culprit score and accuracy metrics
                        valid_scores = [r['score'] for r in culprit_results if r['score'] > 0]
                        if valid_scores:
                            avg_culprit_score = np.mean(valid_scores)
                            
                            # Calculate accuracy metrics - compare culprits vs all other candidates
                            all_scores = list(cand_scores.values())
                            non_culprit_scores = []
                            culprit_names_lower = [str(c).lower().strip() for c in culprits_raw]
                            
                            for cand_name, score in cand_scores.items():
                                is_culprit = any(
                                    culprit_name.lower() in cand_name.lower() or 
                                    cand_name.lower() in culprit_name.lower() or
                                    culprit_name.lower() == cand_name.lower()
                                    for culprit_name in culprit_names_lower
                                )
                                if not is_culprit:
                                    non_culprit_scores.append(score)
                            
                            avg_non_culprit_score = np.mean(non_culprit_scores) if non_culprit_scores else 0.0
                            
                            # Display metrics
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📊 Avg Culprit Score", f"{avg_culprit_score:.3f}")
                            with col2:
                                st.metric("📊 Avg Non-Culprit Score", f"{avg_non_culprit_score:.3f}")
                            with col3:
                                discrimination = avg_culprit_score / avg_non_culprit_score if avg_non_culprit_score > 0 else float('inf')
                                st.metric("📊 Discrimination Ratio", f"{discrimination:.2f}x")
                            
                            # Performance assessment with discrimination context
                            st.markdown(f"\n**🎯 Model Performance Analysis:**")
                            if discrimination > 2.0 and avg_culprit_score > 0.6:
                                st.success(f"🎯 Excellent! Culprits score {discrimination:.1f}x higher than non-culprits.")
                            elif discrimination > 1.5:
                                st.warning(f"🟡 Moderate. Culprits score {discrimination:.1f}x higher than non-culprits.")
                            elif discrimination > 1.1:
                                st.info(f"🔵 Weak discrimination. Culprits only score {discrimination:.1f}x higher.")
                            else:
                                st.error("🔴 Poor! Model may be giving high scores to everyone.")
                            
                            # Show score distribution info
                            st.markdown(f"- **Total candidates:** {len(all_scores)}")
                            st.markdown(f"- **Culprits found:** {len(valid_scores)}/{len(culprits_raw)}")
                            st.markdown(f"- **Score range:** {min(all_scores):.3f} - {max(all_scores):.3f}")
                        
                        # Detailed explanations for each culprit with scores > 0
                        if culprit_results:
                            st.markdown("\n**🔍 Detailed Explanations:**")
                            
                            # Create explanation tabs for each culprit that has a score
                            culprits_with_scores = [r for r in culprit_results if r['score'] > 0 and r['details']]
                            
                            if culprits_with_scores:
                                # Create tabs for each culprit
                                tab_names = [f"{r['culprit']} ({r['score']:.3f})" for r in culprits_with_scores]
                                tabs = st.tabs(tab_names)
                                
                                for tab, result in zip(tabs, culprits_with_scores):
                                    with tab:
                                        details = result['details']
                                        features = details['features']
                                        
                                        st.markdown(f"**🎯 Analysis for '{result['culprit']}'**")
                                        if result['matched_candidate'] != result['culprit']:
                                            st.markdown(f"*Matched via candidate: {result['matched_candidate']}*")
                                        
                                        # Show top contributing trope categories
                                        categories = set(name.rsplit('_', 2)[0] for name in features.keys())
                                        contributions = []
                                        for category in categories:
                                            count = features.get(f"{category}_count_within_50", 0)
                                            min_dist = features.get(f"{category}_min_dist", 999999)
                                            kernel = features.get(f"{category}_kernel_sum", 0.0)
                                            if count > 0:
                                                weight = 1.0
                                                if hasattr(st.session_state.trope_classifier, 'category_weights') and st.session_state.trope_classifier.category_weights:
                                                    weight = st.session_state.trope_classifier.category_weights.get(category, 1.0)
                                                contributions.append((category, count * weight, count, min_dist, kernel, weight))
                                        
                                        contributions.sort(key=lambda x: x[1], reverse=True)
                                        if contributions:
                                            st.markdown("**🔍 Top Contributing Trope Categories:**")
                                            for i, (cat, contrib, cnt, dist, ker, w) in enumerate(contributions[:8], start=1):
                                                st.markdown(f"{i}. **{cat}** — {cnt} tropes within 50 tokens, min_dist: {int(dist)}, discriminative_weight: {w:.2f}")
                                            
                                            # Show interpretation
                                            st.markdown("\n**📖 Interpretation:**")
                                            top_categories = [c[0] for c in contributions[:3]]
                                            if 'poison_medical' in top_categories:
                                                st.markdown("- 🧪 **Medical/poison themes** detected near this character")
                                            if 'financial_motive' in top_categories:
                                                st.markdown("- 💰 **Financial motives** detected near this character")
                                            if 'crime_theme' in top_categories:
                                                st.markdown("- 🔍 **Crime-related language** detected near this character")
                                            if 'locations_places' in top_categories:
                                                st.markdown("- 📍 **Significant locations** mentioned near this character")
                                            if 'character_names' in top_categories:
                                                st.markdown("- 👤 **Character interactions** detected near this character")
                                        else:
                                            st.write("❌ No significant trope activity detected for this culprit.")
                            else:
                                st.info("No culprits found with trope-based scores to explain.")
                        
                except Exception as e:
                    st.error(f"Error running pure alias culprit-focused trope analysis: {e}")

        # Show previous results if they exist
        if 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction

            st.markdown("---")
            st.markdown("**📊 Summary:**")

            # Status indicator
            if pred.is_correct:
                st.success(f"✅ **CORRECT** (Match Score: {pred.match_score}%)")
            else:
                st.error(f"❌ **INCORRECT** (Match Score: {pred.match_score}%)")

            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{pred.confidence}%")
            with col_b:
                st.metric("Match Score", f"{pred.match_score}%")

    # Footer with model info
    st.markdown("---")
    st.markdown(
        f"**Model:** {st.session_state.detector.model_name} | **Total Stories:** {len(st.session_state.dataset)} | **Temperature:** {temperature} | **Max Tokens:** {max_tokens}")


if __name__ == "__main__":
    main()

