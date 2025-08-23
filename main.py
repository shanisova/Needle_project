import streamlit as st
import json
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import ollama
from datasets import load_dataset
from pydantic import BaseModel, Field

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
    match_score: float
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
- match_score: 0.0 to 1.0 score (1.0 = perfect match, 0.0 = no match)
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
                        match_score=0.0,
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
                    match_score=0.0,
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
        - **Match Score:** {judging_result.match_score:.2f}
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
- match_score: 0.0 to 1.0 score (1.0 = perfect match, 0.0 = no match)
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
    max_tokens = st.sidebar.slider("Max Tokens", 100, 2000, 800, 50)

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

        # Show previous results if they exist
        if 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction

            st.markdown("---")
            st.markdown("**📊 Summary:**")

            # Status indicator
            if pred.is_correct:
                st.success(f"✅ **CORRECT** (Match Score: {pred.match_score:.2f})")
            else:
                st.error(f"❌ **INCORRECT** (Match Score: {pred.match_score:.2f})")

            # Metrics
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Confidence", f"{pred.confidence}%")
            with col_b:
                st.metric("Match Score", f"{pred.match_score:.2f}")

    # Footer with model info
    st.markdown("---")
    st.markdown(
        f"**Model:** {st.session_state.detector.model_name} | **Total Stories:** {len(st.session_state.dataset)} | **Temperature:** {temperature} | **Max Tokens:** {max_tokens}")


if __name__ == "__main__":
    main()

