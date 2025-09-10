import streamlit as st
import json
import time
import subprocess
import shlex
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import ollama
from datasets import load_dataset
from pydantic import BaseModel, Field
import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import ast

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

Evaluation Guidelines:
- A prediction is CORRECT if it identifies all actual culprits, even with name variations
- A prediction is PARTIALLY CORRECT if it identifies some but not all culprits
- A prediction is INCORRECT if it misses all major culprits or identifies wrong people
- Consider common name variations (nicknames, titles, full names vs partial names)

Provide:
- is_correct: True if the prediction substantially matches the actual culprits
- match_score: 00 to 1.0 score (100 = perfect match, 0 = no match)
- reasoning: Detailed explanation of your judgment
- matched_culprits: List of predicted culprits that match actual ones
- missed_culprits: List of actual culprits not predicted
- extra_culprits: List of predicted culprits that are not actual culprits

Be generous with partial name matches but make sure to identify the right characters."""

    return default_detection_prompt, default_judging_prompt


def load_priority_stories():
    """Load priority stories from CSV"""
    try:
        priority_df = pd.read_csv('priority_stories.csv')
        return priority_df
    except FileNotFoundError:
        st.warning("Priority stories CSV not found. Using default order.")
        return None


def create_story_selection_options(dataset, priority_df):
    """Create organized story selection with priorities first"""
    story_options = []
    story_indices = []
    
    if priority_df is not None:
        # Add priority stories first (no stars)
        for _, row in priority_df.iterrows():
            story_idx = int(row["story_index"])
            if story_idx < len(dataset):
                story_title = dataset[story_idx]["title"]
                story_options.append(story_title)
                story_indices.append(story_idx)
        
        # Add remaining stories (no separator)
        priority_indices = set(priority_df["story_index"].tolist())
        for i in range(len(dataset)):
            if i not in priority_indices:
                story_title = dataset[i]["title"]
                story_options.append(story_title)
                story_indices.append(i)
    else:
        # Fallback to original format
        for i in range(len(dataset)):
            story_title = dataset[i]["title"]
            story_options.append(story_title)
            story_indices.append(i)
    
    return story_options, story_indices


# Character Graph Analysis Functions
def clean_dir_name(title: str) -> str:
    """Clean story title for directory naming."""
    return title.replace(' ', '_').replace("'", '').replace('"', '').replace(':', '').replace(';', '')


def run_pipeline_for_story(story_index: int, progress_callback=None) -> bool:
    """Run the pipeline for a story and return success status."""
    try:
        if progress_callback:
            progress_callback("🚀 Starting character data collection pipeline...")
        
        # Load story info for better progress messages
        try:
            dataset = load_dataset("kjgpta/WhoDunIt", split="train")
            story_title = dataset[story_index].get("title", f"Story_{story_index}")
            if progress_callback:
                progress_callback(f"📖 Processing: {story_title}")
        except:
            story_title = f"Story {story_index}"
        
        if progress_callback:
            progress_callback("🔍 Step 1/3: Starting Character Extraction...")
        
        # Create command
        cmd = f"python3 run_pipeline.py {story_index} --reuse-chars"
        
        # Run the pipeline with real-time output
        import threading
        import time
        
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                 stderr=subprocess.STDOUT, text=True, cwd=Path.cwd())
        
        # Variables to track progress
        step1_started = False
        step1_done = False
        step2_done = False
        step3_done = False
        heartbeat_counter = 0
        
        # Function to send periodic heartbeat during long operations
        def heartbeat_thread():
            nonlocal heartbeat_counter
            while process.poll() is None:
                time.sleep(10)  # Wait 10 seconds
                if not step1_done and step1_started and progress_callback:
                    heartbeat_counter += 1
                    dots = "." * ((heartbeat_counter % 4) + 1)
                    progress_callback(f"🔍 Step 1/3: Character Extraction in progress{dots} (LLM is working, please wait)")
        
        # Start heartbeat thread
        heartbeat = threading.Thread(target=heartbeat_thread, daemon=True)
        heartbeat.start()
        
        # Read output line by line
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            line = line.strip()
            if not line:
                continue
            
            output_lines.append(line)
            
            if progress_callback:
                # More detailed progress tracking
                if "Character Extraction" in line and "Running" in line and not step1_started:
                    progress_callback("🔍 Step 1/3: Character Extraction starting...")
                    step1_started = True
                elif "Character Extraction" in line and "completed" in line and not step1_done:
                    progress_callback("✅ Step 1/3: Character Extraction completed!")
                    step1_done = True
                elif "Alias Building" in line and "Running" in line and not step2_done:
                    progress_callback("🔗 Step 2/3: Alias Building in progress...")
                elif "Alias Building" in line and "completed" in line and not step2_done:
                    progress_callback("✅ Step 2/3: Alias Building completed!")
                    step2_done = True
                elif "Interaction Extraction" in line and "Running" in line and not step3_done:
                    progress_callback("📊 Step 3/3: Interaction Analysis in progress...")
                elif "Interaction Extraction" in line and "completed" in line and not step3_done:
                    progress_callback("✅ Step 3/3: Interaction Analysis completed!")
                    step3_done = True
                elif "Pipeline Completed Successfully" in line:
                    progress_callback("🎉 All steps completed successfully!")
                elif "ERROR" in line or "❌" in line:
                    progress_callback(f"⚠️ {line}")
        
        # Wait for process to complete
        return_code = process.wait()
        
        # Create a result-like object for compatibility
        class Result:
            def __init__(self, returncode, stdout_lines):
                self.returncode = returncode
                self.stdout = '\n'.join(stdout_lines)
                self.stderr = ''
        
        result = Result(return_code, output_lines)
        
        if result.returncode == 0:
            if progress_callback:
                # Parse output to provide step-by-step progress updates
                output_lines = (result.stdout + result.stderr).split('\n')
                step1_done = False
                step2_done = False
                step3_done = False
                
                for line in output_lines:
                    if "Character Extraction" in line and "Running" in line and not step1_done:
                        progress_callback("🔍 Step 1/3: Character Extraction in progress...")
                    elif "Character Extraction" in line and "completed" in line and not step1_done:
                        progress_callback("✅ Step 1/3: Character Extraction completed!")
                        step1_done = True
                    elif "Alias Building" in line and "Running" in line and not step2_done:
                        progress_callback("🔗 Step 2/3: Alias Building in progress...")
                    elif "Alias Building" in line and "completed" in line and not step2_done:
                        progress_callback("✅ Step 2/3: Alias Building completed!")
                        step2_done = True
                    elif "Interaction Extraction" in line and "Running" in line and not step3_done:
                        progress_callback("📊 Step 3/3: Interaction Analysis in progress...")
                    elif "Interaction Extraction" in line and "completed" in line and not step3_done:
                        progress_callback("✅ Step 3/3: Interaction Analysis completed!")
                        step3_done = True
                
                progress_callback("🎉 All steps completed successfully!")
                progress_callback("📊 Generated: Character list, Name aliases, Character interactions")
            return True
        else:
            if progress_callback:
                # Parse the output to provide more specific progress updates
                output_lines = (result.stdout + result.stderr).split('\n')
                for line in output_lines:
                    if "Character Extraction" in line and "Running" in line:
                        progress_callback("🔍 Step 1/3: Character Extraction in progress...")
                    elif "Alias Building" in line and "Running" in line:
                        progress_callback("🔗 Step 2/3: Alias Building in progress...")
                    elif "Interaction Extraction" in line and "Running" in line:
                        progress_callback("📊 Step 3/3: Interaction Analysis in progress...")
                    elif "ERROR" in line.upper() or "FAILED" in line.upper():
                        progress_callback(f"❌ Error: {line}")
                
                # Show final error
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                progress_callback(f"❌ Pipeline failed: {error_msg}")
            return False
            
    except Exception as e:
        if progress_callback:
            progress_callback(f"❌ Error running pipeline: {str(e)}")
        return False


def load_story_graph_data(story_index: int, dataset):
    """Load graph data for a story."""
    story = dataset[story_index]
    story_title = story.get("title", f"Story_{story_index}")
    
    # Locate pipeline outputs
    story_dir = Path("out") / f"{clean_dir_name(story_title)}_{story_index}"
    aliases_csv = story_dir / f"{story_title}_aliases.csv"
    interactions_csv = story_dir / f"{story_title}_interactions.csv"
    chars_csv = story_dir / f"{story_title}_chars.csv"
    
    return story_title, story_dir, aliases_csv, interactions_csv, chars_csv


def check_pipeline_outputs_exist(aliases_csv: Path, interactions_csv: Path) -> bool:
    """Check if pipeline outputs exist."""
    return aliases_csv.exists() and interactions_csv.exists()


def load_alias_mapping(aliases_csv_path: Path):
    """Load canonical->aliases and alias->canonical from CSV."""
    if not aliases_csv_path.exists():
        return {}, {}
        
    df = pd.read_csv(aliases_csv_path)
    can_to_aliases = {}
    alias_to_can = {}
    
    for _, row in df.iterrows():
        can = str(row['canonical_name'])
        al = str(row['alias_name']) if not (pd.isna(row['alias_name']) or row['alias_name'] == '') else None
        can_to_aliases.setdefault(can, [])
        if al:
            can_to_aliases[can].append(al)
            alias_to_can[al] = can
        # also map canonical to itself for resolution convenience
        alias_to_can.setdefault(can, can)
        
    return can_to_aliases, alias_to_can


def check_name_compatibility_with_alias_rules(victim_name: str, canonical_name: str) -> bool:
    """Check if victim_name would be grouped with canonical_name using alias_builder rules."""
    try:
        # Import the functions from alias_builder
        import sys
        import importlib.util
        
        # Load alias_builder module
        spec = importlib.util.spec_from_file_location("alias_builder", "alias_builder.py")
        alias_builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(alias_builder)
        
        # Use the same compatibility check as alias_builder
        return alias_builder.persons_compatible(victim_name, canonical_name)
    except Exception:
        # Fallback to simple substring matching if import fails
        def norm_alpha(s: str) -> str:
            return re.sub(r"[^A-Za-z]", "", (s or "")).lower()
        
        victim_norm = norm_alpha(victim_name)
        canonical_norm = norm_alpha(canonical_name)
        return victim_norm in canonical_norm or canonical_norm in victim_norm


def find_victims_from_csv(story_title: str, alias_to_can: dict, story_index: int = None) -> set:
    """Find victims from the curated CSV file."""
    victims_can = set()
    curated_path = Path("victim_list.csv")
    
    if not curated_path.exists():
        return victims_can

    def norm_alpha(s: str) -> str:
        return re.sub(r"[^A-Za-z]", "", (s or "")).lower()

    alias_norm_to_can = {norm_alpha(a): c for a, c in alias_to_can.items()}
    # Ensure canonicals also resolve to themselves
    for can in set(alias_to_can.values()):
        alias_norm_to_can.setdefault(norm_alpha(can), can)

    try:
        df = pd.read_csv(curated_path)
        row = df.loc[df['Story'] == story_title]
        if row.empty:
            return victims_can
            
        victims_field = str(row.iloc[0].get('Victim(s)', '') or '')

        def strip_brackets(text: str) -> str:
            return re.sub(r"\s*[\(\[].*?[\)\]]\s*", " ", text).strip()

        def parse_list_field(text: str) -> list:
            t = text.strip().strip('{}')
            parts = re.split(r"[;,]", t) if t else []
            cleaned = []
            for p in parts:
                p = p.strip()
                if not p:
                    continue
                p = strip_brackets(p)
                if p:
                    cleaned.append(p)
            return cleaned

        # Treat 'none' as no victims
        if victims_field.strip().lower() in {'none', '', '{none}', '—', '-'}:
            victim_aliases = []
        else:
            victim_aliases = parse_list_field(victims_field)

        for alias in victim_aliases:
            alias = strip_brackets(alias)
            key = norm_alpha(alias)
            can = alias_norm_to_can.get(key)
            if can:
                victims_can.add(can)
            else:
                # Try to resolve through metadata mapping if direct alias lookup fails
                if story_index is not None:
                    try:
                        from datasets import load_dataset
                        ds = load_dataset("kjgpta/WhoDunIt", split="train")
                        meta = ds[story_index].get("metadata", {})
                        if isinstance(meta, str):
                            try:
                                meta = json.loads(meta)
                            except Exception:
                                try:
                                    meta = ast.literal_eval(meta)
                                except Exception:
                                    meta = {}
                        
                        name_id_map = meta.get('name_id_map', {}) if isinstance(meta, dict) else {}
                        if name_id_map:
                            # Split victim name and try to map each part
                            victim_parts = alias.lower().split()
                            mapped_parts = []
                            for part in victim_parts:
                                # Look for the part in metadata keys
                                for meta_key, meta_value in name_id_map.items():
                                    if part == meta_key.lower():
                                        mapped_parts.append(meta_value)
                                        break
                                else:
                                    mapped_parts.append(part)  # Keep original if no mapping found
                            
                            # Try to find the mapped name in aliases
                            if mapped_parts:
                                mapped_name = ' '.join(mapped_parts)
                                mapped_key = norm_alpha(mapped_name)
                                mapped_can = alias_norm_to_can.get(mapped_key)
                                if mapped_can:
                                    victims_can.add(mapped_can)
                                else:
                                    # Try different combinations and capitalizations
                                    for possible_name in [mapped_name, mapped_name.title(), ' '.join(mapped_parts).title()]:
                                        possible_key = norm_alpha(possible_name)
                                        possible_can = alias_norm_to_can.get(possible_key)
                                        if possible_can:
                                            victims_can.add(possible_can)
                                            break
                                    else:
                                        # Use alias builder matching rules to check compatibility
                                        for canonical in set(alias_to_can.values()):
                                            if check_name_compatibility_with_alias_rules(mapped_name, canonical):
                                                victims_can.add(canonical)
                                                break
                    except Exception:
                        pass
                
    except Exception:
        pass

    return victims_can


def build_graph_from_interactions(interactions_csv: Path):
    """Build NetworkX graph from interactions CSV."""
    if not interactions_csv.exists():
        return nx.Graph()
        
    df = pd.read_csv(interactions_csv)
    G = nx.Graph()
    
    for _, row in df.iterrows():
        a = str(row['character1'])
        b = str(row['character2'])
        w = int(row['count'])
        if a == b:
            continue
        if G.has_edge(a, b):
            G[a][b]['weight'] += w
        else:
            G.add_edge(a, b, weight=w)
            
    return G


def compute_victim_connection_scores(G: nx.Graph, victims: set) -> dict:
    """Compute connection scores to victims for each node."""
    scores = {}
    for node in G.nodes():
        total = 0
        for v in victims:
            if G.has_edge(node, v):
                total += int(G[node][v].get('weight', 0))
        scores[node] = total
    return scores


def create_graph_plot(G: nx.Graph, victims: set, story_title: str):
    """Create and return a matplotlib figure for the graph."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    if len(G.nodes()) == 0:
        ax.text(0.5, 0.5, 'No character interactions found', 
                ha='center', va='center', transform=ax.transAxes, fontsize=16)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        return fig
    
    pos = nx.spring_layout(G, weight='weight', seed=42, k=1, iterations=50)
    
    # Calculate PageRank and victim connection scores
    pagerank = nx.pagerank(G, weight='weight') if len(G) > 0 else {}
    victim_scores = compute_victim_connection_scores(G, victims)
    
    # Node sizes: if victims known, use victim connection score; else use PageRank
    sizes = []
    if victims:
        for n in G.nodes():
            s = victim_scores.get(n, 0)
            sizes.append(max(300, 100 * (1 + s)))
    else:
        # Scale by PageRank
        pr_vals = [pagerank.get(n, 0.0) for n in G.nodes()]
        max_pr = max(pr_vals) if pr_vals else 1.0
        for n in G.nodes():
            pr = pagerank.get(n, 0.0) / max_pr if max_pr > 0 else 0.0
            sizes.append(max(300, 3000 * pr))

    # Colors: highlight victims in red; others by PageRank (blue intensity)
    pr_vals = [pagerank.get(n, 0.0) for n in G.nodes()]
    max_pr = max(pr_vals) if pr_vals else 1.0
    node_colors = []
    for n in G.nodes():
        if n in victims:
            node_colors.append('#d62728')  # red
        else:
            pr = pagerank.get(n, 0.0) / max_pr if max_pr > 0 else 0
            # map to light->dark blue
            node_colors.append((0.2, 0.2, 0.8 * (0.3 + 0.7 * pr)))

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=sizes, node_color=node_colors, 
                          alpha=0.85, linewidths=1, edgecolors='black', ax=ax)
    
    # Edge widths by weight
    weights = [G[e[0]][e[1]].get('weight', 1) for e in G.edges()]
    max_w = max(weights) if weights else 1
    scaled_w = [1 + 4 * (w / max_w) for w in weights]
    nx.draw_networkx_edges(G, pos, width=scaled_w, alpha=0.6, ax=ax)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', ax=ax)

    ax.set_title(f"{story_title} - Character Interaction Graph", fontsize=14, fontweight='bold')
    ax.axis('off')
    
    return fig


def get_node_metrics(G: nx.Graph, victims: set, story_dir: Path):
    """Get node metrics and optionally save to CSV."""
    if len(G.nodes()) == 0:
        return pd.DataFrame()
        
    pagerank = nx.pagerank(G, weight='weight')
    victim_scores = compute_victim_connection_scores(G, victims)
    
    rows = []
    for n in sorted(G.nodes()):
        rows.append({
            'node': n,
            'is_victim': int(n in victims),
            'pagerank': pagerank.get(n, 0.0),
            'victim_connection_weight': victim_scores.get(n, 0),
            'degree': int(G.degree(n, weight=None)),
            'strength': float(sum(G[n][nbr].get('weight', 0) for nbr in G.neighbors(n))),
        })
    
    df = pd.DataFrame(rows)
    
    # Save to CSV in story directory
    metrics_csv = story_dir / "node_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    
    return df


def analyze_character_metrics_for_culprits(story_index: int, dataset) -> str:
    """Analyze character graph metrics to provide culprit insights."""
    try:
        # Load story data
        story_title, story_dir, aliases_csv, interactions_csv, chars_csv = load_story_graph_data(
            story_index, dataset
        )
        
        # Check if pipeline outputs exist
        if not check_pipeline_outputs_exist(aliases_csv, interactions_csv):
            return "⚠️ **Graph data not available.** Run the character pipeline first in the Graph tab to enable graph-based insights."
        
        # Load graph data
        can_to_aliases, alias_to_can = load_alias_mapping(aliases_csv)
        G = build_graph_from_interactions(interactions_csv)
        victims = find_victims_from_csv(story_title, alias_to_can, story_index)
        
        if len(G.nodes()) == 0:
            return "⚠️ **No character interactions found** in the graph data."
        
        # Get metrics
        df_metrics = get_node_metrics(G, victims, story_dir)
        if df_metrics.empty:
            return "⚠️ **No character metrics available.**"
        
        # Sort by different criteria for analysis
        df_sorted = df_metrics.sort_values('pagerank', ascending=False)
        
        # Build insights text
        insights = []
        insights.append("## 📊 **Character Network Analysis**\n")
        
        # Victims section
        if victims:
            victim_list = sorted(list(victims))
            insights.append(f"**🔴 Identified Victims:** {', '.join(victim_list)}\n")
        else:
            insights.append("**⚪ No victims identified** in the graph data.\n")
        
        # Top characters by importance
        insights.append("### **🎯 Top Characters by Network Importance (PageRank):**")
        top_chars = df_sorted.head(5)
        for i, (_, row) in enumerate(top_chars.iterrows(), 1):
            victim_indicator = "🔴" if row['is_victim'] else "🔵"
            insights.append(
                f"{i}. {victim_indicator} **{row['node']}** — PageRank: {row['pagerank']:.3f}, "
                f"Connections: {int(row['degree'])}, Total Interactions: {int(row['strength'])}"
            )
        
        # Victim connection analysis
        if victims:
            insights.append("\n### **🔗 Connections to Victims Analysis:**")
            victim_connections = df_metrics[df_metrics['victim_connection_weight'] > 0].sort_values(
                'victim_connection_weight', ascending=False
            )
            
            if len(victim_connections) > 0:
                insights.append("*Characters with direct interactions to victims (suspicious connections):*")
                for i, (_, row) in enumerate(victim_connections.head(5).iterrows(), 1):
                    connection_strength = int(row['victim_connection_weight'])
                    strength_desc = "VERY HIGH" if connection_strength >= 10 else "HIGH" if connection_strength >= 5 else "MODERATE"
                    insights.append(
                        f"{i}. **{row['node']}** — {connection_strength} victim interactions ({strength_desc} connection)"
                    )
            else:
                insights.append("*No direct victim connections found in the data.*")
        
        # Culprit candidate recommendations
        insights.append("\n### **🕵️ Culprit Candidate Analysis:**")
        
        # High PageRank non-victims (influential characters)
        high_importance = df_metrics[
            (df_metrics['is_victim'] == 0) & 
            (df_metrics['pagerank'] > df_metrics['pagerank'].median())
        ].sort_values('pagerank', ascending=False)
        
        # Characters with victim connections (suspicious)
        victim_connected = df_metrics[
            (df_metrics['is_victim'] == 0) & 
            (df_metrics['victim_connection_weight'] > 0)
        ].sort_values('victim_connection_weight', ascending=False)
        
        # Well-connected characters (network hubs)
        highly_connected = df_metrics[
            (df_metrics['is_victim'] == 0) & 
            (df_metrics['degree'] >= df_metrics['degree'].quantile(0.7))
        ].sort_values('degree', ascending=False)
        
        insights.append("**🎯 PRIMARY SUSPECTS (Recommended focus):**")
        
        # Combine different criteria for recommendations
        primary_suspects = set()
        
        # Add top importance characters
        primary_suspects.update(high_importance.head(3)['node'].tolist())
        
        # Add characters with victim connections
        primary_suspects.update(victim_connected.head(3)['node'].tolist())
        
        # Add highly connected characters
        primary_suspects.update(highly_connected.head(2)['node'].tolist())
        
        if primary_suspects:
            for i, suspect in enumerate(sorted(primary_suspects)[:5], 1):
                suspect_data = df_metrics[df_metrics['node'] == suspect].iloc[0]
                
                reasons = []
                if suspect_data['pagerank'] > df_metrics['pagerank'].median():
                    reasons.append(f"High network importance (PageRank: {suspect_data['pagerank']:.3f})")
                if suspect_data['victim_connection_weight'] > 0:
                    reasons.append(f"Connected to victims ({int(suspect_data['victim_connection_weight'])} interactions)")
                if suspect_data['degree'] >= df_metrics['degree'].quantile(0.7):
                    reasons.append(f"Highly connected ({int(suspect_data['degree'])} different characters)")
                
                insights.append(f"{i}. **{suspect}** — {'; '.join(reasons)}")
        else:
            insights.append("*Unable to identify clear primary suspects from graph data.*")
        
        # Secondary considerations
        insights.append("\n**🔍 SECONDARY CONSIDERATIONS:**")
        
        # Characters with medium influence
        medium_importance = df_metrics[
            (df_metrics['is_victim'] == 0) & 
            (df_metrics['pagerank'] <= df_metrics['pagerank'].median()) &
            (df_metrics['pagerank'] > df_metrics['pagerank'].quantile(0.25))
        ].sort_values('pagerank', ascending=False)
        
        if len(medium_importance) > 0:
            insights.append("*Characters with moderate network influence:*")
            for suspect in medium_importance.head(3)['node'].tolist():
                suspect_data = df_metrics[df_metrics['node'] == suspect].iloc[0]
                insights.append(f"- **{suspect}** (PageRank: {suspect_data['pagerank']:.3f})")
        
        insights.append("\n---")
        insights.append("**💡 Analysis Tips:**")
        insights.append("- **High PageRank** suggests central importance in the story")
        insights.append("- **Victim connections** may indicate motive, opportunity, or involvement")
        insights.append("- **High connectivity** suggests active participation in events")
        insights.append("- **Consider combining** graph insights with story context and evidence")
        
        return "\n".join(insights)
        
    except Exception as e:
        return f"❌ **Error analyzing graph data:** {str(e)}"


def create_enhanced_prompt_with_graph_insights(original_prompt: str, story_text: str, story_title: str, 
                                             story_author: str, graph_insights: str) -> str:
    """Create an enhanced prompt that includes graph-based character insights."""
    
    # Insert graph insights before the main analysis instructions
    enhanced_prompt = original_prompt.replace(
        "Your task is to analyze this mystery story",
        f"""**ADDITIONAL CONTEXT - CHARACTER NETWORK ANALYSIS:**

{graph_insights}

---

Your task is to analyze this mystery story"""
    )
    
    # Add instruction to consider graph insights
    enhanced_prompt = enhanced_prompt.replace(
        "Focus on finding the actual perpetrator(s) based on evidence presented in the story.",
        """Focus on finding the actual perpetrator(s) based on evidence presented in the story.

**IMPORTANT:** Consider the character network analysis provided above. Pay special attention to:
- Characters identified as PRIMARY SUSPECTS based on network metrics
- Characters with high victim connections (suspicious relationships)
- Characters with high network importance (central to the story)
- Use this graph-based intelligence alongside story evidence for your analysis."""
    )
    
    return enhanced_prompt


def translate_and_match_culprit_names(predicted_culprits: List[str], actual_culprits: List[str], 
                                    story_index: int) -> dict:
    """Check if predicted and actual culprits would be grouped together using alias rules."""
    try:
        # Load story data for metadata
        dataset = load_dataset("kjgpta/WhoDunIt", split="train")
        story = dataset[story_index]
        
        # Get metadata for translation
        meta = story.get("metadata", {})
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except Exception:
                try:
                    meta = ast.literal_eval(meta)
                except Exception:
                    meta = {}
        
        name_id_map = meta.get('name_id_map', {}) if isinstance(meta, dict) else {}
        
        # Function to translate a name using metadata
        def translate_name_with_metadata(name: str) -> str:
            if not name_id_map:
                return name
                
            # Split name and try to map each part
            name_parts = name.lower().split()
            mapped_parts = []
            for part in name_parts:
                # Look for the part in metadata keys
                for meta_key, meta_value in name_id_map.items():
                    if part == meta_key.lower():
                        mapped_parts.append(meta_value)
                        break
                else:
                    mapped_parts.append(part)  # Keep original if no mapping found
            
            return ' '.join(mapped_parts)
        
        # Function to check if two names would be grouped together
        def would_be_grouped(name1: str, name2: str) -> bool:
            # Try direct alias builder compatibility
            if check_name_compatibility_with_alias_rules(name1, name2):
                return True
            
            # Try with metadata translation of name1
            translated_name1 = translate_name_with_metadata(name1)
            if translated_name1 != name1:
                if check_name_compatibility_with_alias_rules(translated_name1, name2):
                    return True
            
            # Try with metadata translation of name2
            translated_name2 = translate_name_with_metadata(name2)
            if translated_name2 != name2:
                if check_name_compatibility_with_alias_rules(name1, translated_name2):
                    return True
            
            # Try with both translated
            if translated_name1 != name1 and translated_name2 != name2:
                if check_name_compatibility_with_alias_rules(translated_name1, translated_name2):
                    return True
            
            return False
        
        # Match predicted culprits against actual culprits
        matched_pairs = []
        extra_culprits = []
        translation_notes = []
        
        # For each predicted culprit, try to match with any actual culprit
        for pred_culprit in predicted_culprits:
            matched = False
            for actual_culprit in actual_culprits:
                if would_be_grouped(pred_culprit, actual_culprit):
                    matched_pairs.append({
                        'predicted': pred_culprit,
                        'actual': actual_culprit,
                        'predicted_translated': translate_name_with_metadata(pred_culprit),
                        'actual_translated': translate_name_with_metadata(actual_culprit)
                    })
                    
                    # Add translation note if there was a translation
                    pred_trans = translate_name_with_metadata(pred_culprit)
                    actual_trans = translate_name_with_metadata(actual_culprit)
                    
                    if pred_trans != pred_culprit or actual_trans != actual_culprit:
                        translation_notes.append(
                            f"'{pred_culprit}' → '{pred_trans}' matches '{actual_culprit}' → '{actual_trans}'"
                        )
                    else:
                        translation_notes.append(f"'{pred_culprit}' matches '{actual_culprit}' (direct)")
                    
                    matched = True
                    break
            
            if not matched:
                extra_culprits.append(pred_culprit)
                pred_trans = translate_name_with_metadata(pred_culprit)
                if pred_trans != pred_culprit:
                    translation_notes.append(f"'{pred_culprit}' → '{pred_trans}' (no match found)")
                else:
                    translation_notes.append(f"'{pred_culprit}' (no match found)")
        
        # Find which actual culprits were correctly identified
        matched_actual_culprits = [pair['actual'] for pair in matched_pairs]
        missed_culprits = [actual for actual in actual_culprits if actual not in matched_actual_culprits]
        
        return {
            'matched_pairs': matched_pairs,
            'matched_culprits': [pair['actual'] for pair in matched_pairs],
            'missed_culprits': missed_culprits,
            'extra_culprits': extra_culprits,
            'translation_notes': translation_notes
        }
        
    except Exception as e:
        return {
            'matched_pairs': [],
            'matched_culprits': [],
            'missed_culprits': actual_culprits,
            'extra_culprits': predicted_culprits,
            'translation_notes': [f"Error in translation: {str(e)}"]
        }


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

    # Load priority stories
    if 'priority_df' not in st.session_state:
        st.session_state.priority_df = load_priority_stories()

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
    max_tokens = st.sidebar.slider("Max Tokens", 100, 120000, 10000, 50)
    
    # Graph-based analysis option
    st.sidebar.subheader("Graph-Enhanced Analysis")
    use_graph_insights = st.sidebar.checkbox("Use character graph insights", value=False, 
                                            help="Include character network analysis in the AI prediction")
    
    if use_graph_insights:
        st.sidebar.markdown("📊 **Graph insights will provide:**")
        st.sidebar.markdown("- Character importance (PageRank)")
        st.sidebar.markdown("- Connections to victims") 
        st.sidebar.markdown("- Interaction patterns")
        st.sidebar.markdown("- Culprit candidate recommendations")

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

    # Enhanced Story selection
    st.sidebar.subheader("📚 Story Selection")
    
    # Create story options with priorities
    story_options, story_indices = create_story_selection_options(st.session_state.dataset, st.session_state.priority_df)
    
    selected_option = st.sidebar.selectbox(
        "Choose a story for analysis:",
        story_options,
        help="This story will be used in both AI Detective and Character Graph tabs"
    )
    
    # Get the actual story index
    selected_idx = story_options.index(selected_option)
    story_index = story_indices[selected_idx]
    

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

    # Create tabs for different features
    tab1, tab2 = st.tabs(["🔍 AI Detective", "📊 Character Graph"])
    
    with tab1:
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

            # Graph insights display (if enabled)
            if use_graph_insights:
                with st.expander("📊 Character Network Insights", expanded=False):
                    graph_insights = analyze_character_metrics_for_culprits(story_index, st.session_state.dataset)
                    st.markdown(graph_insights)

        # Generate button
            if st.button("🔍 Generate Analysis", type="primary", width="stretch"):
            # Clear previous results
            st.session_state.pop('current_prediction', None)

            # Create placeholders for streaming
            prediction_placeholder = st.empty()
            judging_placeholder = st.empty()

            # Show loading message
            prediction_placeholder.markdown("**🔄 Generating prediction...**")
            judging_placeholder.markdown("**🔄 Preparing evaluation...**")

                # Prepare prompt (enhanced with graph insights if enabled)
                detection_prompt = st.session_state.custom_detection_prompt if edit_prompts else None
                
                if use_graph_insights:
                    # Get graph insights for prompt enhancement
                    graph_insights = analyze_character_metrics_for_culprits(story_index, st.session_state.dataset)
                    
                    if not graph_insights.startswith("⚠️") and not graph_insights.startswith("❌"):
                        # Create enhanced prompt with graph insights
                        base_prompt = detection_prompt or st.session_state.detector.create_prompt(
                            current_story['text'], current_story['title'], current_story['author']
                        )
                        detection_prompt = create_enhanced_prompt_with_graph_insights(
                            base_prompt, current_story['text'], current_story['title'], 
                            current_story['author'], graph_insights
                        )
                        
                        # Show that graph insights are being used
                        st.info("🔗 **Graph-enhanced analysis activated** - Using character network insights for better prediction")
                    else:
                        st.warning("⚠️ Graph insights unavailable - using standard analysis")

            # Process story with streaming
            try:
                prediction = st.session_state.detector.process_story_streaming(
                    current_story,
                    prediction_placeholder,
                    judging_placeholder,
                        custom_prompt=detection_prompt,
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
                        
                            for cand_name, score in cand_scores.items():
                                # Enhanced word-based matching
                                culprit_words = set(culprit_name.lower().split())
                                cand_words = set(cand_name.lower().split())
                                
                                # Match if:
                                # 1. Exact match
                                # 2. Candidate is substring of culprit (e.g., "diaz" in "diaz fanning")
                                # 3. Culprit is substring of candidate (e.g., "smith" matches "john smith")
                                # 4. Any word overlap between culprit and candidate
                                is_match = (
                                    culprit_name.lower() == cand_name.lower() or
                                    culprit_name.lower() in cand_name.lower() or
                                    cand_name.lower() in culprit_name.lower() or
                                    len(culprit_words & cand_words) > 0  # Word overlap
                                )
                                
                                if is_match:
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
                                culprit_words_sets = [set(culprit_name.lower().split()) for culprit_name in culprit_names_lower]
                                cand_words = set(cand_name.lower().split())
                                
                                is_culprit = any(
                                    culprit_name.lower() == cand_name.lower() or
                                    culprit_name.lower() in cand_name.lower() or
                                    cand_name.lower() in culprit_name.lower() or
                                    len(culprit_words & cand_words) > 0
                                    for culprit_name, culprit_words in zip(culprit_names_lower, culprit_words_sets)

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
                # Add a clear results button
                if st.button("🗑️ Clear Results", help="Clear previous prediction to run a fresh analysis"):
                    del st.session_state.current_prediction
                    st.rerun()
            pred = st.session_state.current_prediction

            st.markdown("---")
                st.markdown("**📊 Analysis Results:**")

                # Enhanced matching with metadata translation
                translation_results = translate_and_match_culprit_names(
                    pred.predicted_culprits, pred.actual_culprits, story_index
                )

                # Status indicator with enhanced matching
                correct_matches = len(translation_results['matched_culprits'])
                total_actual = len(pred.actual_culprits)
                enhanced_score = (correct_matches / total_actual * 100) if total_actual > 0 else 0

                if correct_matches == total_actual and len(translation_results['extra_culprits']) == 0:
                    st.success(f"✅ **PERFECT MATCH** (Enhanced Score: {enhanced_score:.0f}%)")
                elif correct_matches > 0:
                    st.warning(f"🟡 **PARTIAL MATCH** (Enhanced Score: {enhanced_score:.0f}%)")
            else:
                    st.error(f"❌ **NO MATCH** (Enhanced Score: {enhanced_score:.0f}%)")

                # Detailed Translation Results
                with st.expander("🔍 **Name Translation & Matching Analysis**", expanded=True):
                    st.markdown("### 📝 **AI Predictions vs Actual Culprits:**")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🤖 AI Predicted:**")
                        for pred_culprit in pred.predicted_culprits:
                            st.markdown(f"• **{pred_culprit}**")
                    
                    with col2:
                        st.markdown("**🎯 Actual Culprits:**")
                        for actual_culprit in pred.actual_culprits:
                            st.markdown(f"• **{actual_culprit}**")
                    
                    if translation_results['matched_pairs']:
                        st.markdown("### ✅ **Successful Matches:**")
                        for pair in translation_results['matched_pairs']:
                            if (pair['predicted_translated'] != pair['predicted'] or 
                                pair['actual_translated'] != pair['actual']):
                                st.markdown(
                                    f"• **{pair['predicted']}** → *{pair['predicted_translated']}* ≈ "
                                    f"*{pair['actual_translated']}* ← **{pair['actual']}**"
                                )
                            else:
                                st.markdown(f"• **{pair['predicted']}** ≈ **{pair['actual']}** (direct match)")
                    
                    if translation_results['translation_notes']:
                        st.markdown("### 📋 **Matching Details:**")
                        for note in translation_results['translation_notes']:
                            st.markdown(f"• {note}")
                    
                    st.markdown("### ⚖️ **Summary:**")
                    if translation_results['matched_culprits']:
                        st.markdown("**✅ Correctly Identified:**")
                        for matched in translation_results['matched_culprits']:
                            st.markdown(f"• **{matched}**")
                    
                    if translation_results['missed_culprits']:
                        st.markdown("**❌ Missed Culprits:**")
                        for missed in translation_results['missed_culprits']:
                            st.markdown(f"• **{missed}**")
                    
                    if translation_results['extra_culprits']:
                        st.markdown("**🚫 Incorrect Predictions:**")
                        for extra in translation_results['extra_culprits']:
                            st.markdown(f"• **{extra}**")

                # Metrics comparison
                col_a, col_b, col_c = st.columns(3)
            with col_a:
                    st.metric("AI Confidence", f"{pred.confidence}%")
            with col_b:
                    st.metric("Original Score", f"{pred.match_score}%")
                with col_c:
                    st.metric("Enhanced Score", f"{enhanced_score:.0f}%")

    with tab2:
        st.header("📊 Character Interaction Analysis")
        
        # Use the same story selection from sidebar
        graph_story_index = story_index
        
        # Load story data
        story_title, story_dir, aliases_csv, interactions_csv, chars_csv = load_story_graph_data(
            graph_story_index, st.session_state.dataset
        )
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("🛠️ Pipeline Controls")
            
            # Check if pipeline outputs exist
            outputs_exist = check_pipeline_outputs_exist(aliases_csv, interactions_csv)
            
            if outputs_exist:
                st.success("✅ Pipeline outputs found")
                st.markdown(f"**Story:** {story_title}")
                st.markdown(f"**Output directory:** `{story_dir}`")
                
                # Show file info
                if aliases_csv.exists():
                    aliases_df = pd.read_csv(aliases_csv)
                    st.markdown(f"**Characters:** {len(aliases_df)} canonical names")
                    
                if interactions_csv.exists():
                    interactions_df = pd.read_csv(interactions_csv)
                    st.markdown(f"**Interactions:** {len(interactions_df)} pairs")
            else:
                st.warning("⚠️ Character data not found")
                st.markdown("**What the pipeline does:**")
                st.markdown("• 🔍 **Extract characters** from story text using AI")
                st.markdown("• 🔗 **Build aliases** for different name variations")  
                st.markdown("• 📊 **Analyze interactions** between characters")
                st.markdown("• 🎯 **Identify victims** using curated data")
                st.markdown("\n👇 **Click below to start the process**")
            
            # Pipeline control buttons
            if st.button("📊 Collect Character Data & Connections", type="primary", key="run_pipeline"):
                # Create expandable progress section
                with st.expander("📋 Pipeline Progress", expanded=True):
                    # Overall progress
                    overall_progress = st.progress(0)
                    overall_status = st.empty()
                    
                    # Step progress bars
                    st.markdown("### 📊 Step Progress")
                    
                    # Step 1: Character Extraction
                    step1_col1, step1_col2 = st.columns([3, 1])
                    with step1_col1:
                        st.markdown("🔍 **Step 1:** Character Extraction")
                        step1_progress = st.progress(0)
                    with step1_col2:
                        step1_status = st.empty()
                        step1_status.markdown("⏳ Waiting...")
                    
                    # Step 2: Alias Building  
                    step2_col1, step2_col2 = st.columns([3, 1])
                    with step2_col1:
                        st.markdown("🔗 **Step 2:** Alias Building")
                        step2_progress = st.progress(0)
                    with step2_col2:
                        step2_status = st.empty()
                        step2_status.markdown("⏳ Waiting...")
                    
                    # Step 3: Interaction Extraction
                    step3_col1, step3_col2 = st.columns([3, 1])
                    with step3_col1:
                        st.markdown("📊 **Step 3:** Interaction Analysis")
                        step3_progress = st.progress(0)
                    with step3_col2:
                        step3_status = st.empty()
                        step3_status.markdown("⏳ Waiting...")
                    
                    # Detailed logs
                    st.markdown("### 📝 Detailed Logs")
                    progress_placeholder = st.empty()
                    
                    def update_progress(msg):
                        # Accumulate messages in session state
                        if 'pipeline_logs' not in st.session_state:
                            st.session_state.pipeline_logs = []
                        st.session_state.pipeline_logs.append(msg)
                        
                        # Update overall status
                        overall_status.markdown(f"**Current:** {msg}")
                        
                        # Update step progress based on message content
                        if "Step 1" in msg or "Character Extraction" in msg or "Extracting character" in msg:
                            if "starting" in msg.lower() or "step 1" in msg.lower():
                                step1_progress.progress(0.3)
                                step1_status.markdown("🔄 Running...")
                                overall_progress.progress(0.1)
                            elif "completed" in msg.lower() or "✅" in msg:
                                step1_progress.progress(1.0)
                                step1_status.markdown("✅ Done")
                                overall_progress.progress(0.33)
                        
                        elif "Step 2" in msg or "Alias Building" in msg or "alias" in msg.lower():
                            if "starting" in msg.lower() or "step 2" in msg.lower():
                                step2_progress.progress(0.3)
                                step2_status.markdown("🔄 Running...")
                                overall_progress.progress(0.4)
                            elif "completed" in msg.lower() or "✅" in msg:
                                step2_progress.progress(1.0)
                                step2_status.markdown("✅ Done")
                                overall_progress.progress(0.66)
                        
                        elif "Step 3" in msg or "Interaction" in msg or "interaction" in msg.lower():
                            if "starting" in msg.lower() or "step 3" in msg.lower():
                                step3_progress.progress(0.3)
                                step3_status.markdown("🔄 Running...")
                                overall_progress.progress(0.7)
                            elif "completed" in msg.lower() or "✅" in msg:
                                step3_progress.progress(1.0)
                                step3_status.markdown("✅ Done")
                                overall_progress.progress(1.0)
                        
                        # Show all logs
                        log_text = "\n".join([f"• {log}" for log in st.session_state.pipeline_logs[-10:]])  # Show last 10 messages
                        progress_placeholder.markdown(f"```\n{log_text}\n```")
                    
                    # Clear previous logs
                    st.session_state.pipeline_logs = []
                    
                    success = run_pipeline_for_story(graph_story_index, update_progress)
                    
                    if success:
                        # Complete all progress bars
                        overall_progress.progress(1.0)
                        step1_progress.progress(1.0)
                        step2_progress.progress(1.0) 
                        step3_progress.progress(1.0)
                        step1_status.markdown("✅ Done")
                        step2_status.markdown("✅ Done")
                        step3_status.markdown("✅ Done")
                        overall_status.markdown("**Status:** 🎉 All steps completed!")
                        
                        st.success("✅ Character data collection completed successfully!")
                        st.balloons()  # Celebration animation
                        st.rerun()
                    else:
                        # Show failed status
                        overall_status.markdown("**Status:** ❌ Pipeline failed")
                        st.error("❌ Character data collection failed. Check the logs above.")
            
            if outputs_exist and st.button("🔄 Refresh Graph", key="refresh_graph"):
                st.rerun()
        
        with col1:
            st.subheader("🕸️ Character Interaction Graph")
            
            if outputs_exist:
                try:
                    # Load data
                    can_to_aliases, alias_to_can = load_alias_mapping(aliases_csv)
                    G = build_graph_from_interactions(interactions_csv)
                    victims = find_victims_from_csv(story_title, alias_to_can, graph_story_index)
                    
                    if len(G.nodes()) == 0:
                        st.warning("No character interactions found for this story.")
                    else:
                        # Create and display the graph
                        fig = create_graph_plot(G, victims, story_title)
                        st.pyplot(fig)
                        plt.close(fig)  # Clean up to avoid memory issues
                        
                        # Graph statistics
                        st.markdown("### 📈 Graph Statistics")
                        
                        total_nodes = len(G.nodes())
                        total_edges = len(G.edges())
                        total_victims = len(victims)
                        
                        col_stats1, col_stats2, col_stats3 = st.columns(3)
                        with col_stats1:
                            st.metric("Characters", total_nodes)
                        with col_stats2:
                            st.metric("Interactions", total_edges)
                        with col_stats3:
                            st.metric("Victims", total_victims)
                        
                        # Victims list
                        if victims:
                            st.markdown("**🔴 Identified Victims:**")
                            for victim in sorted(victims):
                                st.markdown(f"- {victim}")
                        
                        # Node metrics table
                        st.markdown("### 📋 Character Metrics")
                        
                        df_metrics = get_node_metrics(G, victims, story_dir)
                        if not df_metrics.empty:
                            # Sort by PageRank descending
                            df_display = df_metrics.sort_values('pagerank', ascending=False)
                            
                            # Color code the victims
                            def highlight_victims(row):
                                if row['is_victim'] == 1:
                                    return ['background-color: #ffcccc'] * len(row)
                                return [''] * len(row)
                            
                            styled_df = df_display.style.apply(highlight_victims, axis=1)
                            st.dataframe(styled_df, width="stretch")
                            
                            st.markdown("**Legend:**")
                            st.markdown("- 🔴 Red background = Victim")
                            st.markdown("- **PageRank**: Importance based on network position")
                            st.markdown("- **Victim Connection Weight**: Total interaction strength with victims")
                            st.markdown("- **Degree**: Number of different characters interacted with")
                            st.markdown("- **Strength**: Total interaction weight")
                        
                except Exception as e:
                    st.error(f"Error creating graph: {str(e)}")
                    st.markdown("Please check that the pipeline has completed successfully.")
            else:
                st.info("👆 Run the pipeline first to generate the character interaction graph.")

    # Footer with model info and story index
    st.markdown("---")
    footer_col1, footer_col2 = st.columns([3, 1])
    
    with footer_col1:
    st.markdown(
        f"**Model:** {st.session_state.detector.model_name} | **Total Stories:** {len(st.session_state.dataset)} | **Temperature:** {temperature} | **Max Tokens:** {max_tokens}")
    
    with footer_col2:
        st.markdown(f"**Story Index:** {story_index}")


if __name__ == "__main__":
    main()
