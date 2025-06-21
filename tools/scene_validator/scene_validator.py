#!/usr/bin/env python3
"""
SceneValidator - A tool for validating scene structure and continuity in media productions.

This module provides functionality to analyze scripts and storyboards for continuity errors,
structural issues, and other common problems in media production.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import yaml
from datetime import datetime

# Optional imports - will be loaded dynamically when needed
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from google.cloud import vision
    VISION_API_AVAILABLE = True
except ImportError:
    VISION_API_AVAILABLE = False

try:
    import firebase_admin
    from firebase_admin import firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("SceneValidator")


class ValidationError:
    """Represents a validation error found in a script or storyboard."""
    
    SEVERITY_LOW = "low"
    SEVERITY_MEDIUM = "medium"
    SEVERITY_HIGH = "high"
    
    def __init__(self, 
                 error_type: str, 
                 message: str, 
                 location: Dict[str, Any], 
                 severity: str = SEVERITY_MEDIUM,
                 suggestions: List[str] = None):
        """
        Initialize a validation error.
        
        Args:
            error_type: Type of error (e.g., "continuity", "structure")
            message: Human-readable error message
            location: Dictionary with information about error location
            severity: Error severity level
            suggestions: List of suggested fixes
        """
        self.error_type = error_type
        self.message = message
        self.location = location
        self.severity = severity
        self.suggestions = suggestions or []
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary for serialization."""
        return {
            "error_type": self.error_type,
            "message": self.message,
            "location": self.location,
            "severity": self.severity,
            "suggestions": self.suggestions,
            "timestamp": self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValidationError':
        """Create error from dictionary."""
        error = cls(
            error_type=data["error_type"],
            message=data["message"],
            location=data["location"],
            severity=data.get("severity", cls.SEVERITY_MEDIUM),
            suggestions=data.get("suggestions", [])
        )
        error.timestamp = data.get("timestamp", datetime.now().isoformat())
        return error


class SceneValidator:
    """Main class for validating scenes in scripts and storyboards."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the validator with configuration.
        
        Args:
            config_path: Path to configuration file
        """
        self.config = self._load_config(config_path)
        self.script_data = None
        self.storyboard_data = None
        self.validation_results = None
        self._initialize_apis()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "gemini": {
                "api_key": os.environ.get("GEMINI_API_KEY", ""),
                "model": "gemini-pro",
                "max_tokens": 8192
            },
            "vision_api": {
                "enabled": False
            },
            "firebase": {
                "enabled": False,
                "collection": "validation_results"
            },
            "validation": {
                "strict_mode": False,
                "rules": {
                    "structure": True,
                    "continuity": True,
                    "character": True,
                    "setting": True,
                    "timeline": True,
                    "technical": True
                }
            },
            "reporting": {
                "include_suggestions": True,
                "max_suggestions_per_error": 3,
                "default_format": "json"
            }
        }
        
        if not config_path:
            logger.info("No config path provided, using default configuration")
            return default_config
        
        try:
            if config_path.endswith(".json"):
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
            elif config_path.endswith((".yaml", ".yml")):
                with open(config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
            else:
                logger.warning(f"Unsupported config format: {config_path}")
                return default_config
                
            # Merge configs, preferring user values
            for key, value in user_config.items():
                if key in default_config and isinstance(value, dict):
                    default_config[key].update(value)
                else:
                    default_config[key] = value
                    
            return default_config
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return default_config
    
    def _initialize_apis(self):
        """Initialize external APIs based on configuration."""
        # Initialize Gemini if available
        if GEMINI_AVAILABLE and self.config["gemini"]["api_key"]:
            try:
                genai.configure(api_key=self.config["gemini"]["api_key"])
                logger.info("Gemini API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini API: {e}")
        
        # Initialize Vision API if enabled and available
        if VISION_API_AVAILABLE and self.config["vision_api"]["enabled"]:
            try:
                self.vision_client = vision.ImageAnnotatorClient()
                logger.info("Vision API initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Vision API: {e}")
        
        # Initialize Firebase if enabled and available
        if FIREBASE_AVAILABLE and self.config["firebase"]["enabled"]:
            try:
                if not firebase_admin._apps:
                    firebase_admin.initialize_app()
                self.db = firestore.client()
                logger.info("Firebase initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Firebase: {e}")
    
    def load_script(self, script_path: str) -> bool:
        """
        Load a script for validation.
        
        Args:
            script_path: Path to script file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Basic implementation - will need to handle different formats
            extension = os.path.splitext(script_path)[1].lower()
            
            if extension == '.json':
                with open(script_path, 'r') as f:
                    self.script_data = json.load(f)
            elif extension == '.fountain':
                # TODO: Implement Fountain format parser
                logger.warning("Fountain format support not fully implemented")
                with open(script_path, 'r') as f:
                    self.script_data = self._parse_fountain(f.read())
            elif extension in ['.fdx', '.xml']:
                # TODO: Implement Final Draft XML parser
                logger.warning("Final Draft format support not fully implemented")
                with open(script_path, 'r') as f:
                    self.script_data = self._parse_final_draft(f.read())
            elif extension in ['.txt', '.md']:
                # Simple text format
                with open(script_path, 'r') as f:
                    self.script_data = self._parse_text_script(f.read())
            else:
                logger.error(f"Unsupported script format: {extension}")
                return False
                
            logger.info(f"Script loaded from {script_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading script from {script_path}: {e}")
            return False
    
    def _parse_fountain(self, content: str) -> Dict[str, Any]:
        """Parse Fountain format script."""
        # Placeholder implementation
        scenes = []
        current_scene = None
        
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("INT.") or line.startswith("EXT."):
                # Scene heading
                if current_scene:
                    scenes.append(current_scene)
                current_scene = {
                    "heading": line,
                    "description": "",
                    "characters": [],
                    "dialogue": []
                }
            elif current_scene:
                # Add content to current scene
                current_scene["description"] += line + "\n"
        
        if current_scene:
            scenes.append(current_scene)
            
        return {"scenes": scenes}
    
    def _parse_final_draft(self, content: str) -> Dict[str, Any]:
        """Parse Final Draft XML format."""
        # Placeholder implementation
        return {"scenes": []}
    
    def _parse_text_script(self, content: str) -> Dict[str, Any]:
        """Parse simple text format script."""
        scenes = []
        current_scene = None
        
        for line in content.splitlines():
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("INT.") or line.startswith("EXT."):
                # Scene heading
                if current_scene:
                    scenes.append(current_scene)
                current_scene = {
                    "heading": line,
                    "description": "",
                    "characters": [],
                    "dialogue": []
                }
            elif current_scene:
                # Simple heuristic: all caps might be character names
                if line.isupper():
                    character = line
                    if character not in current_scene["characters"]:
                        current_scene["characters"].append(character)
                else:
                    current_scene["description"] += line + "\n"
        
        if current_scene:
            scenes.append(current_scene)
            
        return {"scenes": scenes}
        
    def load_storyboard(self, storyboard_path: str) -> bool:
        """
        Load a storyboard for validation.
        
        Args:
            storyboard_path: Path to storyboard directory or zip file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if it's a directory or zip file
            if os.path.isdir(storyboard_path):
                self.storyboard_data = self._load_storyboard_from_directory(storyboard_path)
            elif storyboard_path.endswith('.zip'):
                self.storyboard_data = self._load_storyboard_from_zip(storyboard_path)
            elif storyboard_path.endswith('.json'):
                # Load storyboard metadata directly from JSON
                with open(storyboard_path, 'r') as f:
                    self.storyboard_data = json.load(f)
            else:
                logger.error(f"Unsupported storyboard format: {storyboard_path}")
                return False
            
            logger.info(f"Storyboard loaded from {storyboard_path}")
            return True
        except Exception as e:
            logger.error(f"Error loading storyboard from {storyboard_path}: {e}")
            return False
    
    def _load_storyboard_from_directory(self, directory: str) -> Dict[str, Any]:
        """Load storyboard from directory of images with metadata."""
        # Placeholder implementation
        storyboard = {
            "frames": []
        }
        
        # Look for metadata file first
        metadata_path = os.path.join(directory, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                storyboard.update(json.load(f))
        
        # Process image files
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                frame_path = os.path.join(directory, filename)
                
                # Look for frame-specific metadata
                frame_metadata_path = os.path.join(
                    directory, 
                    os.path.splitext(filename)[0] + ".json"
                )
                
                frame_data = {
                    "path": frame_path,
                    "filename": filename
                }
                
                if os.path.exists(frame_metadata_path):
                    with open(frame_metadata_path, 'r') as f:
                        frame_data.update(json.load(f))
                
                storyboard["frames"].append(frame_data)
        
        return storyboard
    
    def _load_storyboard_from_zip(self, zip_path: str) -> Dict[str, Any]:
        """Load storyboard from zip file."""
        # Placeholder implementation
        import zipfile
        import tempfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return self._load_storyboard_from_directory(temp_dir)
    
    def validate(self) -> Dict[str, Any]:
        """
        Perform validation on loaded content.
        
        Returns:
            Dict: Validation results
        """
        # Reset validation results
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "errors": [],
            "warnings": [],
            "info": [],
            "summary": {
                "total_errors": 0,
                "total_warnings": 0,
                "error_types": {},
                "status": "success"
            }
        }
        
        try:
            # Validate script if loaded
            if self.script_data:
                self._validate_script()
            
            # Validate storyboard if loaded
            if self.storyboard_data:
                self._validate_storyboard()
            
            # Validate continuity between script and storyboard if both loaded
            if self.script_data and self.storyboard_data:
                self._validate_continuity()
            
            # Use Gemini for enhanced validation if available
            if GEMINI_AVAILABLE and self.config["gemini"]["api_key"]:
                self._enhance_validation_with_gemini()
            
            # Update summary
            self.validation_results["summary"]["total_errors"] = len(self.validation_results["errors"])
            self.validation_results["summary"]["total_warnings"] = len(self.validation_results["warnings"])
            
            # Count error types
            error_types = {}
            for error in self.validation_results["errors"]:
                error_type = error["error_type"]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            self.validation_results["summary"]["error_types"] = error_types
            
            # Set status based on errors
            if len(self.validation_results["errors"]) > 0:
                self.validation_results["summary"]["status"] = "failed"
            
            logger.info(f"Validation completed with {len(self.validation_results['errors'])} errors and {len(self.validation_results['warnings'])} warnings")
            return self.validation_results
            
        except Exception as e:
            logger.error(f"Error during validation: {e}")
            self.validation_results["summary"]["status"] = "error"
            self.validation_results["summary"]["error_message"] = str(e)
            return self.validation_results
    
    def _validate_script(self):
        """Validate script structure and content."""
        if not self.script_data or "scenes" not in self.script_data:
            self._add_error(
                "structure", 
                "Invalid script format - missing scenes", 
                {"file": "script", "position": "global"},
                ValidationError.SEVERITY_HIGH
            )
            return
        
        # Validate scene structure
        self._validate_scene_structure()
        
        # Validate characters
        self._validate_characters()
        
        # Validate settings and locations
        self._validate_settings()
        
        # Validate timeline
        self._validate_timeline()
    
    def _validate_scene_structure(self):
        """Validate the structure of each scene."""
        for i, scene in enumerate(self.script_data.get("scenes", [])):
            # Check for scene heading
            if not scene.get("heading"):
                self._add_error(
                    "structure", 
                    f"Scene {i+1} is missing a heading", 
                    {"scene": i+1},
                    ValidationError.SEVERITY_MEDIUM,
                    ["Add a clear INT./EXT. scene heading"]
                )
            
            # Check for scene description
            if not scene.get("description"):
                self._add_warning(
                    "structure", 
                    f"Scene {i+1} has no description", 
                    {"scene": i+1},
                    ["Add a description of the scene setting and action"]
                )
            
            # Check for characters without dialogue
            if scene.get("characters") and not scene.get("dialogue"):
                self._add_warning(
                    "structure", 
                    f"Scene {i+1} has characters but no dialogue", 
                    {"scene": i+1, "characters": scene.get("characters", [])},
                    ["Add dialogue for the characters", "Remove unused characters"]
                )
    
    def _validate_characters(self):
        """Validate character consistency across scenes."""
        # Track characters and their appearances
        characters = {}
        
        # First pass: collect all characters
        for i, scene in enumerate(self.script_data.get("scenes", [])):
            for character in scene.get("characters", []):
                if character not in characters:
                    characters[character] = []
                characters[character].append(i+1)
        
        # Second pass: check for character consistency issues
        all_scenes = len(self.script_data.get("scenes", []))
        for character, appearances in characters.items():
            # Check for one-off characters (potential continuity issues)
            if len(appearances) == 1 and all_scenes > 3:
                self._add_info(
                    "character", 
                    f"Character '{character}' only appears in scene {appearances[0]}", 
                    {"character": character, "appearances": appearances},
                    ["Verify if this character is important to the story", 
                     "Consider expanding their role or removing them"]
                )
            
            # Check for large gaps in character appearances
            if len(appearances) > 1:
                gaps = [appearances[i+1] - appearances[i] for i in range(len(appearances)-1)]
                max_gap = max(gaps)
                if max_gap > 5:  # Arbitrary threshold
                    gap_start = appearances[gaps.index(max_gap)]
                    gap_end = appearances[gaps.index(max_gap) + 1]
                    self._add_warning(
                        "character", 
                        f"Character '{character}' has a large gap in appearances between scenes {gap_start} and {gap_end}", 
                        {"character": character, "gap_start": gap_start, "gap_end": gap_end},
                        ["Verify character's absence makes sense for the story",
                         "Consider adding a brief appearance to maintain continuity"]
                    )
    
    def _validate_settings(self):
        """Validate consistency of settings and locations."""
        settings = {}
        
        # Track settings and extract location information
        for i, scene in enumerate(self.script_data.get("scenes", [])):
            heading = scene.get("heading", "")
            
            # Basic regex-like extraction of location
            location = None
            if "INT." in heading:
                location_part = heading.split("INT.")[1].strip()
                location = location_part.split("-")[0].strip() if "-" in location_part else location_part
            elif "EXT." in heading:
                location_part = heading.split("EXT.")[1].strip()
                location = location_part.split("-")[0].strip() if "-" in location_part else location_part
            
            if location:
                if location not in settings:
                    settings[location] = []
                settings[location].append(i+1)
        
        # Check for potential setting inconsistencies
        for i, scene in enumerate(self.script_data.get("scenes", [])):
            heading = scene.get("heading", "")
            description = scene.get("description", "")
            
            # Look for location mentions in description that don't match heading
            for location in settings.keys():
                if location not in heading and location in description:
                    self._add_warning(
                        "setting", 
                        f"Scene {i+1} description mentions location '{location}' not found in heading", 
                        {"scene": i+1, "location": location, "heading": heading},
                        ["Update scene heading to match location in description",
                         "Clarify location reference in description"]
                    )
    
    def _validate_timeline(self):
        """Validate timeline consistency."""
        # Look for time indicators in scene headings
        time_indicators = ["MORNING", "AFTERNOON", "EVENING", "NIGHT", "DAY", "LATER", "CONTINUOUS"]
        
        current_time = None
        current_location = None
        
        for i, scene in enumerate(self.script_data.get("scenes", [])):
            heading = scene.get("heading", "")
            
            # Extract time information
            scene_time = None
            for indicator in time_indicators:
                if indicator in heading:
                    scene_time = indicator
                    break
            
            # Extract location
            location = None
            if "INT." in heading:
                location = heading.split("INT.")[1].strip()
            elif "EXT." in heading:
                location = heading.split("EXT.")[1].strip()
            
            # Check for time continuity
            if scene_time and current_time:
                # Check for impossible time transitions
                impossible_transitions = [
                    ("MORNING", "NIGHT"),
                    ("NIGHT", "MORNING"),
                    ("EVENING", "MORNING"),
                    ("AFTERNOON", "MORNING")
                ]
                
                if (current_time, scene_time) in impossible_transitions and "LATER" not in heading:
                    self._add_error(
                        "timeline", 
                        f"Impossible time transition from {current_time} to {scene_time} between scenes {i} and {i+1}", 
                        {"scene": i+1, "previous_scene": i, "current_time": scene_time, "previous_time": current_time},
                        ValidationError.SEVERITY_MEDIUM,
                        ["Add a transitional scene", f"Add 'LATER' to scene {i+1} heading"]
                    )
            
            # Check for "CONTINUOUS" without matching location
            if "CONTINUOUS" in heading and current_location and location != current_location:
                self._add_error(
                    "timeline", 
                    f"Scene {i+1} marked as CONTINUOUS but location changes from '{current_location}' to '{location}'", 
                    {"scene": i+1, "previous_scene": i, "current_location": location, "previous_location": current_location},
                    ValidationError.SEVERITY_MEDIUM,
                    ["Remove CONTINUOUS from heading", "Correct the location to match previous scene"]
                )
            
            # Update trackers
            current_time = scene_time
            current_location = location
    
    def _validate_storyboard(self):
        """Validate storyboard structure and content."""
        if not self.storyboard_data or "frames" not in self.storyboard_data:
            self._add_error(
                "structure", 
                "Invalid storyboard format - missing frames", 
                {"file": "storyboard", "position": "global"},
                ValidationError.SEVERITY_HIGH
            )
            return
        
        # Check for missing frame numbers
        frames = self.storyboard_data.get("frames", [])
        for i, frame in enumerate(frames):
            if "number" not in frame:
                self._add_warning(
                    "structure", 
                    f"Storyboard frame at position {i+1} is missing a frame number", 
                    {"frame_position": i+1},
                    ["Add sequential frame numbers to all storyboard frames"]
                )
            
            # Check for missing scene reference
            if "scene" not in frame:
                self._add_warning(
                    "structure", 
                    f"Storyboard frame at position {i+1} is missing a scene reference", 
                    {"frame_position": i+1},
                    ["Add scene references to all storyboard frames"]
                )
    
    def _validate_continuity(self):
        """Validate continuity between script and storyboard."""
        if not self.script_data or not self.storyboard_data:
            return
        
        # Map scenes to frames
        scene_to_frames = {}
        for i, frame in enumerate(self.storyboard_data.get("frames", [])):
            scene_num = frame.get("scene")
            if scene_num:
                if scene_num not in scene_to_frames:
                    scene_to_frames[scene_num] = []
                scene_to_frames[scene_num].append(i+1)
        
        # Check for scenes without frames
        script_scenes = len(self.script_data.get("scenes", []))
        for i in range(1, script_scenes + 1):
            if i not in scene_to_frames:
                self._add_warning(
                    "continuity", 
                    f"Script scene {i} has no corresponding storyboard frames", 
                    {"scene": i},
                    ["Create storyboard frames for this scene", 
                     "Mark scene as 'no visual' if intentionally excluded"]
                )
        
        # Check for frames without scenes
        for i, frame in enumerate(self.storyboard_data.get("frames", [])):
            scene_num = frame.get("scene")
            if scene_num and scene_num > script_scenes:
                self._add_error(
                    "continuity", 
                    f"Storyboard frame {i+1} references non-existent scene {scene_num}", 
                    {"frame": i+1, "referenced_scene": scene_num},
                    ValidationError.SEVERITY_HIGH,
                    ["Update frame to reference correct scene number", 
                     "Add missing scene to script"]
                )
    
    def _enhance_validation_with_gemini(self):
        """Use Gemini API to enhance validation with AI-powered insights."""
        if not GEMINI_AVAILABLE or not self.config["gemini"]["api_key"]:
            return
            
        try:
            model = genai.GenerativeModel(
                model_name=self.config["gemini"]["model"],
                generation_config={
                    "max_output_tokens": self.config["gemini"]["max_tokens"],
                    "temperature": 0.2,
                }
            )
            
            # Only process if we have script data
            if not self.script_data or "scenes" not in self.script_data:
                return
                
            # Prepare script data for Gemini
            script_content = "SCRIPT ANALYSIS REQUEST:\n\n"
            
            for i, scene in enumerate(self.script_data.get("scenes", [])[:10]):  # Limit to 10 scenes to stay within token limits
                script_content += f"SCENE {i+1}: {scene.get('heading', 'NO HEADING')}\n"
                script_content += f"DESCRIPTION: {scene.get('description', 'NO DESCRIPTION')}\n"
                script_content += "CHARACTERS: " + ", ".join(scene.get("characters", [])) + "\n"
                script_content += "---\n"
            
            prompt = f"""
            {script_content}
            
            Please analyze this script for potential issues in the following categories:
            1. Structure issues
            2. Continuity problems
            3. Character consistency
            4. Setting/location problems
            5. Timeline issues
            
            For each issue you find, provide:
            - Category
            - Description of the issue
            - Scene number(s) affected
            - Suggested fix
            
            Format your response as a JSON array of objects with keys: category, description, scenes, suggestion
            """
            
            # Generate analysis
            response = model.generate_content(prompt)
            
            if response.text:
                # Parse Gemini response
                try:
                    # Extract JSON from response
                    import re
                    json_text = re.search(r'\[\s*{.*}\s*\]', response.text, re.DOTALL)
                    if json_text:
                        issues = json.loads(json_text.group(0))
                    else:
                        # Try another approach - sometimes Gemini doesn't wrap in JSON format
                        json_text = re.search(r'{.*}', response.text, re.DOTALL)
                        if json_text:
                            issues = [json.loads(json_text.group(0))]
                        else:
                            logger.warning("Could not extract JSON from Gemini response")
                            return
                        
                    # Add Gemini findings to validation results
                    for issue in issues:
                        category = issue.get("category", "").lower()
                        if "structure" in category:
                            error_type = "structure"
                        elif "continuity" in category:
                            error_type = "continuity"
                        elif "character" in category:
                            error_type = "character"
                        elif "setting" in category or "location" in category:
                            error_type = "setting"
                        elif "timeline" in category or "time" in category:
                            error_type = "timeline"
                        else:
                            error_type = "other"
                            
                        scenes = issue.get("scenes", [])
                        if isinstance(scenes, str):
                            # Try to parse scene numbers from string
                            import re
                            scenes = [int(s) for s in re.findall(r'\d+', scenes)]
                        
                        location = {"scenes": scenes}
                        
                        self._add_info(
                            error_type,
                            issue.get("description", "Unspecified issue"),
                            location,
                            [issue.get("suggestion", "No suggestion provided")],
                        )
                except Exception as e:
                    logger.error(f"Error processing Gemini response: {e}")
                    
        except Exception as e:
            logger.error(f"Error using Gemini API for validation enhancement: {e}")
    
    def _add_error(self, error_type: str, message: str, location: Dict[str, Any], 
                 severity: str = ValidationError.SEVERITY_MEDIUM, suggestions: List[str] = None):
        """Add an error to validation results."""
        error = ValidationError(error_type, message, location, severity, suggestions)
        self.validation_results["errors"].append(error.to_dict())
    
    def _add_warning(self, error_type: str, message: str, location: Dict[str, Any], 
                    suggestions: List[str] = None):
        """Add a warning to validation results."""
        error = ValidationError(error_type, message, location, ValidationError.SEVERITY_LOW, suggestions)
        self.validation_results["warnings"].append(error.to_dict())
    
    def _add_info(self, error_type: str, message: str, location: Dict[str, Any], 
                 suggestions: List[str] = None):
        """Add an informational note to validation results."""
        error = ValidationError(error_type, message, location, ValidationError.SEVERITY_LOW, suggestions)
        self.validation_results["info"].append(error.to_dict())
    
    def export_report(self, format: str = "json", output_path: str = None) -> str:
        """
        Export validation report.
        
        Args:
            format: Output format (json, html, pdf)
            output_path: Path to save the report
            
        Returns:
            str: Path to generated report
        """
        if not self.validation_results:
            logger.warning("No validation results to export")
            return None
            
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"validation_report_{timestamp}.{format}"
            
        try:
            if format.lower() == "json":
                with open(output_path, 'w') as f:
                    json.dump(self.validation_results, f, indent=2)
                    
            elif format.lower() == "html":
                html_content = self._generate_html_report()
                with open(output_path, 'w') as f:
                    f.write(html_content)
                    
            elif format.lower() == "pdf":
                # Placeholder for PDF generation
                logger.warning("PDF export not fully implemented")
                # Generate HTML first
                html_content = self._generate_html_report()
                # TODO: Convert HTML to PDF
                
            else:
                logger.error(f"Unsupported export format: {format}")
                return None
                
            logger.info(f"Report exported to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return None
    
    def _generate_html_report(self) -> str:
        """Generate HTML report from validation results."""
        # Simple HTML template
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Scene Validation Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 20px; }
                h1 { color: #333; }
                .summary { background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }
                .error { background-color: #ffebee; padding: 10px; margin: 10px 0; border-left: 4px solid #f44336; }
                .warning { background-color: #fff8e1; padding: 10px; margin: 10px 0; border-left: 4px solid #ffca28; }
                .info { background-color: #e3f2fd; padding: 10px; margin: 10px 0; border-left: 4px solid #2196f3; }
                .suggestions { margin-top: 10px; padding-left: 20px; }
                .suggestion { margin: 5px 0; }
                .high { color: #d32f2f; }
                .medium { color: #f57c00; }
                .low { color: #0288d1; }
            </style>
        </head>
        <body>
            <h1>Scene Validation Report</h1>
            
            <div class="summary">
                <h2>Summary</h2>
                <p>Timestamp: {timestamp}</p>
                <p>Status: <strong>{status}</strong></p>
                <p>Total Errors: <strong class="high">{total_errors}</strong></p>
                <p>Total Warnings: <strong class="medium">{total_warnings}</strong></p>
                <p>Error Types: {error_types}</p>
            </div>
            
            <h2>Errors</h2>
            {errors}
            
            <h2>Warnings</h2>
            {warnings}
            
            <h2>Information</h2>
            {info}
        </body>
        </html>
        """
        
        # Format timestamps
        timestamp = self.validation_results.get("timestamp", datetime.now().isoformat())
        try:
            dt = datetime.fromisoformat(timestamp)
            timestamp = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            pass
        
        # Format error types
        error_types = self.validation_results.get("summary", {}).get("error_types", {})
        error_types_html = ", ".join([f"{k}: {v}" for k, v in error_types.items()]) if error_types else "None"
        
        # Format errors
        errors_html = ""
        for error in self.validation_results.get("errors", []):
            severity_class = error.get("severity", "medium")
            suggestions_html = ""
            for suggestion in error.get("suggestions", []):
                suggestions_html += f'<div class="suggestion">{suggestion}</div>'
                
            location_html = ""
            for k, v in error.get("location", {}).items():
                location_html += f"{k}: {v}, "
            location_html = location_html.rstrip(", ")
                
            errors_html += f"""
            <div class="error">
                <h3 class="{severity_class}">[{error.get("error_type", "Unknown")}] {error.get("message", "")}</h3>
                <p>Location: {location_html}</p>
                <div class="suggestions">
                    <strong>Suggestions:</strong>
                    {suggestions_html}
                </div>
            </div>
            """
        
        if not errors_html:
            errors_html = "<p>No errors found.</p>"
            
        # Format warnings
        warnings_html = ""
        for warning in self.validation_results.get("warnings", []):
            suggestions_html = ""
            for suggestion in warning.get("suggestions", []):
                suggestions_html += f'<div class="suggestion">{suggestion}</div>'
                
            location_html = ""
            for k, v in warning.get("location", {}).items():
                location_html += f"{k}: {v}, "
            location_html = location_html.rstrip(", ")
                
            warnings_html += f"""
            <div class="warning">
                <h3>[{warning.get("error_type", "Unknown")}] {warning.get("message", "")}</h3>
                <p>Location: {location_html}</p>
                <div class="suggestions">
                    <strong>Suggestions:</strong>
                    {suggestions_html}
                </div>
            </div>
            """
            
        if not warnings_html:
            warnings_html = "<p>No warnings found.</p>"
            
        # Format info
        info_html = ""
        for info in self.validation_results.get("info", []):
            suggestions_html = ""
            for suggestion in info.get("suggestions", []):
                suggestions_html += f'<div class="suggestion">{suggestion}</div>'
                
            location_html = ""
            for k, v in info.get("location", {}).items():
                location_html += f"{k}: {v}, "
            location_html = location_html.rstrip(", ")
                
            info_html += f"""
            <div class="info">
                <h3>[{info.get("error_type", "Unknown")}] {info.get("message", "")}</h3>
                <p>Location: {location_html}</p>
                <div class="suggestions">
                    <strong>Suggestions:</strong>
                    {suggestions_html}
                </div>
            </div>
            """
            
        if not info_html:
            info_html = "<p>No additional information.</p>"
        
        # Fill in template
        html = html.format(
            timestamp=timestamp,
            status=self.validation_results.get("summary", {}).get("status", "unknown"),
            total_errors=self.validation_results.get("summary", {}).get("total_errors", 0),
            total_warnings=self.validation_results.get("summary", {}).get("total_warnings", 0),
            error_types=error_types_html,
            errors=errors_html,
            warnings=warnings_html,
            info=info_html
        )
        
        return html


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate scene structure and continuity in scripts and storyboards")
    parser.add_argument("-c", "--config", help="Path to configuration file")
    parser.add_argument("-s", "--script", help="Path to script file")
    parser.add_argument("-b", "--storyboard", help="Path to storyboard directory or zip file")
    parser.add_argument("-o", "--output", help="Path to output report file")
    parser.add_argument("-f", "--format", choices=["json", "html", "pdf"], default="json", help="Output format")
    
    args = parser.parse_args()
    
    validator = SceneValidator(args.config)
    
    if args.script:
        validator.load_script(args.script)
        
    if args.storyboard:
        validator.load_storyboard(args.storyboard)
        
    if args.script or args.storyboard:
        validator.validate()
        validator.export_report(args.format, args.output)
    else:
        parser.print_help()