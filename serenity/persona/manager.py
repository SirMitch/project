# File: serenity/persona/manager.py
# Created: 2024-12-24 15:54:40
# Purpose: Enhanced Persona Management System
# Version: 1.0.0

"""
Serenity Persona Manager
Advanced persona and personality management with dynamic adaptation
"""

import logging
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

class PersonaManager:
    def __init__(self):
        self.logger = logging.getLogger('Serenity.Persona')
        self.persona_path = Path('serenity/data/personas')
        self.persona_path.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.active_persona = None
        self.persona_history = []
        self.initialize_personas()

    def initialize_personas(self):
        """Initialize enhanced persona system"""
        try:
            self.logger.info("Initializing persona manager...")
            self.load_personas()
            self.setup_default_persona()
            self.initialize_learning_system()
            self.logger.info("Persona manager initialized successfully")
        except Exception as e:
            self.logger.error(f"Persona initialization failed: {str(e)}")
            raise

    def load_personas(self):
        """Load all available personas with history"""
        try:
            self.personas = {}
            self.persona_embeddings = {}
            
            for persona_file in self.persona_path.glob('*.json'):
                with open(persona_file, 'r') as f:
                    persona_data = json.load(f)
                    self.personas[persona_data['name']] = persona_data
                    if 'embedding' in persona_data:
                        self.persona_embeddings[persona_data['name']] = torch.tensor(
                            persona_data['embedding']
                        ).to(self.device)
                        
            self.logger.info(f"Loaded {len(self.personas)} personas")
        except Exception as e:
            self.logger.error(f"Failed to load personas: {str(e)}")
            raise

    def setup_default_persona(self):
        """Create and set up enhanced default persona"""
        default_persona = {
            'name': 'default',
            'traits': {
                'friendliness': 0.8,
                'professionalism': 0.9,
                'creativity': 0.7,
                'efficiency': 0.9,
                'empathy': 0.8,
                'curiosity': 0.7,
                'adaptability': 0.9
            },
            'voice': {
                'pitch': 1.0,
                'speed': 1.0,
                'tone': 'neutral',
                'modulation': 0.5
            },
            'preferences': {
                'communication_style': 'balanced',
                'response_length': 'adaptive',
                'technical_level': 'adaptive',
                'formality': 'professional'
            },
            'learning': {
                'learning_rate': 0.01,
                'adaptation_speed': 0.5,
                'memory_retention': 0.8
            },
            'created_at': datetime.now().isoformat(),
            'embedding': self._generate_embedding({
                'friendliness': 0.8,
                'professionalism': 0.9,
                'creativity': 0.7
            })
        }

        if 'default' not in self.personas:
            self.save_persona(default_persona)
            self.personas['default'] = default_persona

        self.active_persona = 'default'

    def initialize_learning_system(self):
        """Initialize persona learning and adaptation system"""
        try:
            self.learning_params = {
                'base_learning_rate': 0.01,
                'adaptation_threshold': 0.1,
                'history_weight': 0.7,
                'novelty_weight': 0.3
            }
            
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.logger.info("Learning system initialized")
        except Exception as e:
            self.logger.error(f"Learning system initialization failed: {str(e)}")

    def _generate_embedding(self, traits: Dict[str, float]) -> List[float]:
        """Generate embedding vector for persona traits"""
        try:
            # Convert traits to normalized vector
            trait_values = np.array(list(traits.values()))
            normalized = (trait_values - np.mean(trait_values)) / np.std(trait_values)
            return normalized.tolist()
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return [0.0] * len(traits)

    def save_persona(self, persona_data: Dict) -> bool:
        """Save persona with history tracking"""
        try:
            file_path = self.persona_path / f"{persona_data['name']}.json"
            
            # Add timestamp and version
            persona_data['last_modified'] = datetime.now().isoformat()
            persona_data['version'] = persona_data.get('version', 0) + 1
            
            # Save backup if exists
            if file_path.exists():
                backup_path = self.persona_path / f"{persona_data['name']}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                with open(file_path, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
            
            with open(file_path, 'w') as f:
                json.dump(persona_data, f, indent=4)
                
            self.logger.info(f"Saved persona: {persona_data['name']}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to save persona: {str(e)}")
            return False

    def get_active_persona(self) -> Dict:
        """Get currently active persona with real-time adaptation"""
        try:
            persona = self.personas.get(self.active_persona)
            if persona:
                # Apply real-time adaptations
                adapted_persona = self._adapt_persona(persona)
                return adapted_persona
            return None
        except Exception as e:
            self.logger.error(f"Failed to get active persona: {str(e)}")
            return None

    def _adapt_persona(self, persona: Dict) -> Dict:
        """Apply real-time adaptations to persona"""
        try:
            # Deep copy persona
            adapted = json.loads(json.dumps(persona))
            
            # Apply recent learning
            if hasattr(self, 'recent_interactions'):
                for trait, value in adapted['traits'].items():
                    if trait in self.recent_interactions:
                        adapted['traits'][trait] = (
                            value * (1 - self.learning_params['adaptation_threshold']) +
                            self.recent_interactions[trait] * self.learning_params['adaptation_threshold']
                        )
            
            return adapted
        except Exception as e:
            self.logger.error(f"Persona adaptation failed: {str(e)}")
            return persona

    def switch_persona(self, persona_name: str) -> bool:
        """Switch to different persona with transition handling"""
        try:
            if persona_name in self.personas:
                # Save current state
                if self.active_persona:
                    self._save_persona_state(self.active_persona)
                
                self.active_persona = persona_name
                self._load_persona_state(persona_name)
                
                self.logger.info(f"Switched to persona: {persona_name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to switch persona: {str(e)}")
            return False

    def _save_persona_state(self, persona_name: str):
        """Save persona state and learning progress"""
        try:
            state = {
                'name': persona_name,
                'timestamp': datetime.now().isoformat(),
                'learning_progress': getattr(self, 'recent_interactions', {}),
                'active_duration': getattr(self, 'active_duration', 0)
            }
            
            state_file = self.persona_path / f"{persona_name}_state.json"
            with open(state_file, 'w') as f:
                json.dump(state, f, indent=4)
        except Exception as e:
            self.logger.error(f"Failed to save persona state: {str(e)}")

    def _load_persona_state(self, persona_name: str):
        """Load persona state and learning progress"""
        try:
            state_file = self.persona_path / f"{persona_name}_state.json"
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state = json.load(f)
                    self.recent_interactions = state.get('learning_progress', {})
                    self.active_duration = state.get('active_duration', 0)
        except Exception as e:
            self.logger.error(f"Failed to load persona state: {str(e)}")

    def create_persona(self, name: str, traits: Dict[str, float], 
                      voice: Dict[str, Any], preferences: Dict[str, Any]) -> bool:
        """Create new persona with enhanced capabilities"""
        try:
            # Validate inputs
            if not all(0 <= v <= 1 for v in traits.values()):
                raise ValueError("Trait values must be between 0 and 1")
                
            new_persona = {
                'name': name,
                'traits': traits,
                'voice': voice,
                'preferences': preferences,
                'learning': {
                    'learning_rate': 0.01,
                    'adaptation_speed': 0.5,
                    'memory_retention': 0.8
                },
                'created_at': datetime.now().isoformat(),
                'embedding': self._generate_embedding(traits)
            }
            
            if self.save_persona(new_persona):
                self.personas[name] = new_persona
                self.persona_embeddings[name] = torch.tensor(
                    new_persona['embedding']
                ).to(self.device)
                return True
            return False
        except Exception as e:
            self.logger.error(f"Failed to create persona: {str(e)}")
            return False

    def update_persona(self, name: str, updates: Dict[str, Any]) -> bool:
        """Update existing persona with learning integration"""
        try:
            if name not in self.personas:
                return False
                
            persona = self.personas[name]
            
            # Apply updates
            for category, values in updates.items():
                if category in persona:
                    if isinstance(values, dict):
                        persona[category].update(values)
                    else:
                        persona[category] = values
            
            # Update embedding if traits changed
            if 'traits' in updates:
                persona['embedding'] = self._generate_embedding(persona['traits'])
                self.persona_embeddings[name] = torch.tensor(
                    persona['embedding']
                ).to(self.device)
            
            return self.save_persona(persona)
        except Exception as e:
            self.logger.error(f"Failed to update persona: {str(e)}")
            return False

    def learn_from_interaction(self, interaction_data: Dict[str, Any]):
        """Learn and adapt from interactions"""
        try:
            if not self.active_persona:
                return
                
            # Update recent interactions
            if not hasattr(self, 'recent_interactions'):
                self.recent_interactions = {}
                
            # Process interaction data
            for trait, value in interaction_data.items():
                if trait in self.personas[self.active_persona]['traits']:
                    current = self.recent_interactions.get(trait, 
                        self.personas[self.active_persona]['traits'][trait])
                    self.recent_interactions[trait] = (
                        current * (1 - self.learning_params['base_learning_rate']) +
                        value * self.learning_params['base_learning_rate']
                    )
            
            # Periodically save learned adaptations
            if len(self.recent_interactions) % 10 == 0:
                self._save_persona_state(self.active_persona)
                
        except Exception as e:
            self.logger.error(f"Failed to learn from interaction: {str(e)}")

    def get_persona_stats(self) -> Dict[str, Any]:
        """Get persona system statistics"""
        try:
            return {
                'total_personas': len(self.personas),
                'active_persona': self.active_persona,
                'learning_rate': self.learning_params['base_learning_rate'],
                'adaptation_threshold': self.learning_params['adaptation_threshold'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get persona stats: {str(e)}")
            return {}

if __name__ == "__main__":
    # Test persona manager
    try:
        manager = PersonaManager()
        
        # Test persona creation
        test_persona = manager.create_persona(
            "test_persona",
            {'friendliness': 0.9, 'professionalism': 0.8, 'creativity': 0.7},
            {'pitch': 1.0, 'speed': 1.1, 'tone': 'friendly'},
            {'communication_style': 'casual', 'response_length': 'medium'}
        )
        
        if test_persona:
            print("Persona creation successful!")
            
        # Test persona switching
        if manager.switch_persona("test_persona"):
            print("Persona switch successful!")
            
        # Get active persona
        active = manager.get_active_persona()
        print("Active persona:", json.dumps(active, indent=2))
        
        print("Persona manager test successful!")
    except Exception as e:
        print(f"Persona manager test failed: {str(e)}")