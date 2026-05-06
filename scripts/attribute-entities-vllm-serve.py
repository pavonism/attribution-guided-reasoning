import datetime
from math import ceil
import os
import traceback
from datasets import Dataset, load_dataset, load_from_disk
import argparse
from PIL.Image import Image
from tqdm import tqdm
from dataclasses import field, dataclass

import asyncio
from pathlib import Path
import random
import re
from typing import Any

from src.configuration import get_vllm_sampling_params
from src.messenger.content import ImageContent, TextContent
from src.messenger.vllm_api_messenger import VllmApiMessengerFactory
from src.messenger.async_llm_messenger import AsyncLLMMessenger
from src.model.entity_attributions import (
    EntityAttributionResults,
    ProblemAttributions,
    ImageAttributions,
)
from src.dataset_adapters import DatasetAdapter, AutoDatasetAdapter

PROMPTS_POSITIVE = {
    "v1": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify and describe the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be indivisible and strictly exclude visual noise (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspetive does not allow for finer localization.
""".strip(),
    "v2": """
You are provided with two images:
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Generate a JSON-formatted list of strict visual proofs found in the Positive Image.

Instructions:
1. First, briefly contrast the Positive Image against the Negative Image to find the differentiating factors.
2. Select the specific, physical objects or body parts (Entities) in the Positive Image responsible for these factors.
3. Apply the "Minimum Viable Entity" rule: Focus on the smallest distinct body part or object necessary (e.g., select "Hand" instead of "Person" if only the grip matters).
4. Combine the Entity and its State/Action into a single descriptive string.

Constraint Checklist:
- Do NOT list abstract concepts (e.g., "Freedom", "Fun"). List only visible elements.
- Stop reasoning immediately after identifying the entities and output the final format.
""".strip(),
    "v3": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify and describe the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be indivisible and strictly exclude visual noise (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspetive does not allow for finer localization.
4. Stop reasoning immediately after identifying the entities and output the final format.
""".strip(),
    "v4": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify and describe the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be indivisible (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspetive does not allow for finer localization.
4. Stop reasoning immediately after identifying the entities and output the final format.
""".strip(),
    "v5": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify and describe the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be indivisible (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspetive does not allow for finer localization.
""".strip(),
    "v6": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify and describe the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective.
""".strip(),
    "v7": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective.
5. Your entity captions should be precise and concise, consisting of maximum 3 words (e.g "Left Hand", "Red Ball", "Holding Hand").
""".strip(),
    "v8": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective.
5. Your entity captions should be precise and concise, consisting of maximum 3 words (e.g "Left Hand", "Red Ball", "Holding Hand").
""".strip(),
    "v9": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept {concept}.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Your entity captions should be precise and concise, consisting of maximum 3 words (e.g "left hand", "red ball", "foreground human").
""".strip(),
    "v10": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. They must be minimal (e.g., exclude the torso if only the arm is acting).
3. The prmpts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text here (e.g., say "empty hand" instead of "hand not holding", "distant ball" instead of "ball thrown away").
""".strip(),
    "v11": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm").
3. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text (e.g., for the concept 'drinking water', say "right hand", "glass", "mouth" instead of "person drinking").
""".strip(),
    "v12": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible, in which case you can provide a holistic description like "human".
3. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text (e.g., for the concept 'drinking water', say "right hand", "glass", "mouth" instead of "person drinking").
""".strip(),
    "v13": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible, in which case you can provide a holistic description like "human".
3. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text.
4. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, or motions.

Good examples: "right hand", "glass", "water", "person", "tennis ball"
Bad examples: "water splash effect", "person drinking", "holding hand", "ball motion"
""".strip(),
    "v14": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible, in which case you can provide a holistic description like "human".
3. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
4. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, or motions.

Good examples: "right hand", "glass", "water", "person", "tennis ball", "umbrella", "backpack"
Bad examples: "water splash effect", "person drinking", "holding hand", "ball motion", "hand holding umbrella", "held umbrella", "backpack on the floor"
""".strip(),
    "v15": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. The entities should be minimal. For example, if the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible.
3. Avoid holistic entities or vast background surfaces unless they are the only way to grasp the provided concept.
4. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
5. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, or motions.

Good examples: "right hand", "glass", "water", "person", "tennis ball", "umbrella", "backpack"
Bad examples: "water splash effect", "person drinking", "holding hand", "ball motion", "hand holding umbrella", "held umbrella", "backpack on the floor"
""".strip(),
    "v16": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. You must break these down into their most specific, localized physical parts. For example, if the concept implies an action involving a human, you MUST output the specific body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible.
3. Avoid holistic entities or vast background surfaces unless they are the only way to grasp the provided concept.
4. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
5. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, or motions.

Good examples: "right hand", "glass", "water", "person", "tennis ball", "umbrella", "backpack"
Bad examples: "water splash effect", "person drinking", "holding hand", "ball motion", "hand holding umbrella", "held umbrella", "backpack on the floor"
""".strip(),
    "v17": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept (used for contrast).

Task: Identify the specific visual elements in the Positive Image that strictly define the concept '{concept}'.

Instructions:
1. Compare the Positive Image against the Negative Image to identify the specific visual features that distinguish the concept.
2. Identify the key entities that are necessary to verify the concept. You must break these down into their most specific, localized physical parts. For example, if the concept implies an action involving a human, you MUST output the specific body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible.
3. Avoid holistic entities or vast background surfaces (snow, grass, ground, mountains).
4. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
5. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, or motions.

Good examples: "right hand", "glass", "water", "person", "tennis ball", "umbrella", "backpack"
Bad examples: "water splash effect", "person drinking", "holding hand", "ball motion", "hand holding umbrella", "held umbrella", "backpack on the floor"
""".strip(),
}

PROMPTS_NEGATIVE = {
    "v1": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify and describe the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be indivisible and strictly exclude visual noise (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspective does not allow for finer localization.
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
""".strip(),
    "v2": """
You are provided with two images:
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Generate a JSON-formatted list of strict visual proofs found in the Negative Image that create the violation.

Instructions:
1. Contrast the Negative Image against the Positive Image to find the specific element that breaks the rule of '{concept}'.
2. Select the specific "Minimum Viable Entity" responsible for the violation (e.g., if the concept is "Holding" and the hand is open, select the "Open Hand").
3. Describe the Entity and its State/Action combined. Focus on *what is actually happening* instead of what is missing.
4. Distinguish related actions: If the concept implies static control (e.g., 'Holding') but the image shows dynamic release (e.g., 'Throwing'), explicitly describe the dynamic action as the violation.

Constraint Checklist:
- Do NOT list what is "missing" (e.g., do not say "No kite"). List what is *present* that proves the absence (e.g., "Empty hands").
- Stop reasoning immediately after identifying the entities and output the final format.
""".strip(),
    "v3": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify and describe the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be indivisible and strictly exclude visual noise (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspective does not allow for finer localization.
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
6. Stop reasoning immediately after identifying the entities and output the final format.
""".strip(),
    "v4": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify and describe the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be indivisible (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspective does not allow for finer localization.
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
6. Stop reasoning immediately after identifying the entities and output the final format.
""".strip(),
    "v5": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify and describe the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting). You can break this rule if the concept is holistic or the perspective does not allow for finer localization.
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
""".strip(),
    "v6": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify and describe the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective. 
5. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
6. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
""".strip(),
    "v7": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify and describe the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective. 
5. Your descriptions should be precise and concise, making it easy to locate the entity in the image.
6. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
7. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
""".strip(),
    "v8": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept. Include relevant states or actions in the description.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective. 
5. Your entity captions should be precise and concise, consisting of maximum 3 words (e.g "Left Hand", "Red Ball", "Holding Hand").
6. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
7. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
""".strip(),
    "v9": """
You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Don't provide holistic descriptions unless it's necessary due to the nature of the concept or perspective. 
5. Your entity captions should be precise and concise, consisting of maximum 3 words (e.g "left hand", "red ball", "foreground human").
6. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
7. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
""".strip(),
    "v10": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the specific entities that violate the concept.
3. The entities should be minimal (e.g., exclude the torso if only the arm is acting).
4. Do not provide holistic descriptions unless it is necessary due to the nature of the concept or perspective. 
5. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
6. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
7. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text (e.g., say "empty hand" instead of "hand not holding", "distant ball" instead of "ball thrown away").
""".strip(),
    "v11": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm").
3. Do not provide holistic descriptions unless it is necessary due to the nature of the concept or perspective. 
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
6. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text (e.g., for the concept 'drinking water', say "right hand", "glass", "mouth" instead of "person drinking").
""".strip(),
    "v12": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible, in which case you can provide a holistic description like "human".
3. Do not provide holistic descriptions unless it is necessary due to the nature of the concept or perspective. 
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. If no direct violation is found, identify the entities that are semantically most relevant to the incompatibility between the image and the concept.
6. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text (e.g., for the concept 'drinking water', say "right hand", "glass", "mouth" instead of "person drinking").
""".strip(),
    "v13": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible, in which case you can provide a holistic description like "human".
3. Do not provide holistic descriptions unless it is necessary due to the nature of the concept or perspective. 
4. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept.
5. ONLY name objects that are physically present in the image. If an expected object is missing, name the physical part that is exposed instead (e.g., say "head" instead of "missing hat"). NEVER use words like "missing", "no", "empty", or "without".
6. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text (e.g., for the concept 'drinking water', say "right hand", "glass", "mouth" instead of "person drinking").
7. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, or motions.

Good examples: "right hand", "glass", "water", "person", "tennis ball"
Bad examples: "water splash effect", "person drinking", "holding hand", "ball motion", "missing hat", "no glasses"
""".strip(),
    "v14": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. If the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible, in which case you can provide a holistic description like "human".
3. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
4. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, motions, or absences.
5. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept. Identify the specific physical objects or body parts that visually prove the difference in action. (e.g., if the difference between running and standing is the legs, output "left leg" and "right leg", NEVER "leg positions" or "running legs").
6. ONLY name objects that are physically present in the image. If an expected object is missing, name the physical part that is exposed instead (e.g., say "head" instead of "missing hat"). NEVER use words like "missing", "no", "empty", or "without".

Good examples: "right hand", "glass", "water", "person", "bare head", "tennis ball", "umbrella", "backpack", "legs"
Bad examples: "missing hat", "no glasses", "person drinking", "ball motion", "water splash effect", "hand holding umbrella", "held umbrella", "backpack on the floor", "leg positions"
""".strip(),
    "v15": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. The entities should be minimal. For example, if the concept implies an action involving a human, you MUST include the specific minimal body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible.
3. Avoid holistic entities or vast background surfaces unless they are the only way to grasp the provided concept.
4. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
5. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, motions, or absences.
6. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept. Identify the specific physical objects or body parts that visually prove the difference in action. (e.g., if the difference between running and standing is the legs, output "left leg" and "right leg", NEVER "leg positions" or "running legs").
7. ONLY name objects that are physically present in the image. If an expected object is missing, name the physical part that is exposed instead (e.g., say "head" instead of "missing hat"). NEVER use words like "missing", "no", "empty", or "without".

Good examples: "right hand", "glass", "water", "person", "bare head", "tennis ball", "umbrella", "backpack", "legs"
Bad examples: "missing hat", "no glasses", "person drinking", "ball motion", "water splash effect", "hand holding umbrella", "held umbrella", "backpack on the floor", "leg positions"
""".strip(),
    "v16": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. You must break these down into their most specific, localized physical parts. For example, if the concept implies an action involving a human, you MUST output the specific body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible.
3. Avoid holistic entities or vast background surfaces unless they are the only way to grasp the provided concept.
4. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
5. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, motions, or absences.
6. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept. Identify the specific physical objects or body parts that visually prove the difference in action. (e.g., if the difference between running and standing is the legs, output "left leg" and "right leg", NEVER "leg positions" or "running legs").
7. ONLY name objects that are physically present in the image. If an expected object is missing, name the physical part that is exposed instead (e.g., say "head" instead of "missing hat"). NEVER use words like "missing", "no", "empty", or "without".

Good examples: "right hand", "glass", "water", "person", "bare head", "tennis ball", "umbrella", "backpack", "legs"
Bad examples: "missing hat", "no glasses", "person drinking", "ball motion", "water splash effect", "hand holding umbrella", "held umbrella", "backpack on the floor", "leg positions"
""".strip(),
    "v17": """
You are an expert computer vision annotator preparing text prompts for a Segment Anything Model (SAM). 

You are provided with two images: 
1. Positive Image: Depicts the concept '{concept}'.
2. Negative Image: Violates this concept.

Task: Identify the specific visual elements in the Negative Image that cause it to violate the concept '{concept}'.

Instructions:
1. Compare the Negative Image against the Positive Image to find the specific feature, object, or relationship that violates the concept.
2. Identify the key entities that are necessary to verify the concept. You must break these down into their most specific, localized physical parts. For example, if the concept implies an action involving a human, you MUST output the specific body parts or tools executing the action (e.g., exclude the torso if only the arm is acting, but include "hand" or "arm") unless the human is too small or too far away to be clearly visible.
3. Avoid holistic entities or vast background surfaces (snow, grass, ground, mountains).
4. The prompts for SAM MUST be highly concise noun phrases of EXACTLY 1 to 3 words maximum. Do not include verbs or conversational text. You MUST deconstruct complex interactions into single, independent atomic nouns. NEVER use spatial prepositions (e.g., in, on, at) or state descriptors (e.g., held, holding, flying). For example, split "hand holding umbrella" into separate outputs: "hand" and "umbrella".
5. Every output MUST be a tangible, physical object with visible boundaries that can be explicitly masked. You MUST NOT output abstract concepts, forces, effects, motions, or absences.
6. Sometimes an action is related to the concept but visually distinct. If the concept is 'Holding' (static control) and the image shows 'Throwing' (releasing), this is a violation. Do not assume related actions satisfy the concept. Identify the specific physical objects or body parts that visually prove the difference in action. (e.g., if the difference between running and standing is the legs, output "left leg" and "right leg", NEVER "leg positions" or "running legs").
7. ONLY name objects that are physically present in the image. If an expected object is missing, name the physical part that is exposed instead (e.g., say "head" instead of "missing hat"). NEVER use words like "missing", "no", "empty", or "without".

Good examples: "right hand", "glass", "water", "person", "bare head", "tennis ball", "umbrella", "backpack", "legs"
Bad examples: "missing hat", "no glasses", "person drinking", "ball motion", "water splash effect", "hand holding umbrella", "held umbrella", "backpack on the floor", "leg positions"
""".strip(),
}

FINAL_ANSWER_PROMPT = """
Now, stop your reasoning and give the final answer.
Put your identified entities in the string list.
""".strip()

FORMAT = """
Use the format: <answer>["entity1", "entity2", ...]</answer>.
""".strip()


@dataclass
class ConversationContext:
    messenger: AsyncLLMMessenger
    problem_id: str
    img_column_name: str
    contrasting_img_column_name: str

    positive_image: Image
    negative_image: Image

    question: str
    reasoning: str = ""
    final_answer: str = ""
    entities: set[str] = field(default_factory=set)

    def contents(self):
        return [
            TextContent(f"{self.question}\n\n{FORMAT}"),
            TextContent("Positive Image:"),
            ImageContent(image=self.positive_image),
            TextContent("Negative Image:"),
            ImageContent(image=self.negative_image),
        ]

    def final_answer_contents(self):
        return [
            TextContent(f"{FINAL_ANSWER_PROMPT}\n\n{FORMAT}"),
        ]

    def parse_entities(self, text: str):
        answer_matches = re.findall(
            r"<answer>\[(.*?)\]</answer>",
            text,
            re.DOTALL,
        )

        if not answer_matches:
            return

        entities = re.findall(r"\"(.*?)\"", answer_matches[-1], re.DOTALL)
        entities = [entity.lower().strip() for entity in entities]

        if "entity1" in entities:
            entities.remove("entity1")
        if "entity2" in entities:
            entities.remove("entity2")

        if entities:
            self.entities.update(entities)


async def attribute_entities(cc: ConversationContext) -> ConversationContext:

    cc.messenger.open_context()

    try:
        cc.reasoning = await cc.messenger.ask(cc.contents())
        cc.parse_entities(cc.reasoning)

        if not cc.entities:
            cc.final_answer = await cc.messenger.ask(cc.final_answer_contents())
            cc.parse_entities(cc.final_answer)
    except Exception as e:
        print(f"Error in problem {cc.problem_id} - {cc.img_column_name}: {e}")
        print(traceback.format_exc())
    finally:
        cc.messenger.close_context()

    return cc


def prepare_contrasting_images(
    images: list[Image],
    image_columns: list[str],
) -> tuple[list[list[Image]], list[str]]:
    left_images = images[0 : len(images) // 2]
    right_images = images[len(images) // 2 :]
    left_columns = image_columns[0 : len(image_columns) // 2]
    right_columns = image_columns[len(image_columns) // 2 :]

    contrasting_images = []
    contrasting_columns = []
    for img in left_images:
        index = random.choice(range(len(right_images)))
        contrast_img = right_images[index]
        contrasting_images.append([img, contrast_img])
        contrasting_columns.append(right_columns[index])
    for img in right_images:
        index = random.choice(range(len(left_images)))
        contrast_img = left_images[index]
        contrasting_images.append([contrast_img, img])
        contrasting_columns.append(left_columns[index])

    return contrasting_images, contrasting_columns


async def attribute(
    messengers: list[AsyncLLMMessenger],
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    output_dir: str,
    n_problems: int,
    prompt_version: str,
    n_samples: int,
    job_id: int,
    job_count: int,
):
    output_path = Path(output_dir, f"{job_id}_of_{job_count}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_results = EntityAttributionResults(attributions=[])

    if output_path.exists():
        print(f"Resuming evaluation from {output_path}")
        attribution_results = EntityAttributionResults.from_file(output_path)

    image_columns = dataset_adapter.get_image_columns()
    all_problem_ids = dataset_adapter.get_problem_ids(dataset)

    problems_ids_to_attribute = all_problem_ids[:n_problems]

    problem_ids_per_job = ceil(len(problems_ids_to_attribute) / job_count)
    start_index = (job_id - 1) * problem_ids_per_job
    job_problem_indexes = list(
        range(
            start_index,
            min(start_index + problem_ids_per_job, len(problems_ids_to_attribute)),
        )
    )

    available_messengers = messengers.copy()
    running_tasks = set()

    problem_attr_by_problem_id = {
        a.problem_id: a for a in attribution_results.attributions
    }

    print(f"[{datetime.datetime.now()}] Starting attribution...")

    for index in tqdm(job_problem_indexes):
        batch = dataset[index : index + 1]

        problem_id = dataset_adapter.get_problem_ids(batch)[0]
        concept = dataset_adapter.get_answers(batch)[0].strip()
        images = dataset_adapter.get_images(batch)[0]
        images, contrasting_columns = prepare_contrasting_images(images, image_columns)
        source_dataset = batch["dataset"][0]

        if source_dataset == "bongard-hoi":
            words = concept.split()
            words[0] = f"{words[0]}ing"
            concept = " ".join(words)

        question = PROMPTS_POSITIVE[prompt_version].format(concept=concept)
        negative_question = PROMPTS_NEGATIVE[prompt_version].format(concept=concept)
        questions = [question] * (len(images) // 2) + [negative_question] * (
            len(images) // 2
        )
        print("Question:", question)
        print("Negative Question:", negative_question)

        problem_attr = problem_attr_by_problem_id.get(problem_id)

        for i in range(len(images)):
            if problem_attr:
                img_attr = get_or_create_image_attr(
                    problem_attr,
                    image_columns[i],
                )
                if img_attr.entities:
                    print(
                        f"Skipping problem {problem_id} - {image_columns[i]} since it already has attributed entities."
                    )
                    continue

            for _ in range(n_samples):
                conversation_context = ConversationContext(
                    messenger=available_messengers.pop(),
                    problem_id=problem_id,
                    img_column_name=image_columns[i],
                    contrasting_img_column_name=contrasting_columns[i],
                    positive_image=images[i][0],
                    negative_image=images[i][1],
                    question=questions[i],
                )

                task = asyncio.create_task(attribute_entities(conversation_context))
                running_tasks.add(task)

                if len(available_messengers) == 0:
                    running_tasks = await process_completed_tasks(
                        running_tasks,
                        available_messengers,
                        attribution_results,
                        problem_attr_by_problem_id,
                    )

        attribution_results.to_file(output_path)

    if running_tasks:
        await process_completed_tasks(
            running_tasks,
            available_messengers,
            attribution_results,
            problem_attr_by_problem_id,
            return_when=asyncio.ALL_COMPLETED,
        )

    print(f"[{datetime.datetime.now()}] Attribution completed.")

    attribution_results.to_file(output_path)
    print(f"Attribution results saved to {output_path}")


async def process_completed_tasks(
    running_tasks: set,
    available_messengers: list[AsyncLLMMessenger],
    attribution_results: EntityAttributionResults,
    problem_attr_by_problem_id: dict,
    return_when=asyncio.FIRST_COMPLETED,
):
    done_tasks, running_tasks = await asyncio.wait(
        running_tasks,
        return_when=return_when,
    )

    for done in done_tasks:
        try:
            cc: ConversationContext = done.result()
            print(
                f"Task completed: {cc.problem_id} - {cc.img_column_name} - Entities: {cc.entities}"
            )

            available_messengers.append(cc.messenger)

            problem_attr = get_or_create_problem_attr(
                attribution_results,
                problem_attr_by_problem_id,
                cc.problem_id,
            )

            img_attr = get_or_create_image_attr(
                problem_attr,
                cc.img_column_name,
            )

            cc.entities.update(img_attr.entities)
            img_attr.entities = list(cc.entities)

        except Exception as e:
            print(f"Error processing task result: {e}")
            print(traceback.format_exc())

    return running_tasks


def get_or_create_problem_attr(
    attribution_results: EntityAttributionResults,
    problem_attr_by_problem_id: dict,
    problem_id: str,
) -> ProblemAttributions:
    problem_attr = problem_attr_by_problem_id.get(problem_id)

    if not problem_attr:
        problem_attr = ProblemAttributions(
            problem_id=problem_id,
            images=[],
        )
        problem_attr_by_problem_id[problem_id] = problem_attr
        attribution_results.attributions.append(problem_attr)
    return problem_attr


def get_or_create_image_attr(
    problem_attr: ProblemAttributions,
    img_column_name: str,
) -> ImageAttributions:
    for img_attr in problem_attr.images:
        if img_attr.column_name == img_column_name:
            return img_attr

    new_img_attr = ImageAttributions(column_name=img_column_name, entities=[])
    problem_attr.images.append(new_img_attr)
    return new_img_attr


async def main():
    parser = argparse.ArgumentParser(description="Attribute bounding boxes using VLLM.")
    parser.add_argument("--model", help="Name or path to model.")
    parser.add_argument("--dataset", help="Path to test dataset.")
    parser.add_argument("--output", help="Path to attribution output directory.")
    parser.add_argument(
        "--job-id",
        type=int,
        default=1,
        help="ID of the current job for distributed execution.",
    )
    parser.add_argument(
        "--job-count",
        type=int,
        default=1,
        help="Total number of jobs for distributed execution.",
    )
    parser.add_argument(
        "--tasks-per-gpu",
        type=int,
        default=25,
        help="Number of concurrent tasks per GPU.",
    )
    parser.add_argument(
        "--data-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to use for data parallelism.",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        help="Number of GPUs to use for tensor parallelism.",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-tokens",
        help="Maximum number of tokens to generate.",
        type=int,
        default=4096,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--n_problems",
        help="Number of problems to attribute.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--prompt-version",
        help="Version of the prompt to use.",
        type=str,
        default="v17",
        choices=PROMPTS_POSITIVE.keys(),
    )
    parser.add_argument(
        "--n_samples",
        default=10,
        type=int,
        help="Number of samples to generate per problem.",
    )

    args = parser.parse_args()
    model_name_or_path = args.model
    dataset_name_or_path = args.dataset
    output_file = args.output
    job_id = args.job_id
    job_count = args.job_count
    tasks_per_gpu = args.tasks_per_gpu
    data_parallel_size = args.data_parallel_size
    tensor_parallel_size = args.tensor_parallel_size
    max_tokens = args.max_tokens
    seed = args.seed
    n_problems = args.n_problems
    prompt_version = args.prompt_version
    n_samples = args.n_samples

    print(f"Model: {model_name_or_path}")
    print(f"Dataset: {dataset_name_or_path}")
    print(f"Output file: {output_file}")
    print(f"Job ID: {job_id}")
    print(f"Job count: {job_count}")
    print(f"Maximum concurrent tasks: {tasks_per_gpu}")
    print(f"Data parallel size: {data_parallel_size}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Max tokens: {max_tokens}")
    print(f"Seed: {seed}")
    print(f"Number of problems: {n_problems}")
    print(f"Prompt version: {prompt_version}")
    print(f"Number of samples: {n_samples}")

    random.seed(seed)

    sampling_params = get_vllm_sampling_params(model_name_or_path)

    messengers = VllmApiMessengerFactory(
        model_name_or_path,
        max_tokens=10_240 + 2 * max_tokens,
        limit_mm_per_prompt=2,
        custom_args=[
            "--gpu-memory-utilization",
            "0.95",
            "--seed",
            str(seed),
            "--enforce-eager",
            "--dtype",
            "bfloat16",
            "--enable-prefix-caching",
            "--tensor-parallel-size",
            str(tensor_parallel_size),
            "--data-parallel-size",
            str(data_parallel_size),
            "--mm-processor-cache-gb",
            "32",
        ],
    ).make_messengers(
        tasks_per_gpu * data_parallel_size,
        temperature=sampling_params.temperature,
        top_p=sampling_params.top_p,
    )

    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, split="test")

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        dataset_name_or_path,
        dataset,
        text_only=False,
    )

    await attribute(
        messengers,
        dataset,
        dataset_adapter,
        output_file,
        n_problems,
        prompt_version,
        n_samples,
        job_id,
        job_count,
    )


if __name__ == "__main__":
    asyncio.run(main())
