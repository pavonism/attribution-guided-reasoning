import os
from datasets import Dataset, load_dataset, load_from_disk
import argparse
from PIL.Image import Image
from tqdm import tqdm
from dataclasses import field, dataclass

from pathlib import Path
import random
import re
import torch
from qwen_vl_utils import process_vision_info

from src.model.entity_attributions import (
    EntityAttributionResults,
    ProblemAttributions,
    ImageAttributions,
)
from src.dataset_adapters import DatasetAdapter, AutoDatasetAdapter
from src.dataset_adapters.common import (
    make_image_content,
    make_text_content,
    ConversationFormat,
)
from src.v1 import V1ForConditionalGeneration, get_processor
from src.v1.processor import V1Processor

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
    "v16": r"""
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

Use the format: 

**Final Answer**

\[ \boxed{\text{[\text{"entity1 caption"}, \text{"entity2 caption"}, ...]}} \]
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
}

FINAL_ANSWER_PROMPT = """
Now, stop your reasoning and give the final answer.
Ensure each entity is a concise noun phrase of EXACTLY 1 to 3 words maximum, without any verbs or conversational text. 
""".strip()


ANSWER_FORMAT = r"""
Use the format: 

**Final Answer**

\[ \boxed{\text{[\text{"entity1"}, \text{"entity2"}, ...]}} \]
""".strip()


@dataclass
class ConversationContext:
    img_column_name: str
    contrasting_img_column_name: str

    positive_image: Image
    negative_image: Image

    question: str
    reasoning: str = ""
    final_answer: str = ""
    entities: set[str] = field(default_factory=set)

    ask_for_final_answer: bool = False

    def get_main_image(self) -> Image:
        return (
            self.positive_image
            if "left" in self.img_column_name
            else self.negative_image
        )

    def get_images(self) -> list[Image]:
        return [self.positive_image, self.negative_image]

    def make_initial_question(self) -> str:
        return (
            self.question
            if self.ask_for_final_answer
            else f"{self.question}\n{ANSWER_FORMAT}"
        )

    def make_conversation(self):
        return [
            {
                "role": "user",
                "content": [
                    make_text_content(self.make_initial_question()),
                    make_text_content("Positive Image:"),
                    make_image_content(
                        self.positive_image, ConversationFormat.TRANSFORMERS
                    ),
                    make_text_content("Negative Image:"),
                    make_image_content(
                        self.negative_image, ConversationFormat.TRANSFORMERS
                    ),
                ],
            },
        ]

    def make_final_answer_conversation(self):
        return [
            {
                "role": "user",
                "content": [
                    make_text_content(self.make_initial_question()),
                    make_text_content("Positive Image:"),
                    make_image_content(
                        self.positive_image, ConversationFormat.TRANSFORMERS
                    ),
                    make_text_content("Negative Image:"),
                    make_image_content(
                        self.negative_image, ConversationFormat.TRANSFORMERS
                    ),
                ],
            },
            {
                "role": "assistant",
                "content": [make_text_content(self.reasoning)],
            },
            {
                "role": "user",
                "content": [
                    make_text_content(f"{FINAL_ANSWER_PROMPT}\n{ANSWER_FORMAT}")
                ],
            },
        ]


def chat(
    model: V1ForConditionalGeneration,
    processor: V1Processor,
    conversations: list[dict],
    sampling_params: dict,
) -> list[str]:
    images = [process_vision_info(c)[0] for c in conversations]

    text = processor.apply_chat_template(
        conversations,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=text,
        images=images,
        return_tensors="pt",
        padding=True,
        padding_side="left",
        add_special_tokens=False,
    ).to(model.device)

    gen_kwargs = {
        "do_sample": True,
        "pad_token_id": processor.tokenizer.pad_token_id,
        "use_cache": True,
        **sampling_params,
    }

    with torch.no_grad():
        input_ids_len = inputs["input_ids"].shape[1]
        outputs = model.generate(**inputs, **gen_kwargs)
        generated_tokens = outputs[:, input_ids_len:]
        responses = processor.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return responses


def parse_entities(response: str) -> list[str]:
    answer_matches = re.findall(
        r"\*\*Final Answer\*\*\n*\\\[\s*\\boxed{\\text{(.*?)}}\s*\\\]",
        response,
        re.DOTALL,
    )

    if not answer_matches:
        return []

    print("Raw model output for entities:", answer_matches[-1])

    entities = re.findall(r"\\text{(.*?)}", answer_matches[-1], re.DOTALL)

    if entities:
        return [entity.lower().strip() for entity in entities]
    else:
        return []


def attribute_entities(
    model: V1ForConditionalGeneration,
    processor: V1Processor,
    sampling_params: dict,
    conversation_contexts: list[ConversationContext],
    ask_for_final_answer: bool,
) -> ImageAttributions:
    responses = chat(
        model=model,
        processor=processor,
        conversations=[c.make_conversation() for c in conversation_contexts],
        sampling_params=sampling_params,
    )

    if ask_for_final_answer:
        for response, conversation_context in zip(responses, conversation_contexts):
            conversation_context.reasoning = response

        responses = chat(
            model=model,
            processor=processor,
            conversations=[
                c.make_final_answer_conversation() for c in conversation_contexts
            ],
            sampling_params=sampling_params,
        )

    cc_with_entities_dict: dict[str, ConversationContext] = {}
    for response, conversation_context in zip(responses, conversation_contexts):
        conversation_context = cc_with_entities_dict.get(
            conversation_context.img_column_name, conversation_context
        )
        conversation_context.entities.update(parse_entities(response))
        cc_with_entities_dict[conversation_context.img_column_name] = (
            conversation_context
        )

    cc_with_entities_list: list[ConversationContext] = list(
        cc_with_entities_dict.values()
    )

    attributions = []

    for conversation_context in cc_with_entities_list:
        print(conversation_context.img_column_name, conversation_context.entities)
        attribution = ImageAttributions(
            column_name=conversation_context.img_column_name,
            entities=list(conversation_context.entities),
        )

        attributions.append(attribution)

    return attributions


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


def attribute(
    model: V1ForConditionalGeneration,
    processor: V1Processor,
    sampling_params: dict,
    dataset: Dataset,
    dataset_adapter: DatasetAdapter,
    output_file: str,
    batch_size: int,
    n_problems: int,
    prompt_version: str,
    n_samples: int,
    ask_for_final_answer: bool,
):
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    attribution_results = EntityAttributionResults(attributions=[])

    if output_path.exists():
        print(f"Resuming evaluation from {output_file}")
        attribution_results = EntityAttributionResults.from_file(output_file)

    image_columns = dataset_adapter.get_image_columns()
    all_problem_ids = dataset_adapter.get_problem_ids(dataset)
    attributed_problem_ids = set(a.problem_id for a in attribution_results.attributions)
    not_attributed_indexes = [
        i
        for i in range(len(dataset))
        if all_problem_ids[i] not in attributed_problem_ids
    ]

    for index in tqdm(not_attributed_indexes[:n_problems]):
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

        question = PROMPTS_POSITIVE[prompt_version].replace("{concept}", concept)
        negative_question = PROMPTS_NEGATIVE[prompt_version].replace(
            "{concept}", concept
        )
        questions = [question] * (len(images) // 2) + [negative_question] * (
            len(images) // 2
        )
        print("Question:", question)
        print("Negative Question:", negative_question)

        problem_attributions = ProblemAttributions(
            problem_id=problem_id,
            images=[],
        )
        image_batch_indexes = list(range(0, len(images), batch_size))

        for batch_index in image_batch_indexes:
            current_indexes = list(
                range(batch_index, min(batch_index + batch_size, len(images)))
            )

            conversation_contexts = [
                ConversationContext(
                    question=questions[i],
                    positive_image=images[i][0],
                    negative_image=images[i][1],
                    img_column_name=image_columns[i],
                    contrasting_img_column_name=contrasting_columns[i],
                    ask_for_final_answer=ask_for_final_answer,
                )
                for i in current_indexes
                for _ in range(n_samples)
            ]

            valid_image_attributions = attribute_entities(
                model,
                processor,
                sampling_params,
                conversation_contexts,
                ask_for_final_answer,
            )

            problem_attributions.images.extend(valid_image_attributions)

        attribution_results.attributions.append(problem_attributions)
        attribution_results.to_file(output_path)

    print(f"Attribution results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Attribute bounding boxes using VLLM.")
    parser.add_argument("--model", help="Name or path to model.")
    parser.add_argument("--dataset", help="Path to test dataset.")
    parser.add_argument("--output", help="Path to attribution output file.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=12,
        help="Batch size for image attribution.",
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
        default="v16",
        choices=PROMPTS_POSITIVE.keys(),
    )
    parser.add_argument(
        "--n_samples",
        default=1,
        type=int,
        help="Number of samples to generate per problem.",
    )
    parser.add_argument(
        "--ask-for-final-answer",
        help="Whether to include the final answer prompt in the second round of generation.",
        action="store_true",
    )

    args = parser.parse_args()
    model_name_or_path = args.model
    dataset_name_or_path = args.dataset
    output_file = args.output
    batch_size = args.batch_size
    tensor_parallel_size = args.tensor_parallel_size
    max_tokens = args.max_tokens
    seed = args.seed
    n_problems = args.n_problems
    prompt_version = args.prompt_version
    n_samples = args.n_samples
    ask_for_final_answer = args.ask_for_final_answer

    print(f"Model: {model_name_or_path}")
    print(f"Dataset: {dataset_name_or_path}")
    print(f"Output file: {output_file}")
    print(f"Batch size: {batch_size}")
    print(f"Tensor parallel size: {tensor_parallel_size}")
    print(f"Max tokens: {max_tokens}")
    print(f"Seed: {seed}")
    print(f"Number of problems: {n_problems}")
    print(f"Prompt version: {prompt_version}")
    print(f"Number of samples: {n_samples}")
    print(f"Final answer prompt: {ask_for_final_answer}")

    random.seed(seed)

    model = V1ForConditionalGeneration.from_pretrained(
        model_name_or_path,
        device_map="cuda",
        torch_dtype=torch.float16,
        attn_implementation="flash_attention_2",
    )

    processor = get_processor(model_name_or_path)

    sampling_params = {
        "max_new_tokens": max_tokens,
    }

    if os.path.exists(dataset_name_or_path):
        dataset = load_from_disk(dataset_name_or_path)
    else:
        dataset = load_dataset(dataset_name_or_path, split="test")

    dataset_adapter = AutoDatasetAdapter.from_dataset(
        dataset_name_or_path,
        dataset,
        text_only=False,
    )

    attribute(
        model,
        processor,
        sampling_params,
        dataset,
        dataset_adapter,
        output_file,
        batch_size,
        n_problems,
        prompt_version,
        n_samples,
        ask_for_final_answer,
    )


if __name__ == "__main__":
    main()
