#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import itertools
import logging
import os
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from statistics import mean

from veomni.utils.reward_score.ifeval import instructions_registry

def test_instruction_following_strict(inp, response):
    """Tests response to see if instrutions are followed."""
    instruction_list = inp['instruction_id_list']
    is_following_list = []

    for index, instruction_id in enumerate(instruction_list):
        instruction_cls = instructions_registry.INSTRUCTION_DICT[instruction_id]
        instruction = instruction_cls(instruction_id)

        instruction.build_description(**inp['kwargs'][index])
        args = instruction.get_instruction_args()
        if args and "prompt" in args:
            instruction.build_description(prompt=inp['prompt'])

        if response.strip() and instruction.check_following(response):
            is_following_list.append(True)
        else:
            is_following_list.append(False)

    return {
        "strict_prompt_acc": all(is_following_list),
        "strict_instruction_acc": is_following_list
    }

# def compute_scores(jobs, cache_path):
#     for job in jobs:
#         assert len(job["gen"]) == 1
#         gen = job['gen'][0]
#         job.update(test_instruction_following_strict(job, gen))
#     save_cache(jobs, cache_path)
#     return mean(x['strict_prompt_acc'] for x in jobs)

def compute_score(solution_str, prompt_id):
    with open('/data/lipsync/andrew/VeOmni/veomni/utils/reward_score/ifeval/scoring.json', 'r') as f:
        scoring = json.load(f)
    data = scoring[prompt_id]
    result = test_instruction_following_strict(data, solution_str)
    return int(result['strict_prompt_acc'])


if __name__ == "__main__":
    # Example usage
    solution_str = """
Raymond III (1140–1187) was Count of Tripoli from 1152 to 1187 and a key figure in the Crusader states. Born to Raymond II and Hodierna of Jerusalem he succeeded his father at age twelve after Nizari Assassins murdered Raymond II. His mother served as regent under King Baldwin III of Jerusalem who oversaw his early years at the royal court. Reaching majority in 1155 Raymond engaged in military campaigns against Nur ad-Din the Zengid ruler of Damascus. In 1161 he retaliated against Byzantine Emperor Manuel I Komnenos for rejecting his sister Melisende’s marriage by hiring pirates to raid Byzantine territories. Captured in the 1164 Battle of Harim he was imprisoned in Aleppo for nearly a decade with King Amalric I managing Tripoli in his absence.

Released in 1172 after a hefty ransom from the Knights Hospitaller Raymond married Eschiva of Bures becoming Prince of Galilee and one of Jerusalem’s wealthiest nobles. After Amalric’s 1174 death Raymond became regent for the young leper king Baldwin IV. He maintained neutrality in conflicts between Nur ad-Din’s successors and Saladin aiding Saladin’s unification of Egypt and Syria. His regency ended in 1176 when Baldwin IV came of age but Raymond continued influencing Jerusalem’s politics opposing Baldwin’s mother Agnes of Courtenay and her allies. In 1180 Baldwin exiled him briefly but Raymond returned as regent for Baldwin V from 1185 to 1186. After Baldwin V’s death he opposed Guy of Lusignan’s coronation retreating to Tiberias. Reconciling with Guy to face Saladin’s 1187 invasion Raymond survived the disastrous Battle of Hattin but died soon after in Tripoli possibly of pleurisy leaving no heirs. His godson Raymond IV succeeded him.

*xaiArtifact artifact_id="f3b7e8a5-1c2d-4e8f-b9c3-7d8e4a1b2f3e" title="Summary of Raymond III Count of Tripoli" contentType="text/markdown"*

*Early Life and Accession*
Raymond III was born in 1140 to Raymond II and Hodierna of Jerusalem. His father’s assassination by Nizari Assassins in 1152 left him count at twelve. Baldwin III appointed Hodierna regent ignoring Raymond II’s wishes for a Hospitaller official. Raymond spent his youth in Jerusalem’s court likely under Baldwin’s mentorship.

*Regencies and Political Influence*
As Baldwin IV’s closest male relative Raymond served as regent from 1174 to 1176 and for Baldwin V from 1185 to 1186. His neutrality in Muslim conflicts allowed Saladin’s rise but he opposed Agnes of Courtenay’s faction. Exiled briefly in 1180 he later resisted Guy of Lusignan’s rule.

*Conflict with Saladin and Death*
Raymond reconciled with Guy to counter Saladin’s 1187 invasion. Surviving Hattin he retired to Tripoli dying in September or October 1187. His death without heirs led to Raymond IV’s succession under Bohemond III’s influence.

*/xaiArtifact*
""".strip()
    print(compute_score(solution_str, '0'))