import re
from collections import Counter
from traceback import print_exc

from loguru import logger
from openai import BadRequestError

from api.api_db_client import get_actions_api
from common.common_utils import colorful, save_to_json
from common.constant import LLM_FINETUNING_SERVER_MAP

SearchNodes, SearchGraphPatterns, ExecuteSPARQL = None, None, None


def rm_duplicate_actions(content: str):
    """
    if multi \nAction: in content, only keep the first one.
    """
    if "\nAction:" in content:
        _cs = content.split("\nAction: ")
        content = "\nAction: ".join(_cs[:2])
    return content.strip()


def parse_action(text: str, db: str = None, execute=False):
    """
    Handling single quotes
        "Critics' Choice Movie Award"
        "Man of Steel's narrative location"
    """
    try:
        text = text.strip()

        if "Action:" not in text:
            return "No valid action found. Please provide an action after `Action:` ."
        action_str = text.split("Action:")
        if len(action_str) < 2:
            return "No valid action found. Please provide an action after `Action:` ."

        # find the start index of SearchNodes, SearchGraphPatterns, ExecuteSPARQL, Done
        action_str = action_str[1]
        _s = [action_str.find(i) for i in ["SearchNodes", "SearchGraphPatterns", "ExecuteSPARQL", "Done"]]
        _s = min([i for i in _s if i != -1])
        if not _s:
            raise
        action_str = action_str[_s:].strip()
        if action_str.startswith("Done"):
            return "Done"

        # Handling single quotes
        if "\\'" not in action_str:
            action_str = action_str.replace("' ", "\\' ").replace("'s ", "\\'s ")
        # Handling \_
        action_str = action_str.replace("\\_", "_")

        # find the last ) as the end.
        action_str = action_str[: action_str.rfind(")") + 1]

        if execute:
            assert db is not None, "db must be provided when execute=True"
            global SearchNodes, SearchGraphPatterns, ExecuteSPARQL
            if SearchNodes is None:
                logger.info(f"init actions for db: {db}")
                SearchNodes, SearchGraphPatterns, ExecuteSPARQL = get_actions_api(db)

            # may time out, return None
            obs = eval(action_str)
            return obs
        else:
            return action_str
    except Exception as e:
        print_exc()
        err = "Action parsing error, action must be one of [SearchNodes, SearchGraphPatterns, ExecuteSPARQL, Done], One at a time."
        logger.error(f"Action parsing error. Raw: {text}")
        print(e)
        return err


def _is_valid_action(action_str):
    if (
        not action_str
        or action_str.startswith("Action parsing error")
        or action_str.startswith("No valid action found")
        or action_str.startswith("Done")
    ):
        return False
    return True


def extract_predicates(sparql_query, db):
    assert db in ["kqapro", "fb"], "db must be one of [kqapro, fb]"
    if db == "kqapro":
        matches = re.compile(r"<([^\s]*?)>").findall(sparql_query)
        matches = [i for i in matches if not i.startswith("pred:")]
        for k in ["UNION", "LIMIT"]:
            if k in sparql_query:
                matches.append(k)

    elif db == "fb":
        matches = re.compile(r"ns:([^\s]*?) ").findall(sparql_query)
        matches = [i for i in matches if i and i != "type.object.name"]

    matches = "; ".join(sorted(set(matches), key=matches.index))
    return matches


def extract_entity(sparql):
    """
    use regex to extract:
        'PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE\n{\n?c ns:film.film.other_crew ?k .\n?e0 ns:type.object.name "Barry Angus"@en .\n?k ns:film.film_crew_gig.crewmember ?e0 .\n?c ns:film.film.starring ?y .\n?y ns:film.performance.actor ?x .\n?e1 ns:type.object.name "Darth Vader"@en .\n?y ns:film.performance.character ?e1 .\n}'
        -> ['Barry Angus', 'Darth Vader']
    """
    # after "ns:type.object.name" and before "@en"
    matches = re.compile(r'type.object.name "(.*?)"@en').findall(sparql)
    return matches


def self_consistency_for_action(choices):
    """
    return: ranked (content, action)
    """
    actions = [parse_action(i, execute=False) for i in choices]
    valid_actions = [i for i in actions if _is_valid_action(i)]
    actions_counter = Counter(valid_actions).most_common()

    # rank actions by actions_counter
    ranked_choicess = []
    for action, _count in actions_counter:
        for choice in choices:
            if action in choice:
                ranked_choicess.append(choice)
                break
    return ranked_choicess


def chat_with_LLM(
    d: dict,
    db: str,
    model_name: str,
    save_dir: str = None,
    tooldesc_demos: str = None,
    max_round_num: int = 12,
    entity=None,
):
    """
    different between apis:
    same:
        - messages
        - temperature
        - top_p
        - stop
    chatgpt:
        - max_tokens
        - n
        - presence_penalty
        - frequency_penalty
    llama:
        "max_new_tokens": 512,
        "do_sample": True,
        "repetition_penalty": 1,
        "num_return_sequences": 1,
    """
    assert "id" in d, "id must be provided."
    assert "question" in d, "question must be provided."
    assert save_dir is not None, "save_dir must be provided."

    if model_name.startswith("gpt-"):
        from tool.openai import chatgpt

        _llm_func = chatgpt
        config = {
            "max_tokens": 384,
            "n": 6,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
        }

    # for local LLM server.
    elif model_name in LLM_FINETUNING_SERVER_MAP:
        if model_name.startswith("LLMs/"):
            from api.api_llm_client import local_vLLM_api as _llm_func

            logger.warning("Using vLLM as the LLM server.")
        else:
            from api.api_llm_client import local_LLM_api as _llm_func

        config = {
            "max_new_tokens": 384,
            "do_sample": True,
            "repetition_penalty": 1,
            "num_return_sequences": 6,
        }
    else:
        raise ValueError(f"model_name: {model_name} is not supported.")

    question = d["question"].strip()

    # for no fine-tuned model, add tool desc.
    if tooldesc_demos:
        tooldesc_demos = tooldesc_demos.strip()
        if entity and entity == "golden":
            assert db == "fb", "golden entity only support fb db."
            # add: Golden entity: ['xxx']
            entity_list = extract_entity(d["sparql"])
            entity_line = f"Golden entity: {entity_list}"
            start_text = tooldesc_demos + f"\n\nQ: {question}\n{entity_line}\nThought:"
        else:
            start_text = tooldesc_demos + f"\n\nQ: {question}\nThought:"

    # for fine-tuned model, api server will add tool desc.
    else:
        start_text = f"Q: {question}"

    # mistral chat does not support system role.
    if model_name == "LLMs/mistralai/Mistral-7B-Instruct-v0.2":
        messages = [{"role": "user", "content": start_text}]
    else:
        messages = [
            {"role": "system", "content": "You are an AI assistant."},
            {"role": "user", "content": start_text},
        ]

    round_id = 0
    _last_out = ""

    # info
    completion_tokens = []
    prompt_tokens = []

    # history: all inp and out
    history = []

    while round_id < max_round_num:
        if round_id > 0:
            logger.debug(f"round_id: {round_id}")

        try:
            response = _llm_func(
                model=model_name,
                db=db,
                messages=messages,
                stop=["\nObservation", "\nThought"],
                temperature=0.7,  # self-consistency:0.5 chat-KBQA:0.7
                **config,
            )
        except BadRequestError as e:
            logger.error(f"BadRequestError: {e}")
            return
        if response is None:
            logger.error(f"response is None. id: {d['id']}")
            return
        if "usage" not in response:
            logger.error(response["error"])
            return

        prompt_tokens.append(response["usage"]["prompt_tokens"])
        completion_tokens.append(response["usage"]["completion_tokens"])

        # prepare choices
        choices = [r["message"]["content"].strip() for r in response["choices"]]
        choices = [c.split("Observation")[0] for c in choices if c]

        # add self-consistency
        ranked_choicess = self_consistency_for_action(choices)

        history.append({"round_id": round_id, "choices": choices})

        # Attempt to execute the first valid action in the actions.
        out_thought_action = choices[0].strip()
        Observation = parse_action(out_thought_action, db=db, execute=True)

        # time out
        if Observation is None:
            return

        for content in ranked_choicess:
            _obs = parse_action(content, db=db, execute=True)
            if _obs and "Error:" not in _obs:
                Observation = _obs
                out_thought_action = content.strip()
                break
        Observation = str(Observation).strip()

        if out_thought_action == _last_out:
            messages.append({"role": "user", "content": "STOP because of repetition."})
            break
        _last_out = out_thought_action

        if not out_thought_action.startswith("Thought: "):
            out_thought_action = "Thought: " + out_thought_action
        messages.append({"role": "assistant", "content": out_thought_action})

        # debug
        print()
        print(colorful("LLM: ", color="yellow"), end="")
        print(out_thought_action.replace("\n", "\\n"))
        print(colorful("Observation: ", color="yellow"), end="")
        print(Observation)

        if Observation != "Done":
            messages.append(
                {
                    "role": "user",
                    "content": "Observation: " + Observation,
                }
            )
            round_id += 1
        else:
            messages.append({"role": "user", "content": "Stop condition detected."})
            break

    d["dialog"] = messages

    # add model info
    d["model_name"] = model_name
    d["completion_tokens"] = completion_tokens
    d["prompt_tokens"] = prompt_tokens

    save_to_json(d, f"{save_dir}/{d['id']}.json")
    if "gpt-" in model_name:
        save_to_json(history, f"{save_dir}/{d['id']}.history", _print=False)


if __name__ == "__main__":
    # python tools/action_execution.py
    x = """Raw: To find the location of the Battle of Shiloh and then filter locations with a population larger than 5703719, I need to Search for the node representing the Battle of Shiloh.\nAction: SearchNodes("Battle of Shiloh")"""
    print(parse_action(x, db="fb", execute=True))
