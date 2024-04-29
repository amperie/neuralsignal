import re


def wrap_with_prompt(prompt_template: str, values: dict) -> str:
    """Creates a prompt from the template and values

    Args:
        prompt_template (str): Template for the prompt which contains
        placeholders for values in the form of {name}
        values (dict): contains the values to replace the placeholders.
        Should be of the form {"name": value}
        Example:
        "prompt": "From this question {input} tell me if this
            output is correct: {output}"

    Returns:
        str: fully formed prompt
    """
    retVal = prompt_template
    vals = re.findall(r'\{.*?\}', prompt_template)
    for v in vals:
        val = v[1:-1]
        retVal = retVal.replace(v, values[val])
    return retVal
