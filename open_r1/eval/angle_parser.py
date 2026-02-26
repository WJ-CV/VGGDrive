import re
import ast

def extract_and_parse_token(text, token_left, token_right):
    pattern = token_left + r".*?" + token_right
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        return False, ''
    try:
        content = match.group()[len(token_left): -len(token_right)].strip()
    except Exception as e:
        print(e)
        return False, ''
    return True, content

class AngleParserFilter(object):
    def __init__(self, **kwargs) -> None:
        pass

    def apply(self, response, **kwargs):
        if isinstance(response, list):
            response = response[0]

        content = response.strip()
        preds = {}
        for k in ['lateral_control', 'longitudinal_control', 'trajectory']:
            preds[k] = ''
            try:
                is_extracted, preds[k] = extract_and_parse_token(content, f'<{k}>', f'</{k}>')
            except Exception as e:
                print(f'[Error] {k}', e)
        if len(preds['trajectory']) > 0:
            try:
                preds['trajectory'] = ast.literal_eval(preds['trajectory'])
                preds['trajectory'] = [] if not isinstance(preds['trajectory'], list) else preds['trajectory'][:6]
            except Exception as e:
                print(f'[Error] trajectory ast.literal_eval', e)

        return {
            'filtered_response': preds,
            'is_filtered': True if len(preds['trajectory']) > 0 else False
        }
